import { useState, useEffect } from "react";
import { Link } from "react-router-dom";

export default function Image() {
  const [modalOpen, setModalOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [heatmapImage, setHeatmapImage] = useState(null);

  // map each model to the input size the backend expects
  const MODEL_INPUT_SIZES = {
    brain: [150, 150], // deploy4
    lung: [224, 224], // deploy1
    lung_size: [256, 256], // deploy3 (nodule size)
    skin: [224, 224], // deploy12
    pancreas: [256, 256], // deploy13
  };

  const models = [
    { name: "Brain Tumor", value: "brain", image: "/images/brain.jpg" },
    { name: "Lung Cancer", value: "lung", image: "/images/lung.jpg" },
    {
      name: "Lung Tumor Size",
      value: "lung_size",
      image: "/images\\lung-cancer.jpg",
    },
    { name: "Skin Tumor", value: "skin", image: "/images/skin.jpg" },
    {
      name: "Pancreas Tumor",
      value: "pancreas",
      image: "/images/pancreas.jpg",
    },
  ];

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  // Resize a File client-side to target width/height using canvas (center-crop -> fill)
  const resizeImageFile = (file, targetW, targetH) => {
    return new Promise((resolve, reject) => {
      const img = new window.Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = targetW;
        canvas.height = targetH;
        const ctx = canvas.getContext("2d");

        // cover (center-crop) while preserving aspect ratio
        const scale = Math.max(targetW / img.width, targetH / img.height);
        const sw = targetW / scale;
        const sh = targetH / scale;
        const sx = Math.max(0, (img.width - sw) / 2);
        const sy = Math.max(0, (img.height - sh) / 2);

        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, targetW, targetH);
        // Always output a JPEG from canvas (broader browser support). If the
        // browser fails to create the blob for any reason, try PNG as a fallback.
        canvas.toBlob(
          (blob) => {
            if (blob) {
              const outType = "image/jpeg";
              const newName = file.name.replace(/\.[^.]+$/, ".jpg");
              const resizedFile = new File([blob], newName, { type: outType });
              resolve(resizedFile);
              return;
            }

            // Fallback: try PNG then JPEG
            canvas.toBlob(
              (b2) => {
                if (!b2) return reject(new Error("Canvas toBlob failed"));
                const newName = file.name.replace(/\.[^.]+$/, ".jpg");
                resolve(new File([b2], newName, { type: "image/jpeg" }));
              },
              "image/png",
              0.92,
            );
          },
          "image/jpeg",
          0.92,
        );
      };
      img.onerror = (err) => reject(err);
      img.src = URL.createObjectURL(file);
    });
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPrediction(null);
      setHeatmapImage(null);
      // show original preview (before resizing) so user can confirm image
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
    }
  };

  const handlePredict = async () => {
    if (!file || !selectedModel) {
      alert("Please select a file and a model!");
      return;
    }

    const [w, h] = MODEL_INPUT_SIZES[selectedModel] || [224, 224];

    setLoading(true);
    try {
      const resized = await resizeImageFile(file, w, h);

      const formData = new FormData();
      formData.append("file", resized, resized.name);
      formData.append("model_name", selectedModel);

      const response = await fetch("http://127.0.0.1:9001/predict/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errBody = null;
        try {
          errBody = await response.json();
        } catch (e) {
          /* ignore */
        }
        const msg =
          errBody?.error ||
          errBody?.detail ||
          `Prediction failed (status ${response.status})`;
        throw new Error(msg);
      }

      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setPrediction(`Prediction: ${data.prediction}`);
    } catch (err) {
      console.error("Predict error:", err);
      alert(`Error processing the image: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleLocate = async () => {
    if (!file || !selectedModel) {
      alert("Please select a file and a model before locating!");
      return;
    }

    const [w, h] = MODEL_INPUT_SIZES[selectedModel] || [224, 224];

    setLoading(true);
    try {
      const resized = await resizeImageFile(file, w, h);

      const formData = new FormData();
      formData.append("file", resized, resized.name);
      formData.append("model_name", selectedModel);

      const response = await fetch("http://127.0.0.1:9001/gradcam/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errBody = null;
        try {
          errBody = await response.json();
        } catch (e) {
          /* ignore */
        }
        const msg =
          errBody?.error ||
          errBody?.detail ||
          `Grad-CAM request failed (status ${response.status})`;
        throw new Error(msg);
      }

      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setHeatmapImage(data.heatmap_image);
    } catch (err) {
      console.error("Grad-CAM error:", err);
      alert(`Error generating heatmap: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCloseModal = () => {
    setModalOpen(false);
    setFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setHeatmapImage(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black to-gray-900 p-8 flex flex-col items-center justify-center relative">
      <Link to="/">
        <div className="absolute top-6 left-8 flex items-center cursor-pointer">
          <img
            src="/images/logo1.jpg"
            alt="logo"
            className="h-10 w-10 max-w-[40px] object-contain"
          />
          <span className="text-2xl font-bold text-white">ragon.</span>
        </div>
      </Link>

      <Link
        to="/"
        className="absolute top-6 right-6 text-white px-6 py-2 rounded-lg text-lg font-semibold shadow-md border border-white 
        hover:bg-white hover:text-black transition duration-300"
      >
        Home
      </Link>

      <h1
        className="text-5xl font-semibold text-white text-center mb-10 tracking-tight 
        drop-shadow-[0_0_20px_rgba(255,255,255,0.7)] animate-pulse"
      >
        Image Models
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {models.map((model) => (
          <button
            key={model.value}
            onClick={() => {
              setSelectedModel(model.value);
              setModalOpen(true);
            }}
            className="bg-gray-900 rounded-xl shadow-md p-4 flex flex-col items-center justify-center
            w-52 h-60 border border-gray-700 transition-transform duration-300 hover:-translate-y-2"
          >
            <img
              src={model.image}
              alt={model.name}
              className="w-32 max-w-full h-auto rounded-lg mb-3 shadow-md object-contain"
            />
            <p className="text-white text-xl font-semibold">{model.name}</p>
          </button>
        ))}
      </div>

      {modalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-gray-800 p-6 rounded-xl shadow-2xl w-[450px] text-white">
            <h2 className="text-2xl font-bold mb-6 text-center">
              Upload Image for{" "}
              {models.find((m) => m.value === selectedModel)?.name}
            </h2>

            <div className="border-dashed border-2 border-gray-500 rounded-lg p-6 text-center bg-gray-900">
              <p className="text-gray-400 text-lg">Drag and Drop</p>
              <p className="text-gray-500 text-sm">or</p>
              <label
                htmlFor="file-upload"
                className="cursor-pointer bg-purple-600 text-white px-3 py-0.5 rounded text-md hover:bg-purple-500 transition mt-4 inline-block"
              >
                Browse file
              </label>
              <input
                id="file-upload"
                type="file"
                className="hidden"
                onChange={handleFileChange}
              />
              {file && (
                <p className="mt-3 text-gray-300 text-lg">
                  Selected file: {file.name}
                </p>
              )}

              {/* Preview + resized-size hint */}
              {previewUrl && (
                <div className="mt-4">
                  <img
                    src={previewUrl}
                    alt="preview"
                    className="mx-auto w-64 max-w-full h-auto object-contain rounded-md shadow"
                  />
                  <p className="text-sm text-gray-400 mt-2">
                    Image will be resized and converted client-side to{" "}
                    <strong>
                      {(MODEL_INPUT_SIZES[selectedModel] || [224, 224]).join(
                        "×",
                      )}{" "}
                      (JPEG)
                    </strong>{" "}
                    before upload.
                  </p>
                </div>
              )}
            </div>

            <div className="flex justify-end mt-6 space-x-2">
              <button
                className="px-4 py-2 bg-red-500 text-white rounded-lg text-lg hover:bg-red-400 transition"
                onClick={handleCloseModal}
              >
                Close
              </button>
              <button
                className="px-4 py-2 bg-blue-500 text-white rounded-lg text-lg hover:bg-blue-400 transition"
                onClick={handleLocate}
                disabled={loading}
              >
                {loading ? "Locating..." : "Locate"}
              </button>
              <button
                className="px-4 py-2 bg-green-500 text-white rounded-lg text-lg hover:bg-green-400 transition"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading ? "Processing..." : "Predict"}
              </button>
            </div>

            {prediction && (
              <div className="mt-4 p-4 bg-gray-700 rounded-lg">
                <h3 className="text-lg font-semibold">Prediction Result:</h3>
                <p className="text-gray-300 text-lg">{prediction}</p>
              </div>
            )}

            {heatmapImage && (
              <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-75 z-50">
                <div className="bg-white p-4 rounded-lg shadow-lg max-w-2xl">
                  <img
                    src={heatmapImage}
                    alt="Tumor Heatmap"
                    className="max-w-full h-auto"
                  />
                  <button
                    className="mt-4 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-400 transition"
                    onClick={() => setHeatmapImage(null)}
                  >
                    Close
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
