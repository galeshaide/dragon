import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Simple dataset
data = {
    "BMI": [22, 30, 35, 28, 40, 25, 27],
    "HighBP": [0,1,1,0,1,0,0],
    "Age": [25,45,50,35,60,30,40],
    "Diabetes": [0,1,1,0,1,0,0]
}

df = pd.DataFrame(data)

X = df[["BMI","HighBP","Age"]]
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

joblib.dump(model,"diabetes_model.pkl")

print("✅ Diabetes model created")