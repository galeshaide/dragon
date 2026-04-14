import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Image from './components/Image';
import Report from './components/Report';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/images" element={<Image />} />
        <Route path="/reports" element={<Report />} />
      </Routes>
    </BrowserRouter>
  );
}
