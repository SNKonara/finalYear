import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
/*import './pages/css/Login.css';
import Login from './pages/Login.tsx'; */

/*import App from './pages/dashboard.tsx';
import './pages/css/dashboard.css';  */

import Streaming from './pages/streaming.tsx';
import './pages/css/streaming.css';

import FraudDetectionDashboard from './pages/aereal.tsx';
import './pages/css/aereal.css'; 

import LSTMFraudDetectionDashboard from './pages/lstmreal.tsx';
import './pages/css/lstmreal.css';

import SNNFraudDetectionDashboard from './pages/snnreal.tsx';

import BatchProcessing from './pages/batch_upload.tsx';
import './pages/css/batch_upload.css'; 

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<FraudDetectionDashboard />} />
        <Route path="/lstmreal" element={<LSTMFraudDetectionDashboard />} />
        <Route path="/snnreal" element={<SNNFraudDetectionDashboard />} />
        <Route path="/streaming" element={<Streaming />} />
        <Route path="/batch-upload" element={<BatchProcessing />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
  

/*
ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <Login />
  </React.StrictMode>
);
*/



