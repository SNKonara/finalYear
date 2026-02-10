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

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<FraudDetectionDashboard />} />
        <Route path="/streaming" element={<Streaming />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);

