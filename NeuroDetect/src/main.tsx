import React from 'react';
import ReactDOM from 'react-dom/client';
/*import './pages/css/Login.css';
import Login from './pages/Login.tsx'; */

import App from './pages/dashboard.tsx';
import './pages/css/dashboard.css'; 

/*import Streaming from './pages/streaming.tsx';*/

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

