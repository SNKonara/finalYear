import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Navigate, Routes, Route } from 'react-router-dom';

import { AuthProvider } from './auth/AuthContext';
import ProtectedRoute, { PublicOnlyRoute } from './auth/ProtectedRoute';
import { ThemeProvider } from './theme/ThemeContext';
import './theme/theme.css';

import Login from './pages/Login.tsx';
import UnifiedModelsDashboard from './pages/models.tsx';
import '../pages/css/aereal.css';

import SNNAlertsInvestigation from './pages/snn_alerts.tsx';

import Streaming from './pages/streaming.tsx';
import '../pages/css/streaming.css';

import BatchProcessing from './pages/batch_upload.tsx';
import '../pages/css/batch_upload.css';

import Reports from './pages/Reports.tsx';
import AuditDetails from './pages/AuditDetails.tsx';
import UserManagement from './pages/UserManagement.tsx';
import Unauthorized from './pages/Unauthorized.tsx';
import Profile from './pages/Profile.tsx';
import Investigations from './pages/Investigations.tsx';
import InvestigationDetail from './pages/InvestigationDetail.tsx';
import SystemOverview from './pages/SystemOverview.tsx';
import AppLayout from './layout/AppLayout.tsx';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            {/* Public login route */}
            <Route element={<PublicOnlyRoute />}>
              <Route path="/login" element={<Login />} />
            </Route>

            {/* Protected application routes */}
            <Route element={<ProtectedRoute />}>
              <Route element={<AppLayout />}>
                <Route element={<ProtectedRoute allowedRoles={['admin', 'analyst', 'senior_analyst']} />}>
                  <Route path="/" element={<UnifiedModelsDashboard />} />
                  <Route path="/lstmreal" element={<UnifiedModelsDashboard />} />
                  <Route path="/snnreal" element={<UnifiedModelsDashboard />} />
                  <Route path="/streaming" element={<Streaming />} />
                  <Route path="/system" element={<SystemOverview />} />
                </Route>
                <Route element={<ProtectedRoute allowedRoles={['analyst', 'senior_analyst']} />}>
                  <Route path="/snn-alerts" element={<SNNAlertsInvestigation />} />
                  <Route path="/batch-upload" element={<BatchProcessing />} />
                  <Route path="/investigations" element={<Investigations />} />
                  <Route path="/senior-alerts" element={<Investigations />} />
                  <Route path="/investigations/:alertId" element={<InvestigationDetail />} />
                  <Route path="/senior-alerts/:alertId" element={<InvestigationDetail />} />
                </Route>
                <Route path="/reports" element={<Reports />} />
                <Route path="/summary" element={<Reports />} />
                <Route path="/audit" element={<AuditDetails />} />
                <Route element={<ProtectedRoute allowedRoles={['admin']} />}>
                  <Route path="/user-management" element={<UserManagement />} />
                </Route>
                <Route path="/profile" element={<Profile />} />
                <Route path="/unauthorized" element={<Unauthorized />} />
                <Route path="*" element={<Navigate to="/snnreal" replace />} />
              </Route>
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  </React.StrictMode>
);


