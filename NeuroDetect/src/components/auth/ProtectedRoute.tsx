import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { getDefaultRouteByRole, useAuth } from './AuthContext';
import type { UserRole } from '../../auth/types';

interface ProtectedRouteProps {
  allowedRoles?: UserRole[];
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ allowedRoles }) => {
  const { isAuthenticated, currentUser } = useAuth();

  if (!isAuthenticated || !currentUser) {
    return <Navigate to="/login" replace />;
  }

  if (allowedRoles && !allowedRoles.includes(currentUser.role)) {
    return <Navigate to={getDefaultRouteByRole(currentUser.role)} replace />;
  }

  return <Outlet />;
};

export const PublicOnlyRoute: React.FC = () => {
  const { currentUser } = useAuth();
  if (currentUser) {
    return <Navigate to={getDefaultRouteByRole(currentUser.role)} replace />;
  }
  return <Outlet />;
};

export default ProtectedRoute;
