import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import {
  authenticateUser,
  clearSession,
  fetchCurrentUser,
  getSessionToken,
  getSessionUser,
  logoutUser,
  saveSession,
} from '../../auth/storage';
import type { AuthUser, UserRole } from '../../auth/types';

interface LoginResult {
  success: boolean;
  user?: AuthUser;
  error?: string;
}

interface AuthContextValue {
  currentUser: AuthUser | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<LoginResult>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export const getDefaultRouteByRole = (role: UserRole): string => {
  switch (role) {
    case 'admin':
      return '/user-management';
    case 'analyst':
      return '/';
    case 'viewer':
      return '/reports';
    default:
      return '/login';
  }
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(() => getSessionUser());

  useEffect(() => {
    const token = getSessionToken();
    if (!token) {
      return;
    }

    fetchCurrentUser(token)
      .then((user) => {
        setCurrentUser(user);
        saveSession(user, token);
      })
      .catch(() => {
        setCurrentUser(null);
        clearSession();
      });
  }, []);

  const login = useCallback(async (email: string, password: string): Promise<LoginResult> => {
    try {
      const payload = await authenticateUser(email, password);
      setCurrentUser(payload.user);
      saveSession(payload.user, payload.token);
      return { success: true, user: payload.user };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Login failed',
      };
    }
  }, []);

  const logout = () => {
    const token = getSessionToken();
    if (token) {
      logoutUser(token).catch(() => {
        // Ignore API logout failure and clear local session anyway
      });
    }
    setCurrentUser(null);
    clearSession();
  };

  const value = useMemo(
    () => ({
      currentUser,
      isAuthenticated: Boolean(currentUser),
      login,
      logout,
    }),
    [currentUser],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextValue => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used inside AuthProvider');
  }
  return context;
};
