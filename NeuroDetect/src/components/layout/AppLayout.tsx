import React from 'react';
import { Outlet } from 'react-router-dom';
import AppNavbar from './AppNavbar';
import { ModelNavbarProvider } from './ModelNavbarContext';
import { useTheme } from '../theme/ThemeContext';

const AppLayout: React.FC = () => {
  const { currentTheme } = useTheme();

  return (
    <ModelNavbarProvider>
      <div style={{ minHeight: '100vh', backgroundColor: currentTheme.bgPrimary, color: currentTheme.textPrimary }}>
        <AppNavbar />
        <Outlet />
      </div>
    </ModelNavbarProvider>
  );
};

export default AppLayout;
