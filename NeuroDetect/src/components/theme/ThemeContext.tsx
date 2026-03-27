import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';

type ThemeMode = 'dark' | 'light';

export interface AppThemePalette {
  bgPrimary: string;
  bgSecondary: string;
  bgTertiary: string;
  bgCard: string;
  bgSidebar: string;
  textPrimary: string;
  textSecondary: string;
  textMuted: string;
  borderColor: string;
  borderLight: string;
}

const THEME_KEY = 'neurodetect_theme_mode';

const darkPalette: AppThemePalette = {
  bgPrimary: '#0f0f23',
  bgSecondary: '#1a1a2e',
  bgTertiary: '#252547',
  bgCard: '#1e1e3f',
  bgSidebar: '#14142b',
  textPrimary: '#ffffff',
  textSecondary: '#cbd5e1',
  textMuted: '#94a3b8',
  borderColor: '#2d2d5a',
  borderLight: '#3a3a6a',
};

const lightPalette: AppThemePalette = {
  bgPrimary: '#f5f7fa',
  bgSecondary: '#ffffff',
  bgTertiary: '#f0f4f8',
  bgCard: '#ffffff',
  bgSidebar: '#f9fafb',
  textPrimary: '#1a202c',
  textSecondary: '#4a5568',
  textMuted: '#718096',
  borderColor: '#cbd5e0',
  borderLight: '#e2e8f0',
};

interface ThemeContextValue {
  mode: ThemeMode;
  isDarkTheme: boolean;
  currentTheme: AppThemePalette;
  toggleTheme: () => void;
  setThemeMode: (mode: ThemeMode) => void;
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [mode, setMode] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem(THEME_KEY);
    return saved === 'light' ? 'light' : 'dark';
  });

  useEffect(() => {
    localStorage.setItem(THEME_KEY, mode);
    document.body.dataset.theme = mode;
    document.documentElement.dataset.theme = mode;
    document.body.style.backgroundColor = mode === 'dark' ? darkPalette.bgPrimary : lightPalette.bgPrimary;
    document.body.style.color = mode === 'dark' ? darkPalette.textPrimary : lightPalette.textPrimary;
  }, [mode]);

  const value = useMemo<ThemeContextValue>(
    () => ({
      mode,
      isDarkTheme: mode === 'dark',
      currentTheme: mode === 'dark' ? darkPalette : lightPalette,
      toggleTheme: () => setMode((prev) => (prev === 'dark' ? 'light' : 'dark')),
      setThemeMode: setMode,
    }),
    [mode],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};

export const useTheme = (): ThemeContextValue => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used inside ThemeProvider');
  }
  return context;
};
