import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';

export type NavbarModelType = 'autoencoder' | 'lstm' | 'snn';
export type NavbarConnectionStatus = 'disconnected' | 'connecting' | 'connected';

export interface NavbarThemeConfig {
  headerBackground: string;
  panelBackground: string;
  borderColor: string;
  textPrimary: string;
  textMuted: string;
  accent: string;
  success: string;
  danger: string;
  shadow: string;
}

export interface ModelNavbarConfig {
  selectedModel: NavbarModelType;
  availableModels: NavbarModelType[];
  accuracyText: string;
  isStreaming: boolean;
  isConnected: boolean;
  connectionStatus: NavbarConnectionStatus;
  onSelectModel: (model: NavbarModelType) => void;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  theme?: Partial<NavbarThemeConfig>;
}

interface ModelNavbarContextValue {
  navbarConfig: ModelNavbarConfig | null;
  setNavbarConfig: (config: ModelNavbarConfig | null) => void;
  clearNavbarConfig: () => void;
}

const ModelNavbarContext = createContext<ModelNavbarContextValue | undefined>(undefined);

export const ModelNavbarProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [navbarConfig, setNavbarConfigState] = useState<ModelNavbarConfig | null>(null);

  const setNavbarConfig = useCallback((config: ModelNavbarConfig | null) => {
    setNavbarConfigState(config);
  }, []);

  const clearNavbarConfig = useCallback(() => {
    setNavbarConfigState(null);
  }, []);

  const value = useMemo(
    () => ({
      navbarConfig,
      setNavbarConfig,
      clearNavbarConfig,
    }),
    [clearNavbarConfig, navbarConfig, setNavbarConfig],
  );

  return <ModelNavbarContext.Provider value={value}>{children}</ModelNavbarContext.Provider>;
};

export const useModelNavbar = (): ModelNavbarContextValue => {
  const context = useContext(ModelNavbarContext);
  if (!context) {
    throw new Error('useModelNavbar must be used within ModelNavbarProvider');
  }
  return context;
};