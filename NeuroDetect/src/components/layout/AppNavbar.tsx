import React, { useEffect, useMemo, useRef, useState } from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { Activity, ChevronDown, LogOut, Moon, Play, RefreshCw, Square, Sun, User } from 'lucide-react';
import { useModelNavbar } from './ModelNavbarContext';
import type { NavbarThemeConfig } from './ModelNavbarContext';
import { useTheme } from '../theme/ThemeContext';
import { useAuth } from '../auth/AuthContext';

type NavItem = {
  to: string;
  label: string;
  match?: string[];
};

const navItems: NavItem[] = [
  { to: '/', label: 'Dashboard', match: ['/', '/lstmreal', '/snnreal'] },
  { to: '/streaming', label: 'Streaming' },
  { to: '/batch-upload', label: 'Batch Upload' },
  { to: '/snn-alerts', label: 'Alerts' },
  { to: '/reports', label: 'Reports' },
  { to: '/audit', label: 'Audit' },
  { to: '/user-management', label: 'Users' },
];

const getCurrentLabel = (pathname: string) => {
  const item = navItems.find((entry) => entry.match?.includes(pathname) || entry.to === pathname);
  return item?.label || 'Workspace';
};

const modelLabels = {
  autoencoder: 'Autoencoder',
  lstm: 'LSTM',
  snn: 'SNN',
} as const;

const defaultTheme: NavbarThemeConfig = {
  headerBackground: '#020617',
  panelBackground: '#0f172a',
  borderColor: 'rgba(148, 163, 184, 0.18)',
  textPrimary: '#f8fafc',
  textMuted: '#94a3b8',
  accent: '#8b5cf6',
  success: '#10b981',
  danger: '#ef4444',
  shadow: '0 16px 40px rgba(2, 6, 23, 0.36)',
};

const AppNavbar: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { navbarConfig } = useModelNavbar();
  const { isDarkTheme, toggleTheme } = useTheme();
  const { currentUser, logout } = useAuth();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);
  const currentLabel = getCurrentLabel(location.pathname);
  const isModelRoute = ['/', '/lstmreal', '/snnreal'].includes(location.pathname);
  const theme = useMemo(
    () => ({
      ...{
        ...defaultTheme,
        headerBackground: isDarkTheme ? '#020617' : '#eef2f7',
        panelBackground: isDarkTheme ? '#0f172a' : '#ffffff',
        borderColor: isDarkTheme ? 'rgba(148, 163, 184, 0.18)' : 'rgba(203, 213, 225, 0.88)',
        textPrimary: isDarkTheme ? '#f8fafc' : '#0f172a',
        textMuted: isDarkTheme ? '#94a3b8' : '#64748b',
        shadow: isDarkTheme ? '0 16px 40px rgba(2, 6, 23, 0.36)' : '0 16px 38px rgba(148, 163, 184, 0.16)',
      },
      ...(navbarConfig?.theme || {}),
    }),
    [isDarkTheme, navbarConfig?.theme],
  );

  useEffect(() => {
    setDropdownOpen(false);
    setProfileOpen(false);
  }, [location.pathname, navbarConfig?.selectedModel]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (profileRef.current && !profileRef.current.contains(e.target as Node)) {
        setProfileOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const initials = currentUser
    ? currentUser.name
        .split(' ')
        .map((n) => n[0])
        .join('')
        .toUpperCase()
        .slice(0, 2)
    : '?';

  return (
    <header
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 3000,
        padding: '18px 24px 0',
        background: theme.headerBackground,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '18px',
          width: '100%',
          margin: '0 auto',
          padding: '14px 20px',
          borderRadius: '18px',
          background: theme.panelBackground,
          border: `1px solid ${theme.borderColor}`,
          boxShadow: theme.shadow,
          flexWrap: 'wrap',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '24px',
            minWidth: 0,
            flex: 1,
          }}
        >
          <NavLink
            to="/"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              color: theme.textPrimary,
              textDecoration: 'none',
              fontWeight: 800,
              fontSize: '1rem',
              letterSpacing: '-0.02em',
              whiteSpace: 'nowrap',
            }}
          >
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '38px',
                height: '38px',
                borderRadius: '12px',
                background: `linear-gradient(135deg, ${theme.accent}, #2563eb)`,
                color: '#ffffff',
                boxShadow: '0 12px 28px rgba(59, 130, 246, 0.2)',
              }}
            >
              <Activity size={18} />
            </span>
            <span style={{ fontSize: '0.96rem' }}>NeuroDetect</span>
          </NavLink>

          <div
            style={{
              width: '1px',
              height: '28px',
              background: theme.borderColor,
              flexShrink: 0,
            }}
          />

          <div
            style={{
              color: theme.textMuted,
              fontSize: '0.76rem',
              fontWeight: 800,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              whiteSpace: 'nowrap',
            }}
          >
            {currentLabel}
          </div>
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '14px',
            flexShrink: 0,
          }}
        >
          {isModelRoute && navbarConfig && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                flexWrap: 'wrap',
              }}
            >
              <div style={{ position: 'relative' }}>
                <button
                  type="button"
                  onClick={() => setDropdownOpen((open) => !open)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    minWidth: '188px',
                    padding: '10px 14px',
                    borderRadius: '12px',
                    border: `1px solid ${theme.borderColor}`,
                    background: 'rgba(15, 23, 42, 0.24)',
                    color: theme.textPrimary,
                    cursor: 'pointer',
                    fontSize: '0.9rem',
                    fontWeight: 700,
                  }}
                >
                  <span
                    style={{
                      width: '9px',
                      height: '9px',
                      borderRadius: '999px',
                      background:
                        navbarConfig.connectionStatus === 'connected'
                          ? theme.success
                          : navbarConfig.connectionStatus === 'connecting'
                            ? '#f59e0b'
                            : theme.danger,
                    }}
                  />
                  <span style={{ flex: 1, textAlign: 'left' }}>{modelLabels[navbarConfig.selectedModel]}</span>
                  <ChevronDown
                    size={16}
                    color={theme.textMuted}
                    style={{ transform: dropdownOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s ease' }}
                  />
                </button>

                {dropdownOpen && (
                  <div
                    style={{
                      position: 'absolute',
                      top: 'calc(100% + 8px)',
                      left: 0,
                      right: 0,
                      background: theme.panelBackground,
                      border: `1px solid ${theme.borderColor}`,
                      borderRadius: '12px',
                      boxShadow: theme.shadow,
                      overflow: 'hidden',
                    }}
                  >
                    {navbarConfig.availableModels.map((model) => (
                      <button
                        key={model}
                        type="button"
                        onClick={() => {
                          navbarConfig.onSelectModel(model);
                          setDropdownOpen(false);
                        }}
                        style={{
                          width: '100%',
                          padding: '10px 14px',
                          border: 'none',
                          textAlign: 'left',
                          background: navbarConfig.selectedModel === model ? 'rgba(139, 92, 246, 0.16)' : 'transparent',
                          color: theme.textPrimary,
                          fontSize: '0.9rem',
                          cursor: 'pointer',
                        }}
                      >
                        {modelLabels[model]}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '10px 14px',
                  borderRadius: '12px',
                  background: 'rgba(15, 23, 42, 0.24)',
                  border: `1px solid ${theme.borderColor}`,
                  color: theme.textMuted,
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  whiteSpace: 'nowrap',
                }}
              >
                <span
                  style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '999px',
                    background: theme.success,
                    boxShadow: `0 0 0 4px ${theme.success}22`,
                  }}
                />
                <span>{navbarConfig.accuracyText}</span>
              </div>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  paddingLeft: '6px',
                }}
              >
                <button
                  type="button"
                  aria-label="Start streaming"
                  disabled={!navbarConfig.isConnected || navbarConfig.isStreaming}
                  onClick={navbarConfig.onStart}
                  style={{
                    border: `1px solid ${theme.borderColor}`,
                    background: 'rgba(16, 185, 129, 0.12)',
                    color: theme.success,
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '38px',
                    height: '38px',
                    borderRadius: '10px',
                    cursor: !navbarConfig.isConnected || navbarConfig.isStreaming ? 'not-allowed' : 'pointer',
                    opacity: !navbarConfig.isConnected || navbarConfig.isStreaming ? 0.45 : 1,
                  }}
                >
                  <Play size={16} />
                </button>
                <button
                  type="button"
                  aria-label="Stop streaming"
                  disabled={!navbarConfig.isConnected || !navbarConfig.isStreaming}
                  onClick={navbarConfig.onStop}
                  style={{
                    border: `1px solid ${theme.borderColor}`,
                    background: 'rgba(239, 68, 68, 0.12)',
                    color: theme.danger,
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '38px',
                    height: '38px',
                    borderRadius: '10px',
                    cursor: !navbarConfig.isConnected || !navbarConfig.isStreaming ? 'not-allowed' : 'pointer',
                    opacity: !navbarConfig.isConnected || !navbarConfig.isStreaming ? 0.45 : 1,
                  }}
                >
                  <Square size={14} />
                </button>
                <button
                  type="button"
                  aria-label="Reset streaming"
                  onClick={navbarConfig.onReset}
                  style={{
                    border: `1px solid ${theme.borderColor}`,
                    background: 'rgba(148, 163, 184, 0.12)',
                    color: theme.textMuted,
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '38px',
                    height: '38px',
                    borderRadius: '10px',
                    cursor: 'pointer',
                  }}
                >
                  <RefreshCw size={15} />
                </button>
              </div>
            </div>
          )}

          <button
            type="button"
            onClick={toggleTheme}
            aria-label={isDarkTheme ? 'Switch to light mode' : 'Switch to dark mode'}
            style={{
              border: `1px solid ${theme.borderColor}`,
              background: isDarkTheme ? 'rgba(148, 163, 184, 0.12)' : '#f8fafc',
              color: theme.textPrimary,
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '38px',
              height: '38px',
              borderRadius: '10px',
              cursor: 'pointer',
            }}
          >
            {isDarkTheme ? <Sun size={16} /> : <Moon size={16} />}
          </button>

          {/* User profile dropdown */}
          <div ref={profileRef} style={{ position: 'relative' }}>
            <button
              type="button"
              onClick={() => setProfileOpen((o) => !o)}
              aria-label="User profile"
              title={currentUser?.name}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '0 12px 0 4px',
                height: '38px',
                borderRadius: '10px',
                border: `1px solid ${theme.borderColor}`,
                background: isDarkTheme ? 'rgba(139, 92, 246, 0.14)' : 'rgba(139, 92, 246, 0.08)',
                color: theme.textPrimary,
                cursor: 'pointer',
              }}
            >
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '28px',
                  height: '28px',
                  borderRadius: '8px',
                  background: `linear-gradient(135deg, ${theme.accent}, #2563eb)`,
                  color: '#fff',
                  fontSize: '0.7rem',
                  fontWeight: 800,
                  letterSpacing: '0.02em',
                }}
              >
                {initials}
              </span>
              <span style={{ fontSize: '0.82rem', fontWeight: 600, maxWidth: '90px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {currentUser?.name ?? 'Account'}
              </span>
              <ChevronDown size={13} color={theme.textMuted} style={{ transform: profileOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s ease' }} />
            </button>

            {profileOpen && (
              <div
                style={{
                  position: 'absolute',
                  top: 'calc(100% + 10px)',
                  right: 0,
                  minWidth: '210px',
                  background: theme.panelBackground,
                  border: `1px solid ${theme.borderColor}`,
                  borderRadius: '14px',
                  boxShadow: theme.shadow,
                  overflow: 'hidden',
                  zIndex: 4000,
                }}
              >
                {/* Header */}
                <div style={{ padding: '14px 16px 12px', borderBottom: `1px solid ${theme.borderColor}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: '36px',
                        height: '36px',
                        borderRadius: '10px',
                        background: `linear-gradient(135deg, ${theme.accent}, #2563eb)`,
                        color: '#fff',
                        fontSize: '0.82rem',
                        fontWeight: 800,
                        flexShrink: 0,
                      }}
                    >
                      {initials}
                    </span>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: theme.textPrimary, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {currentUser?.name}
                      </div>
                      <div style={{ fontSize: '0.74rem', color: theme.textMuted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {currentUser?.email}
                      </div>
                    </div>
                  </div>
                  <div style={{
                    marginTop: '10px',
                    display: 'inline-block',
                    padding: '2px 10px',
                    borderRadius: '999px',
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    textTransform: 'capitalize',
                    letterSpacing: '0.06em',
                    background: currentUser?.role === 'admin' ? 'rgba(239,68,68,0.14)' : currentUser?.role === 'analyst' ? 'rgba(245,158,11,0.14)' : 'rgba(16,185,129,0.14)',
                    color: currentUser?.role === 'admin' ? '#ef4444' : currentUser?.role === 'analyst' ? '#f59e0b' : '#10b981',
                  }}>
                    {currentUser?.role}
                  </div>
                </div>

                {/* Actions */}
                <div style={{ padding: '6px' }}>
                  <NavLink
                    to="/profile"
                    onClick={() => setProfileOpen(false)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '10px',
                      padding: '9px 12px',
                      borderRadius: '9px',
                      color: theme.textPrimary,
                      textDecoration: 'none',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = isDarkTheme ? 'rgba(148,163,184,0.08)' : 'rgba(0,0,0,0.04)')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    <User size={15} color={theme.textMuted} />
                    View Profile
                  </NavLink>

                  <button
                    type="button"
                    onClick={handleLogout}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '10px',
                      padding: '9px 12px',
                      borderRadius: '9px',
                      border: 'none',
                      background: 'transparent',
                      color: '#ef4444',
                      textAlign: 'left',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(239,68,68,0.08)')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    <LogOut size={15} />
                    Logout
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default AppNavbar;
