import React from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut, Mail, Shield, User } from 'lucide-react';
import { useAuth } from '../auth/AuthContext';
import { useTheme } from '../theme/ThemeContext';

const roleColors: Record<string, { bg: string; text: string }> = {
  admin:   { bg: 'rgba(239,68,68,0.14)',   text: '#ef4444' },
  analyst: { bg: 'rgba(245,158,11,0.14)',  text: '#f59e0b' },
  viewer:  { bg: 'rgba(16,185,129,0.14)',  text: '#10b981' },
};

const Profile: React.FC = () => {
  const { currentUser, logout } = useAuth();
  const { currentTheme, isDarkTheme } = useTheme();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const initials = currentUser
    ? currentUser.name.split(' ').map((n) => n[0]).join('').toUpperCase().slice(0, 2)
    : '?';

  const role = currentUser?.role ?? 'viewer';
  const roleColor = roleColors[role] ?? roleColors.viewer;

  const card: React.CSSProperties = {
    background: currentTheme.bgSecondary,
    border: `1px solid ${currentTheme.borderColor}`,
    borderRadius: '18px',
    padding: '28px',
  };

  const label: React.CSSProperties = {
    fontSize: '0.72rem',
    fontWeight: 700,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: currentTheme.textMuted,
    marginBottom: '4px',
  };

  const value: React.CSSProperties = {
    fontSize: '0.95rem',
    fontWeight: 600,
    color: currentTheme.textPrimary,
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        background: currentTheme.bgPrimary,
        color: currentTheme.textPrimary,
        padding: '40px 24px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      <div style={{ width: '100%', maxWidth: '520px', display: 'flex', flexDirection: 'column', gap: '24px' }}>

        {/* Page title */}
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 800, margin: 0 }}>My Profile</h1>
          <p style={{ margin: '4px 0 0', fontSize: '0.85rem', color: currentTheme.textMuted }}>
            Account details and session information
          </p>
        </div>

        {/* Avatar card */}
        <div style={{ ...card, display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '72px',
              height: '72px',
              borderRadius: '20px',
              background: 'linear-gradient(135deg, #8b5cf6, #2563eb)',
              color: '#fff',
              fontSize: '1.5rem',
              fontWeight: 800,
              flexShrink: 0,
              boxShadow: '0 12px 30px rgba(139,92,246,0.25)',
            }}
          >
            {initials}
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: '1.25rem', fontWeight: 800, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {currentUser?.name ?? '—'}
            </div>
            <div style={{ fontSize: '0.85rem', color: currentTheme.textMuted, marginTop: '2px' }}>
              {currentUser?.email ?? '—'}
            </div>
            <div
              style={{
                marginTop: '8px',
                display: 'inline-block',
                padding: '3px 12px',
                borderRadius: '999px',
                fontSize: '0.72rem',
                fontWeight: 700,
                textTransform: 'capitalize',
                letterSpacing: '0.06em',
                background: roleColor.bg,
                color: roleColor.text,
              }}
            >
              {role}
            </div>
          </div>
        </div>

        {/* Details card */}
        <div style={card}>
          <h2 style={{ margin: '0 0 20px', fontSize: '0.95rem', fontWeight: 700 }}>Account Details</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            <div>
              <div style={label}>Full Name</div>
              <div style={value}>
                <User size={15} color={currentTheme.textMuted} />
                {currentUser?.name ?? '—'}
              </div>
            </div>
            <div>
              <div style={label}>Email Address</div>
              <div style={value}>
                <Mail size={15} color={currentTheme.textMuted} />
                {currentUser?.email ?? '—'}
              </div>
            </div>
            <div>
              <div style={label}>Role</div>
              <div style={value}>
                <Shield size={15} color={currentTheme.textMuted} />
                <span style={{ textTransform: 'capitalize' }}>{role}</span>
              </div>
            </div>
            <div>
              <div style={label}>User ID</div>
              <div style={{ ...value, fontFamily: 'monospace', fontSize: '0.82rem', color: currentTheme.textMuted }}>
                {currentUser?.id ?? '—'}
              </div>
            </div>
          </div>
        </div>

        {/* Logout button */}
        <button
          type="button"
          onClick={handleLogout}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            width: '100%',
            padding: '14px',
            borderRadius: '14px',
            border: '1px solid rgba(239,68,68,0.35)',
            background: isDarkTheme ? 'rgba(239,68,68,0.1)' : 'rgba(239,68,68,0.06)',
            color: '#ef4444',
            fontSize: '0.95rem',
            fontWeight: 700,
            cursor: 'pointer',
            transition: 'background 0.15s ease',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(239,68,68,0.18)')}
          onMouseLeave={(e) => (e.currentTarget.style.background = isDarkTheme ? 'rgba(239,68,68,0.1)' : 'rgba(239,68,68,0.06)')}
        >
          <LogOut size={17} />
          Logout
        </button>

      </div>
    </div>
  );
};

export default Profile;
