import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import { useTheme } from '../theme/ThemeContext';

const Unauthorized: React.FC = () => {
  const { currentUser } = useAuth();
  const { currentTheme, isDarkTheme } = useTheme();

  return (
    <div style={{ minHeight: '100vh', background: currentTheme.bgPrimary, color: currentTheme.textPrimary, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px' }}>
      <div style={{ maxWidth: '560px', width: '100%', background: isDarkTheme ? currentTheme.bgCard : '#ffffff', border: `1px solid ${currentTheme.borderColor}`, borderRadius: '22px', padding: '32px', boxShadow: isDarkTheme ? 'none' : '0 18px 36px rgba(148,163,184,.14)' }}>
        <h1 style={{ margin: '0 0 12px', fontSize: '2rem', fontWeight: 800 }}>Access denied</h1>
        <p style={{ margin: '0 0 22px', color: currentTheme.textSecondary, lineHeight: 1.7 }}>
          Your role <span style={{ fontWeight: 700, color: '#2563eb' }}>{currentUser?.role ?? 'unknown'}</span> does not have permission to open this section.
        </p>
        <Link to="/" style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', padding: '12px 16px', borderRadius: '12px', background: '#2563eb', color: '#ffffff', textDecoration: 'none', fontWeight: 700 }}>
          Back to model page
        </Link>
      </div>
    </div>
  );
};

export default Unauthorized;
