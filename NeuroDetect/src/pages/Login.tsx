import React, { useState } from 'react';
import { 
  Lock, 
  Mail, 
  Eye, 
  EyeOff, 
  Shield,
  Brain,
  AlertCircle,
  Fingerprint
} from 'lucide-react';
import './css/login.css';

const Login: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [twoFAEnabled, setTwoFAEnabled] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      console.log('Login attempt:', { email, password, twoFAEnabled });
      setIsLoading(false);
      // In real app, redirect to dashboard
    }, 1500);
  };

  return (
    <div className="login-container">
      {/* Background Effects */}
      <div className="background-effects">
        <div className="gradient-circle top-left"></div>
        <div className="gradient-circle bottom-right"></div>
        <div className="grid-overlay"></div>
      </div>

      {/* Login Card */}
      <div className="login-card">
        {/* Header */}
        <div className="login-header">
          <div className="logo-container">
            <div className="logo-icon">
              <Brain className="logo-svg" />
              <div className="logo-glow"></div>
            </div>
            <div className="logo-text">
              <h1>NeuroDetect</h1>
              <p className="tagline">Autoencoder Fraud Detection</p>
            </div>
          </div>
          <div className="security-badge">
            <Shield size={16} />
            <span>Secure Login</span>
          </div>
        </div>

        {/* Welcome Message */}
        <div className="welcome-section">
          <h2>Welcome Back</h2>
          <p className="subtitle">Sign in to access the fraud detection dashboard</p>
        </div>

        {/* Form */}
        <form className="login-form" onSubmit={handleSubmit}>
          {/* Email Field */}
          <div className="form-group">
            <label htmlFor="email" className="form-label">
              <Mail className="label-icon" size={16} />
              <span>Email Address</span>
            </label>
            <div className="input-container">
              <input
                type="email"
                id="email"
                placeholder="admin@neurodetect.ai"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
                className="form-input"
                disabled={isLoading}
              />
              <div className="input-border"></div>
            </div>
          </div>

          {/* Password Field */}
          <div className="form-group">
            <label htmlFor="password" className="form-label">
              <Lock className="label-icon" size={16} />
              <span>Password</span>
            </label>
            <div className="input-container">
              <input
                type={passwordVisible ? 'text' : 'password'}
                id="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                className="form-input"
                disabled={isLoading}
              />
              <div className="input-border"></div>
              <button
                type="button"
                aria-label={passwordVisible ? 'Hide password' : 'Show password'}
                className="password-toggle"
                onClick={() => setPasswordVisible(!passwordVisible)}
                disabled={isLoading}
              >
                {passwordVisible ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>

          {/* Options Row */}
          <div className="options-row">
            <label className="checkbox-label">
              <input type="checkbox" className="checkbox-input" />
              <span className="checkbox-custom"></span>
              <span className="checkbox-text">Remember me</span>
            </label>
            <a href="#" className="forgot-link">
              Forgot password?
            </a>
          </div>

          {/* 2FA Toggle */}
          <div className="twofa-section">
            <div className="twofa-header">
              <Fingerprint className="twofa-icon" size={18} />
              <div>
                <div className="twofa-title">Two-Factor Authentication</div>
                <div className="twofa-subtitle">Enhanced security for your account</div>
              </div>
            </div>
            <ToggleSwitch
              enabled={twoFAEnabled}
              setEnabled={setTwoFAEnabled}
              disabled={isLoading}
            />
          </div>

          {/* Login Button */}
          <button
            type="submit"
            className={`login-button ${isLoading ? 'loading' : ''}`}
            disabled={isLoading}
          >
            {isLoading ? (
              <div className="button-loader">
                <div className="spinner"></div>
                <span>Authenticating...</span>
              </div>
            ) : (
              <>
                <Shield className="button-icon" size={18} />
                <span>Secure Login</span>
              </>
            )}
          </button>

          {/* Divider */}
          <div className="divider">
            <span className="divider-text">or continue with</span>
          </div>

          {/* Social Login */}
          <div className="social-login">
            <button type="button" className="social-button google" disabled={isLoading}>
              <svg className="social-icon" viewBox="0 0 24 24">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
              </svg>
              Google
            </button>
            <button type="button" className="social-button microsoft" disabled={isLoading}>
              <svg className="social-icon" viewBox="0 0 24 24">
                <path d="M0 0h11v11H0zM13 0h11v11H13zM0 13h11v11H0zM13 13h11v11H13z" fill="#7FBA00"/>
              </svg>
              Microsoft
            </button>
          </div>
        </form>

        {/* Footer */}
        <div className="login-footer">
          <p className="signup-text">
            Don't have an account?{' '}
            <a href="#" className="signup-link">
              Request access
            </a>
          </p>
          <div className="security-info">
            <AlertCircle size={14} />
            <span>All connections are encrypted with TLS 1.3</span>
          </div>
        </div>
      </div>
    </div>
  );
};

interface ToggleSwitchProps {
  enabled: boolean;
  setEnabled: (enabled: boolean) => void;
  disabled?: boolean;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ enabled, setEnabled, disabled }) => (
  <button
    type="button"
    role="switch"
    aria-checked={enabled}
    onClick={() => !disabled && setEnabled(!enabled)}
    className={`toggle-switch ${enabled ? 'enabled' : ''} ${disabled ? 'disabled' : ''}`}
    disabled={disabled}
  >
    <span className="toggle-thumb" />
  </button>
);

export default Login;