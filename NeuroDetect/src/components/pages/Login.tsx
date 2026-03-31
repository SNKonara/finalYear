import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Lock, 
  Mail, 
  Eye, 
  EyeOff, 
  Shield,
  Activity,
  AlertCircle,
  Fingerprint,
  ArrowRight
} from 'lucide-react';
import '../../pages/css/login.css';
import { getDefaultRouteByRole, useAuth } from '../auth/AuthContext';

const Login: React.FC = () => {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [rememberDevice, setRememberDevice] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setIsLoading(true);

    const result = await login(email, password);
    if (!result.success) {
      setIsLoading(false);
      setErrorMessage(result.error ?? 'Login failed');
      return;
    }

    const authenticatedRole = result.user?.role ?? 'viewer';

    setTimeout(() => {
      setIsLoading(false);
      navigate(getDefaultRouteByRole(authenticatedRole));
    }, 400);
  };

  return (
    <div className="login-container">
      <div className="login-backdrop-glow login-backdrop-glow-left" aria-hidden="true" />
      <div className="login-backdrop-glow login-backdrop-glow-right" aria-hidden="true" />

      <div className="login-layout">
        <aside className="login-side-panel" aria-hidden="true">
          <h3>Credit Card Fraud Detection</h3>
          <p>Use your assigned credentials to continue.</p>
          <div className="side-panel-pill-row">
            <span className="side-panel-pill">Real-time Detection</span>
            <span className="side-panel-pill">Batch Upload</span>
          </div>
        </aside>

        <div className="login-card">
          <div className="login-header">
            <div className="brand-row">
              <div className="logo-icon">
                <Activity className="logo-svg" />
              </div>
              <h1>NeuroDetect</h1>
            </div>
          </div>

          <div className="welcome-section">
            <h2>Welcome back</h2>
            <p className="subtitle">Sign in with your administrator-issued account</p>
          </div>

          <form className="login-form" onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="email" className="form-label">Email Address</label>
              <div className="input-container">
                <Mail className="input-leading-icon" size={16} />
                <input
                  type="email"
                  id="email"
                  placeholder="sarah.analyst@neurodetect.io"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  autoComplete="email"
                  className="form-input"
                  disabled={isLoading}
                />
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="password" className="form-label">Password</label>
              <div className="input-container">
                <Lock className="input-leading-icon" size={16} />
                <input
                  type={passwordVisible ? 'text' : 'password'}
                  id="password"
                  placeholder="password123"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  autoComplete="current-password"
                  className="form-input"
                  disabled={isLoading}
                />
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

            <div className="options-row">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  className="checkbox-input"
                  checked={rememberDevice}
                  onChange={(event) => setRememberDevice(event.target.checked)}
                />
                <span className="checkbox-custom"></span>
                <span className="checkbox-text">Remember this device</span>
              </label>
              <a href="#" className="forgot-link" onClick={(event) => event.preventDefault()}>
                Forgot password?
              </a>
            </div>

            {errorMessage && (
              <div className="security-info" style={{ marginBottom: '12px' }}>
                <AlertCircle size={14} />
                <span>{errorMessage}</span>
              </div>
            )}

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
                  <span>Sign In</span>
                  <ArrowRight className="button-icon" size={18} />
                </>
              )}
            </button>

            <div className="divider">
              <span className="divider-text">or continue with</span>
            </div>

            <div className="social-login enterprise-login">
              <button type="button" className="social-button" disabled={isLoading}>
                <Fingerprint className="social-icon" size={16} />
                Corporate ID
              </button>
              <button type="button" className="social-button" disabled={isLoading}>
                <Shield className="social-icon" size={16} />
                SSO Login
              </button>
            </div>
          </form>

          <div className="login-footer">
            <div className="security-info">
              <Shield size={14} />
              <span>Secure gateway active</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
