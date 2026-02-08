import React, { useState } from 'react';
import bgImg from '../assets/images/bg1.png'; // Vite-compatible import

const Login: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [twoFAEnabled, setTwoFAEnabled] = useState(false);

  return (
    <div
      className="login-page flex items-center justify-center brightness-55"
      style={{ backgroundImage: `url(${bgImg})`, }}
    >
      <div className="login-card ">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-2xl font-semibold text-indigo-400 flex items-center justify-center space-x-2">
            <IconLogo />
            <span className="italic font-semibold">NeuroDetect</span>
          </h1>
          <p className="mt-3 text-white font-bold text-lg">
            Welcome Back to NeuroDetect
          </p>
        </div>

        {/* Form */}
        <form className="space-y-5" onSubmit={(e) => e.preventDefault()}>
          {/* Email */}
          <div>
            <label
              htmlFor="email"
              className="block text-gray-300 font-medium text-sm mb-1"
            >
              Email Address
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-gray-400">
                <IconEnvelope />
              </span>
              <input
                type="email"
                id="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
                className="input-field"
              />
            </div>
          </div>

          {/* Password */}
          <div>
            <label
              htmlFor="password"
              className="block text-gray-300 font-medium text-sm mb-1"
            >
              Password
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-gray-400">
                <IconLock />
              </span>
              <input
                type={passwordVisible ? 'text' : 'password'}
                id="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                className="input-field pr-10"
              />
              <button
                type="button"
                aria-label={passwordVisible ? 'Hide password' : 'Show password'}
                className="absolute inset-y-0 right-0 flex items-center pr-3 text-gray-400 hover:text-gray-200"
                onClick={() => setPasswordVisible(!passwordVisible)}
              >
                {passwordVisible ? <IconEyeOff /> : <IconEye />}
              </button>
            </div>
          </div>

          {/* Forgot Password */}
          <div className="text-right">
            <a
              href="#"
              className="text-indigo-400 text-sm hover:underline"
            >
              Forgot password?
            </a>
          </div>

          {/* 2FA Toggle */}
          <div className="flex items-center justify-between mt-2">
            <label
              htmlFor="toggle-2fa"
              className="text-gray-300 text-sm cursor-pointer select-none"
            >
              Enable Two-Factor Authentication (2FA)
            </label>
            <ToggleSwitch
              id="toggle-2fa"
              enabled={twoFAEnabled}
              setEnabled={setTwoFAEnabled}
            />
          </div>

          {/* Login Button */}
          <button
            type="submit"
            className="w-full mt-6 bg-indigo-500 hover:bg-indigo-600 transition rounded px-4 py-3 text-white font-semibold"
          >
            Secure Login
          </button>
        </form>

        {/* Signup Link */}
        <p className="mt-6 text-center text-gray-400 text-sm">
          Don&apos;t have an account?{' '}
          <a href="#" className="text-indigo-400 hover:underline font-medium">
            Sign up
          </a>
        </p>
      </div>
    </div>
  );
};

interface ToggleSwitchProps {
  id: string;
  enabled: boolean;
  setEnabled: (enabled: boolean) => void;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ id, enabled, setEnabled }) => (
  <button
    type="button"
    role="switch"
    aria-checked={enabled}
    aria-labelledby={id}
    onClick={() => setEnabled(!enabled)}
    className={`toggle-switch ${enabled ? 'toggle-switch-enabled' : ''}`}
  >
    <span className="toggle-thumb" />
  </button>
);

/* Icons */

const IconLogo: React.FC = () => (
  <svg
    width="24"
    height="24"
    fill="none"
    stroke="#8b84ff"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    viewBox="0 0 24 24"
    aria-hidden="true"
  >
    <path d="M12 2v20M2 12h20M4.5 4.5l15 15M4.5 19.5l15-15" />
  </svg>
);

const IconEnvelope: React.FC = () => (
  <svg
    className="icon"
    fill="none"
    stroke="currentColor"
    width={16}
    height={16}
    viewBox="0 0 24 24"
  >
    <polyline points="22,6 12,13 2,6" />
  </svg>
);

const IconLock: React.FC = () => (
  <svg
    className="icon"
    fill="none"
    stroke="currentColor"
    width={16}
    height={16}
    viewBox="0 0 24 24"
  >
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
  </svg>
);

const IconEye: React.FC = () => (
  <svg
    className="icon"
    fill="none"
    stroke="currentColor"
    width={16}
    height={16}
    viewBox="0 0 24 24"
  >
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const IconEyeOff: React.FC = () => (
  <svg
    className="icon"
    fill="none"
    stroke="currentColor"
    width={16}
    height={16}
    viewBox="0 0 24 24"
  >
    <path d="M17.94 17.94a10.03 10.03 0 0 1-5.94 2.06C6.06 20 2 12 2 12s1.73-3.7 5-6.2M9.88 9.88l4.24 4.24M1 1l22 22" />
  </svg>
);

export default Login;
