import type { AuthUser, ManagedUser, UserRole } from './types';

const SESSION_KEY = 'neurodetect_session';
const TOKEN_KEY = 'neurodetect_token';
const AUTH_API_BASE = 'http://localhost:8000';

const AUTH_API_UNREACHABLE_MESSAGE =
  'Cannot reach auth server at http://localhost:8000. Start backend/server/batch_api.py and try again.';

const normalizeFetchError = (error: unknown): Error => {
  if (error instanceof TypeError) {
    return new Error(AUTH_API_UNREACHABLE_MESSAGE);
  }
  if (error instanceof Error) {
    return error;
  }
  return new Error('Request failed');
};

interface LoginApiResponse {
  token: string;
  user: AuthUser;
}

const parseErrorMessage = async (response: Response, fallback: string): Promise<string> => {
  try {
    const payload = await response.json();
    const detail = payload?.detail;
    if (typeof detail === 'string' && detail.length > 0) {
      return detail;
    }
  } catch {
    // Ignore parse errors and use fallback
  }
  return fallback;
};

export const saveSession = (user: AuthUser, token: string): void => {
  localStorage.setItem(SESSION_KEY, JSON.stringify(user));
  localStorage.setItem(TOKEN_KEY, token);
};

export const getSessionUser = (): AuthUser | null => {
  const raw = localStorage.getItem(SESSION_KEY);
  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw) as AuthUser;
  } catch {
    localStorage.removeItem(SESSION_KEY);
    return null;
  }
};

export const getSessionToken = (): string | null => localStorage.getItem(TOKEN_KEY);

export const clearSession = (): void => {
  localStorage.removeItem(SESSION_KEY);
  localStorage.removeItem(TOKEN_KEY);
};

export const authenticateUser = async (email: string, password: string): Promise<LoginApiResponse> => {
  try {
    const response = await fetch(`${AUTH_API_BASE}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const message = await parseErrorMessage(response, 'Invalid email or password');
      throw new Error(message);
    }

    return response.json() as Promise<LoginApiResponse>;
  } catch (error) {
    throw normalizeFetchError(error);
  }
};

export const fetchCurrentUser = async (token: string): Promise<AuthUser> => {
  try {
    const response = await fetch(`${AUTH_API_BASE}/auth/me`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      const message = await parseErrorMessage(response, 'Session expired');
      throw new Error(message);
    }

    const payload = (await response.json()) as { user: AuthUser };
    return payload.user;
  } catch (error) {
    throw normalizeFetchError(error);
  }
};

export const logoutUser = async (token: string): Promise<void> => {
  await fetch(`${AUTH_API_BASE}/auth/logout`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
};

export const getAllUsers = async (token: string): Promise<ManagedUser[]> => {
  const response = await fetch(`${AUTH_API_BASE}/auth/users`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    const message = await parseErrorMessage(response, 'Unable to load users');
    throw new Error(message);
  }

  const payload = (await response.json()) as { users: ManagedUser[] };
  return payload.users;
};

export const updateUserRole = async (token: string, userId: string, role: UserRole): Promise<ManagedUser> => {
  const response = await fetch(`${AUTH_API_BASE}/auth/users/${userId}/role`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ role }),
  });

  if (!response.ok) {
    const message = await parseErrorMessage(response, 'Unable to update role');
    throw new Error(message);
  }

  const payload = (await response.json()) as { user: ManagedUser };
  return payload.user;
};

export const createUser = async (
  token: string,
  payload: { name: string; email: string; phone_number: string; password: string; role: UserRole },
): Promise<ManagedUser> => {
  const response = await fetch(`${AUTH_API_BASE}/auth/users`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const message = await parseErrorMessage(response, 'Unable to create user');
    throw new Error(message);
  }

  const result = (await response.json()) as { user: ManagedUser };
  return result.user;
};

export const deleteUser = async (token: string, userId: string): Promise<void> => {
  const response = await fetch(`${AUTH_API_BASE}/auth/users/${userId}`, {
    method: 'DELETE',
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    const message = await parseErrorMessage(response, 'Unable to delete user');
    throw new Error(message);
  }
};
