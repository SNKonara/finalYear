export type UserRole = 'admin' | 'analyst' | 'viewer';

export interface ManagedUser {
  id: string;
  name: string;
  email: string;
  role: UserRole;
}

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  role: UserRole;
}
