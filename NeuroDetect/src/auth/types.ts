export type UserRole = 'admin' | 'analyst' | 'senior_analyst' | 'viewer';

export interface ManagedUser {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  phone_number?: string;
}

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  phone_number?: string;
}
