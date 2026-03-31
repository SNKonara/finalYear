import type { UserRole } from '../../auth/types';

export type RoleNavItem = {
  label: string;
  to: string;
};

export const getDefaultRouteByRole = (role: UserRole): string => {
  switch (role) {
    case 'admin':
      return '/snnreal';
    case 'analyst':
      return '/snnreal';
    case 'viewer':
      return '/summary';
    default:
      return '/login';
  }
};

export const getRoleSidebarItems = (role: UserRole): RoleNavItem[] => {
  switch (role) {
    case 'admin':
      return [
        { label: 'Dashboard', to: '/snnreal' },
        { label: 'User', to: '/user-management' },
        { label: 'Report', to: '/reports' },
        { label: 'System', to: '/system' },
      ];
    case 'analyst':
      return [
        { label: 'Dashboard', to: '/snnreal' },
        { label: 'Investigation', to: '/investigations' },
        { label: 'System', to: '/system' },
        { label: 'Batch Upload', to: '/batch-upload' },
        { label: 'Reports', to: '/reports' },
      ];
    case 'viewer':
      return [
        { label: 'Reports', to: '/reports' },
        { label: 'Summary', to: '/summary' },
      ];
    default:
      return [];
  }
};

export const getBatchUploadSidebarItems = (role: UserRole): RoleNavItem[] => {
  if (role === 'viewer') {
    return getRoleSidebarItems(role);
  }

  return [
    { label: 'Dashboard', to: '/snnreal' },
    { label: 'Investigation', to: '/investigations' },
    { label: 'Batch Upload', to: '/batch-upload' },
    { label: 'Model Tune', to: '/batch-upload?tuning=1' },
    { label: 'Reports', to: '/reports' },
  ];
};

export const getCurrentSectionLabel = (pathname: string, role: UserRole | null | undefined): string => {
  const resolvedRole = role ?? 'viewer';
  const items = [
    ...getRoleSidebarItems(resolvedRole),
    { label: 'Profile', to: '/profile' },
    { label: 'Unauthorized', to: '/unauthorized' },
  ];
  const match = items.find((item) => item.to === pathname);
  if (match) return match.label;
  if (pathname === '/lstmreal' || pathname === '/snnreal') return 'Dashboard';
  if (pathname.startsWith('/investigations/')) return 'Investigation';
  if (pathname === '/snn-alerts') return 'Investigation';
  return 'Workspace';
};