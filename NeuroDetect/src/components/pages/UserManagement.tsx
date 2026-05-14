import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  CheckCircle,
  Edit2,
  LogOut,
  Plus,
  Search,
  Settings2,
  ShieldCheck,
  Trash2,
  Users,
  XCircle,
} from 'lucide-react';
import { useAuth } from '../auth/AuthContext';
import { getRoleSidebarItems } from '../layout/roleNavigation';
import { useTheme } from '../theme/ThemeContext';
import { createUser, deleteUser, getAllUsers, getSessionToken, updateUserRole } from '../../auth/storage';
import type { ManagedUser, UserRole } from '../../auth/types';

interface FormData {
  name: string;
  email: string;
  phone_number: string;
  password: string;
  role: UserRole;
}

const panel = {
  background: '#fff',
  border: '1px solid #e6ebf4',
  borderRadius: '18px',
  boxShadow: '0 18px 38px rgba(148,163,184,.12)',
} satisfies React.CSSProperties;

const inputBase = {
  width: '100%',
  padding: '12px 14px',
  borderRadius: '12px',
  border: '1px solid #d5ddeb',
  background: '#fff',
  color: '#0f172a',
  boxSizing: 'border-box',
  outline: 'none',
} satisfies React.CSSProperties;

const roles: UserRole[] = ['admin', 'analyst', 'senior_analyst', 'viewer'];
const ORGANIZATION_EMAIL_PATTERN = /^[^\s@]+@neurodetect\.ai$/i;
const PHONE_NUMBER_PATTERN = /^\+\d{1,3}\s\d{6,15}$/;

const roleStyles: Record<UserRole, React.CSSProperties> = {
  admin: { background: '#f4f0ff', color: '#6d28d9', border: '1px solid #e6dbff' },
  analyst: { background: '#ecfdf5', color: '#059669', border: '1px solid #c7f2de' },
  senior_analyst: { background: '#fff7ed', color: '#c2410c', border: '1px solid #fed7aa' },
  viewer: { background: '#eef6ff', color: '#0284c7', border: '1px solid #cfe5ff' },
};

const roleCards = [
  ['Admin', 'User lifecycle and access control'],
  ['Analyst', 'Dashboards, reviews, and reports'],
  ['Senior Analyst', 'Escalated fraud review and phone notifications'],
  ['Viewer', 'Read-only operational visibility'],
] as const;

const StatCard = ({ label, value, color }: { label: string; value: number; color: string }) => (
  <div style={{ ...panel, padding: 18 }}>
    <div style={{ color: '#6b7280', fontSize: '.82rem', fontWeight: 700, letterSpacing: '.08em', textTransform: 'uppercase', marginBottom: 10 }}>{label}</div>
    <div style={{ color, fontSize: '1.95rem', fontWeight: 800 }}>{value}</div>
  </div>
);

const UserManagement: React.FC = () => {
  const { currentUser, logout } = useAuth();
  const { isDarkTheme } = useTheme();
  const navigate = useNavigate();
  const navItems = getRoleSidebarItems(currentUser?.role ?? 'admin');
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);
  const [editingUserId, setEditingUserId] = useState<string | null>(null);
  const [selectedUser, setSelectedUser] = useState<ManagedUser | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [formData, setFormData] = useState<FormData>({ name: '', email: '', phone_number: '', password: '', role: 'viewer' });

  useEffect(() => {
    if (currentUser && currentUser.role !== 'admin') navigate('/unauthorized');
  }, [currentUser, navigate]);

  const loadUsers = useCallback(async () => {
    const token = getSessionToken();
    if (!token) {
      setError('Session expired. Please login again.');
      setLoading(false);
      return;
    }
    try {
      setLoading(true);
      setError(null);
      setUsers(await getAllUsers(token));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load users');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  const stats = useMemo(() => ({
    total: users.length,
    admins: users.filter((u) => u.role === 'admin').length,
    analysts: users.filter((u) => u.role === 'analyst').length,
    seniorAnalysts: users.filter((u) => u.role === 'senior_analyst').length,
    viewers: users.filter((u) => u.role === 'viewer').length,
  }), [users]);

  const filteredUsers = useMemo(() => {
    const q = searchTerm.trim().toLowerCase();
    if (!q) return users;
    return users.filter((u) => [u.name, u.email, u.role].some((v) => v.toLowerCase().includes(q)));
  }, [searchTerm, users]);

  const flashSuccess = (message: string) => {
    setSuccess(message);
    setTimeout(() => setSuccess(null), 3000);
  };

  const handleAddUser = async (e: React.FormEvent) => {
    e.preventDefault();
    const token = getSessionToken();
    if (!token) return setError('Session expired');
    if (!ORGANIZATION_EMAIL_PATTERN.test(formData.email.trim())) {
      return setError('Email must use the format xxx@neurodetect.ai');
    }
    if (!PHONE_NUMBER_PATTERN.test(formData.phone_number.trim())) {
      return setError('Phone number must use +CC XXXXXXXX format, for example +94 704209968');
    }
    try {
      setError(null);
      await createUser(token, formData);
      setFormData({ name: '', email: '', phone_number: '', password: '', role: 'viewer' });
      setShowAddModal(false);
      await loadUsers();
      flashSuccess(`User ${formData.email} created successfully`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create user');
    }
  };

  const handleUpdateRole = async (userId: string, role: UserRole) => {
    const token = getSessionToken();
    if (!token) return setError('Session expired');
    try {
      setError(null);
      await updateUserRole(token, userId, role);
      setEditingUserId(null);
      setSelectedUser(null);
      await loadUsers();
      flashSuccess('User role updated successfully');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update role');
    }
  };

  const handleDeleteUser = async (userId: string) => {
    const token = getSessionToken();
    if (!token) return setError('Session expired');
    try {
      setError(null);
      await deleteUser(token, userId);
      setShowDeleteConfirm(null);
      await loadUsers();
      flashSuccess('User deleted successfully');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete user');
    }
  };

  if (!currentUser) return null;

  const shellBg = isDarkTheme ? '#0f172a' : '#f5f6fb';
  const shellPanel = isDarkTheme ? '#111827' : '#ffffff';
  const shellSoft = isDarkTheme ? '#1f2937' : '#fbfbfe';
  const shellBorder = isDarkTheme ? 'rgba(148,163,184,.18)' : '#e6ebf4';
  const shellText = isDarkTheme ? '#f8fafc' : '#111827';
  const shellMuted = isDarkTheme ? '#94a3b8' : '#7c8599';
  const cardPanel = isDarkTheme ? '#111827' : '#ffffff';
  const cardBorder = shellBorder;
  const cardHeaderBg = isDarkTheme ? '#0f172a' : '#fbfcfe';
  const cardRowBorder = isDarkTheme ? 'rgba(148,163,184,.12)' : '#f0f2f7';
  const cardText = shellText;
  const cardMuted = shellMuted;
  const inputBg = isDarkTheme ? '#0b1220' : '#ffffff';
  const inputText = shellText;
  const inputBorder = shellBorder;
  const mutedSurface = isDarkTheme ? '#0f172a' : '#f5f6fb';
  const accentButtonText = '#ffffff';
  const roleDirectoryBg = isDarkTheme ? '#111827' : '#fbfcfe';
  const modalBg = isDarkTheme ? '#0f172a' : '#ffffff';
  const modalText = shellText;
  const modalMuted = shellMuted;

  return (
    <div style={{ minHeight: '100vh', background: shellBg, color: shellText }}>
      <div style={{ width: '100%', padding: '26px 24px 34px' }}>
        <div style={{ ...panel, overflow: 'hidden', background: shellPanel, border: `1px solid ${shellBorder}` }}>
          <div style={{ display: 'grid', gridTemplateColumns: '220px minmax(0,1fr)', minHeight: 'calc(100vh - 110px)' }}>
            <aside style={{ borderRight: `1px solid ${shellBorder}`, background: isDarkTheme ? 'linear-gradient(180deg,#0f172a 0%,#111827 100%)' : 'linear-gradient(180deg,#fcfcff 0%,#f6f7fb 100%)', minHeight: 'calc(100vh - 110px)', display: 'flex', flexDirection: 'column' }}>
              <div style={{ padding: '28px 24px 18px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ width: 42, height: 42, borderRadius: 14, background: 'linear-gradient(135deg,#7c3aed,#2563eb)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 14px 28px rgba(99,102,241,.22)' }}>
                    <Users size={20} />
                  </div>
                  <div>
                    <div style={{ fontSize: '1.02rem', fontWeight: 800, color: shellText }}>NeuroDetect</div>
                    <div style={{ fontSize: '.78rem', color: shellMuted, letterSpacing: '.08em', textTransform: 'uppercase' }}>Admin Settings</div>
                  </div>
                </div>
              </div>

              <nav style={{ padding: '8px 16px 18px', display: 'grid', gap: 8 }}>
                {navItems.map((item) => (
                  <Link key={item.label} to={item.to} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 14px', borderRadius: 14, color: item.to === '/user-management' ? shellText : shellMuted, background: item.to === '/user-management' ? (isDarkTheme ? '#1f2937' : '#f2f4f8') : 'transparent', textDecoration: 'none', fontWeight: 600 }}>
                    {item.label}
                  </Link>
                ))}
              </nav>

              <div style={{ marginTop: 'auto', padding: '18px 16px 20px', borderTop: `1px solid ${shellBorder}` }}>
                <div style={{ padding: '12px 14px', borderRadius: 14, background: isDarkTheme ? '#1f2937' : '#f2f4f8', color: shellText, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                  <Settings2 size={18} /> Admin Settings
                </div>
                <button onClick={() => { logout(); navigate('/login'); }} style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 12, padding: '12px 14px', borderRadius: 14, background: 'transparent', border: 'none', color: shellMuted, fontWeight: 600, cursor: 'pointer' }}>
                  <LogOut size={18} /> Sign Out
                </button>
              </div>
            </aside>

            <main style={{ background: shellSoft }}>
              <div style={{ padding: '28px 30px 32px' }}>
                {error && <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '14px 16px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 14, marginBottom: 18 }}><XCircle size={18} color="#dc2626" /><div style={{ color: '#b91c1c', fontWeight: 600 }}>{error}</div></div>}
                {success && <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '14px 16px', background: '#ecfdf5', border: '1px solid #bbf7d0', borderRadius: 14, marginBottom: 18 }}><CheckCircle size={18} color="#10b981" /><div style={{ color: '#047857', fontWeight: 600 }}>{success}</div></div>}

                <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', marginBottom: 20 }}>
                  <div>
                    <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 800, color: shellText }}>Team Members</h2>
                    <p style={{ margin: '8px 0 0', color: shellMuted, fontSize: '1rem' }}>Manage your team&apos;s access and review role assignments.</p>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '0 14px', height: 42, minWidth: 260, borderRadius: 14, border: `1px solid ${inputBorder}`, background: inputBg }}>
                      <Search size={16} color={cardMuted} />
                      <input value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} placeholder="Search users..." style={{ border: 'none', outline: 'none', width: '100%', fontSize: '.95rem', color: inputText, background: 'transparent' }} />
                    </div>
                    <button type="button" onClick={() => setShowAddModal(true)} style={{ display: 'inline-flex', alignItems: 'center', gap: 10, height: 42, padding: '0 18px', borderRadius: 14, border: 'none', background: 'linear-gradient(135deg,#7c3aed,#8b5cf6)', color: '#fff', fontWeight: 700, cursor: 'pointer', boxShadow: '0 16px 30px rgba(124,58,237,.22)' }}>
                      <Plus size={16} /> Add Member
                    </button>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', gap: 16, marginBottom: 18 }}>
                  <StatCard label="Total Users" value={stats.total} color="#2563eb" />
                  <StatCard label="Administrators" value={stats.admins} color="#7c3aed" />
                  <StatCard label="Analysts" value={stats.analysts} color="#059669" />
                  <StatCard label="Senior Analysts" value={stats.seniorAnalysts} color="#c2410c" />
                  <StatCard label="Viewers" value={stats.viewers} color="#0284c7" />
                </div>

                <div style={{ ...panel, overflow: 'hidden', marginBottom: 22, background: cardPanel, border: `1px solid ${cardBorder}` }}>
                  {loading ? (
                    <div style={{ padding: 42, textAlign: 'center', color: cardMuted }}>Loading users...</div>
                  ) : (
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead style={{ background: cardHeaderBg, borderBottom: `1px solid ${cardBorder}` }}>
                          <tr>
                            {['User', 'Role', 'Status', 'Last Active', 'Actions'].map((h, i) => (
                              <th key={h} style={{ padding: '16px 18px', textAlign: i === 4 ? 'center' : 'left', color: cardMuted, fontSize: '.9rem' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {filteredUsers.map((user) => (
                            <tr key={user.id} style={{ borderTop: `1px solid ${cardRowBorder}` }}>
                              <td style={{ padding: '16px 18px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                                  <div style={{ width: 38, height: 38, borderRadius: 999, background: user.role === 'admin' ? 'linear-gradient(135deg,#c4b5fd,#8b5cf6)' : user.role === 'analyst' ? 'linear-gradient(135deg,#86efac,#10b981)' : user.role === 'senior_analyst' ? 'linear-gradient(135deg,#fdba74,#f97316)' : 'linear-gradient(135deg,#bfdbfe,#60a5fa)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800 }}>
                                    {user.name.slice(0, 1).toUpperCase()}
                                  </div>
                                  <div>
                                    <div style={{ fontWeight: 700, color: cardText }}>{user.name}</div>
                                    <div style={{ color: cardMuted, fontSize: '.92rem' }}>{user.email}</div>
                                    {user.phone_number ? <div style={{ color: cardMuted, fontSize: '.82rem' }}>{user.phone_number}</div> : null}
                                  </div>
                                </div>
                              </td>
                              <td style={{ padding: '16px 18px' }}>
                                {editingUserId === user.id ? (
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                                    <select value={selectedUser?.role || user.role} onChange={(e) => selectedUser && setSelectedUser({ ...selectedUser, role: e.target.value as UserRole })} style={{ ...inputBase, width: 140, padding: '10px 12px', background: inputBg, color: inputText, borderColor: inputBorder }}>
                                      {roles.map((role) => <option key={role} value={role}>{role}</option>)}
                                    </select>
                                    <button type="button" onClick={() => selectedUser && handleUpdateRole(user.id, selectedUser.role)} style={{ padding: '9px 12px', borderRadius: 10, border: 'none', background: '#10b981', color: accentButtonText, fontWeight: 700, cursor: 'pointer' }}>Save</button>
                                  </div>
                                ) : (
                                  <span style={{ display: 'inline-flex', padding: '8px 12px', borderRadius: 999, fontSize: '.82rem', fontWeight: 700, ...roleStyles[user.role] }}>{user.role === 'admin' ? 'Admin' : user.role === 'analyst' ? 'Analyst' : user.role === 'senior_analyst' ? 'Senior Analyst' : 'Viewer'}</span>
                                )}
                              </td>
                              <td style={{ padding: '16px 18px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: cardMuted }}>
                                  <span style={{ width: 8, height: 8, borderRadius: 999, background: user.role === 'viewer' ? '#cbd5e1' : '#10b981' }} />
                                  {user.role === 'viewer' ? 'offline' : 'online'}
                                </div>
                              </td>
                              <td style={{ padding: '16px 18px', color: cardMuted }}>{user.role === 'viewer' ? '2 hours ago' : 'Active now'}</td>
                              <td style={{ padding: '16px 18px' }}>
                                <div style={{ display: 'flex', justifyContent: 'center', gap: 10 }}>
                                  <button type="button" onClick={() => { setSelectedUser(user); setEditingUserId(user.id); }} style={{ width: 38, height: 38, borderRadius: 12, border: `1px solid ${cardBorder}`, background: inputBg, color: shellText, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
                                    <Edit2 size={16} />
                                  </button>
                                  <button type="button" onClick={() => setShowDeleteConfirm(user.id)} style={{ width: 38, height: 38, borderRadius: 12, border: '1px solid #fecaca', background: isDarkTheme ? 'rgba(127,29,29,.22)' : '#fff5f5', color: '#dc2626', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
                                    <Trash2 size={16} />
                                  </button>
                                </div>
                              </td>
                            </tr>
                          ))}
                          {!loading && filteredUsers.length === 0 && (
                            <tr><td colSpan={5} style={{ padding: 28, textAlign: 'center', color: cardMuted }}>No users match your search.</td></tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '340px minmax(0,1fr)', gap: 18 }}>
                  <div style={{ ...panel, padding: 22, background: cardPanel, border: `1px solid ${cardBorder}` }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                      <ShieldCheck size={18} color="#7c3aed" />
                      <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 800, color: cardText }}>Role Directory</h3>
                    </div>
                    <div style={{ display: 'grid', gap: 12 }}>
                      {roleCards.map(([title, caption]) => (
                        <div key={title} style={{ padding: '14px 15px', borderRadius: 14, border: `1px solid ${cardBorder}`, background: roleDirectoryBg, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                          <div>
                            <div style={{ fontWeight: 700, color: cardText }}>{title}</div>
                            <div style={{ color: cardMuted, fontSize: '.88rem', marginTop: 4 }}>{caption}</div>
                          </div>
                          <Edit2 size={15} color={cardMuted} />
                        </div>
                      ))}
                      <button type="button" onClick={() => setShowAddModal(true)} style={{ marginTop: 6, height: 42, borderRadius: 14, border: `1px solid ${cardBorder}`, background: inputBg, color: shellText, fontWeight: 700, cursor: 'pointer' }}>
                        Create New Role
                      </button>
                    </div>
                  </div>

                  <div style={{ ...panel, padding: 22, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center', minHeight: 280, background: cardPanel, border: `1px solid ${cardBorder}` }}>
                    <div style={{ width: 52, height: 52, borderRadius: 999, background: mutedSurface, border: `1px solid ${cardBorder}`, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 18 }}>
                      <ShieldCheck size={22} color="#6b7280" />
                    </div>
                    <h3 style={{ margin: '0 0 10px', fontSize: '1.55rem', fontWeight: 800, color: cardText }}>Security Policies</h3>
                    <p style={{ maxWidth: 480, margin: '0 0 22px', color: cardMuted, lineHeight: 1.7 }}>
                      Manage MFA requirements, session timeouts, and access boundaries for your organization with a read-only security overview for admins.
                    </p>
                    <button type="button" style={{ height: 44, padding: '0 20px', borderRadius: 14, border: `1px solid ${cardBorder}`, background: inputBg, color: shellText, fontWeight: 700, cursor: 'pointer' }}>
                      Configure Policies
                    </button>
                  </div>
                </div>
              </div>
            </main>
          </div>
        </div>
      </div>

      {showAddModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(15,23,42,.48)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 60, padding: 16 }}>
          <div style={{ width: '100%', maxWidth: 480, background: modalBg, borderRadius: 22, border: `1px solid ${cardBorder}`, boxShadow: '0 24px 60px rgba(15,23,42,.2)', padding: 24 }}>
            <h2 style={{ margin: '0 0 18px', fontSize: '1.4rem', fontWeight: 800, color: modalText }}>Add New User</h2>
            <form onSubmit={handleAddUser} style={{ display: 'grid', gap: 14 }}>
              <div><label style={{ display: 'block', fontSize: '.9rem', fontWeight: 700, marginBottom: 8, color: modalMuted }}>Name</label><input type="text" value={formData.name} onChange={(e) => setFormData({ ...formData, name: e.target.value })} style={{ ...inputBase, background: inputBg, color: inputText, borderColor: inputBorder }} required /></div>
              <div>
                <label style={{ display: 'block', fontSize: '.9rem', fontWeight: 700, marginBottom: 8, color: modalMuted }}>Email</label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  style={{ ...inputBase, background: inputBg, color: inputText, borderColor: inputBorder }}
                  required
                  pattern="^[^\s@]+@neurodetect\.ai$"
                  placeholder="name@neurodetect.ai"
                />
                <div style={{ marginTop: 6, fontSize: '.8rem', color: modalMuted }}>Only organization emails ending with @neurodetect.ai are allowed.</div>
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '.9rem', fontWeight: 700, marginBottom: 8, color: modalMuted }}>Phone Number</label>
                <input
                  type="tel"
                  value={formData.phone_number}
                  onChange={(e) => setFormData({ ...formData, phone_number: e.target.value })}
                  style={{ ...inputBase, background: inputBg, color: inputText, borderColor: inputBorder }}
                  required
                  placeholder="+94 704209968"
                />
                <div style={{ marginTop: 6, fontSize: '.8rem', color: modalMuted }}>Use country code format with a space after it, for example +94 704209968.</div>
              </div>
              <div><label style={{ display: 'block', fontSize: '.9rem', fontWeight: 700, marginBottom: 8, color: modalMuted }}>Password</label><input type="password" value={formData.password} onChange={(e) => setFormData({ ...formData, password: e.target.value })} style={{ ...inputBase, background: inputBg, color: inputText, borderColor: inputBorder }} required minLength={6} /></div>
              <div><label style={{ display: 'block', fontSize: '.9rem', fontWeight: 700, marginBottom: 8, color: modalMuted }}>Role</label><select value={formData.role} onChange={(e) => setFormData({ ...formData, role: e.target.value as UserRole })} style={{ ...inputBase, background: inputBg, color: inputText, borderColor: inputBorder }}>{roles.map((role) => <option key={role} value={role}>{role}</option>)}</select></div>
              <div style={{ display: 'flex', gap: 12, paddingTop: 6 }}>
                <button type="submit" style={{ flex: 1, padding: '12px 14px', borderRadius: 14, background: 'linear-gradient(135deg,#7c3aed,#8b5cf6)', color: '#fff', border: 'none', fontWeight: 700, cursor: 'pointer' }}>Create User</button>
                <button type="button" onClick={() => setShowAddModal(false)} style={{ flex: 1, padding: '12px 14px', borderRadius: 14, background: inputBg, color: shellText, border: `1px solid ${inputBorder}`, fontWeight: 700, cursor: 'pointer' }}>Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showDeleteConfirm && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(15,23,42,.48)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 60, padding: 16 }}>
          <div style={{ width: '100%', maxWidth: 400, background: modalBg, borderRadius: 22, border: `1px solid ${cardBorder}`, boxShadow: '0 24px 60px rgba(15,23,42,.2)', padding: 24 }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 18 }}>
              <XCircle size={24} color="#ef4444" />
              <div>
                <h2 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 800, color: modalText }}>Delete User?</h2>
                <p style={{ margin: '6px 0 0', fontSize: '.94rem', color: modalMuted }}>This action cannot be undone.</p>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 12 }}>
              <button onClick={() => handleDeleteUser(showDeleteConfirm)} style={{ flex: 1, padding: '12px 14px', borderRadius: 14, background: '#ef4444', color: '#fff', border: 'none', fontWeight: 700, cursor: 'pointer' }}>Delete</button>
              <button onClick={() => setShowDeleteConfirm(null)} style={{ flex: 1, padding: '12px 14px', borderRadius: 14, background: inputBg, color: shellText, border: `1px solid ${inputBorder}`, fontWeight: 700, cursor: 'pointer' }}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UserManagement;
