import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, AlertTriangle, Brain, Database, Lock, RefreshCw, Server, ShieldCheck, Users } from 'lucide-react';
import { useTheme } from '../theme/ThemeContext';
import { useAuth } from '../auth/AuthContext';
import { getRoleSidebarItems } from '../layout/roleNavigation';

const API_BASE = 'http://localhost:8000';

type ServiceStatus = 'healthy' | 'degraded' | 'offline';

interface OverviewPayload {
  generated_at: string;
  read_only: boolean;
  status: ServiceStatus;
  services: {
    api: { status: ServiceStatus; detail: string };
    mongodb: { status: ServiceStatus; detail: string };
    auth: { status: ServiceStatus; detail: string };
  };
  kpis: {
    loaded_models: number;
    expected_models: number;
    open_alerts: number;
    resolved_frauds: number;
    reports_total: number;
    batch_runs_last_24h: number;
    active_sessions: number;
    users_total: number;
  };
  models: Array<{
    model_type: string;
    display_name: string;
    loaded: boolean;
    threshold?: number;
    feature_count?: number;
    sequence_length?: number;
    time_steps?: number;
    architecture?: string;
    performance?: {
      accuracy?: number;
      precision?: number;
      recall?: number;
      f1_score?: number;
    };
  }>;
  monitoring: {
    open_alerts: number;
    dismissed_alerts: number;
    false_positives: number;
    escalated_alerts: number;
    resolved_frauds_total: number;
    resolved_frauds_last_24h: number;
    alert_mix: Record<string, number>;
  };
  reports: {
    model_reports_total: number;
    hourly_reports_total: number;
    latest_generated_at: string | null;
  };
  operations: {
    batch_runs_total: number;
    recent_batches: Array<{
      batch_id: string;
      model_type: string;
      timestamp: string | null;
      fraud_count: number;
      total_transactions: number;
      threshold?: number;
    }>;
    recent_events: Array<{
      type: string;
      title: string;
      timestamp: string | null;
      detail: string;
    }>;
  };
  sources: string[];
}

const SystemOverview: React.FC = () => {
  const { currentTheme, isDarkTheme } = useTheme();
  const { currentUser } = useAuth();
  const location = useLocation();
  const navLinks = getRoleSidebarItems(currentUser?.role ?? 'viewer');

  const border = isDarkTheme ? 'rgba(148, 163, 184, 0.18)' : '#e2e8f0';
  const panel = isDarkTheme ? '#111827' : '#ffffff';
  const panelSoft = isDarkTheme ? '#172033' : '#f8fafc';
  const shell = isDarkTheme ? '#0f172a' : '#ffffff';
  const sidebarBg = isDarkTheme ? '#111827' : '#fbfbfe';
  const text = currentTheme.textPrimary;
  const textSoft = currentTheme.textSecondary;
  const muted = currentTheme.textMuted;

  const [overview, setOverview] = useState<OverviewPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const formatDateTime = (value?: string | null): string => {
    if (!value) return 'Unavailable';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleString();
  };

  const formatMetric = (value?: number | null, digits = 2): string => {
    if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
    return Number(value).toFixed(digits);
  };

  const statusColors: Record<ServiceStatus, { tone: string; border: string; glow: string }> = {
    healthy: { tone: '#10b981', border: 'rgba(16,185,129,0.24)', glow: 'rgba(16,185,129,0.12)' },
    degraded: { tone: '#f59e0b', border: 'rgba(245,158,11,0.24)', glow: 'rgba(245,158,11,0.12)' },
    offline: { tone: '#ef4444', border: 'rgba(239,68,68,0.24)', glow: 'rgba(239,68,68,0.12)' },
  };

  const loadOverview = useCallback(async (mode: 'initial' | 'refresh' = 'initial') => {
    try {
      if (mode === 'refresh') {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      setError(null);

      const response = await fetch(`${API_BASE}/system/overview`);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload?.detail || 'Failed to load system overview');
      }

      const payload = (await response.json()) as OverviewPayload;
      setOverview(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load system overview');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadOverview('initial');
    const timer = window.setInterval(() => {
      void loadOverview('refresh');
    }, 60000);
    return () => window.clearInterval(timer);
  }, [loadOverview]);

  const topCards = useMemo(() => {
    const kpis = overview?.kpis;
    return [
      { label: 'Platform Status', value: overview?.status ?? 'degraded', icon: Activity, color: statusColors[overview?.status ?? 'degraded'].tone, accent: 'Operational posture' },
      { label: 'Model Coverage', value: `${kpis?.loaded_models ?? 0}/${kpis?.expected_models ?? 3}`, icon: Brain, color: '#7c3aed', accent: 'Models loaded' },
      { label: 'Open Alerts', value: `${kpis?.open_alerts ?? 0}`, icon: AlertTriangle, color: '#f97316', accent: 'Active monitoring queue' },
      { label: 'Resolved Frauds', value: `${kpis?.resolved_frauds ?? 0}`, icon: ShieldCheck, color: '#10b981', accent: 'Confirmed archive' },
      { label: 'Reports Library', value: `${kpis?.reports_total ?? 0}`, icon: Server, color: '#2563eb', accent: 'Templates and hourly summaries' },
      { label: 'Active Sessions', value: `${kpis?.active_sessions ?? 0}`, icon: Users, color: '#14b8a6', accent: 'Authenticated users online' },
    ];
  }, [overview]);

  const serviceCards = useMemo(
    () =>
      overview
        ? [
            { label: 'API Core', detail: overview.services.api.detail, status: overview.services.api.status },
            { label: 'MongoDB', detail: overview.services.mongodb.detail, status: overview.services.mongodb.status },
            { label: 'Auth Sessions', detail: overview.services.auth.detail, status: overview.services.auth.status },
          ]
        : [],
    [overview],
  );

  const modelMixTotal = useMemo(
    () => Object.values(overview?.monitoring.alert_mix ?? {}).reduce((sum, value) => sum + value, 0),
    [overview],
  );

  return (
    <div style={{ minHeight: '100vh', background: currentTheme.bgPrimary, color: text }}>
      <div style={{ width: '100%', padding: '0 20px 28px' }}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '220px minmax(0, 1fr)',
            background: shell,
            borderLeft: `1px solid ${border}`,
            borderRight: `1px solid ${border}`,
            minHeight: 'calc(100vh - 110px)',
          }}
        >
          <aside
            style={{
              borderRight: `1px solid ${border}`,
              background: sidebarBg,
              padding: '22px 0',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <nav style={{ display: 'grid', gap: '8px', padding: '0 16px' }}>
              {navLinks.map(({ label, to }) => {
                const active = location.pathname === to;
                return (
                  <Link
                    key={label}
                    to={to}
                    style={{
                      padding: '12px 14px',
                      borderRadius: '14px',
                      textDecoration: 'none',
                      fontWeight: 700,
                      color: active ? text : textSoft,
                      background: active ? (isDarkTheme ? 'rgba(124, 58, 237, 0.16)' : '#f2efff') : 'transparent',
                      boxShadow: active
                        ? `inset 0 0 0 1px ${isDarkTheme ? 'rgba(124, 58, 237, 0.28)' : '#ded3ff'}`
                        : 'none',
                    }}
                  >
                    {label}
                  </Link>
                );
              })}
            </nav>
          </aside>

          <main style={{ padding: '24px', minWidth: 0 }}>
            <div
              style={{
                marginBottom: '22px',
                border: `1px solid ${border}`,
                borderRadius: '24px',
                background: isDarkTheme
                  ? 'linear-gradient(135deg, rgba(30,41,59,0.96), rgba(15,23,42,0.98))'
                  : 'linear-gradient(135deg, #ffffff, #f7fbff)',
                padding: '24px',
                boxShadow: isDarkTheme ? '0 22px 44px rgba(2,6,23,0.26)' : '0 20px 44px rgba(148,163,184,0.16)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ color: muted, fontSize: '.72rem', fontWeight: 700, letterSpacing: '.12em', textTransform: 'uppercase', marginBottom: '8px' }}>
                    System
                  </div>
                  <h1 style={{ margin: 0, fontSize: '2rem', fontWeight: 800 }}>System Overview</h1>
                  <p style={{ margin: '10px 0 0', color: muted, fontSize: '.98rem', maxWidth: 720, lineHeight: 1.7 }}>
                    Centralized read-only visibility into platform health, model readiness, monitoring queues, report generation, and recent operational activity.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '10px 14px', borderRadius: 999, border: `1px solid ${border}`, background: panelSoft, color: text, fontWeight: 700 }}>
                    <Lock size={15} color="#f59e0b" /> Read only
                  </div>
                  <button
                    type="button"
                    onClick={() => void loadOverview('refresh')}
                    disabled={refreshing}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '10px 14px', borderRadius: 999, border: `1px solid ${border}`, background: panel, color: text, fontWeight: 700, cursor: refreshing ? 'wait' : 'pointer' }}
                  >
                    <RefreshCw size={15} style={{ opacity: refreshing ? 0.6 : 1 }} />
                    {refreshing ? 'Refreshing...' : 'Refresh'}
                  </button>
                </div>
              </div>

              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 18 }}>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '10px 14px', borderRadius: 16, background: panelSoft, border: `1px solid ${border}` }}>
                  <ShieldCheck size={16} color="#2563eb" />
                  <span style={{ color: muted }}>Role</span>
                  <strong style={{ color: text, textTransform: 'capitalize' }}>{currentUser?.role ?? 'viewer'}</strong>
                </div>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '10px 14px', borderRadius: 16, background: panelSoft, border: `1px solid ${border}` }}>
                  <Activity size={16} color={statusColors[overview?.status ?? 'degraded'].tone} />
                  <span style={{ color: muted }}>Last refresh</span>
                  <strong style={{ color: text }}>{formatDateTime(overview?.generated_at)}</strong>
                </div>
              </div>
            </div>

            {error && (
              <div style={{ marginBottom: 18, borderRadius: 18, border: '1px solid rgba(239,68,68,0.24)', background: isDarkTheme ? 'rgba(127,29,29,0.22)' : '#fef2f2', padding: '14px 16px', color: isDarkTheme ? '#fecaca' : '#b91c1c', display: 'flex', alignItems: 'center', gap: 10 }}>
                <AlertTriangle size={18} />
                <span>{error}</span>
              </div>
            )}

            {loading ? (
              <div style={{ border: `1px solid ${border}`, borderRadius: '22px', background: panel, padding: '40px 24px', textAlign: 'center', color: muted }}>
                Loading system overview...
              </div>
            ) : (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 18 }}>
                  {topCards.map((card) => {
                    const Icon = card.icon;
                    return (
                      <div key={card.label} style={{ border: `1px solid ${border}`, borderRadius: '20px', background: panel, padding: '18px', boxShadow: isDarkTheme ? '0 18px 32px rgba(2,6,23,0.18)' : '0 14px 28px rgba(148,163,184,0.12)' }}>
                        <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: '40px', height: '40px', borderRadius: '14px', background: panelSoft, color: card.color, marginBottom: '14px' }}>
                          <Icon size={18} />
                        </div>
                        <div style={{ color: muted, fontSize: '.72rem', fontWeight: 700, letterSpacing: '.08em', textTransform: 'uppercase', marginBottom: '8px' }}>{card.label}</div>
                        <div style={{ fontSize: '1.3rem', fontWeight: 800, textTransform: 'capitalize' }}>{card.value}</div>
                        <div style={{ marginTop: 8, color: muted, fontSize: '.84rem' }}>{card.accent}</div>
                      </div>
                    );
                  })}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.15fr) minmax(0, 0.85fr)', gap: 18, marginBottom: 18 }}>
                  <section style={{ border: `1px solid ${border}`, borderRadius: '22px', background: panel, padding: '22px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                      <Server size={18} color="#2563eb" />
                      <div style={{ fontSize: '1.04rem', fontWeight: 800 }}>Service Health</div>
                    </div>
                    <div style={{ display: 'grid', gap: 12 }}>
                      {serviceCards.map((service) => {
                        const statusStyle = statusColors[service.status];
                        return (
                          <div key={service.label} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', padding: '14px 16px', borderRadius: '18px', background: isDarkTheme ? 'rgba(15,23,42,0.5)' : '#fbfdff', border: `1px solid ${statusStyle.border}` }}>
                            <div>
                              <div style={{ fontWeight: 700, color: text }}>{service.label}</div>
                              <div style={{ marginTop: 5, color: muted, fontSize: '.9rem' }}>{service.detail}</div>
                            </div>
                            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '8px 12px', borderRadius: 999, background: statusStyle.glow, color: statusStyle.tone, border: `1px solid ${statusStyle.border}`, fontWeight: 800, textTransform: 'capitalize' }}>
                              <span style={{ width: 8, height: 8, borderRadius: 999, background: statusStyle.tone }} />
                              {service.status}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </section>

                  <section style={{ border: `1px solid ${border}`, borderRadius: '22px', background: panel, padding: '22px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                      <Database size={18} color="#14b8a6" />
                      <div style={{ fontSize: '1.04rem', fontWeight: 800 }}>Operational Context</div>
                    </div>
                    <div style={{ display: 'grid', gap: 12 }}>
                      {[
                        { label: 'Users provisioned', value: `${overview?.kpis.users_total ?? 0}` },
                        { label: 'Batch runs in last 24h', value: `${overview?.kpis.batch_runs_last_24h ?? 0}` },
                        { label: 'Latest report generated', value: formatDateTime(overview?.reports.latest_generated_at) },
                        { label: 'Data sources', value: `${overview?.sources.length ?? 0} collections` },
                      ].map((item) => (
                        <div key={item.label} style={{ padding: '14px 16px', borderRadius: '18px', background: panelSoft, border: `1px solid ${border}` }}>
                          <div style={{ color: muted, fontSize: '.76rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 6 }}>{item.label}</div>
                          <div style={{ fontWeight: 700, color: text }}>{item.value}</div>
                        </div>
                      ))}
                    </div>
                  </section>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: 18, marginBottom: 18 }}>
                  <section style={{ border: `1px solid ${border}`, borderRadius: '22px', background: panel, padding: '22px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                      <AlertTriangle size={18} color="#f97316" />
                      <div style={{ fontSize: '1.04rem', fontWeight: 800 }}>Monitoring Summary</div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 16 }}>
                      {[
                        { label: 'Open', value: overview?.monitoring.open_alerts ?? 0, color: '#f97316' },
                        { label: 'Dismissed', value: overview?.monitoring.dismissed_alerts ?? 0, color: '#64748b' },
                        { label: 'False Positives', value: overview?.monitoring.false_positives ?? 0, color: '#0ea5e9' },
                        { label: 'Escalated', value: overview?.monitoring.escalated_alerts ?? 0, color: '#8b5cf6' },
                        { label: 'Resolved 24h', value: overview?.monitoring.resolved_frauds_last_24h ?? 0, color: '#10b981' },
                      ].map((item) => (
                        <div key={item.label} style={{ padding: '14px 14px 12px', borderRadius: '18px', border: `1px solid ${border}`, background: panelSoft }}>
                          <div style={{ color: muted, fontSize: '.76rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 8 }}>{item.label}</div>
                          <div style={{ fontSize: '1.45rem', fontWeight: 800, color: item.color }}>{item.value}</div>
                        </div>
                      ))}
                    </div>

                    <div style={{ display: 'grid', gap: 10 }}>
                      {Object.entries(overview?.monitoring.alert_mix ?? {}).map(([modelType, count]) => {
                        const width = modelMixTotal > 0 ? `${Math.max(12, (count / modelMixTotal) * 100)}%` : '12%';
                        return (
                          <div key={modelType}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 6, fontSize: '.92rem' }}>
                              <span style={{ color: text, fontWeight: 700 }}>{modelType === 'autoencoder' ? 'Autoencoder' : modelType.toUpperCase()}</span>
                              <span style={{ color: muted }}>{count} queued</span>
                            </div>
                            <div style={{ height: 10, borderRadius: 999, background: isDarkTheme ? '#0f172a' : '#e9eef5', overflow: 'hidden' }}>
                              <div style={{ width, height: '100%', borderRadius: 999, background: modelType === 'autoencoder' ? '#2563eb' : modelType === 'lstm' ? '#7c3aed' : '#f97316' }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </section>

                  <section style={{ border: `1px solid ${border}`, borderRadius: '22px', background: panel, padding: '22px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                      <Server size={18} color="#7c3aed" />
                      <div style={{ fontSize: '1.04rem', fontWeight: 800 }}>Reporting Context</div>
                    </div>
                    <div style={{ display: 'grid', gap: 12, marginBottom: 16 }}>
                      {[
                        { label: 'Model reports', value: `${overview?.reports.model_reports_total ?? 0}` },
                        { label: 'Hourly reports', value: `${overview?.reports.hourly_reports_total ?? 0}` },
                        { label: 'Latest output', value: formatDateTime(overview?.reports.latest_generated_at) },
                        { label: 'Batch runs archived', value: `${overview?.operations.batch_runs_total ?? 0}` },
                      ].map((item) => (
                        <div key={item.label} style={{ padding: '14px 16px', borderRadius: '18px', background: panelSoft, border: `1px solid ${border}` }}>
                          <div style={{ color: muted, fontSize: '.76rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 6 }}>{item.label}</div>
                          <div style={{ fontWeight: 700, color: text }}>{item.value}</div>
                        </div>
                      ))}
                    </div>

                    <div style={{ borderRadius: '18px', padding: '16px', background: isDarkTheme ? 'rgba(37,99,235,0.12)' : 'rgba(239,246,255,0.96)', border: '1px solid rgba(37,99,235,0.18)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                        <Lock size={16} color="#2563eb" />
                        <div style={{ fontWeight: 800, color: text }}>Read-only control boundary</div>
                      </div>
                      <div style={{ color: muted, lineHeight: 1.7, fontSize: '.92rem' }}>
                        This workspace summarizes platform state only. Administrative changes remain on their dedicated pages, keeping operational visibility separate from write actions.
                      </div>
                    </div>
                  </section>
                </div>

                <section style={{ border: `1px solid ${border}`, borderRadius: '22px', background: panel, padding: '22px', marginBottom: 18 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                    <Brain size={18} color="#7c3aed" />
                    <div style={{ fontSize: '1.04rem', fontWeight: 800 }}>Model Registry</div>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14 }}>
                    {overview?.models.map((model) => {
                      const health = model.loaded ? statusColors.healthy : statusColors.degraded;
                      return (
                        <div key={model.model_type} style={{ borderRadius: '20px', border: `1px solid ${health.border}`, background: panelSoft, padding: '18px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 12 }}>
                            <div>
                              <div style={{ fontWeight: 800, fontSize: '1.02rem', color: text }}>{model.display_name}</div>
                              <div style={{ color: muted, fontSize: '.86rem', marginTop: 4 }}>{model.architecture || 'Model metadata unavailable'}</div>
                            </div>
                            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '7px 10px', borderRadius: 999, background: health.glow, color: health.tone, border: `1px solid ${health.border}`, fontWeight: 800 }}>
                              <span style={{ width: 8, height: 8, borderRadius: 999, background: health.tone }} />
                              {model.loaded ? 'Loaded' : 'Standby'}
                            </div>
                          </div>

                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 10 }}>
                            {[
                              { label: 'Threshold', value: formatMetric(model.threshold, 4) },
                              { label: 'Features', value: String(model.feature_count ?? 'N/A') },
                              { label: 'Sequence', value: String(model.sequence_length ?? 'N/A') },
                              { label: 'Time Steps', value: String(model.time_steps ?? 'N/A') },
                            ].map((stat) => (
                              <div key={stat.label} style={{ padding: '12px 12px 10px', borderRadius: '14px', background: panel, border: `1px solid ${border}` }}>
                                <div style={{ color: muted, fontSize: '.72rem', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 6 }}>{stat.label}</div>
                                <div style={{ fontWeight: 800 }}>{stat.value}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>

                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.1fr) minmax(0, 0.9fr)', gap: 18 }}>
                  <section style={{ border: `1px solid ${border}`, borderRadius: '22px', background: panel, padding: '22px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                      <Activity size={18} color="#2563eb" />
                      <div style={{ fontSize: '1.04rem', fontWeight: 800 }}>Recent Batch Activity</div>
                    </div>
                    <div style={{ display: 'grid', gap: 12 }}>
                      {(overview?.operations.recent_batches.length ?? 0) === 0 ? (
                        <div style={{ color: muted }}>No batch history available yet.</div>
                      ) : (
                        overview?.operations.recent_batches.map((batch) => (
                          <div key={batch.batch_id} style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.2fr) repeat(3, minmax(90px, 0.55fr))', gap: 12, alignItems: 'center', padding: '14px 16px', borderRadius: '18px', background: panelSoft, border: `1px solid ${border}` }}>
                            <div style={{ minWidth: 0 }}>
                              <div style={{ fontWeight: 800, color: text }}>{batch.model_type === 'autoencoder' ? 'Autoencoder' : batch.model_type.toUpperCase()}</div>
                              <div style={{ marginTop: 4, color: muted, fontSize: '.88rem', overflowWrap: 'anywhere' }}>{batch.batch_id}</div>
                            </div>
                            <div>
                              <div style={{ color: muted, fontSize: '.72rem', textTransform: 'uppercase', letterSpacing: '.08em' }}>Flagged</div>
                              <div style={{ fontWeight: 800, color: '#ef4444' }}>{batch.fraud_count}</div>
                            </div>
                            <div>
                              <div style={{ color: muted, fontSize: '.72rem', textTransform: 'uppercase', letterSpacing: '.08em' }}>Records</div>
                              <div style={{ fontWeight: 800 }}>{batch.total_transactions}</div>
                            </div>
                            <div>
                              <div style={{ color: muted, fontSize: '.72rem', textTransform: 'uppercase', letterSpacing: '.08em' }}>Processed</div>
                              <div style={{ fontWeight: 800, fontSize: '.9rem' }}>{formatDateTime(batch.timestamp)}</div>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </section>

                  <section style={{ border: `1px solid ${border}`, borderRadius: '22px', background: panel, padding: '22px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                      <Activity size={18} color="#14b8a6" />
                      <div style={{ fontSize: '1.04rem', fontWeight: 800 }}>Recent Operational Events</div>
                    </div>
                    <div style={{ display: 'grid', gap: 14 }}>
                      {(overview?.operations.recent_events.length ?? 0) === 0 ? (
                        <div style={{ color: muted }}>No recent events available.</div>
                      ) : (
                        overview?.operations.recent_events.map((event, index) => (
                          <div key={`${event.type}-${index}`} style={{ display: 'grid', gridTemplateColumns: '18px minmax(0, 1fr)', gap: 12, alignItems: 'flex-start' }}>
                            <div style={{ display: 'flex', justifyContent: 'center' }}>
                              <div style={{ width: 10, height: 10, marginTop: 6, borderRadius: 999, background: event.type === 'resolution' ? '#10b981' : event.type === 'alert' ? '#f97316' : '#2563eb' }} />
                            </div>
                            <div style={{ padding: '12px 14px', borderRadius: '18px', background: panelSoft, border: `1px solid ${border}` }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 6, flexWrap: 'wrap' }}>
                                <div style={{ fontWeight: 800, color: text }}>{event.title}</div>
                                <div style={{ color: muted, fontSize: '.82rem' }}>{formatDateTime(event.timestamp)}</div>
                              </div>
                              <div style={{ color: muted, lineHeight: 1.6, fontSize: '.92rem' }}>{event.detail}</div>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </section>
                </div>
              </>
            )}
          </main>
        </div>
      </div>
    </div>
  );
};

export default SystemOverview;
