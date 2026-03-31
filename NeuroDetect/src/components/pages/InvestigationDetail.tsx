import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  ChevronRight,
  Clock3,
  FileText,
  ShieldAlert,
  Siren,
  Sparkles,
} from 'lucide-react';
import { useTheme } from '../theme/ThemeContext';
import { useAuth } from '../auth/AuthContext';
import { getRoleSidebarItems } from '../layout/roleNavigation';

type InvestigationDetailResponse = {
  alert_id: string;
  transaction_id: string;
  title: string;
  model_label: string;
  model_type: string;
  risk_level: string;
  fraud_score: number;
  decision_threshold: number;
  risk_percent: number;
  is_fraud: boolean;
  status: string;
  alerted_at: string;
  dismissed_at?: string;
  resolved_at?: string;
  resolved_by?: {
    id?: string;
    name?: string;
    email?: string;
    role?: string;
  };
  amount: number;
  merchant: string;
  category: string;
  city: string;
  state: string;
  job: string;
  raw_transaction: Record<string, unknown>;
  raw_alert: Record<string, unknown>;
  evidence: Array<{
    event_id: string;
    timestamp: string;
    attribute: string;
    raw_value: string;
    deviation: string;
  }>;
  confidence_pct?: number;
  score_series?: number[];
  explainability?: {
    reasons: string[];
    top_factors: Array<{
      feature: string;
      value: string;
      contribution_pct: number;
    }>;
    feature_contribution_chart: { labels: string[]; values: number[] };
    threshold_chart: { probability: number; threshold: number; margin: number };
    risk_dimension_chart: { labels: string[]; values: number[] };
  };
};

type ResolveFraudResponse = {
  success: boolean;
  message: string;
  result: {
    status: string;
    transaction_id: string;
    resolved_by: {
      id?: string;
      name?: string;
      email?: string;
      role?: string;
    };
  };
};

const API_BASE = 'http://localhost:8000';

const formatMoney = (value: number): string =>
  `$${Number(value || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const formatStatus = (value: string): string => {
  const normalized = String(value || '').replace(/_/g, ' ').trim();
  if (!normalized) return 'Pending Review';
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
};

const buildFeatureContributions = (detail: InvestigationDetailResponse) => {
  if (detail.explainability?.top_factors && detail.explainability.top_factors.length > 0) {
    return detail.explainability.top_factors.map((f) => ({
      label: f.feature,
      value: Math.round(f.contribution_pct),
    }));
  }
  const risk = Math.max(0, Math.min(100, Number(detail.risk_percent) || 0));
  return [
    { label: 'Amount Deviation', value: Math.min(98, Math.round(risk * 0.96)) },
    { label: 'Geo Velocity', value: Math.min(95, Math.round(risk * 0.84)) },
    { label: 'Merchant Risk', value: Math.min(88, Math.round(risk * 0.58)) },
    { label: 'Temporal Anomaly', value: Math.min(72, Math.round(risk * 0.4)) },
    { label: 'Device Integrity', value: Math.min(64, Math.round(risk * 0.22)) },
  ];
};

const InvestigationDetail: React.FC = () => {
  const { alertId } = useParams();
  const navigate = useNavigate();
  const { currentTheme, isDarkTheme } = useTheme();
  const { currentUser } = useAuth();
  const [detail, setDetail] = useState<InvestigationDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!alertId) {
      setError('Missing alert id');
      setLoading(false);
      return;
    }

    const loadDetail = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(`${API_BASE}/investigations/alerts/${encodeURIComponent(alertId)}`);
        if (!response.ok) {
          throw new Error('Failed to load investigation detail');
        }
        const payload: InvestigationDetailResponse = await response.json();
        setDetail(payload);
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : 'Failed to load investigation detail');
      } finally {
        setLoading(false);
      }
    };

    loadDetail();
  }, [alertId]);

  const handleConfirmAsFraud = useCallback(async () => {
    if (!detail?.alert_id || actionLoading) {
      return;
    }

    setActionLoading(true);
    setActionError(null);
    setActionMessage(null);

    try {
      const response = await fetch(`${API_BASE}/investigations/alerts/resolve-fraud`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          alert_id: detail.alert_id,
          transaction_id: detail.transaction_id,
          analyst_id: currentUser?.id || '',
          analyst_name: currentUser?.name || 'Unknown Analyst',
          analyst_email: currentUser?.email || '',
          analyst_role: currentUser?.role || 'analyst',
          resolution_note: `Confirmed as fraud by ${currentUser?.name || 'analyst'} from investigation detail.`,
        }),
      });

      const payload = (await response.json()) as ResolveFraudResponse | { detail?: string };
      if (!response.ok) {
        const errorDetail = 'detail' in payload && typeof payload.detail === 'string' ? payload.detail : 'Failed to confirm alert as fraud';
        throw new Error(errorDetail);
      }

      const successPayload = payload as ResolveFraudResponse;
      setDetail((previous) =>
        previous
          ? {
              ...previous,
              status: successPayload.result.status,
              resolved_at: new Date().toISOString(),
              resolved_by: successPayload.result.resolved_by,
            }
          : previous,
      );
      setActionMessage(successPayload.message);
      window.setTimeout(() => {
        navigate('/investigations');
      }, 900);
    } catch (requestError) {
      setActionError(requestError instanceof Error ? requestError.message : 'Failed to confirm alert as fraud');
    } finally {
      setActionLoading(false);
    }
  }, [actionLoading, currentUser, detail, navigate]);

  const handleFalsePositive = useCallback(async () => {
    if (!detail?.alert_id || actionLoading) return;
    setActionLoading(true);
    setActionError(null);
    setActionMessage(null);
    try {
      const response = await fetch(`${API_BASE}/investigations/alerts/false-positive`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          alert_id: detail.alert_id,
          transaction_id: detail.transaction_id,
          analyst_name: currentUser?.name || 'investigator',
          analyst_note: `Marked as false positive by ${currentUser?.name || 'investigator'}.`,
        }),
      });
      const data = (await response.json()) as { success?: boolean; message?: string; detail?: string };
      if (!response.ok) throw new Error(data.detail || 'Failed to mark as false positive');
      setDetail((prev) => (prev ? { ...prev, status: 'false_positive' } : prev));
      setActionMessage(data.message || 'Alert marked as false positive');
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to mark as false positive');
    } finally {
      setActionLoading(false);
    }
  }, [actionLoading, currentUser, detail]);

  const handleEscalate = useCallback(async () => {
    if (!detail?.alert_id || actionLoading) return;
    setActionLoading(true);
    setActionError(null);
    setActionMessage(null);
    try {
      const response = await fetch(`${API_BASE}/investigations/alerts/escalate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          alert_id: detail.alert_id,
          transaction_id: detail.transaction_id,
          analyst_name: currentUser?.name || 'investigator',
          escalation_note: `Escalated by ${currentUser?.name || 'investigator'} from investigation detail.`,
        }),
      });
      const data = (await response.json()) as { success?: boolean; message?: string; detail?: string };
      if (!response.ok) throw new Error(data.detail || 'Failed to escalate alert');
      setDetail((prev) => (prev ? { ...prev, status: 'escalated' } : prev));
      setActionMessage(data.message || 'Alert escalated to senior analyst');
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to escalate alert');
    } finally {
      setActionLoading(false);
    }
  }, [actionLoading, currentUser, detail]);

  const border = isDarkTheme ? 'rgba(148, 163, 184, 0.16)' : '#e6ebf4';
  const shell = isDarkTheme ? '#101827' : '#fbfbfc';
  const sidebarBg = isDarkTheme ? '#111827' : '#fcfcff';
  const panel = isDarkTheme ? '#0f172a' : '#ffffff';
  const panelSoft = isDarkTheme ? '#172033' : '#f7f8fc';
  const text = currentTheme.textPrimary;
  const textSoft = currentTheme.textSecondary;
  const muted = currentTheme.textMuted;
  const accent = '#7c3aed';
  const accentSoft = isDarkTheme ? 'rgba(124,58,237,.18)' : '#efe7ff';
  const coral = '#ff6363';
  const shadow = isDarkTheme ? '0 18px 38px rgba(2, 6, 23, 0.28)' : '0 16px 34px rgba(148, 163, 184, 0.12)';
  const navLinks = getRoleSidebarItems(currentUser?.role ?? 'analyst');

  const chartSeries = useMemo(() => {
    if (detail?.score_series && detail.score_series.length > 0) {
      return detail.score_series;
    }
    const risk = Math.max(0.1, Math.min(0.98, detail?.fraud_score || 0.5));
    return [0.12, 0.15, 0.16, 0.17, 0.22, risk, risk * 0.8, risk * 0.52, risk * 0.34];
  }, [detail]);

  const chartWidth = 620;
  const chartHeight = 210;
  const chartPath = chartSeries
    .map((value, index) => {
      const x = 18 + (index * (chartWidth - 36)) / Math.max(chartSeries.length - 1, 1);
      const y = chartHeight - 18 - value * 150;
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
    })
    .join(' ');

  const contributionBars = detail ? buildFeatureContributions(detail) : [];
  const prettySource = JSON.stringify(detail?.raw_transaction || {}, null, 2);

  const timelineRows = [
    {
      icon: CheckCircle2,
      label: 'Authorization Requested',
      description: `Incoming payload received from ${detail?.merchant || 'merchant gateway'}.`,
      color: '#22c55e',
      time: detail?.alerted_at || '--:--',
    },
    {
      icon: Sparkles,
      label: 'Feature Extraction',
      description: `${detail?.evidence?.length || 0} source attributes prepared for model scoring.`,
      color: '#06b6d4',
      time: detail?.alerted_at || '--:--',
    },
    {
      icon: ShieldAlert,
      label: 'Model Detection',
      description: `${detail?.model_label || 'Model'} exceeded threshold ${Number(detail?.decision_threshold || 0).toFixed(2)}.`,
      color: '#f59e0b',
      time: detail?.alerted_at || '--:--',
    },
    {
      icon: Siren,
      label: 'Decision: Flagged',
      description: `Risk score finalized at ${detail?.risk_percent || 0}/100 for analyst review.`,
      color: accent,
      time: detail?.alerted_at || '--:--',
    },
  ];

  const activityRows = [
    { actor: 'System Agent', action: 'Flagged high risk', time: detail?.alerted_at || '--:--' },
    { actor: 'Analyst Queue', action: 'Opened investigation', time: detail?.alerted_at || '--:--' },
    { actor: detail?.model_label || 'Model', action: 'Attached forensic evidence', time: detail?.alerted_at || '--:--' },
  ];

  if (detail?.resolved_by?.name) {
    activityRows.push({
      actor: detail.resolved_by.name,
      action: 'Confirmed and archived as resolved fraud',
      time: detail.resolved_at || '--:--',
    });
  }

  return (
    <div style={{ minHeight: '100vh', background: currentTheme.bgPrimary, color: text }}>
      <div style={{ width: '100%', padding: '0 20px 28px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '220px minmax(0, 1fr)', borderLeft: `1px solid ${border}`, borderRight: `1px solid ${border}`, background: shell, minHeight: 'calc(100vh - 110px)' }}>
          <aside style={{ borderRight: `1px solid ${border}`, background: sidebarBg, padding: '22px 0', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <nav style={{ display: 'grid', gap: '8px', padding: '0 16px' }}>
              {navLinks.map(({ label, to }) => {
                const active = to === '/investigations';
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
                      boxShadow: active ? `inset 0 0 0 1px ${isDarkTheme ? 'rgba(124,58,237,.28)' : '#ded3ff'}` : 'none',
                    }}
                  >
                    {label}
                  </Link>
                );
              })}
            </nav>

            <div style={{ padding: '0 16px', color: muted, fontSize: '.72rem', fontWeight: 700 }}>
              Role: {(currentUser?.role ?? 'analyst').toUpperCase()}
            </div>
          </aside>

          <main style={{ background: shell, display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '16px 22px', borderBottom: `1px solid ${border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '18px', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '14px', minWidth: 0 }}>
                <button
                  type="button"
                  onClick={() => navigate('/investigations')}
                  style={{ width: '36px', height: '36px', borderRadius: '12px', border: `1px solid ${border}`, background: panel, color: text, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
                >
                  <ArrowLeft size={16} />
                </button>
                <div>
                  <div style={{ color: muted, fontSize: '.68rem', fontWeight: 700, letterSpacing: '.12em', textTransform: 'uppercase' }}>Forensic Detail</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                    <div style={{ fontWeight: 800 }}>{detail?.transaction_id || 'Loading...'}</div>
                    <div style={{ fontSize: '1.18rem', fontWeight: 800 }}>{formatMoney(detail?.amount || 0)}</div>
                    <div style={{ color: textSoft, fontSize: '.88rem' }}>{detail?.merchant || 'Unknown merchant'} {detail?.category ? `• ${detail.category}` : ''}</div>
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '18px', flexWrap: 'wrap' }}>
                <div>
                  <div style={{ color: muted, fontSize: '.66rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '6px' }}>Risk Score</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ width: '96px', height: '6px', borderRadius: '999px', background: isDarkTheme ? 'rgba(255,255,255,.08)' : '#eceff7', overflow: 'hidden' }}>
                      <div style={{ width: `${detail?.risk_percent || 0}%`, height: '100%', background: coral }} />
                    </div>
                    <div style={{ color: coral, fontWeight: 700, fontSize: '.84rem' }}>{detail?.risk_percent || 0}/100</div>
                  </div>
                </div>
                <div>
                  <div style={{ color: muted, fontSize: '.66rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '6px' }}>Status</div>
                  <div style={{ padding: '7px 12px', borderRadius: '999px', background: isDarkTheme ? 'rgba(255,99,99,.12)' : '#fff1f1', color: coral, fontWeight: 700, fontSize: '.76rem' }}>
                    {formatStatus(detail?.status || 'pending review')}
                  </div>
                </div>
              </div>
            </div>

            <div style={{ padding: '22px', display: 'grid', gap: '18px' }}>
              {loading ? (
                <div style={{ border: `1px solid ${border}`, borderRadius: '18px', background: panel, padding: '18px', color: muted }}>Loading investigation detail...</div>
              ) : null}
              {error ? (
                <div style={{ border: `1px solid ${border}`, borderRadius: '18px', background: panel, padding: '18px', color: coral }}>{error}</div>
              ) : null}

              {!loading && !error && detail ? (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.9fr) minmax(260px, .9fr)', gap: '18px' }}>
                    <section style={{ border: `1px solid ${border}`, borderRadius: '18px', background: panel, padding: '18px 18px 14px', boxShadow: shadow }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '10px', flexWrap: 'wrap' }}>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                            <Sparkles size={14} color={accent} />
                            <div style={{ fontWeight: 700, fontSize: '.92rem' }}>{detail.model_label} Reconstruction Error</div>
                          </div>
                          <div style={{ color: muted, fontSize: '.82rem' }}>Neural anomaly detection over transaction processing cycle</div>
                        </div>
                        <div style={{ padding: '5px 10px', borderRadius: '999px', background: accentSoft, color: accent, fontSize: '.68rem', fontWeight: 700 }}>
                          Model: {detail.model_label}
                        </div>
                      </div>

                      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} style={{ width: '100%', height: '220px', display: 'block' }}>
                        {[0.25, 0.5, 0.75].map((tick) => {
                          const y = chartHeight - 18 - tick * 150;
                          return (
                            <g key={tick}>
                              <line x1="18" x2={chartWidth - 18} y1={y} y2={y} stroke={isDarkTheme ? 'rgba(148,163,184,.16)' : '#e5e7eb'} strokeDasharray="4 5" />
                              <text x="2" y={y + 4} fill={muted} fontSize="11">{tick.toFixed(2)}</text>
                            </g>
                          );
                        })}
                        <line x1="18" x2={chartWidth - 18} y1={chartHeight - 18 - (detail.decision_threshold || 0.5) * 150} y2={chartHeight - 18 - (detail.decision_threshold || 0.5) * 150} stroke="#ff9a8b" strokeDasharray="5 5" />
                        <path d={`${chartPath} L ${chartWidth - 18} ${chartHeight - 18} L 18 ${chartHeight - 18} Z`} fill={isDarkTheme ? 'rgba(124,58,237,.16)' : 'rgba(124,58,237,.14)'} />
                        <path d={chartPath} fill="none" stroke={accent} strokeWidth="3" />
                      </svg>
                    </section>

                    <div style={{ display: 'grid', gap: '18px' }}>
                      <section style={{ border: `1px solid ${border}`, borderRadius: '18px', background: accentSoft, padding: '18px', boxShadow: shadow }}>
                        <div style={{ color: accent, fontSize: '.7rem', fontWeight: 700, letterSpacing: '.12em', textTransform: 'uppercase', marginBottom: '18px' }}>Model Confidence</div>
                        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '14px' }}>
                          <div style={{ width: '120px', height: '120px', borderRadius: '999px', border: `8px solid ${accent}`, display: 'flex', alignItems: 'center', justifyContent: 'center', background: panel }}>
                            <div style={{ textAlign: 'center' }}>
                              <div style={{ color: accent, fontSize: '1.7rem', fontWeight: 800 }}>{(detail.confidence_pct ?? Number(detail.fraud_score || 0) * 100).toFixed(1)}%</div>
                              <div style={{ color: muted, fontSize: '.62rem', letterSpacing: '.08em', textTransform: 'uppercase' }}>Certainty</div>
                            </div>
                          </div>
                        </div>
                        <div style={{ color: muted, fontSize: '.8rem', lineHeight: 1.65, textAlign: 'center' }}>
                          Detection certainty based on sequence deviation and anomaly concentration.
                        </div>
                      </section>

                      <section style={{ border: `1px solid ${border}`, borderRadius: '18px', background: panel, padding: '18px', boxShadow: shadow }}>
                        <div style={{ fontWeight: 700, fontSize: '.86rem', marginBottom: '14px' }}>Feature Contributions</div>
                        <div style={{ display: 'grid', gap: '12px' }}>
                          {contributionBars.map((item, index) => (
                            <div key={item.label}>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px', fontSize: '.76rem' }}>
                                <span style={{ color: textSoft }}>{item.label}</span>
                                <span style={{ color: muted }}>{item.value}%</span>
                              </div>
                              <div style={{ height: '6px', borderRadius: '999px', background: panelSoft, overflow: 'hidden' }}>
                                <div style={{ width: `${item.value}%`, height: '100%', background: index < 2 ? coral : index === 2 ? '#f59e0b' : accent }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      </section>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 1fr) minmax(320px, 1.1fr) minmax(260px, .9fr)', gap: '18px' }}>
                    <section style={{ border: `1px solid ${border}`, borderRadius: '18px', background: panel, padding: '18px', boxShadow: shadow }}>
                      <div style={{ fontWeight: 700, fontSize: '.84rem', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}><Clock3 size={14} /> Event Lifecycle</div>
                      <div style={{ display: 'grid', gap: '16px' }}>
                        {timelineRows.map((row) => {
                          const Icon = row.icon;
                          return (
                            <div key={row.label} style={{ display: 'grid', gridTemplateColumns: '28px minmax(0, 1fr)', gap: '12px' }}>
                              <div style={{ width: '28px', height: '28px', borderRadius: '999px', border: `1px solid ${row.color}`, color: row.color, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <Icon size={14} />
                              </div>
                              <div>
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
                                  <div style={{ fontWeight: 700, fontSize: '.8rem' }}>{row.label}</div>
                                  <div style={{ color: muted, fontSize: '.7rem' }}>{row.time}</div>
                                </div>
                                <div style={{ color: muted, fontSize: '.76rem', lineHeight: 1.65, marginTop: '4px' }}>{row.description}</div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </section>

                    <section style={{ border: `1px solid ${border}`, borderRadius: '18px', background: panel, padding: '18px', boxShadow: shadow }}>
                      <div style={{ fontWeight: 700, fontSize: '.84rem', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}><FileText size={14} /> Source Evidence</div>
                      <div style={{ padding: '14px', borderRadius: '14px', background: isDarkTheme ? '#0b1020' : '#0f172a', color: '#dbe4ff', fontSize: '.75rem', lineHeight: 1.6, minHeight: '264px', overflow: 'auto', whiteSpace: 'pre-wrap', fontFamily: 'Consolas, Monaco, monospace' }}>
                        {prettySource}
                      </div>
                    </section>

                    <section style={{ display: 'grid', gap: '18px' }}>
                      <div style={{ border: `1px solid ${isDarkTheme ? 'rgba(255,99,99,.22)' : '#ffd4d4'}`, borderRadius: '18px', background: panel, padding: '18px', boxShadow: shadow }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: coral, fontWeight: 700, fontSize: '.84rem', marginBottom: '14px' }}>
                          <AlertTriangle size={15} /> Action Center
                        </div>
                        <div style={{ display: 'grid', gap: '10px' }}>
                          <button
                            type="button"
                            onClick={() => {
                              void handleConfirmAsFraud();
                            }}
                            disabled={actionLoading || detail.status === 'resolved_fraud'}
                            style={{
                              height: '42px',
                              borderRadius: '12px',
                              border: 'none',
                              background: actionLoading || detail.status === 'resolved_fraud' ? '#f59b9b' : coral,
                              color: '#fff',
                              fontWeight: 700,
                              cursor: actionLoading || detail.status === 'resolved_fraud' ? 'not-allowed' : 'pointer',
                            }}
                          >
                            {detail.status === 'resolved_fraud' ? 'Resolved as Fraud' : actionLoading ? 'Confirming...' : 'Confirm as Fraud'}
                          </button>
                          <button
                            type="button"
                            onClick={() => { void handleFalsePositive(); }}
                            disabled={actionLoading || detail.status === 'false_positive'}
                            style={{ height: '42px', borderRadius: '12px', border: `1px solid ${border}`, background: actionLoading || detail.status === 'false_positive' ? panelSoft : panelSoft, color: detail.status === 'false_positive' ? '#22c55e' : text, fontWeight: 700, cursor: actionLoading || detail.status === 'false_positive' ? 'not-allowed' : 'pointer' }}
                          >
                            {detail.status === 'false_positive' ? 'Marked as False Positive' : actionLoading ? 'Processing...' : 'Mark as False Positive'}
                          </button>
                          <button
                            type="button"
                            onClick={() => { void handleEscalate(); }}
                            disabled={actionLoading || detail.status === 'escalated'}
                            style={{ height: '42px', borderRadius: '12px', border: `1px solid ${border}`, background: 'transparent', color: detail.status === 'escalated' ? '#f59e0b' : textSoft, fontWeight: 700, cursor: actionLoading || detail.status === 'escalated' ? 'not-allowed' : 'pointer' }}
                          >
                            {detail.status === 'escalated' ? 'Escalated' : actionLoading ? 'Processing...' : 'Escalate to Senior Analyst'}
                          </button>
                        </div>
                        {actionError ? <div style={{ marginTop: '12px', color: coral, fontSize: '.76rem', lineHeight: 1.5 }}>{actionError}</div> : null}
                        {actionMessage ? <div style={{ marginTop: '12px', color: '#22c55e', fontSize: '.76rem', lineHeight: 1.5 }}>{actionMessage}</div> : null}
                        <div style={{ marginTop: '16px', paddingTop: '14px', borderTop: `1px solid ${border}` }}>
                          <div style={{ color: muted, fontSize: '.66rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '8px' }}>Investigation Queue</div>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 12px', borderRadius: '12px', background: panelSoft, fontSize: '.78rem' }}>
                            <span>Priority: {detail.risk_level}</span>
                            <ChevronRight size={14} color={muted} />
                          </div>
                          <div style={{ marginTop: '12px', color: textSoft, fontSize: '.76rem', lineHeight: 1.6 }}>
                            Analyst: {currentUser?.name || 'Unknown Analyst'}{currentUser?.email ? ` • ${currentUser.email}` : ''}
                          </div>
                        </div>
                      </div>

                      <div style={{ border: `1px solid ${border}`, borderRadius: '18px', background: panel, padding: '18px', boxShadow: shadow }}>
                        <div style={{ fontWeight: 700, fontSize: '.84rem', marginBottom: '12px' }}>Transaction Context</div>
                        <div style={{ display: 'grid', gap: '10px', fontSize: '.78rem' }}>
                          {[
                            ['City', detail.city || 'N/A'],
                            ['State', detail.state || 'N/A'],
                            ['Category', detail.category || 'N/A'],
                            ['Profile', detail.job || 'N/A'],
                          ].map(([label, value]) => (
                            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: '10px' }}>
                              <span style={{ color: muted }}>{label}</span>
                              <span style={{ color: text, fontWeight: 700 }}>{value}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </section>
                  </div>

                  <section style={{ border: `1px solid ${border}`, borderRadius: '18px', background: panel, padding: '18px', boxShadow: shadow }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '12px' }}>
                      <div>
                        <div style={{ fontWeight: 700, fontSize: '.9rem' }}>Audit Trail & Collaboration</div>
                        <div style={{ color: muted, fontSize: '.76rem' }}>Regulatory log of all automated and manual actions</div>
                      </div>
                      <button type="button" style={{ height: '34px', padding: '0 12px', borderRadius: '10px', border: `1px solid ${border}`, background: panelSoft, color: textSoft, cursor: 'pointer', fontSize: '.76rem', fontWeight: 700 }}>View Full History</button>
                    </div>
                    <div style={{ display: 'grid', gap: '12px' }}>
                      {activityRows.map((row) => (
                        <div key={`${row.actor}-${row.action}`} style={{ display: 'grid', gridTemplateColumns: '40px minmax(0, 1fr) auto', gap: '12px', alignItems: 'center', padding: '10px 0', borderTop: `1px solid ${border}` }}>
                          <div style={{ width: '40px', height: '40px', borderRadius: '999px', background: `linear-gradient(135deg, ${accent}, #2563eb)`, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700 }}>
                            {row.actor.split(' ').map((part) => part[0]).join('').slice(0, 2)}
                          </div>
                          <div>
                            <div style={{ fontWeight: 700, fontSize: '.82rem' }}>{row.actor}</div>
                            <div style={{ color: muted, fontSize: '.75rem' }}>{row.action}</div>
                          </div>
                          <div style={{ color: muted, fontSize: '.72rem' }}>{row.time}</div>
                        </div>
                      ))}
                    </div>
                  </section>
                </>
              ) : null}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};

export default InvestigationDetail;