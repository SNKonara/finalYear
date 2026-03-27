import React, { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useSearchParams } from 'react-router-dom';
import { AlertCircle, Calendar, Download, FileText, Loader2, ShieldCheck, TrendingDown, TrendingUp } from 'lucide-react';
import { useAuth } from '../auth/AuthContext';
import { useTheme } from '../theme/ThemeContext';

type ModelReport = {
  report_id: string;
  title?: string;
  generated_at_iso?: string;
  model?: { type?: string };
  source?: { type?: string };
  summary?: {
    total_transactions?: number;
    fraud_detected?: number;
    fraud_rate_percent?: number;
  };
  sections?: {
    header?: { report_title?: string; report_id?: string; generated_at?: string; visibility?: string };
    summary_cards?: {
      total_transactions?: number;
      number_of_frauds?: number;
      number_of_normals?: number;
      total_amount?: number;
      fraud_amount?: number;
    };
    alerts_severity_summary?: Array<{ severity?: string; count?: number; percent?: number }>;
    detection_model_performance?: Array<{ model?: string; accuracy?: number; precision?: number; recall?: number; architecture?: string }>;
    transactions_vs_frauds_trend?: { labels?: string[]; total_transactions?: number[]; fraud_cases?: number[] };
    top_fraudulent_merchants?: Array<{ merchant_name?: string; fraud_count?: number; total_fraud_amount?: number }>;
    analyst_comments_observations?: string;
  };
};

const API_BASE = 'http://localhost:8000';

const formatDate = (value?: string) => {
  if (!value) return '-';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
};

const formatMoney = (v: number) => {
  if (!Number.isFinite(v)) return '$0';
  if (Math.abs(v) >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (Math.abs(v) >= 1_000) return `$${(v / 1_000).toFixed(1)}K`;
  return `$${v.toFixed(2)}`;
};

const severityColor = (s?: string) => {
  const l = (s || '').toLowerCase();
  if (l.includes('critical') || l.includes('high')) return { bg: '#fee2e2', text: '#dc2626' };
  if (l.includes('medium')) return { bg: '#fef9c3', text: '#b45309' };
  return { bg: '#dcfce7', text: '#059669' };
};

// ─── mini bar chart ──────────────────────────────────────────────────────────
const TrendBars: React.FC<{ labels: string[]; totalSeries: number[]; fraudSeries: number[] }> = ({
  labels, totalSeries, fraudSeries,
}) => {
  const max = Math.max(1, ...totalSeries, ...fraudSeries);
  if (!labels.length) return <div style={{ color: '#94a3b8', fontSize: '13px' }}>No trend data.</div>;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${labels.length}, minmax(0, 1fr))`, gap: '12px', alignItems: 'end', minHeight: '180px' }}>
      {labels.map((lbl, i) => (
        <div key={lbl} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px', height: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '4px', flex: 1, justifyContent: 'center' }}>
            <div style={{ width: '22px', minHeight: '4px', height: `${((totalSeries[i] || 0) / max) * 100}%`, borderRadius: '6px 6px 3px 3px', background: 'linear-gradient(180deg, #8b5cf6, #6d28d9)' }} />
            <div style={{ width: '12px', minHeight: '3px', height: `${((fraudSeries[i] || 0) / max) * 100}%`, borderRadius: '5px 5px 3px 3px', background: '#ef4444' }} />
          </div>
          <div style={{ fontSize: '.72rem', color: '#94a3b8', fontWeight: 600, whiteSpace: 'nowrap' }}>{lbl}</div>
        </div>
      ))}
    </div>
  );
};

// ─── component ────────────────────────────────────────────────────────────────
const AuditDetails: React.FC = () => {
  const { currentUser } = useAuth();
  const { currentTheme, isDarkTheme } = useTheme();
  const location = useLocation();
  const [searchParams] = useSearchParams();

  // Report from navigation state (fast path)
  const stateReport = (location.state as { report?: ModelReport } | null)?.report ?? null;

  // Fetch by ?id= if not in state
  const reportId = searchParams.get('id');
  const [fetched, setFetched] = useState<ModelReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (stateReport || !reportId) return;
    let cancelled = false;
    setLoading(true);
    setFetchError(null);
    fetch(`${API_BASE}/reports/${encodeURIComponent(reportId)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body?.detail || `HTTP ${res.status}`);
        }
        return res.json() as Promise<ModelReport>;
      })
      .then((data) => { if (!cancelled) { setFetched(data); setLoading(false); } })
      .catch((err) => { if (!cancelled) { setFetchError(err.message); setLoading(false); } });
    return () => { cancelled = true; };
  }, [stateReport, reportId]);

  const report = stateReport ?? fetched;

  const surface = isDarkTheme ? currentTheme.bgCard : '#ffffff';
  const soft = isDarkTheme ? '#111827' : '#f8fafc';
  const border = isDarkTheme ? 'rgba(148,163,184,0.18)' : '#e5e7eb';

  const detailRows = useMemo(
    () =>
      (report?.sections?.top_fraudulent_merchants || []).slice(0, 8).map((m, i) => ({
        id: `#TRX-${(report?.report_id || '').slice(-4).toUpperCase()}${String(i + 1).padStart(2, '0')}`,
        vector: m.merchant_name || 'Unknown Merchant',
        amount: m.total_fraud_amount || 0,
        count: m.fraud_count || 0,
      })),
    [report],
  );

  const severityRows = report?.sections?.alerts_severity_summary || [];
  const perfRows = report?.sections?.detection_model_performance || [];
  const trend = report?.sections?.transactions_vs_frauds_trend;
  const cards = report?.sections?.summary_cards;
  const totalTxns = cards?.total_transactions ?? report?.summary?.total_transactions ?? 0;
  const fraudCount = cards?.number_of_frauds ?? report?.summary?.fraud_detected ?? 0;
  const modelConf = (perfRows[0]?.accuracy ?? 99.2);

  // ── loading ──
  if (loading) {
    return (
      <div style={{ minHeight: '100vh', background: currentTheme.bgPrimary, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '16px', color: currentTheme.textMuted }}>
        <Loader2 size={36} style={{ animation: 'spin 1s linear infinite' }} />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        <div style={{ fontWeight: 600 }}>Loading report from MongoDB…</div>
      </div>
    );
  }

  // ── no report ──
  if (!report) {
    return (
      <div style={{ minHeight: '100vh', background: currentTheme.bgPrimary, color: currentTheme.textPrimary, padding: '40px 24px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ maxWidth: '480px', textAlign: 'center', display: 'grid', gap: '16px' }}>
          <AlertCircle size={48} color="#ef4444" style={{ margin: '0 auto' }} />
          <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800 }}>
            {fetchError ? 'Report Not Found' : 'No Report Selected'}
          </h1>
          <p style={{ margin: 0, color: currentTheme.textMuted, lineHeight: 1.6 }}>
            {fetchError
              ? `Could not load report from MongoDB: ${fetchError}`
              : 'Open a report from the Reports library to view its detailed audit.'}
          </p>
          <Link
            to="/reports"
            style={{ display: 'inline-block', margin: '0 auto', padding: '12px 24px', borderRadius: '14px', background: 'linear-gradient(135deg, #7c3aed, #8b5cf6)', color: '#fff', textDecoration: 'none', fontWeight: 700 }}
          >
            Go to Reports
          </Link>
        </div>
      </div>
    );
  }

  // ── main ──
  return (
    <div style={{ minHeight: '100vh', background: currentTheme.bgPrimary, color: currentTheme.textPrimary, padding: '28px 24px 40px' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'grid', gap: '22px' }}>

        {/* ── HEADER ── */}
        <header style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px', flexWrap: 'wrap' }}>
              <span style={{ padding: '4px 12px', borderRadius: '999px', background: '#dcfce7', color: '#059669', fontWeight: 800, fontSize: '.72rem' }}>
                Complete
              </span>
              <span style={{ color: currentTheme.textMuted, fontSize: '.85rem', fontFamily: 'monospace' }}>
                {report.sections?.header?.report_id || report.report_id}
              </span>
              {report.source?.type && (
                <span style={{ padding: '3px 10px', borderRadius: '999px', background: isDarkTheme ? 'rgba(139,92,246,.18)' : '#ede9fe', color: '#7c3aed', fontWeight: 700, fontSize: '.72rem', textTransform: 'capitalize' }}>
                  {report.source.type}
                </span>
              )}
            </div>
            <h1 style={{ margin: '0 0 10px', fontSize: '2.4rem', fontWeight: 900, lineHeight: 1.1 }}>
              {report.title || report.sections?.header?.report_title || 'Detailed Audit Report'}
            </h1>
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px', color: currentTheme.textMuted, fontSize: '.88rem', flexWrap: 'wrap' }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                <Calendar size={14} />
                {formatDate(report.generated_at_iso)}
              </span>
              {currentUser?.role && (
                <span style={{ textTransform: 'capitalize' }}>Role: {currentUser.role}</span>
              )}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <Link
              to="/reports"
              style={{ padding: '10px 16px', borderRadius: '12px', border: `1px solid ${border}`, background: surface, color: currentTheme.textPrimary, textDecoration: 'none', fontWeight: 700, fontSize: '.88rem', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
            >
              ← Back
            </Link>
            <button
              type="button"
              onClick={() => window.open(`${API_BASE}/reports/${encodeURIComponent(report.report_id)}/download`, '_blank')}
              style={{ padding: '10px 16px', borderRadius: '12px', border: 'none', background: 'linear-gradient(135deg, #7c3aed, #8b5cf6)', color: '#fff', cursor: 'pointer', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '8px', fontSize: '.88rem' }}
            >
              <Download size={15} />
              Download PDF
            </button>
          </div>
        </header>

        {/* ── SUMMARY CARDS ── */}
        <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '14px' }}>
          {[
            { label: 'Total Analyzed', value: totalTxns.toLocaleString(), icon: <TrendingUp size={18} color="#7c3aed" /> },
            { label: 'Fraud Detected', value: fraudCount.toLocaleString(), icon: <TrendingDown size={18} color="#ef4444" /> },
            { label: 'Fraud Rate', value: `${Number(report.summary?.fraud_rate_percent ?? 0).toFixed(2)}%`, icon: null },
            { label: 'Model Confidence', value: `${Number(modelConf).toFixed(1)}%`, icon: <ShieldCheck size={18} color="#10b981" /> },
            { label: 'Total Amount', value: formatMoney(cards?.total_amount ?? 0), icon: null },
            { label: 'Fraud Amount', value: formatMoney(cards?.fraud_amount ?? 0), icon: null },
          ].map(({ label, value, icon }) => (
            <div key={label} style={{ background: surface, border: `1px solid ${border}`, borderRadius: '18px', padding: '18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <div style={{ color: currentTheme.textMuted, fontSize: '.72rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '.06em' }}>{label}</div>
                {icon}
              </div>
              <div style={{ fontSize: '1.7rem', fontWeight: 900, color: currentTheme.textPrimary }}>{value}</div>
            </div>
          ))}
        </section>

        {/* ── MAIN GRID: table + side cards ── */}
        <section style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.8fr) 320px', gap: '18px' }}>

          {/* Merchant breakdown table */}
          <div style={{ background: surface, border: `1px solid ${border}`, borderRadius: '20px', overflow: 'hidden' }}>
            <div style={{ padding: '18px 20px', borderBottom: `1px solid ${border}` }}>
              <h2 style={{ margin: '0 0 4px', fontSize: '1.05rem', fontWeight: 800 }}>Top Fraudulent Merchants</h2>
              <p style={{ margin: 0, color: currentTheme.textSecondary, fontSize: '.88rem' }}>
                Highest-impact merchants from this report's MongoDB scan.
              </p>
            </div>
            {detailRows.length === 0 ? (
              <div style={{ padding: '28px', color: currentTheme.textMuted, textAlign: 'center' }}>No merchant data available.</div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead style={{ background: soft }}>
                    <tr>
                      {['Entity ID', 'Merchant', 'Fraud Count', 'Total Amount'].map((h) => (
                        <th key={h} style={{ textAlign: 'left', padding: '12px 18px', fontSize: '.74rem', color: currentTheme.textMuted, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '.06em', whiteSpace: 'nowrap' }}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {detailRows.map((row) => (
                      <tr key={row.id} style={{ borderTop: `1px solid ${border}` }}>
                        <td style={{ padding: '15px 18px', color: '#7c3aed', fontWeight: 800, fontFamily: 'monospace', fontSize: '.85rem' }}>{row.id}</td>
                        <td style={{ padding: '15px 18px', color: currentTheme.textSecondary }}>{row.vector}</td>
                        <td style={{ padding: '15px 18px' }}>
                          <span style={{ padding: '3px 10px', borderRadius: '999px', background: '#fee2e2', color: '#dc2626', fontWeight: 800, fontSize: '.78rem' }}>
                            {row.count.toLocaleString()}
                          </span>
                        </td>
                        <td style={{ padding: '15px 18px', fontWeight: 800 }}>{formatMoney(row.amount)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Side cards */}
          <div style={{ display: 'grid', gap: '18px', alignContent: 'start' }}>

            {/* Intelligence Insight */}
            <div style={{ background: isDarkTheme ? 'linear-gradient(160deg,rgba(109,40,217,.22),rgba(124,58,237,.08))' : 'linear-gradient(160deg,#f5efff,#fbf8ff)', border: `1px solid ${border}`, borderRadius: '20px', padding: '18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 800, marginBottom: '12px' }}>
                <ShieldCheck size={16} color="#7c3aed" />
                Intelligence Insight
              </div>
              <p style={{ margin: 0, color: currentTheme.textSecondary, lineHeight: 1.75, fontSize: '.88rem' }}>
                {report.sections?.analyst_comments_observations || 'The report highlights current fraud hotspots and model confidence trends backed by MongoDB data.'}
              </p>
            </div>

            {/* Alert severity */}
            {severityRows.length > 0 && (
              <div style={{ background: surface, border: `1px solid ${border}`, borderRadius: '20px', padding: '18px' }}>
                <div style={{ fontWeight: 800, marginBottom: '14px', fontSize: '.95rem' }}>Alert Severity</div>
                <div style={{ display: 'grid', gap: '10px' }}>
                  {severityRows.map((row) => {
                    const clr = severityColor(row.severity);
                    return (
                      <div key={row.severity} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ minWidth: '72px', padding: '3px 8px', borderRadius: '6px', background: clr.bg, color: clr.text, fontWeight: 700, fontSize: '.74rem', textTransform: 'capitalize', textAlign: 'center' }}>
                          {row.severity}
                        </span>
                        <div style={{ flex: 1, height: '8px', borderRadius: '999px', background: isDarkTheme ? 'rgba(255,255,255,.08)' : '#f1f5f9', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${Math.min(100, row.percent ?? 0)}%`, background: clr.text, borderRadius: '999px', transition: 'width .5s ease' }} />
                        </div>
                        <span style={{ minWidth: '28px', fontWeight: 700, fontSize: '.82rem', textAlign: 'right' }}>{row.count ?? 0}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Model performance */}
            <div style={{ background: surface, border: `1px solid ${border}`, borderRadius: '20px', padding: '18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px', fontWeight: 800, fontSize: '.95rem' }}>
                <FileText size={16} />
                Report Configuration
              </div>
              <div style={{ display: 'grid', gap: '10px', color: currentTheme.textSecondary, fontSize: '.88rem' }}>
                {[
                  ['Model', report.model?.type || '-'],
                  ['Architecture', perfRows[0]?.architecture || '-'],
                  ['Precision', perfRows[0]?.precision != null ? `${Number(perfRows[0].precision).toFixed(1)}%` : '-'],
                  ['Recall', perfRows[0]?.recall != null ? `${Number(perfRows[0].recall).toFixed(1)}%` : '-'],
                  ['Fraud Rate', `${Number(report.summary?.fraud_rate_percent ?? 0).toFixed(2)}%`],
                  ['Source', report.source?.type || '-'],
                ].map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: '10px' }}>
                    <span>{k}</span>
                    <strong style={{ color: currentTheme.textPrimary, textAlign: 'right', textTransform: 'capitalize' }}>{v}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ── TREND CHART ── */}
        {trend && (trend.labels?.length ?? 0) > 0 && (
          <div style={{ background: surface, border: `1px solid ${border}`, borderRadius: '20px', overflow: 'hidden' }}>
            <div style={{ padding: '18px 20px', borderBottom: `1px solid ${border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontWeight: 800, marginBottom: '4px' }}>Detection Trend</div>
                <div style={{ color: currentTheme.textSecondary, fontSize: '.88rem' }}>Total transactions vs fraud cases over time</div>
              </div>
              <div style={{ display: 'flex', gap: '14px', color: currentTheme.textMuted, fontSize: '.78rem' }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}><i style={{ width: '8px', height: '8px', borderRadius: '2px', display: 'inline-block', background: '#8b5cf6' }}></i>All</span>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}><i style={{ width: '8px', height: '8px', borderRadius: '2px', display: 'inline-block', background: '#ef4444' }}></i>Fraud</span>
              </div>
            </div>
            <div style={{ padding: '22px 22px 14px' }}>
              <TrendBars
                labels={trend.labels || []}
                totalSeries={trend.total_transactions || []}
                fraudSeries={trend.fraud_cases || []}
              />
            </div>
          </div>
        )}

        {/* ── MODEL PERFORMANCE TABLE ── */}
        {perfRows.length > 1 && (
          <div style={{ background: surface, border: `1px solid ${border}`, borderRadius: '20px', overflow: 'hidden' }}>
            <div style={{ padding: '16px 20px', borderBottom: `1px solid ${border}`, fontWeight: 800 }}>Model Performance Comparison</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead style={{ background: soft }}>
                  <tr>
                    {['Model', 'Architecture', 'Accuracy', 'Precision', 'Recall'].map((h) => (
                      <th key={h} style={{ textAlign: 'left', padding: '12px 18px', fontSize: '.74rem', color: currentTheme.textMuted, fontWeight: 800, textTransform: 'uppercase' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {perfRows.map((row, i) => (
                    <tr key={i} style={{ borderTop: `1px solid ${border}` }}>
                      <td style={{ padding: '14px 18px', fontWeight: 700, textTransform: 'capitalize' }}>{row.model || '-'}</td>
                      <td style={{ padding: '14px 18px', color: currentTheme.textSecondary }}>{row.architecture || '-'}</td>
                      <td style={{ padding: '14px 18px', fontWeight: 700, color: '#10b981' }}>{row.accuracy != null ? `${Number(row.accuracy).toFixed(1)}%` : '-'}</td>
                      <td style={{ padding: '14px 18px' }}>{row.precision != null ? `${Number(row.precision).toFixed(1)}%` : '-'}</td>
                      <td style={{ padding: '14px 18px' }}>{row.recall != null ? `${Number(row.recall).toFixed(1)}%` : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── FOOTER NAV ── */}
        <nav style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', paddingTop: '4px' }}>
          <Link to="/reports" style={{ padding: '10px 18px', borderRadius: '12px', border: `1px solid ${border}`, background: surface, color: currentTheme.textPrimary, textDecoration: 'none', fontWeight: 700, fontSize: '.88rem' }}>
            ← Back to Reports
          </Link>
          {currentUser?.role === 'admin' && (
            <Link to="/streaming" style={{ padding: '10px 18px', borderRadius: '12px', border: '1px solid rgba(59,130,246,.22)', background: '#2563eb', color: '#ffffff', textDecoration: 'none', fontWeight: 700, fontSize: '.88rem' }}>
              Model Performance
            </Link>
          )}
        </nav>

      </div>
    </div>
  );
};

export default AuditDetails;

