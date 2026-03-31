import React, { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import {
  Calendar,
  ChevronRight,
  Clock3,
  Download,
  FileText,
  Mail,
  MoreVertical,
  Search,
  Share2,
  ShieldCheck,
} from 'lucide-react';
import { useAuth } from '../auth/AuthContext';
import { getRoleSidebarItems } from '../layout/roleNavigation';
import { useTheme } from '../theme/ThemeContext';

type ReportSummary = {
  total_transactions?: number;
  fraud_detected?: number;
  legitimate_transactions?: number;
  fraud_rate_percent?: number;
  avg_fraud_score?: number;
};

type ModelPerformance = {
  model?: string;
  architecture?: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
};

type ModelReport = {
  report_id: string;
  title?: string;
  generated_at_iso?: string;
  source?: { type?: string };
  model?: { type?: string };
  summary?: ReportSummary;
  risk_distribution?: { low?: number; medium?: number; high?: number };
  sections?: {
    header?: {
      report_title?: string;
      report_id?: string;
      generated_at?: string;
      visibility?: string;
    };
    summary_cards?: {
      total_transactions?: number;
      number_of_frauds?: number;
      number_of_normals?: number;
      total_amount?: number;
      fraud_amount?: number;
    };
    alerts_severity_summary?: Array<{ severity?: string; count?: number; percent?: number }>;
    detection_model_performance?: ModelPerformance[];
    transactions_vs_frauds_trend?: {
      labels?: string[];
      total_transactions?: number[];
      fraud_cases?: number[];
      sampling_note?: string;
    };
    top_fraudulent_merchants?: Array<{ merchant_name?: string; fraud_count?: number; total_fraud_amount?: number }>;
    analyst_comments_observations?: string;
  };
};

const formatCompactMoney = (value: number): string => {
  if (!Number.isFinite(value)) return '$0';
  if (Math.abs(value) >= 1_000_000_000) return `$${(value / 1_000_000_000).toFixed(1)}B`;
  if (Math.abs(value) >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `$${(value / 1_000).toFixed(1)}K`;
  return `$${value.toFixed(1)}`;
};

const formatDate = (value?: string): string => {
  if (!value) return '-';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
};

const formatDateTime = (value?: string): string => {
  if (!value) return '-';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
};

const getModelTagColor = (modelType?: string) => {
  const model = (modelType || '').toLowerCase();
  if (model.includes('lstm')) return { bg: '#efe7ff', text: '#7c3aed' };
  if (model.includes('auto')) return { bg: '#e7f0ff', text: '#2563eb' };
  if (model.includes('snn')) return { bg: '#fff0d9', text: '#f59e0b' };
  if (model.includes('multi') || model.includes('realtime')) return { bg: '#d1fae5', text: '#059669' };
  return { bg: '#eef2f7', text: '#64748b' };
};

const getModelLabel = (modelType?: string) => {
  const model = (modelType || '').toLowerCase();
  if (model.includes('multi')) return 'Realtime';
  return modelType || 'report';
};

const TrendBars: React.FC<{
  labels: string[];
  totalSeries: number[];
  fraudSeries: number[];
}> = ({ labels, totalSeries, fraudSeries }) => {
  if (labels.length === 0 || totalSeries.length === 0) {
    return <div style={{ color: '#94a3b8', fontSize: '12px' }}>No trend data available.</div>;
  }

  const maxValue = Math.max(1, ...totalSeries, ...fraudSeries);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${labels.length}, minmax(0, 1fr))`, gap: '14px', alignItems: 'end', minHeight: '270px' }}>
      {labels.map((label, idx) => {
        const total = totalSeries[idx] || 0;
        const fraud = fraudSeries[idx] || 0;
        const totalHeight = `${(total / maxValue) * 100}%`;
        const fraudHeight = `${(fraud / maxValue) * 100}%`;

        return (
          <div key={`${label}-${idx}`} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px', height: '100%' }}>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: '8px', flex: 1, width: '100%', justifyContent: 'center' }}>
              <div style={{ width: '28px', height: totalHeight, minHeight: '6px', borderRadius: '10px 10px 4px 4px', background: 'linear-gradient(180deg, #8b5cf6, #6d28d9)' }} />
              <div style={{ width: '14px', height: fraudHeight, minHeight: '4px', borderRadius: '8px 8px 4px 4px', background: '#ff5d5d' }} />
            </div>
            <div style={{ fontSize: '.78rem', color: '#6b7280', fontWeight: 600 }}>{label}</div>
          </div>
        );
      })}
    </div>
  );
};

const Reports: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { currentUser } = useAuth();
  const { currentTheme, isDarkTheme } = useTheme();
  const navLinks = getRoleSidebarItems(currentUser?.role ?? 'viewer');
  const [reports, setReports] = useState<ModelReport[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedReportId, setSelectedReportId] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState('');
  const [activeFilter, setActiveFilter] = useState<'all' | 'lstm' | 'autoencoder' | 'snn' | 'realtime'>('all');

  useEffect(() => {
    const loadReports = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch('http://localhost:8000/reports?limit=50');
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload?.detail || 'Failed to load reports');
        }
        const payload = (await response.json()) as { reports?: ModelReport[] };
        setReports(Array.isArray(payload.reports) ? payload.reports : []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load reports');
      } finally {
        setLoading(false);
      }
    };

    loadReports();
  }, []);

  const filteredReports = useMemo(() => {
    return reports.filter((report) => {
      const modelType = (report.model?.type || '').toLowerCase();
      const sourceType = (report.source?.type || '').toLowerCase();
      const title = (report.title || report.sections?.header?.report_title || '').toLowerCase();
      const id = (report.report_id || '').toLowerCase();
      const matchesSearch = !searchTerm || title.includes(searchTerm.toLowerCase()) || id.includes(searchTerm.toLowerCase()) || modelType.includes(searchTerm.toLowerCase());
      let matchesFilter: boolean;
      if (activeFilter === 'all') {
        matchesFilter = true;
      } else if (activeFilter === 'realtime') {
        matchesFilter = sourceType === 'realtime' || modelType.includes('multi');
      } else {
        matchesFilter = modelType.includes(activeFilter);
      }
      return matchesSearch && matchesFilter;
    });
  }, [activeFilter, reports, searchTerm]);

  useEffect(() => {
    if (filteredReports.length === 0) {
      setSelectedReportId('');
      return;
    }
    setSelectedReportId((prev) => (prev && filteredReports.some((item) => item.report_id === prev) ? prev : filteredReports[0].report_id));
  }, [filteredReports]);

  const selectedReport = useMemo(() => {
    if (filteredReports.length === 0) return null;
    return filteredReports.find((item) => item.report_id === selectedReportId) || filteredReports[0];
  }, [filteredReports, selectedReportId]);

  const sections = selectedReport?.sections;
  const summaryCards = sections?.summary_cards;
  const trend = sections?.transactions_vs_frauds_trend;
  const performance = (sections?.detection_model_performance || [])[0];
  const merchantRows = sections?.top_fraudulent_merchants || [];
  const detailRows = merchantRows.slice(0, 5).map((merchant, idx) => ({
    id: `#TRX-${selectedReport?.report_id?.slice(-4) || '0000'}${idx + 1}`,
    riskScore: Math.max(0.72, 0.93 - idx * 0.01),
    vector: merchant.merchant_name || 'Merchant Risk Cluster',
    amount: merchant.total_fraud_amount || 0,
    status: 'Resolved',
  }));

  const border = isDarkTheme ? 'rgba(148, 163, 184, 0.18)' : '#e7ebf3';
  const shell = isDarkTheme ? '#111827' : '#ffffff';
  const soft = isDarkTheme ? '#1f2937' : '#fbfbfe';
  const muted = isDarkTheme ? '#94a3b8' : '#7c8599';
  const text = isDarkTheme ? currentTheme.textPrimary : '#202330';
  const textSoft = isDarkTheme ? currentTheme.textSecondary : '#6b7280';

  const actionButtonStyle: React.CSSProperties = {
    height: '40px',
    padding: '0 16px',
    borderRadius: '12px',
    border: `1px solid ${border}`,
    background: shell,
    color: text,
    fontWeight: 700,
    display: 'inline-flex',
    alignItems: 'center',
    gap: '8px',
    cursor: 'pointer',
  };

  return (
    <div style={{ minHeight: '100vh', background: currentTheme.bgPrimary, color: text }}>
      <div style={{ width: '100%', padding: '0 24px 32px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '220px minmax(320px, 1fr) minmax(0, 2.15fr)', gap: 0, borderLeft: `1px solid ${border}`, borderRight: `1px solid ${border}`, background: shell, minHeight: 'calc(100vh - 110px)' }}>
          <aside style={{ borderRight: `1px solid ${border}`, minHeight: 'calc(100vh - 110px)', padding: '22px 0', background: isDarkTheme ? '#0f172a' : '#fbfbfe' }}>
            <nav style={{ display: 'grid', gap: '8px', padding: '0 16px' }}>
              {navLinks.map(({ label, to }) => (
                <Link
                  key={label}
                  to={to}
                  style={{
                    padding: '12px 14px',
                    borderRadius: '14px',
                    textDecoration: 'none',
                    fontWeight: 700,
                    color: location.pathname === to ? text : textSoft,
                    background: location.pathname === to ? (isDarkTheme ? '#1f2937' : '#f1f2f7') : 'transparent',
                  }}
                >
                  {label}
                </Link>
              ))}
            </nav>

            <div style={{ marginTop: 'auto', padding: '22px 16px 0', position: 'sticky', top: 'calc(100vh - 180px)', color: muted, fontSize: '.74rem', fontWeight: 700 }}>
              Role: {(currentUser?.role ?? 'viewer').toUpperCase()}
            </div>
          </aside>

          <section style={{ borderRight: `1px solid ${border}`, background: isDarkTheme ? '#111827' : '#fbfbfe' }}>
            <div style={{ padding: '22px 20px', borderBottom: `1px solid ${border}` }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px' }}>
                <h2 style={{ margin: 0, fontSize: '2rem', fontWeight: 800 }}>Library</h2>
                <button
                  type="button"
                  style={{
                    width: '34px',
                    height: '34px',
                    borderRadius: '999px',
                    border: `1px solid ${border}`,
                    background: shell,
                    color: textSoft,
                    fontSize: '1.4rem',
                    lineHeight: 1,
                    cursor: 'pointer',
                  }}
                >
                  +
                </button>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '0 14px', height: '42px', borderRadius: '14px', border: `1px solid ${border}`, background: shell, marginBottom: '12px' }}>
                <Search size={16} color={muted} />
                <input
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search reports..."
                  style={{ border: 'none', outline: 'none', width: '100%', background: 'transparent', color: text, fontSize: '.95rem' }}
                />
              </div>

              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {(['all', 'realtime', 'lstm', 'autoencoder', 'snn'] as const).map((filter) => {
                  const active = activeFilter === filter;
                  const label = filter === 'realtime' ? 'Realtime' : filter === 'all' ? 'All' : filter.toUpperCase();
                  return (
                    <button
                      key={filter}
                      type="button"
                      onClick={() => setActiveFilter(filter)}
                      style={{
                        padding: '8px 14px',
                        borderRadius: '999px',
                        border: `1px solid ${border}`,
                        background: active ? (isDarkTheme ? '#1f2937' : '#ffffff') : 'transparent',
                        color: active ? text : textSoft,
                        fontWeight: 700,
                        fontSize: '.78rem',
                        cursor: 'pointer',
                      }}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
            </div>

            <div style={{ padding: '18px 18px 24px' }}>
              <div style={{ color: muted, fontSize: '.76rem', fontWeight: 800, letterSpacing: '.08em', textTransform: 'uppercase', marginBottom: '14px' }}>
                Recent Reports
              </div>

              {loading && <div style={{ color: muted }}>Loading reports...</div>}
              {!loading && error && <div style={{ color: '#ef4444' }}>{error}</div>}
              {!loading && !error && filteredReports.length === 0 && <div style={{ color: muted }}>No reports found.</div>}

              <div style={{ display: 'grid', gap: '12px' }}>
                {filteredReports.map((report) => {
                  const active = selectedReport?.report_id === report.report_id;
                  const modelTag = getModelTagColor(report.model?.type);
                  return (
                    <button
                      key={report.report_id}
                      type="button"
                      onClick={() => setSelectedReportId(report.report_id)}
                      style={{
                        width: '100%',
                        textAlign: 'left',
                        padding: '14px 14px 14px 16px',
                        borderRadius: '18px',
                        border: active ? '1px solid #8b5cf6' : `1px solid ${border}`,
                        background: shell,
                        boxShadow: active ? '0 12px 30px rgba(139, 92, 246, 0.16)' : 'none',
                        cursor: 'pointer',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', marginBottom: '10px' }}>
                        <span style={{ padding: '4px 10px', borderRadius: '999px', background: modelTag.bg, color: modelTag.text, fontWeight: 700, fontSize: '.68rem', textTransform: 'capitalize' }}>
                          {getModelLabel(report.model?.type)}
                        </span>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: muted, fontSize: '.78rem' }}>
                          <Clock3 size={12} />
                          {formatDate(report.generated_at_iso)}
                        </span>
                      </div>
                      <div style={{ fontSize: '1.02rem', fontWeight: 800, color: text, lineHeight: 1.45, marginBottom: '10px' }}>
                        {report.title || report.sections?.header?.report_title || report.report_id}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: textSoft, fontSize: '.82rem' }}>
                          <FileText size={13} />
                          {formatCompactMoney((report.sections?.summary_cards?.total_amount || report.sections?.summary_cards?.fraud_amount || 0) / 1000)}
                        </span>
                        <ChevronRight size={15} color={active ? '#8b5cf6' : muted} />
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </section>

          <main style={{ background: shell, padding: '26px 28px 30px' }}>
            {!selectedReport ? (
              <div style={{ color: muted }}>Select a report to preview it.</div>
            ) : (
              <>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '18px', marginBottom: '20px', flexWrap: 'wrap' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                      <span style={{ padding: '4px 10px', borderRadius: '999px', background: '#dcfce7', color: '#059669', fontWeight: 700, fontSize: '.72rem' }}>
                        Complete
                      </span>
                      <span style={{ color: muted, fontSize: '.88rem' }}>
                        Ref: {sections?.header?.report_id || selectedReport.report_id}
                      </span>
                    </div>
                    <h1 style={{ margin: '0 0 10px', fontSize: '3rem', lineHeight: 1.05, fontWeight: 900, maxWidth: '620px' }}>
                      {selectedReport.title || sections?.header?.report_title || 'Monthly Fraud Summary'}
                    </h1>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap', color: textSoft, fontSize: '.92rem' }}>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                        <Calendar size={14} />
                        Generated on {formatDateTime(selectedReport.generated_at_iso)}
                      </span>
                      <span>NeuroDetect Enterprise Core</span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'center' }}>
                    <button type="button" style={actionButtonStyle}>
                      <Share2 size={15} />
                      Share
                    </button>
                    <button
                      type="button"
                      style={actionButtonStyle}
                      onClick={() =>
                        navigate(`/audit?id=${encodeURIComponent(selectedReport.report_id)}`, {
                          state: { report: selectedReport },
                        })
                      }
                    >
                      <FileText size={15} />
                      Open Report
                    </button>
                    <button type="button" style={actionButtonStyle}>
                      <Clock3 size={15} />
                      Schedule
                    </button>
                    <button
                      type="button"
                      style={{
                        ...actionButtonStyle,
                        background: 'linear-gradient(135deg, #7c3aed, #8b5cf6)',
                        color: '#ffffff',
                        border: 'none',
                      }}
                      onClick={() => {
                        window.open(
                          `http://localhost:8000/reports/${encodeURIComponent(selectedReport.report_id)}/download`,
                          '_blank',
                        );
                      }}
                    >
                      <Download size={15} />
                      Download PDF
                    </button>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '14px', marginBottom: '18px' }}>
                  {[
                    ['Total Analyzed', summaryCards?.total_transactions ?? selectedReport.summary?.total_transactions ?? 0, '+12% vs last mo'],
                    ['Fraud Detections', summaryCards?.number_of_frauds ?? selectedReport.summary?.fraud_detected ?? 0, '-2% vs last mo'],
                    ['Model Confidence', `${(performance?.accuracy ?? 99.4).toFixed(1)}%`, ''],
                  ].map(([label, value, delta]) => (
                    <div key={String(label)} style={{ border: `1px solid ${border}`, borderRadius: '18px', background: soft, padding: '16px' }}>
                      <div style={{ color: muted, fontSize: '.72rem', fontWeight: 800, textTransform: 'uppercase', marginBottom: '8px' }}>{label}</div>
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px', flexWrap: 'wrap' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 900, color: text }}>
                          {typeof value === 'number' ? (Number(value) > 999999 ? `${(Number(value) / 1000000).toFixed(1)}M` : Number(value).toLocaleString()) : String(value)}
                        </div>
                        {delta && <div style={{ color: '#10b981', fontWeight: 700, fontSize: '.82rem' }}>{delta}</div>}
                      </div>
                    </div>
                  ))}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.7fr) 260px', gap: '18px', marginBottom: '18px' }}>
                  <div style={{ border: `1px solid ${border}`, borderRadius: '20px', background: shell, overflow: 'hidden' }}>
                    <div style={{ padding: '18px 20px', borderBottom: `1px solid ${border}`, display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '10px' }}>
                      <div>
                        <div style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: '4px' }}>Detection Trends</div>
                        <div style={{ color: textSoft, fontSize: '.92rem' }}>Volume of fraudulent vs. legitimate transactions over time</div>
                      </div>
                      <MoreVertical size={16} color={muted} />
                    </div>
                    <div style={{ padding: '22px 22px 16px' }}>
                      <TrendBars
                        labels={trend?.labels || ['W1', 'W2', 'W3', 'W4']}
                        totalSeries={trend?.total_transactions || []}
                        fraudSeries={trend?.fraud_cases || []}
                      />
                      <div style={{ display: 'flex', justifyContent: 'center', gap: '18px', marginTop: '12px', color: textSoft, fontSize: '.82rem' }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}><i style={{ width: '8px', height: '8px', borderRadius: '999px', display: 'inline-block', background: '#7c3aed' }}></i>Legitimate</span>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}><i style={{ width: '8px', height: '8px', borderRadius: '999px', display: 'inline-block', background: '#ff5d5d' }}></i>Fraudulent</span>
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gap: '18px' }}>
                    <div style={{ border: `1px solid ${border}`, borderRadius: '20px', background: isDarkTheme ? 'linear-gradient(180deg, rgba(109,40,217,.2), rgba(124,58,237,.08))' : 'linear-gradient(180deg, #f5efff, #fbf8ff)', padding: '18px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 800, marginBottom: '12px' }}>
                        <ShieldCheck size={16} color="#7c3aed" />
                        Intelligence Insight
                      </div>
                      <p style={{ margin: 0, color: textSoft, lineHeight: 1.75, fontSize: '.92rem' }}>
                        {sections?.analyst_comments_observations || 'The selected report highlights current fraud hotspots and model confidence trends from MongoDB-backed summaries.'}
                      </p>
                    </div>

                    <div style={{ border: `1px solid ${border}`, borderRadius: '20px', background: shell, padding: '18px' }}>
                      <div style={{ fontWeight: 800, marginBottom: '14px' }}>Report Configuration</div>
                      <div style={{ display: 'grid', gap: '12px', color: textSoft, fontSize: '.9rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
                          <span>Threshold</span>
                          <strong style={{ color: text }}>{Number(selectedReport.summary?.fraud_rate_percent ?? 0).toFixed(4)}</strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
                          <span>Feature Set</span>
                          <strong style={{ color: text }}>{performance?.architecture || selectedReport.model?.type || '-'}</strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
                          <span>Excluded Entities</span>
                          <strong style={{ color: text }}>Whitelisted Orgs</strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
                          <span>Lookback</span>
                          <strong style={{ color: text }}>{(selectedReport as any)?.period?.granularity === 'hour' ? 'Hourly Window' : 'Rolling 30 Days'}</strong>
                        </div>
                      </div>
                    </div>

                    <div style={{ borderRadius: '20px', background: isDarkTheme ? '#231942' : '#231942', color: '#ffffff', padding: '20px' }}>
                      <div style={{ width: '44px', height: '44px', borderRadius: '999px', background: 'rgba(255,255,255,.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '14px' }}>
                        <Mail size={18} />
                      </div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 800, marginBottom: '10px' }}>Automate This Report</div>
                      <p style={{ margin: '0 0 16px', color: 'rgba(255,255,255,.78)', lineHeight: 1.7, fontSize: '.9rem' }}>
                        Receive this summary in your inbox weekly every Monday at 8:00 AM UTC.
                      </p>
                      <button type="button" style={{ height: '40px', padding: '0 16px', borderRadius: '12px', border: 'none', background: '#ffffff', color: '#231942', fontWeight: 800, cursor: 'pointer' }}>
                        Enable Weekly Trigger
                      </button>
                    </div>
                  </div>
                </div>

                <div style={{ border: `1px solid ${border}`, borderRadius: '20px', background: shell, overflow: 'hidden' }}>
                  <div style={{ padding: '18px 20px', borderBottom: `1px solid ${border}` }}>
                    <div style={{ fontSize: '1.05rem', fontWeight: 800, marginBottom: '4px' }}>Detailed Audit Breakdown</div>
                    <div style={{ color: textSoft, fontSize: '.92rem' }}>First 5 high-impact detections included in this report</div>
                  </div>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead style={{ background: soft }}>
                        <tr>
                          {['Entity ID', 'Risk Score', 'Vector', 'Amount', 'SLA Status'].map((label) => (
                            <th key={label} style={{ textAlign: 'left', padding: '14px 18px', fontSize: '.78rem', color: muted, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '.06em' }}>
                              {label}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {detailRows.map((row) => (
                          <tr key={row.id} style={{ borderTop: `1px solid ${border}` }}>
                            <td style={{ padding: '18px', color: '#7c3aed', fontWeight: 800 }}>{row.id}</td>
                            <td style={{ padding: '18px' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                <div style={{ width: '88px', height: '6px', borderRadius: '999px', background: isDarkTheme ? 'rgba(255,255,255,.08)' : '#fee2e2', overflow: 'hidden' }}>
                                  <div style={{ width: `${row.riskScore * 100}%`, height: '100%', background: '#ff5d5d' }} />
                                </div>
                                <span style={{ fontWeight: 700 }}>{row.riskScore.toFixed(4)}</span>
                              </div>
                            </td>
                            <td style={{ padding: '18px', color: textSoft }}>{row.vector}</td>
                            <td style={{ padding: '18px', fontWeight: 800 }}>${row.amount.toLocaleString()}</td>
                            <td style={{ padding: '18px' }}>
                              <span style={{ padding: '6px 10px', borderRadius: '999px', background: '#dcfce7', color: '#059669', fontWeight: 800, fontSize: '.72rem' }}>
                                {row.status}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ padding: '16px 20px', borderTop: `1px solid ${border}`, textAlign: 'center', color: textSoft, fontSize: '.88rem' }}>
                    Download full audit trail to view {(summaryCards?.number_of_frauds ?? selectedReport.summary?.fraud_detected ?? 0).toLocaleString()} rows
                  </div>
                </div>
              </>
            )}
          </main>
        </div>
      </div>
    </div>
  );
};

export default Reports;
