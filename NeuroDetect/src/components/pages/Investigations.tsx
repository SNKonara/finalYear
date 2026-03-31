import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Clock3, Filter, Search } from 'lucide-react';
import { useTheme } from '../theme/ThemeContext';
import { useAuth } from '../auth/AuthContext';
import { getRoleSidebarItems } from '../layout/roleNavigation';

type CasePriority = 'Critical' | 'High' | 'Medium' | 'Low';
type ModelFilter = 'all' | 'autoencoder' | 'lstm' | 'snn';

type InvestigationAlertItem = {
  alert_id: string;
  transaction_id: string;
  title: string;
  model_type: string;
  model_label: string;
  risk_percent: number;
  fraud_score: number;
  risk_level: string;
  status: string;
  opened_ago_minutes: number | null;
  amount: number;
  merchant: string;
};

type InvestigationAlertsResponse = {
  model_counts: {
    autoencoder: number;
    lstm: number;
    snn: number;
    total: number;
  };
  alerts: InvestigationAlertItem[];
};

type InvestigationQueueItem = {
  alertId: string;
  transactionId: string;
  title: string;
  risk: number;
  amount: number;
  merchant: string;
  modelLabel: string;
  openedAgo: string;
  priority: CasePriority;
  modelType: string;
};

const API_BASE = 'http://localhost:8000';

const formatOpenedAgo = (minutes: number | null | undefined): string => {
  if (typeof minutes !== 'number' || Number.isNaN(minutes)) {
    return 'n/a';
  }
  if (minutes < 60) {
    return `${Math.max(0, Math.round(minutes))}m ago`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
};

const toPriority = (riskPercent: number): CasePriority => {
  if (riskPercent >= 90) return 'Critical';
  if (riskPercent >= 75) return 'High';
  if (riskPercent >= 50) return 'Medium';
  return 'Low';
};

const toQueueItem = (alert: InvestigationAlertItem): InvestigationQueueItem => ({
  alertId: alert.alert_id,
  transactionId: alert.transaction_id,
  title: alert.title,
  risk: Number(alert.risk_percent) || 0,
  amount: Number(alert.amount) || 0,
  merchant: alert.merchant || 'N/A',
  modelLabel: alert.model_label,
  openedAgo: formatOpenedAgo(alert.opened_ago_minutes),
  priority: toPriority(Number(alert.risk_percent) || 0),
  modelType: alert.model_type,
});

const Investigations: React.FC = () => {
  const { currentTheme, isDarkTheme } = useTheme();
  const { currentUser } = useAuth();
  const navigate = useNavigate();
  const [queueItems, setQueueItems] = useState<InvestigationQueueItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [loadError, setLoadError] = useState<string>('');
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);
  const [modelCounts, setModelCounts] = useState<{ autoencoder: number; lstm: number; snn: number }>({
    autoencoder: 0,
    lstm: 0,
    snn: 0,
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedModelFilter, setSelectedModelFilter] = useState<ModelFilter>('all');

  const fetchInvestigations = useCallback(async () => {
    setIsLoading(true);
    setLoadError('');
    try {
      const response = await fetch(`${API_BASE}/investigations/alerts?hours=24&status=open&limit=300`);
      if (!response.ok) {
        throw new Error('Failed to load investigation alerts');
      }
      const payload: InvestigationAlertsResponse = await response.json();

      setModelCounts({
        autoencoder: Number(payload.model_counts?.autoencoder) || 0,
        lstm: Number(payload.model_counts?.lstm) || 0,
        snn: Number(payload.model_counts?.snn) || 0,
      });
      setQueueItems((payload.alerts || []).map(toQueueItem));
      setLastUpdatedAt(new Date());
    } catch {
      setLoadError('Unable to load last 24h fraud alerts from MongoDB.');
      setQueueItems([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchInvestigations();
  }, [fetchInvestigations]);

  const filteredItems = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    return queueItems.filter(
      (item) =>
        (selectedModelFilter === 'all' || item.modelType === selectedModelFilter) &&
        (!query ||
          item.transactionId.toLowerCase().includes(query) ||
          item.title.toLowerCase().includes(query) ||
          item.merchant.toLowerCase().includes(query)),
    );
  }, [queueItems, searchTerm, selectedModelFilter]);

  const border = isDarkTheme ? 'rgba(148, 163, 184, 0.18)' : '#e6ebf4';
  const shell = isDarkTheme ? '#0f172a' : '#ffffff';
  const sidebarBg = isDarkTheme ? '#111827' : '#fbfbfe';
  const panel = isDarkTheme ? '#111827' : '#ffffff';
  const text = currentTheme.textPrimary;
  const textSoft = currentTheme.textSecondary;
  const muted = currentTheme.textMuted;
  const accent = '#7c3aed';
  const accentSoft = isDarkTheme ? 'rgba(124, 58, 237, 0.2)' : '#efe7ff';
  const danger = '#ff5d5d';
  const shadow = isDarkTheme ? '0 14px 34px rgba(2, 6, 23, 0.28)' : '0 12px 26px rgba(148, 163, 184, 0.1)';
  const navLinks = getRoleSidebarItems(currentUser?.role ?? 'analyst');

  const modelCards: Array<{ label: string; key: Exclude<ModelFilter, 'all'>; value: number }> = [
    { label: 'Autoencoder', key: 'autoencoder', value: modelCounts.autoencoder },
    { label: 'LSTM', key: 'lstm', value: modelCounts.lstm },
    { label: 'SNN', key: 'snn', value: modelCounts.snn },
  ];

  const filteredLabel =
    selectedModelFilter === 'all'
      ? 'All models'
      : selectedModelFilter === 'autoencoder'
        ? 'Autoencoder only'
        : selectedModelFilter === 'lstm'
          ? 'LSTM only'
          : 'SNN only';

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
              justifyContent: 'space-between',
            }}
          >
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
                      boxShadow: active ? `inset 0 0 0 1px ${isDarkTheme ? 'rgba(124, 58, 237, 0.28)' : '#ded3ff'}` : 'none',
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

          <main style={{ background: isDarkTheme ? '#0f172a' : '#fcfcfe', padding: '24px 22px 28px' }}>
            <div style={{ maxWidth: '1120px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px', gap: '16px', flexWrap: 'wrap' }}>
                <div>
                  <div style={{ color: muted, fontSize: '.68rem', fontWeight: 700, letterSpacing: '.12em', textTransform: 'uppercase', marginBottom: '8px' }}>
                    Investigations Workflow
                  </div>
                  <h1 style={{ margin: 0, fontSize: '1.86rem', lineHeight: 1.1, fontWeight: 800 }}>Case Queue</h1>
                  <div style={{ marginTop: '10px', color: textSoft, fontSize: '.88rem', lineHeight: 1.6, maxWidth: '660px' }}>
                    Open any case to review the forensic detail, confirm fraud, and move it into the resolved archive with analyst attribution.
                  </div>
                </div>
                <div style={{ color: muted, fontSize: '.8rem', display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                  <Clock3 size={14} />
                  Last update: {lastUpdatedAt ? lastUpdatedAt.toLocaleTimeString() : 'loading...'}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '14px', marginBottom: '18px' }}>
                {modelCards.map((card) => {
                  const active = selectedModelFilter === card.key;
                  return (
                    <button
                      key={card.label}
                      type="button"
                      onClick={() => setSelectedModelFilter((previous) => (previous === card.key ? 'all' : card.key))}
                      style={{
                        textAlign: 'left',
                        padding: '14px 16px',
                        borderRadius: '16px',
                        border: `1px solid ${active ? '#cdb8ff' : border}`,
                        background: active ? accentSoft : panel,
                        boxShadow: active ? '0 10px 22px rgba(124, 58, 237, 0.14)' : 'none',
                        cursor: 'pointer',
                      }}
                    >
                      <div style={{ color: muted, fontSize: '.66rem', fontWeight: 700, letterSpacing: '.08em', textTransform: 'uppercase', marginBottom: '7px' }}>{card.label}</div>
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                        <div style={{ fontSize: '1.65rem', fontWeight: 800, color: active ? accent : text }}>{card.value}</div>
                        <div style={{ padding: '2px 8px', borderRadius: '999px', fontSize: '.66rem', fontWeight: 700, background: isDarkTheme ? 'rgba(255,255,255,.08)' : '#ffffff' }}>24h</div>
                      </div>
                    </button>
                  );
                })}
              </div>

              <div style={{ marginBottom: '12px', color: muted, fontSize: '.76rem' }}>
                Showing: {filteredLabel} ({filteredItems.length} cases)
              </div>

              <div style={{ display: 'flex', gap: '10px', marginBottom: '14px' }}>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '10px', height: '44px', padding: '0 14px', borderRadius: '14px', border: `1px solid ${border}`, background: panel }}>
                  <Search size={16} color={muted} />
                  <input
                    value={searchTerm}
                    onChange={(event) => setSearchTerm(event.target.value)}
                    placeholder="Search queue..."
                    style={{ border: 'none', outline: 'none', width: '100%', background: 'transparent', color: text, fontSize: '.95rem' }}
                  />
                </div>
                <button
                  type="button"
                  style={{
                    width: '44px',
                    height: '44px',
                    borderRadius: '14px',
                    border: `1px solid ${border}`,
                    background: panel,
                    color: textSoft,
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                  }}
                >
                  <Filter size={17} />
                </button>
              </div>

              {loadError ? (
                <div style={{ marginBottom: '12px', border: `1px solid ${border}`, borderRadius: '12px', padding: '10px 12px', color: danger, background: panel, fontSize: '.86rem' }}>
                  {loadError}
                </div>
              ) : null}

              {isLoading ? <div style={{ marginBottom: '12px', color: muted, fontSize: '.84rem' }}>Loading last 24h fraud alerts...</div> : null}

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '12px' }}>
                {filteredItems.map((item) => {
                  const priorityBg =
                    item.priority === 'Critical'
                      ? 'rgba(255, 93, 93, 0.16)'
                      : item.priority === 'High'
                        ? 'rgba(255, 166, 0, 0.14)'
                        : 'rgba(124, 58, 237, 0.12)';
                  const priorityColor =
                    item.priority === 'Critical' ? danger : item.priority === 'High' ? '#f59e0b' : accent;

                  return (
                    <button
                      key={item.alertId}
                      type="button"
                      onClick={() => navigate(`/investigations/${item.alertId}`)}
                      style={{
                        width: '100%',
                        textAlign: 'left',
                        padding: '14px',
                        borderRadius: '16px',
                        border: `1px solid ${border}`,
                        background: panel,
                        boxShadow: shadow,
                        cursor: 'pointer',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px', marginBottom: '12px' }}>
                        <div style={{ color: muted, fontSize: '.78rem', fontWeight: 700 }}>{item.transactionId}</div>
                        <span style={{ padding: '4px 9px', borderRadius: '999px', background: priorityBg, color: priorityColor, fontSize: '.68rem', fontWeight: 800 }}>
                          {item.priority}
                        </span>
                      </div>
                      <div style={{ fontSize: '.95rem', fontWeight: 700, lineHeight: 1.45, color: text, marginBottom: '10px' }}>{item.title}</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px', flexWrap: 'wrap' }}>
                        <span style={{ padding: '4px 8px', borderRadius: '999px', background: accentSoft, color: accent, fontSize: '.68rem', fontWeight: 700 }}>
                          {item.modelLabel}
                        </span>
                        <span style={{ color: textSoft, fontSize: '.8rem' }}>{item.merchant}</span>
                        <span style={{ color: textSoft, fontSize: '.8rem' }}>
                          ${item.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </span>
                      </div>
                      <div style={{ height: '6px', borderRadius: '999px', background: isDarkTheme ? 'rgba(255,255,255,.08)' : '#eceff7', overflow: 'hidden', marginBottom: '10px' }}>
                        <div style={{ width: `${item.risk}%`, height: '100%', borderRadius: 'inherit', background: item.risk > 85 ? danger : accent }} />
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: textSoft, fontSize: '.82rem' }}>
                        <span>{item.risk}% Risk</span>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                          <Clock3 size={12} />
                          {item.openedAgo}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>

              {!isLoading && filteredItems.length === 0 ? (
                <div style={{ marginTop: '12px', border: `1px solid ${border}`, borderRadius: '14px', padding: '14px', color: muted, background: panel }}>
                  No fraud alerts found in the last 24 hours.
                </div>
              ) : null}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};

export default Investigations;
