import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  AlertCircle,
  BarChart3,
  Brain,
  Download,
  RefreshCw,
  Shield,
  TriangleAlert,
} from 'lucide-react';
import { useAuth } from '../auth/AuthContext';
import '../../pages/css/lstmreal.css';
import '../../pages/css/snn_alerts.css';

interface AlertSummary {
  batch_id: string;
  model_type: string;
  timestamp: string;
  total_alerts: number;
  high_risk_alerts: number;
  risk_distribution: Record<string, number>;
}

interface ExplainabilityPayload {
  decision_margin?: number;
  reasons?: string[];
  graph_data?: {
    feature_contribution_chart?: {
      labels: string[];
      values: number[];
    };
    threshold_chart?: {
      probability: number;
      threshold: number;
      margin: number;
    };
    risk_dimension_chart?: {
      labels: string[];
      values: number[];
    };
  };
  top_factors?: Array<{
    feature: string;
    value: number;
    scaled_value: number;
    contribution_pct: number;
  }>;
}

interface HighRiskAlert {
  transaction_id: string;
  amount: number;
  category: string;
  merchant: string;
  city: string;
  fraud_score: number;
  decision_threshold: number;
  risk_level: string;
  prediction: number;
  explainability?: ExplainabilityPayload;
}

interface HighRiskResponse {
  batch_id: string;
  total_high_risk: number;
  high_risk_alerts: HighRiskAlert[];
  graph_data?: {
    top_reasons?: Array<{ reason: string; count: number }>;
    score_vs_threshold?: Array<{ transaction_id: string; score: number; threshold: number }>;
  };
}

interface RasterData {
  rows: string[];
  matrix: number[][];
  timeSteps: number;
}

const API_BASE = 'http://localhost:8000';

const SNNAlertsInvestigation: React.FC = () => {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [searchParams] = useSearchParams();
  const batchId = searchParams.get('batch_id') || undefined;

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<AlertSummary | null>(null);
  const [alertsResponse, setAlertsResponse] = useState<HighRiskResponse | null>(null);
  const [selectedAlertId, setSelectedAlertId] = useState<string | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);
  const [isAutoSyncing, setIsAutoSyncing] = useState(false);
  const isFetchingRef = useRef(false);

  const fetchAlerts = useCallback(async (mode: 'initial' | 'manual' | 'auto' = 'initial') => {
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;

    try {
      if (mode !== 'auto') {
        setError(null);
      }

      if (mode === 'manual') {
        setRefreshing(true);
      } else if (mode === 'initial') {
        setLoading(true);
      } else {
        setIsAutoSyncing(true);
      }

      const summaryUrl = new URL(`${API_BASE}/alerts/summary`);
      summaryUrl.searchParams.set('model_type', 'snn');
      const highRiskUrl = new URL(`${API_BASE}/alerts/high-risk`);
      highRiskUrl.searchParams.set('model_type', 'snn');
      highRiskUrl.searchParams.set('limit', '300');

      if (batchId) {
        summaryUrl.searchParams.set('batch_id', batchId);
        highRiskUrl.searchParams.set('batch_id', batchId);
      }

      const [summaryRes, alertsRes] = await Promise.all([
        fetch(summaryUrl.toString()),
        fetch(highRiskUrl.toString()),
      ]);

      if (!summaryRes.ok) {
        throw new Error('Failed to load alert summary from backend');
      }
      if (!alertsRes.ok) {
        throw new Error('Failed to load high-risk alerts from backend');
      }

      const summaryData: AlertSummary = await summaryRes.json();
      const highRiskData: HighRiskResponse = await alertsRes.json();

      setSummary(summaryData);
      setAlertsResponse(highRiskData);
      setLastUpdatedAt(new Date());

      if (highRiskData.high_risk_alerts.length > 0) {
        setSelectedAlertId((previous) => {
          const stillPresent = highRiskData.high_risk_alerts.some((alert) => alert.transaction_id === previous);
          return stillPresent ? previous : highRiskData.high_risk_alerts[0].transaction_id;
        });
      } else {
        setSelectedAlertId(null);
      }
    } catch (fetchError: any) {
      if (mode !== 'auto') {
        setError(fetchError?.message || 'Unable to load investigation data');
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
      setIsAutoSyncing(false);
      isFetchingRef.current = false;
    }
  }, [batchId]);

  useEffect(() => {
    fetchAlerts('initial');
  }, [fetchAlerts]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        fetchAlerts('auto');
      }
    }, 5000);

    return () => {
      window.clearInterval(interval);
    };
  }, [fetchAlerts]);

  const highRiskAlerts = alertsResponse?.high_risk_alerts || [];

  const selectedAlert = useMemo(() => {
    if (!selectedAlertId) return null;
    return highRiskAlerts.find((alert) => alert.transaction_id === selectedAlertId) || null;
  }, [highRiskAlerts, selectedAlertId]);

  const topReasons = alertsResponse?.graph_data?.top_reasons || [];
  const maxReasonCount = Math.max(...topReasons.map((item) => item.count), 1);

  const selectedExplainability = selectedAlert?.explainability;
  const contributionChart = selectedExplainability?.graph_data?.feature_contribution_chart;
  const dimensionChart = selectedExplainability?.graph_data?.risk_dimension_chart;
  const thresholdChart = selectedExplainability?.graph_data?.threshold_chart;

  const rasterData = useMemo<RasterData | null>(() => {
    if (!selectedAlert) return null;

    const topFactors = selectedExplainability?.top_factors || [];
    const factorLabels =
      topFactors.length > 0
        ? topFactors.slice(0, 6).map((factor) => factor.feature)
        : (contributionChart?.labels || []).slice(0, 6);

    if (factorLabels.length === 0) {
      return null;
    }

    const factorStrengthMap = new Map<string, number>();
    topFactors.forEach((factor) => {
      factorStrengthMap.set(factor.feature, Math.max(0.05, Math.min(1, (factor.contribution_pct || 0) / 100)));
    });

    const chartLabels = contributionChart?.labels || [];
    const chartValues = contributionChart?.values || [];
    chartLabels.forEach((label, idx) => {
      if (!factorStrengthMap.has(label)) {
        factorStrengthMap.set(label, Math.max(0.05, Math.min(1, (chartValues[idx] || 0) / 100)));
      }
    });

    const timeSteps = 20;

    const hashToUnit = (seedText: string): number => {
      let hash = 0;
      for (let index = 0; index < seedText.length; index += 1) {
        hash = (hash * 31 + seedText.charCodeAt(index)) >>> 0;
      }
      const value = Math.sin(hash + 1) * 10000;
      return value - Math.floor(value);
    };

    const matrix = factorLabels.map((featureLabel, rowIndex) => {
      const strength = factorStrengthMap.get(featureLabel) ?? 0.2;
      return Array.from({ length: timeSteps }).map((_, timeIndex) => {
        const seed = `${selectedAlert.transaction_id}-${featureLabel}-${timeIndex}-${rowIndex}`;
        const randomUnit = hashToUnit(seed);
        const activationChance = 0.08 + strength * 0.5;
        if (randomUnit > activationChance) return 0;
        return Math.max(0.25, Math.min(1, strength + randomUnit * 0.35));
      });
    });

    return {
      rows: factorLabels,
      matrix,
      timeSteps,
    };
  }, [selectedAlert, selectedExplainability, contributionChart]);

  return (
    <div className="fraud-dashboard snn-alerts-page">
      <aside className="fraud-sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <Brain className="logo-icon" />
            <span className="logo-text">NeuroDetect</span>
          </div>
          <div className="model-badge">
            <div className="model-type">SNN</div>
            <div className="connection-dot connected"></div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button className="nav-item" onClick={() => navigate('/snnreal')}>
            <BarChart3 className="nav-icon" />
            <span>Dashboard</span>
          </button>
          {currentUser?.role === 'admin' ? (
            <>
              <button className="nav-item" onClick={() => navigate('/user-management')}>
                <Shield className="nav-icon" />
                <span>User</span>
              </button>
              <button className="nav-item" onClick={() => navigate('/reports')}>
                <Download className="nav-icon" />
                <span>Report</span>
              </button>
              <button className="nav-item" onClick={() => navigate('/system')}>
                <RefreshCw className="nav-icon" />
                <span>System</span>
              </button>
            </>
          ) : (
            <>
              <button className="nav-item active">
                <AlertCircle className="nav-icon" />
                <span>Investigation</span>
              </button>
              <button className="nav-item" onClick={() => navigate('/system')}>
                <RefreshCw className="nav-icon" />
                <span>System</span>
              </button>
              <button className="nav-item" onClick={() => navigate('/batch-upload')}>
                <TriangleAlert className="nav-icon" />
                <span>Batch Upload</span>
              </button>
              <button className="nav-item" onClick={() => navigate('/reports')}>
                <Download className="nav-icon" />
                <span>Reports</span>
              </button>
            </>
          )}
        </nav>
      </aside>

      <main className="fraud-main">
        <header className="fraud-topbar">
          <div className="topbar-left">
            <h1>SNN Fraud Investigation</h1>
            <p className="subtitle">High-risk transactions with explainability for analyst review</p>
            <div className="realtime-status">
              <span className={`realtime-dot ${isAutoSyncing ? 'syncing' : 'live'}`}></span>
              <span>
                Real-time updates every 5s
                {lastUpdatedAt ? ` • Last update ${lastUpdatedAt.toLocaleTimeString()}` : ''}
              </span>
            </div>
          </div>
          <div className="control-group">
            <button className="control-btn test-btn" onClick={() => fetchAlerts('manual')} disabled={refreshing}>
              <RefreshCw className="btn-icon" />
              {refreshing ? 'Refreshing...' : 'Refresh'}
            </button>
            <button className="control-btn start-btn" onClick={() => navigate('/snnreal')}>
              <Shield className="btn-icon" />
              Back to SNN Live
            </button>
          </div>
        </header>

        <div className="performance-grid">
          <div className="performance-card large">
            <div className="card-header">
              <h3>Alert Summary</h3>
              <AlertCircle className="card-icon" />
            </div>
            {loading ? (
              <div className="empty-state">Loading summary...</div>
            ) : error ? (
              <div className="empty-state error">{error}</div>
            ) : (
              <div className="alerts-summary">
                <div className="alert-item high">
                  <div className="alert-count">{summary?.high_risk_alerts ?? 0}</div>
                  <div className="alert-info">
                    <div className="alert-title">High Risk Alerts</div>
                    <div className="alert-desc">Immediate analyst investigation</div>
                  </div>
                </div>
                <div className="alert-item medium">
                  <div className="alert-count">{summary?.risk_distribution?.['Medium-High'] ?? 0}</div>
                  <div className="alert-info">
                    <div className="alert-title">Medium-High</div>
                    <div className="alert-desc">Potential escalation candidates</div>
                  </div>
                </div>
                <div className="alert-item low">
                  <div className="alert-count">{summary?.risk_distribution?.Low ?? 0}</div>
                  <div className="alert-info">
                    <div className="alert-title">Low Risk</div>
                    <div className="alert-desc">Background monitoring</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="performance-card">
            <div className="card-header">
              <h3>Top Reasons</h3>
              <TriangleAlert className="card-icon" />
            </div>
            <div className="reasons-chart reasons-scroll">
              {topReasons.length === 0 && <div className="empty-state">No explainability reasons available</div>}
              {topReasons.map((item) => (
                <div key={item.reason} className="reason-row">
                  <div className="reason-label">{item.reason}</div>
                  <div className="reason-bar">
                    <div className="reason-fill" style={{ width: `${(item.count / maxReasonCount) * 100}%` }}></div>
                  </div>
                  <div className="reason-count">{item.count}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="performance-card">
            <div className="card-header">
              <h3>Batch Context</h3>
              <Download className="card-icon" />
            </div>
            <div className="system-stats">
              <div className="stat-item"><span className="stat-label">Batch ID</span><span className="stat-value">{summary?.batch_id || 'N/A'}</span></div>
              <div className="stat-item"><span className="stat-label">Total Alerts</span><span className="stat-value">{summary?.total_alerts ?? 0}</span></div>
              <div className="stat-item"><span className="stat-label">High Risk</span><span className="stat-value">{alertsResponse?.total_high_risk ?? 0}</span></div>
            </div>
          </div>
        </div>

        <div className="content-area">
          <div className="content-column">
            <div className="content-card large">
              <div className="card-header">
                <h2>Fraud Transactions to Investigate</h2>
                <Shield className="card-icon" />
              </div>

              <div className="detections-table">
                <div className="table-header investigation-grid">
                  <div className="table-col">Transaction</div>
                  <div className="table-col">Profile</div>
                  <div className="table-col">Score</div>
                  <div className="table-col">Risk</div>
                  <div className="table-col">Action</div>
                </div>

                <div className="table-body">
                  {highRiskAlerts.length === 0 && <div className="empty-state">No high-risk alerts found in this batch</div>}
                  {highRiskAlerts.map((alert) => {
                    const isSelected = selectedAlertId === alert.transaction_id;
                    const scorePct = Math.min(100, (alert.fraud_score / Math.max(alert.decision_threshold || 1, 0.0001)) * 100);
                    return (
                      <div
                        key={alert.transaction_id}
                        className={`table-row high investigation-grid ${isSelected ? 'selected-row' : ''}`}
                      >
                        <div className="table-col">
                          <div className="transaction-id">{alert.transaction_id}</div>
                          <div className="transaction-category">${alert.amount.toFixed(2)}</div>
                        </div>
                        <div className="table-col">
                          <div className="transaction-id">{alert.category}</div>
                          <div className="transaction-category">{alert.city}</div>
                        </div>
                        <div className="table-col">
                          <div className="fraud-probability">
                            <div className="probability-value">{alert.fraud_score.toFixed(4)}</div>
                            <div className="probability-bar">
                              <div className="probability-fill" style={{ width: `${Math.min(scorePct, 100)}%` }}></div>
                            </div>
                          </div>
                        </div>
                        <div className="table-col">
                          <span className="risk-badge" style={{ backgroundColor: 'var(--danger)' }}>{alert.risk_level}</span>
                        </div>
                        <div className="table-col">
                          <button className="action-btn export-btn" onClick={() => setSelectedAlertId(alert.transaction_id)}>
                            Investigate
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          <div className="content-column">
            <div className="content-card">
              <div className="card-header">
                <h2>Why This Was Flagged</h2>
                <TriangleAlert className="card-icon" />
              </div>

              {!selectedAlert ? (
                <div className="empty-state">Select a high-risk transaction to view explainability</div>
              ) : (
                <>
                  <div className="system-stats">
                    <div className="stat-item"><span className="stat-label">Transaction ID</span><span className="stat-value">{selectedAlert.transaction_id}</span></div>
                    <div className="stat-item"><span className="stat-label">Fraud Score</span><span className="stat-value">{selectedAlert.fraud_score.toFixed(6)}</span></div>
                    <div className="stat-item"><span className="stat-label">Threshold</span><span className="stat-value">{selectedAlert.decision_threshold.toFixed(6)}</span></div>
                    <div className="stat-item"><span className="stat-label">Decision</span><span className="stat-value">{selectedAlert.prediction === 1 ? 'Flagged Fraud' : 'Not Fraud'}</span></div>
                  </div>

                  <div className="explain-section">
                    <h4>Reason Summary</h4>
                    <ul className="reason-list">
                      {(selectedExplainability?.reasons || []).map((reason, idx) => (
                        <li key={`${reason}-${idx}`}>{reason}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="explain-section">
                    <h4>Top Feature Contribution</h4>
                    <div className="mini-chart">
                      {(contributionChart?.labels || []).map((label, idx) => (
                        <div key={label} className="mini-chart-row">
                          <span className="mini-label">{label}</span>
                          <div className="mini-bar">
                            <div
                              className="mini-fill"
                              style={{ width: `${Math.min(100, contributionChart?.values?.[idx] || 0)}%` }}
                            ></div>
                          </div>
                          <span className="mini-value">{(contributionChart?.values?.[idx] || 0).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="explain-section">
                    <h4>Risk Dimension Graph</h4>
                    <div className="mini-chart">
                      {(dimensionChart?.labels || []).map((label, idx) => (
                        <div key={label} className="mini-chart-row">
                          <span className="mini-label">{label}</span>
                          <div className="mini-bar">
                            <div
                              className="mini-fill secondary"
                              style={{ width: `${Math.min(100, (dimensionChart?.values?.[idx] || 0) * 100)}%` }}
                            ></div>
                          </div>
                          <span className="mini-value">{(((dimensionChart?.values?.[idx] || 0) * 100)).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="explain-section">
                    <h4>SNN Spike Raster (Explainability View)</h4>
                    {!rasterData ? (
                      <div className="empty-state">Insufficient feature data to render raster</div>
                    ) : (
                      <>
                        <div className="raster-axis">
                          <span>t1</span>
                          <span>t5</span>
                          <span>t10</span>
                          <span>t15</span>
                          <span>t20</span>
                        </div>
                        <div className="raster-wrapper">
                          {rasterData.rows.map((rowLabel, rowIdx) => (
                            <div className="raster-row" key={`${rowLabel}-${rowIdx}`}>
                              <div className="raster-label" title={rowLabel}>{rowLabel}</div>
                              <div className="raster-track">
                                {rasterData.matrix[rowIdx].map((intensity, colIdx) => (
                                  <div
                                    key={`${rowLabel}-${colIdx}`}
                                    className={`raster-cell ${intensity > 0 ? 'active' : ''}`}
                                    style={{
                                      opacity: intensity > 0 ? Math.max(0.35, intensity) : 0.16,
                                    }}
                                    title={`${rowLabel} @ t${colIdx + 1}: ${intensity > 0 ? 'spike' : 'silent'}`}
                                  ></div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="raster-legend">
                          <span className="legend-dot quiet"></span>
                          <span>Quiet</span>
                          <span className="legend-dot active"></span>
                          <span>High spike activity</span>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="explain-section threshold-box">
                    <h4>Threshold Comparison</h4>
                    <div className="threshold-metrics">
                      <div className="metric-chip">Probability: {(thresholdChart?.probability ?? selectedAlert.fraud_score).toFixed(4)}</div>
                      <div className="metric-chip">Threshold: {(thresholdChart?.threshold ?? selectedAlert.decision_threshold).toFixed(4)}</div>
                      <div className="metric-chip">Margin: {(thresholdChart?.margin ?? 0).toFixed(4)}</div>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default SNNAlertsInvestigation;
