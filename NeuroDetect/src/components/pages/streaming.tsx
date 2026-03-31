import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Zap,
  TrendingUp,
  Database,
  RefreshCw,
  Gauge,
  Users,
  Clock,
  BarChart3,
  Filter,
  Search,
  Settings,
  Shield,
  ChevronRight,
  PieChart,
  Cpu,
  Server,
  ExternalLink,
  Menu,
  X,
  ArrowLeft
} from 'lucide-react';
import '../../pages/css/streaming.css';
import { useAuth } from '../auth/AuthContext';

interface StreamingRecord {
  transaction_id: string;
  stream_index: number;
  stream_timestamp: string;
  is_fraud: boolean;
  amount?: number;
  category?: string;
  gender?: string;
  merchant?: string;
  location?: string;
  [key: string]: any;
}

interface StreamingStats {
  totalReceived: number;
  fraudCount: number;
  normalCount: number;
  totalAmount: number;
  avgAmount: number;
  categories: Map<string, number>;
  fraudRate: number;
}

type RealtimeModel = 'autoencoder' | 'lstm' | 'snn';

const MODEL_CHANNELS: Record<RealtimeModel, { dataKey: string; streamingKey: string }> = {
  autoencoder: {
    dataKey: 'fraud_detection_data',
    streamingKey: 'fraud_detection_streaming'
  },
  lstm: {
    dataKey: 'lstm_detection_data',
    streamingKey: 'lstm_detection_streaming'
  },
  snn: {
    dataKey: 'snn_detection_data',
    streamingKey: 'snn_detection_streaming'
  }
};

const Streaming: React.FC = () => {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  
  // Streaming state (synced from aereal page)
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeModel, setActiveModel] = useState<RealtimeModel>('autoencoder');
  
  // Data state
  const [records, setRecords] = useState<StreamingRecord[]>([]);
  const [stats, setStats] = useState<StreamingStats>({
    totalReceived: 0,
    fraudCount: 0,
    normalCount: 0,
    totalAmount: 0,
    avgAmount: 0,
    categories: new Map(),
    fraudRate: 0
  });
  
  // UI state
  const [streamSpeed, setStreamSpeed] = useState(1);
  const [maxRecords, setMaxRecords] = useState(50);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [showFraudOnly, setShowFraudOnly] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  // Refs
  const recordsRef = useRef<StreamingRecord[]>([]);

  const normalizeRecords = useCallback((input: any[]): StreamingRecord[] => {
    return input.map((row: any, index: number) => {
      const txn = row.transaction_data || row;
      return {
        transaction_id: row.transaction_id || txn.transaction_id || `TXN_${index}`,
        stream_index: row.stream_index ?? index,
        stream_timestamp: row.stream_timestamp || row.timestamp || new Date().toISOString(),
        is_fraud: row.is_fraud === true || row.is_fraud === 1,
        amount: txn.amt ?? row.amount ?? row.amt ?? 0,
        category: txn.category ?? row.category ?? 'N/A',
        gender: txn.gender ?? row.gender,
        merchant: txn.merchant ?? row.merchant,
        location: txn.city ?? row.location,
        ...row,
      } as StreamingRecord;
    });
  }, []);

  const readModelData = useCallback((model: RealtimeModel) => {
    const raw = localStorage.getItem(MODEL_CHANNELS[model].dataKey);
    if (!raw) return [] as StreamingRecord[];

    try {
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [] as StreamingRecord[];
      return normalizeRecords(parsed);
    } catch {
      return [] as StreamingRecord[];
    }
  }, [normalizeRecords]);

  // Update statistics from records data
  const updateStatsFromRecords = useCallback((data: any[]) => {
    if (!data || data.length === 0) {
      setStats({
        totalReceived: 0,
        fraudCount: 0,
        normalCount: 0,
        totalAmount: 0,
        avgAmount: 0,
        categories: new Map(),
        fraudRate: 0
      });
      return;
    }

    const fraudCount = data.filter(r => r.is_fraud).length;
    const normalCount = data.length - fraudCount;
    const totalAmount = data.reduce((sum, r) => sum + (r.transaction_data?.amt || r.amount || 0), 0);
    const avgAmount = totalAmount / data.length;
    const fraudRate = (fraudCount / data.length) * 100;

    const categoryMap = new Map<string, number>();
    data.forEach(r => {
      const category = r.transaction_data?.category || r.category || 'N/A';
      categoryMap.set(category, (categoryMap.get(category) || 0) + 1);
    });

    setStats({
      totalReceived: data.length,
      fraudCount,
      normalCount,
      totalAmount,
      avgAmount,
      categories: categoryMap,
      fraudRate
    });
  }, []);

  const applyModelData = useCallback((model: RealtimeModel, data: StreamingRecord[]) => {
    setActiveModel(model);
    setRecords(data);
    recordsRef.current = data;
    updateStatsFromRecords(data);
  }, [updateStatsFromRecords]);

  // Sync streaming state and data from realtime model pages via localStorage
  useEffect(() => {
    const resolveActiveStreamingModel = (): RealtimeModel | null => {
      if (localStorage.getItem(MODEL_CHANNELS.snn.streamingKey) === 'true') return 'snn';
      if (localStorage.getItem(MODEL_CHANNELS.lstm.streamingKey) === 'true') return 'lstm';
      if (localStorage.getItem(MODEL_CHANNELS.autoencoder.streamingKey) === 'true') return 'autoencoder';
      return null;
    };

    const handleStorageChange = (e: StorageEvent) => {
      const modelFromStreamingKey = (Object.keys(MODEL_CHANNELS) as RealtimeModel[]).find(
        model => MODEL_CHANNELS[model].streamingKey === e.key
      );

      if (modelFromStreamingKey) {
        const newState = e.newValue === 'true';
        if (newState) {
          setActiveModel(modelFromStreamingKey);
          setIsStreaming(true);
          const data = readModelData(modelFromStreamingKey);
          applyModelData(modelFromStreamingKey, data);
          return;
        }

        const stillActiveModel = resolveActiveStreamingModel();
        if (stillActiveModel) {
          setActiveModel(stillActiveModel);
          setIsStreaming(true);
          const data = readModelData(stillActiveModel);
          applyModelData(stillActiveModel, data);
          return;
        }

        setIsStreaming(false);
        return;
      }

      const modelFromDataKey = (Object.keys(MODEL_CHANNELS) as RealtimeModel[]).find(
        model => MODEL_CHANNELS[model].dataKey === e.key
      );

      if (modelFromDataKey) {
        const data = normalizeRecords(JSON.parse(e.newValue || '[]'));
        if (modelFromDataKey === activeModel || data.length > 0) {
          applyModelData(modelFromDataKey, data);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    
    // Load initial model and data from localStorage
    const initialModel = resolveActiveStreamingModel() || 'autoencoder';
    setActiveModel(initialModel);
    setIsStreaming(resolveActiveStreamingModel() !== null);

    const initialData = readModelData(initialModel);
    applyModelData(initialModel, initialData);

    if (initialData.length === 0 && initialModel !== 'snn') {
      const snnData = readModelData('snn');
      if (snnData.length > 0) {
        applyModelData('snn', snnData);
      }
    }

    const pollInterval = setInterval(() => {
      const streamingModel = resolveActiveStreamingModel() || activeModel;
      const data = readModelData(streamingModel);
      if (JSON.stringify(data) !== JSON.stringify(recordsRef.current)) {
        applyModelData(streamingModel, data);
      }
    }, 500);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      clearInterval(pollInterval);
    };
  }, [activeModel, applyModelData, normalizeRecords, readModelData]);

  const updateStreamSpeed = (speed: number) => {
    setStreamSpeed(speed);
    localStorage.setItem('fraud_detection_speed', speed.toString());
  };

  // Filter records based on search and filters
  const filteredRecords = records.filter(record => {
    const matchesSearch = searchTerm === '' || 
      record.transaction_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.merchant?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.location?.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesCategory = selectedCategory === 'all' || 
      record.category === selectedCategory;

    const matchesFraudFilter = !showFraudOnly || record.is_fraud;

    return matchesSearch && matchesCategory && matchesFraudFilter;
  });

  // Get unique categories for filter dropdown
  const uniqueCategories = Array.from(stats.categories.keys());

  // Format currency
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  // Get severity color based on fraud status and amount
  const getSeverityColor = (record: StreamingRecord) => {
    if (record.is_fraud) return 'fraud';
    const amount = record.amount || 0;
    if (amount > 5000) return 'suspicious';
    return 'normal';
  };

  // Chart data for fraud distribution
  const fraudChartData = [
    { label: 'Normal', value: stats.normalCount, color: 'normal' },
    { label: 'Fraud', value: stats.fraudCount, color: 'fraud' }
  ];

  const maxChartValue = Math.max(...fraudChartData.map(d => d.value), 1);
  const activeModelLabel =
    activeModel === 'autoencoder' ? 'Autoencoder' : activeModel === 'lstm' ? 'LSTM' : 'SNN';

  return (
    <div className="streaming-content">
      {/* Hamburger Menu Button */}
      <button 
        className="menu-toggle-btn"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        style={{
          position: 'fixed',
          top: '20px',
          left: '20px',
          zIndex: 1001,
          background: 'rgba(30, 41, 59, 0.9)',
          border: '1px solid rgba(148, 163, 184, 0.2)',
          borderRadius: '8px',
          padding: '10px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.3s ease'
        }}
      >
        {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Collapsible Sidebar */}
      <aside 
        className={`dashboard-sidebar ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}
        style={{
          position: 'fixed',
          top: 0,
          left: sidebarOpen ? 0 : '-280px',
          height: '100vh',
          width: '280px',
          background: 'rgba(15, 23, 42, 0.95)',
          borderRight: '1px solid rgba(148, 163, 184, 0.2)',
          transition: 'left 0.3s ease',
          zIndex: 1000,
          overflowY: 'auto'
        }}
      >
        <div className="sidebar-header" style={{ padding: '24px' }}>
          <div className="logo" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Shield className="logo-icon" style={{ width: '32px', height: '32px', color: '#3b82f6' }} />
            <span className="logo-text" style={{ fontSize: '20px', fontWeight: 'bold' }}>NeuroDetect</span>
          </div>
        </div>

        <nav className="sidebar-nav" style={{ padding: '0 16px' }}>
          <button type="button" onClick={() => navigate('/snnreal')} className="nav-item" style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '12px 16px',
            borderRadius: '8px',
            border: 'none',
            width: '100%',
            color: 'rgba(148, 163, 184, 1)',
            marginBottom: '8px',
            transition: 'all 0.2s',
            background: 'transparent',
            cursor: 'pointer',
            textAlign: 'left'
          }}>
            <BarChart3 className="nav-icon" size={20} />
            <span>Dashboard</span>
          </button>
          <button type="button" onClick={() => navigate('/streaming')} className="nav-item active" style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '12px 16px',
            borderRadius: '8px',
            border: 'none',
            width: '100%',
            background: 'rgba(59, 130, 246, 0.1)',
            color: '#3b82f6',
            marginBottom: '8px',
            cursor: 'pointer',
            textAlign: 'left'
          }}>
            <Activity className="nav-icon" size={20} />
            <span>System</span>
          </button>
          {currentUser?.role === 'admin' ? (
            <>
              <button type="button" onClick={() => navigate('/user-management')} className="nav-item" style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderRadius: '8px', border: 'none', width: '100%', color: 'rgba(148, 163, 184, 1)', marginBottom: '8px', background: 'transparent', cursor: 'pointer', textAlign: 'left' }}>
                <Users className="nav-icon" size={20} />
                <span>User</span>
              </button>
              <button type="button" onClick={() => navigate('/reports')} className="nav-item" style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderRadius: '8px', border: 'none', width: '100%', color: 'rgba(148, 163, 184, 1)', marginBottom: '8px', background: 'transparent', cursor: 'pointer', textAlign: 'left' }}>
                <PieChart className="nav-icon" size={20} />
                <span>Report</span>
              </button>
            </>
          ) : (
            <>
              <button type="button" onClick={() => navigate('/investigations')} className="nav-item" style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderRadius: '8px', border: 'none', width: '100%', color: 'rgba(148, 163, 184, 1)', marginBottom: '8px', background: 'transparent', cursor: 'pointer', textAlign: 'left' }}>
                <Server className="nav-icon" size={20} />
                <span>Investigation</span>
              </button>
              <button type="button" onClick={() => navigate('/batch-upload')} className="nav-item" style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderRadius: '8px', border: 'none', width: '100%', color: 'rgba(148, 163, 184, 1)', marginBottom: '8px', background: 'transparent', cursor: 'pointer', textAlign: 'left' }}>
                <Zap className="nav-icon" size={20} />
                <span>Batch Upload</span>
              </button>
              <button type="button" onClick={() => navigate('/reports')} className="nav-item" style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderRadius: '8px', border: 'none', width: '100%', color: 'rgba(148, 163, 184, 1)', marginBottom: '8px', background: 'transparent', cursor: 'pointer', textAlign: 'left' }}>
                <PieChart className="nav-icon" size={20} />
                <span>Reports</span>
              </button>
            </>
          )}
        </nav>
      
        <div className="sidebar-footer" style={{ padding: '24px', marginTop: 'auto' }}>
          <div className="stream-status" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div className={`status-indicator ${isStreaming ? 'streaming' : 'paused'}`} style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: isStreaming ? '#10b981' : '#6b7280'
            }}>
              <div className="status-pulse"></div>
            </div>
            <span>{isStreaming ? 'Streaming' : 'Idle'}</span>
          </div>
        </div>
      </aside>

      {/* Overlay when sidebar is open */}
      {sidebarOpen && (
        <div 
          onClick={() => setSidebarOpen(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.5)',
            zIndex: 999
          }}
        />
      )}

      <main className="dashboard-main" style={{ marginLeft: 0, paddingLeft: '24px', paddingRight: '24px' }}>
        {/* Stream Status Header */}
        <div className="top-bar" style={{ marginTop: '70px', marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="top-bar-info" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              onClick={() => navigate('/snnreal')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 16px',
                background: 'rgba(30, 41, 59, 0.9)',
                border: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '8px',
                color: '#94a3b8',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '500',
                transition: 'all 0.2s'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = 'rgba(30, 41, 59, 1)';
                e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.4)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = 'rgba(30, 41, 59, 0.9)';
                e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.2)';
              }}
            >
              <ArrowLeft size={18} />
              <span>Back to Dashboard</span>
            </button>
            <h1 style={{ fontSize: '24px', fontWeight: '600', color: '#f8fafc', margin: 0 }}>Live Transaction Stream</h1>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 12px',
              borderRadius: '999px',
              background: 'rgba(59, 130, 246, 0.12)',
              border: '1px solid rgba(59, 130, 246, 0.35)',
              color: '#93c5fd',
              fontSize: '12px',
              fontWeight: 600,
              letterSpacing: '0.02em'
            }}>
              <Cpu size={14} />
              <span>Active Model: {activeModelLabel}</span>
            </div>
            <div className="stream-indicator" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div className={`live-dot ${isStreaming ? 'streaming' : ''}`} style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: isStreaming ? '#10b981' : '#64748b',
                animation: isStreaming ? 'pulse 2s infinite' : 'none'
              }}></div>
              <span style={{ color: '#94a3b8', fontSize: '14px' }}>
                {isStreaming ? 'Streaming Active' : 'Stream Paused'}
              </span>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-header">
              <div className="stat-icon">
                <Database />
              </div>
              <div className="stat-trend positive">+99%</div>
            </div>
            <div className="stat-value">{stats.totalReceived}</div>
            <div className="stat-label">Total Records</div>
            <div className="stat-meta">Streamed in real-time</div>
          </div>

          <div className="stat-card">
            <div className="stat-header">
              <div className="stat-icon">
                <AlertTriangle />
              </div>
            </div>
            <div className="stat-value">{stats.fraudCount}</div>
            <div className="stat-label">Fraud Transactions</div>
            <div className="fraud-rate">{stats.fraudRate.toFixed(1)}% rate</div>
          </div>

          <div className="stat-card">
            <div className="stat-header">
              <div className="stat-icon">
                <TrendingUp />
              </div>
            </div>
            <div className="stat-value">{formatCurrency(stats.avgAmount)}</div>
            <div className="stat-label">Avg Transaction</div>
            <div className="stat-meta">Across all records</div>
          </div>

          <div className="stat-card">
            <div className="stat-header">
              <div className="stat-icon">
                <BarChart3 />
              </div>
            </div>
            <div className="stat-value">{formatCurrency(stats.totalAmount)}</div>
            <div className="stat-label">Total Amount</div>
            <div className="stat-meta">Processed volume</div>
          </div>

          <div className="stat-card wide">
            <div className="stat-header">
              <h3>Fraud Distribution</h3>
              <Shield className="stat-icon" />
            </div>
            <div className="distribution-chart">
              {fraudChartData.map((item, index) => (
                <div key={index} className="distribution-item">
                  <div className="distribution-info">
                    <div className={`distribution-dot ${item.color}`}></div>
                    <span className="distribution-label">{item.label}</span>
                    <span className="distribution-value">{item.value}</span>
                  </div>
                  <div className="distribution-bar">
                    <div
                      className={`distribution-fill ${item.color}`}
                      style={{ width: `${(item.value / maxChartValue) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="content-area">
          {/* Left Column */}
          <div className="content-column">
            {/* Stream Controls Card */}
            <div className="content-card">
              <div className="card-header">
                <h2>Stream Controls</h2>
                <Cpu className="card-icon" />
              </div>
              
              <div className="speed-control">
                <div className="speed-header">
                  <span className="speed-label">Stream Speed</span>
                  <span className="speed-value">{streamSpeed} records/sec</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="10"
                  step="0.5"
                  value={streamSpeed}
                  onChange={(e) => updateStreamSpeed(parseFloat(e.target.value))}
                  className="speed-slider"
                />
                <div className="speed-labels">
                  <span>Slow</span>
                  <span>Medium</span>
                  <span>Fast</span>
                </div>
              </div>

              <div className="record-controls">
                <div className="record-setting">
                  <label>Max Records</label>
                  <div className="record-buttons">
                    <button 
                      className={`record-btn ${maxRecords === 50 ? 'active' : ''}`}
                      onClick={() => setMaxRecords(50)}
                    >
                      50
                    </button>
                    <button 
                      className={`record-btn ${maxRecords === 100 ? 'active' : ''}`}
                      onClick={() => setMaxRecords(100)}
                    >
                      100
                    </button>
                    <button 
                      className={`record-btn ${maxRecords === 200 ? 'active' : ''}`}
                      onClick={() => setMaxRecords(200)}
                    >
                      200
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Live Data Stream Card */}
            <div className="content-card large">
              <div className="card-header">
                <h2>Live Data Stream</h2>
                <div className="stream-indicator">
                  <div className={`live-dot ${isStreaming ? 'streaming' : ''}`}></div>
                  <span>{isStreaming ? 'Live' : 'Paused'}</span>
                </div>
              </div>

              {/* Filters */}
              <div className="filters">
                <div className="search-box">
                  <Search className="search-icon" />
                  <input
                    type="text"
                    placeholder="Search transactions..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                </div>

                <div className="filter-actions">
                  <select
                    className="filter-select"
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                  >
                    <option value="all">All Categories</option>
                    {uniqueCategories.map(category => (
                      <option key={category} value={category}>{category}</option>
                    ))}
                  </select>

                  <button
                    className={`filter-btn ${showFraudOnly ? 'active' : ''}`}
                    onClick={() => setShowFraudOnly(!showFraudOnly)}
                  >
                    <Filter className="btn-icon" />
                    <span>Fraud Only</span>
                  </button>
                </div>
              </div>

              {/* Records Table */}
              <div className="records-table">
                <div className="table-header">
                  <div className="table-col">ID</div>
                  <div className="table-col">Time</div>
                  <div className="table-col">Amount</div>
                  <div className="table-col">Status</div>
                  <div className="table-col">Actions</div>
                </div>
                
                <div className="table-body">
                  {filteredRecords.slice(0, 10).map((record, index) => (
                    <div 
                      key={`${record.transaction_id}-${index}`}
                      className={`table-row ${getSeverityColor(record)}`}
                    >
                      <div className="table-col">
                        <div className="transaction-id">{record.transaction_id}</div>
                      </div>
                      <div className="table-col">
                        {new Date(record.stream_timestamp).toLocaleTimeString([], { 
                          hour: '2-digit', 
                          minute: '2-digit',
                          second: '2-digit'
                        })}
                      </div>
                      <div className="table-col amount">
                        {formatCurrency(record.amount || 0)}
                      </div>
                      <div className="table-col">
                        <div className={`status-badge ${record.is_fraud ? 'fraud' : 'normal'}`}>
                          {record.is_fraud ? (
                            <>
                              <AlertTriangle className="status-icon" />
                              <span>Fraud</span>
                            </>
                          ) : (
                            <>
                              <CheckCircle className="status-icon" />
                              <span>Normal</span>
                            </>
                          )}
                        </div>
                      </div>
                      <div className="table-col">
                        <button 
                          className="action-link"
                          onClick={() => console.log('View details:', record)}
                        >
                          View
                          <ChevronRight className="link-icon" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="table-footer">
                <span>Showing {Math.min(filteredRecords.length, 10)} of {records.length} records</span>
                <button className="view-all">
                  View All Records
                  <ExternalLink className="btn-icon" />
                </button>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="content-column">
            {/* Category Distribution Card */}
            <div className="content-card">
              <div className="card-header">
                <h2>Category Distribution</h2>
                <PieChart className="card-icon" />
              </div>
              
              <div className="category-list">
                {Array.from(stats.categories.entries())
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 6)
                  .map(([category, count]) => (
                    <div key={category} className="category-item">
                      <div className="category-info">
                        <span className="category-name">{category}</span>
                        <span className="category-count">{count}</span>
                      </div>
                      <div className="category-bar">
                        <div 
                          className="category-fill"
                          style={{ 
                            width: `${(count / (stats.totalReceived || 1)) * 100}%`,
                            backgroundColor: `hsl(${Math.random() * 60 + 200}, 70%, 50%)`
                          }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </div>

            {/* System Metrics Card */}
            <div className="content-card">
              <div className="card-header">
                <h2>System Metrics</h2>
                <Server className="card-icon" />
              </div>
              
              <div className="metrics-grid">
                <div className="metric-item">
                  <div className="metric-icon">
                    <Gauge />
                  </div>
                  <div className="metric-content">
                    <div className="metric-value">{streamSpeed}/sec</div>
                    <div className="metric-label">Processing Speed</div>
                  </div>
                </div>

                <div className="metric-item">
                  <div className="metric-icon">
                    <Clock />
                  </div>
                  <div className="metric-content">
                    <div className="metric-value">
                      {Math.floor(stats.totalReceived / (streamSpeed || 1))}s
                    </div>
                    <div className="metric-label">Stream Duration</div>
                  </div>
                </div>

                <div className="metric-item">
                  <div className="metric-icon">
                    <Users />
                  </div>
                  <div className="metric-content">
                    <div className="metric-value">{uniqueCategories.length}</div>
                    <div className="metric-label">Unique Categories</div>
                  </div>
                </div>

                <div className="metric-item">
                  <div className="metric-icon">
                    <Shield />
                  </div>
                  <div className="metric-content">
                    <div className="metric-value">{(100 - stats.fraudRate).toFixed(1)}%</div>
                    <div className="metric-label">Accuracy Rate</div>
                  </div>
                </div>
              </div>
            </div>

            {/* System Status Card */}
            <div className="content-card">
              <div className="card-header">
                <h2>System Status</h2>
                <Settings className="card-icon" />
              </div>
              
              <div className="status-list">
                <div className="status-item">
                  <span className="status-label">Data Source</span>
                  <div className={`status-value ${records.length > 0 ? 'online' : 'offline'}`}>
                    {records.length > 0 ? 'Receiving' : 'Waiting'}
                  </div>
                </div>

                <div className="status-item">
                  <span className="status-label">Stream Status</span>
                  <div className={`status-value ${isStreaming ? 'active' : 'inactive'}`}>
                    {isStreaming ? 'Active' : 'Inactive'}
                  </div>
                </div>

                <div className="status-item">
                  <span className="status-label">Buffer Usage</span>
                  <div className="status-value">
                    {records.length}/{maxRecords}
                  </div>
                </div>

                <div className="status-item">
                  <span className="status-label">Memory</span>
                  <div className="status-value">
                    {(records.length * 0.1).toFixed(1)} MB
                  </div>
                </div>

                <div className="status-item">
                  <span className="status-label">Update Rate</span>
                  <div className="status-value">
                    {isStreaming ? 'Real-time' : '--'}
                  </div>
                </div>
              </div>

              <div className="card-actions">
                <button 
                  className="action-btn clear-btn"
                  onClick={() => {
                    setRecords([]);
                    setStats({
                      totalReceived: 0,
                      fraudCount: 0,
                      normalCount: 0,
                      totalAmount: 0,
                      avgAmount: 0,
                      categories: new Map(),
                      fraudRate: 0
                    });
                  }}
                >
                  <RefreshCw className="btn-icon" />
                  Clear All Data
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="dashboard-footer">
          <div className="footer-info">
            <div className="server-info">
              <span className="info-label">Server:</span>
              <span className="info-value">ws://localhost:8765</span>
            </div>
            <div className="stream-info">
              <span className="info-label">Streaming:</span>
              <span className="info-value">{isStreaming ? 'Active' : 'Paused'}</span>
            </div>
            <div className="stats-info">
              <span className="info-label">Processed:</span>
              <span className="info-value">{stats.totalReceived} records</span>
            </div>
            <div className="stats-info">
              <span className="info-label">Model:</span>
              <span className="info-value">{activeModelLabel}</span>
            </div>
          </div>
          <div className="footer-meta">
            <span>NeuroDetect Dashboard v1.0 • Real-time Monitoring System</span>
          </div>
        </footer>
      </main>
    </div>
  );
};

export default Streaming;
