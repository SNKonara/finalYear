import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Play, 
  Pause, 
  Zap, 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  TrendingUp,
  Database,
  RefreshCw,
  Gauge,
  Users,
  Clock,
  BarChart3,
  Filter,
  Search,
  Download,
  Settings,
  Shield,
  ChevronRight,
  AlertCircle,
  PieChart,
  Cpu,
  Server,
  Network,
  ZapOff,
  ExternalLink
} from 'lucide-react';
import './css/streaming.css';

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

const Streaming: React.FC = () => {
  // WebSocket state
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
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
  const [activeTab, setActiveTab] = useState('overview');
  
  // Refs
  const recordsRef = useRef<StreamingRecord[]>([]);
  const statsRef = useRef<StreamingStats>(stats);
  const wsRef = useRef<WebSocket | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      setConnectionStatus('connecting');
      
      const ws = new WebSocket('ws://localhost:8765');
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setConnectionStatus('connected');
        checkServerStatus();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        setIsConnected(false);
        setIsStreaming(false);
        setConnectionStatus('disconnected');
        
        // Attempt to reconnect after 3 seconds
        setTimeout(() => {
          if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
            connectWebSocket();
          }
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      setSocket(ws);
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((data: any) => {
    // Control messages (server responses)
    if (data.response === 'pong') {
      console.log('PONG from server', data);
      return;
    }

    if (data.error) {
      console.error('Server error:', data.error);
      return;
    }

    // Status / control payload from server
    if (typeof data.streaming === 'boolean' || data.status || data.speed || data.stream_speed) {
      if (typeof data.streaming === 'boolean') {
        setIsStreaming(data.streaming);
      }
      if (data.status === 'stopped') {
        setIsStreaming(false);
      }
      if (data.status === 'already_streaming') {
        setIsStreaming(true);
      }
      const newSpeed = data.speed ?? data.stream_speed;
      if (newSpeed) {
        setStreamSpeed(newSpeed);
      }

      console.log('Control message:', data);
      return;
    }

    // This is a data record
    const newRecord: StreamingRecord = {
      transaction_id: data.transaction_id || `TXN_${Date.now()}`,
      stream_index: data.stream_index || 0,
      stream_timestamp: data.stream_timestamp || new Date().toISOString(),
      is_fraud: data.is_fraud || false,
      amount: data.amount || data.transaction_amt || 0,
      category: data.category || data.merchant_category || 'Unknown',
      gender: data.gender || 'Unknown',
      merchant: data.merchant || data.merchant_name || 'Unknown Merchant',
      location: data.location || data.city || data.state || 'Unknown',
      ...data
    };

    // Update records with new record at the beginning
    const updatedRecords = [newRecord, ...recordsRef.current.slice(0, maxRecords - 1)];
    recordsRef.current = updatedRecords;
    setRecords(updatedRecords);

    // Update statistics
    updateStats(newRecord);
  }, [maxRecords]);

  // Update statistics
  const updateStats = useCallback((record: StreamingRecord) => {
    setStats(prev => {
      const newTotal = prev.totalReceived + 1;
      const newFraudCount = prev.fraudCount + (record.is_fraud ? 1 : 0);
      const newNormalCount = prev.normalCount + (record.is_fraud ? 0 : 1);
      const newTotalAmount = prev.totalAmount + (record.amount || 0);
      const newAvgAmount = newTotalAmount / newTotal;
      const newFraudRate = (newFraudCount / newTotal) * 100;

      // Update category count
      const newCategories = new Map(prev.categories);
      const category = record.category || 'Unknown';
      newCategories.set(category, (newCategories.get(category) || 0) + 1);

      const newStats = {
        totalReceived: newTotal,
        fraudCount: newFraudCount,
        normalCount: newNormalCount,
        totalAmount: newTotalAmount,
        avgAmount: newAvgAmount,
        categories: newCategories,
        fraudRate: newFraudRate
      };

      statsRef.current = newStats;
      return newStats;
    });
  }, []);

  // WebSocket commands
  const sendCommand = useCallback((command: string, data?: any) => {
    if (!wsRef.current) {
      console.warn('WebSocket not initialized — cannot send command:', command);
      return;
    }

    if (wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not open — current state:', wsRef.current.readyState);
      try {
        wsRef.current.send(JSON.stringify({ command, ...data }));
      } catch (err) {
        console.error('Failed to send command, socket not open:', err);
      }
      return;
    }

    wsRef.current.send(JSON.stringify({ command, ...data }));
  }, []);

  const startStreaming = () => {
    sendCommand('start_stream');
  };

  const stopStreaming = () => {
    sendCommand('stop_stream');
  };

  const checkServerStatus = () => {
    sendCommand('get_status');
  };

  const updateStreamSpeed = (speed: number) => {
    setStreamSpeed(speed);
    sendCommand('set_speed', { speed });
  };

  const pingServer = () => {
    sendCommand('ping');
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
    { label: 'Suspicious', value: stats.fraudCount > stats.normalCount ? stats.fraudCount - stats.normalCount : 0, color: 'suspicious' },
    { label: 'Fraud', value: stats.fraudCount, color: 'fraud' }
  ];

  const maxChartValue = Math.max(...fraudChartData.map(d => d.value), 1);

  return (
    <div className="streaming-content">
      <main className="dashboard-main">
        {/* Top Bar */}
        <header className="top-bar">
          <div className="page-title">
            <h1>Streaming Dashboard</h1>
            <p className="subtitle">Real-time transaction monitoring and analysis</p>
          </div>
          
          <div className="top-bar-actions">
            <button 
              className="action-btn test-btn"
              onClick={pingServer}
            >
              <Network className="btn-icon" />
              <span>Test Connection</span>
            </button>
            
            <div className="control-buttons">
              <button
                className={`stream-btn start-btn ${!isConnected || isStreaming ? 'disabled' : ''}`}
                onClick={startStreaming}
                disabled={!isConnected || isStreaming}
              >
                <Play className="btn-icon" />
                <span>Start</span>
              </button>
              <button
                className={`stream-btn stop-btn ${!isConnected || !isStreaming ? 'disabled' : ''}`}
                onClick={stopStreaming}
                disabled={!isConnected || !isStreaming}
              >
                <Pause className="btn-icon" />
                <span>Stop</span>
              </button>
            </div>
          </div>
        </header>

        {/* Stats Grid */}
        <div className="stats-grid">
          <div className="stat-card large">
            <div className="stat-header">
              <div className="stat-icon">
                <Database />
              </div>
              <div className="stat-trend positive">+12%</div>
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
            <div className="stat-label">Fraud Detected</div>
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
                  <span className="status-label">WebSocket</span>
                  <div className={`status-value ${isConnected ? 'online' : 'offline'}`}>
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </div>
                </div>

                <div className="status-item">
                  <span className="status-label">Stream Engine</span>
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
                  <span className="status-label">Latency</span>
                  <div className="status-value">
                    {isConnected ? '< 50ms' : '--'}
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