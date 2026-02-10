import React, { useState, useEffect, useRef, useCallback } from 'react';
import Streaming from './streaming';
import { BarChart3, Server } from 'lucide-react';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  Search, 
  Settings,
  Shield,
  Activity,
  Filter,
  ArrowUp,
  ArrowDown,
  Minus,
  Menu,
  X
} from 'lucide-react';

interface Transaction {
  timestamp: string;
  userId: string;
  amount: string;
  merchant: string;
  location: string;
  score: number;
  severity: 'High-Risk' | 'Suspicious' | 'Normal';
}

interface MetricCardProps {
  title: string;
  value: string;
  change: string;
  icon: React.ReactNode;
  trend: 'up' | 'down' | 'stable';
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, change, icon, trend }) => {
  const getTrendColor = () => {
    switch (trend) {
      case 'up': return 'text-success';
      case 'down': return 'text-danger';
      default: return 'text-warning';
    }
  };

  const getTrendIcon = () => {
    switch (trend) {
      case 'up': return <ArrowUp className="w-4 h-4" />;
      case 'down': return <ArrowDown className="w-4 h-4" />;
      default: return <Minus className="w-4 h-4" />;
    }
  };

  return (
    <div className="metric-card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-300">{title}</h3>
        {icon}
      </div>
      <div className="text-3xl font-bold mb-2">{value}</div>
      <div className={`flex items-center gap-1 ${getTrendColor()}`}>
        {getTrendIcon()}
        <span className="text-sm font-medium">{change}</span>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const [transactions, setTransactions] = useState<Transaction[]>([
    {
      timestamp: "2024-07-02 10:30:15",
      userId: "user_A123",
      amount: "$12,500.00",
      merchant: "Global Tech Inc.",
      location: "London, UK",
      score: 0.95,
      severity: "High-Risk"
    },
    {
      timestamp: "2024-07-02 10:25:00",
      userId: "user_B456",
      amount: "$5,000.00",
      merchant: "Luxury Goods Ltd.",
      location: "New York, USA",
      score: 0.88,
      severity: "High-Risk"
    },
    {
      timestamp: "2024-07-02 10:20:40",
      userId: "user_C789",
      amount: "$150.00",
      merchant: "Online Services Co.",
      location: "Berlin, Germany",
      score: 0.62,
      severity: "Suspicious"
    },
    {
      timestamp: "2024-07-02 10:15:22",
      userId: "user_D012",
      amount: "$50.00",
      merchant: "Local Cafe",
      location: "Paris, France",
      score: 0.30,
      severity: "Normal"
    },
    {
      timestamp: "2024-07-02 10:10:05",
      userId: "user_E345",
      amount: "$250.00",
      merchant: "E-commerce Store",
      location: "Tokyo, Japan",
      score: 0.75,
      severity: "Suspicious"
    },
    {
      timestamp: "2024-07-02 10:05:30",
      userId: "user_F678",
      amount: "$30,000.00",
      merchant: "Investment Platform",
      location: "Singapore",
      score: 0.98,
      severity: "High-Risk"
    },
    {
      timestamp: "2024-07-02 10:01:00",
      userId: "user_G901",
      amount: "$75.00",
      merchant: "Bookstore Chain",
      location: "Sydney, AU",
      score: 0.45,
      severity: "Normal"
    }
  ]);

  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedSeverity, setSelectedSeverity] = useState<string>("All Severities");
  const [activeTab, setActiveTab] = useState<'overview'|'stream'|'analytics'>('overview');
  const [wsConnected, setWsConnected] = useState(false);
  const [detectorStatus, setDetectorStatus] = useState<string>('stopped');
  const [selectedModel, setSelectedModel] = useState<'autoencoder'|'lstm'|'snn'>('autoencoder');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765');
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      ws.send(JSON.stringify({ command: 'get_status' }));
    };

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.status === 'detector_started') setDetectorStatus('running');
        if (data.status === 'detector_stopped' || data.status === 'no_detector_running') setDetectorStatus('stopped');
        if (data.error) console.error('Backend error:', data.error);
      } catch (e) {
        console.error('Invalid WS message', e);
      }
    };

    ws.onclose = () => { setWsConnected(false); setDetectorStatus('stopped'); };
    ws.onerror = () => { /* ignore */ };

    return () => { ws.close(); };
  }, []);

  const sendCommand = useCallback((command: string, payload?: any) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not open');
      return;
    }
    wsRef.current.send(JSON.stringify({ command, ...(payload || {}) }));
  }, []);

  const startDetector = () => {
    sendCommand('start_detector', { model: selectedModel });
  };

  const stopDetector = () => {
    sendCommand('stop_detector');
  };

  // Chart data for real-time activity
  const chartData = [32, 24, 16, 8];
  const chartLabels = ["00:00", "00:10", "00:20", "00:30", "00:40", "00:50", "01:00"];

  const getSeverityColor = (severity: Transaction['severity']) => {
    switch (severity) {
      case 'High-Risk': return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'Suspicious': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      default: return 'bg-green-500/20 text-green-400 border-green-500/30';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-red-400';
    if (score >= 0.6) return 'text-yellow-400';
    return 'text-green-400';
  };

  const filteredTransactions = transactions.filter(tx => {
    const matchesSearch = tx.userId.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         tx.merchant.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         tx.location.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSeverity = selectedSeverity === "All Severities" || tx.severity === selectedSeverity;
    return matchesSearch && matchesSeverity;
  });

  return (
    <div className="streaming-dashboard">
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
          <button 
            className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`} 
            onClick={() => setActiveTab('overview')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: '12px 16px',
              borderRadius: '8px',
              border: 'none',
              width: '100%',
              background: activeTab === 'overview' ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
              color: activeTab === 'overview' ? '#3b82f6' : 'rgba(148, 163, 184, 1)',
              cursor: 'pointer',
              marginBottom: '8px',
              transition: 'all 0.2s'
            }}
          >
            <BarChart3 className="nav-icon" size={20} />
            <span>Overview</span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'stream' ? 'active' : ''}`} 
            onClick={() => setActiveTab('stream')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: '12px 16px',
              borderRadius: '8px',
              border: 'none',
              width: '100%',
              background: activeTab === 'stream' ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
              color: activeTab === 'stream' ? '#3b82f6' : 'rgba(148, 163, 184, 1)',
              cursor: 'pointer',
              marginBottom: '8px',
              transition: 'all 0.2s'
            }}
          >
            <Activity className="nav-icon" size={20} />
            <span>Live Stream</span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'analytics' ? 'active' : ''}`} 
            onClick={() => setActiveTab('analytics')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: '12px 16px',
              borderRadius: '8px',
              border: 'none',
              width: '100%',
              background: activeTab === 'analytics' ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
              color: activeTab === 'analytics' ? '#3b82f6' : 'rgba(148, 163, 184, 1)',
              cursor: 'pointer',
              marginBottom: '8px',
              transition: 'all 0.2s'
            }}
          >
            <Server className="nav-icon" size={20} />
            <span>Analytics</span>
          </button>
        </nav>
      
        <div className="sidebar-footer" style={{ padding: '24px', marginTop: 'auto' }}>
          <div className="stream-status" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div className={`status-indicator ${activeTab === 'stream' ? 'streaming' : 'paused'}`} style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: activeTab === 'stream' ? '#10b981' : '#6b7280'
            }}>
              <div className="status-pulse"></div>
            </div>
            <span>{activeTab === 'stream' ? 'Streaming' : 'Idle'}</span>
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
        {/* Content - Conditional rendering based on active tab */}
        {activeTab === 'stream' ? (
          <div style={{ width: '100%', marginTop: '70px' }}>
            <Streaming />
          </div>
        ) : (
          <>
            {/* Header */}
            <header className="mb-8" style={{ marginTop: '70px' }}>
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-accent/20 rounded-lg">
                    <Shield className="w-8 h-8 text-accent" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold">NeuroDetect Dashboard</h1>
                    <p className="text-gray-400">Real-time anomaly detection monitoring</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-4">
                  <button className="flex items-center gap-2 px-4 py-2 glass-card rounded-lg hover:bg-white/10 transition-colors">
                    <AlertTriangle className="w-5 h-5" />
                    <span>Flagged Cases</span>
                    <span className="bg-red-500 text-white text-xs px-2 py-1 rounded-full">3</span>
                  </button>
                  
                  <button className="flex items-center gap-2 px-4 py-2 glass-card rounded-lg hover:bg-white/10 transition-colors">
                    <Settings className="w-5 h-5" />
                    <span>Model Configuration</span>
                  </button>
                  
                  <button className="flex items-center gap-2 px-4 py-2 glass-card rounded-lg hover:bg-white/10 transition-colors">
                    <Settings className="w-5 h-5" />
                    <span>System Settings</span>
                  </button>
                </div>
              </div>

              {/* Model selector */}
              <div className="mt-4 flex items-center gap-3">
                <label className="text-sm text-gray-300">Model:</label>
                <select value={selectedModel} onChange={e => setSelectedModel(e.target.value as any)} className="px-3 py-2 rounded bg-white/5 text-gray-200">
                  <option value="autoencoder">Autoencoder</option>
                  <option value="lstm">LSTM</option>
                  <option value="snn">SNN</option>
                </select>

                <button onClick={startDetector} className="px-4 py-2 bg-success text-white rounded">Launch Detector</button>
                <button onClick={stopDetector} className="px-4 py-2 bg-danger text-white rounded">Stop Detector</button>
                <div className="text-sm text-gray-300">Detector: {detectorStatus}</div>
              </div>
            </header>

            {/* Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <MetricCard
                title="Recall"
                value="98.2%"
                change="↑ +1.5%"
                icon={<TrendingUp className="w-6 h-6 text-success" />}
                trend="up"
              />
              
              <MetricCard
                title="Precision"
                value="94.5%"
                change="↓ -0.8%"
                icon={<TrendingDown className="w-6 h-6 text-danger" />}
                trend="down"
              />
              
              <MetricCard
                title="F1 Score"
                value="96.3%"
                change="↑ +0.7%"
                icon={<TrendingUp className="w-6 h-6 text-success" />}
                trend="up"
              />
              
              <MetricCard
                title="Latency"
                value="120ms"
                change="→ Stabilized"
                icon={<Activity className="w-6 h-6 text-warning" />}
                trend="stable"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Column - Transactions */}
              <div className="space-y-6">
                {/* Search and Filters */}
                <div className="glass-card p-4">
                  <div className="flex flex-col md:flex-row gap-4 mb-4">
                    <div className="flex-1 relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                      <input
                        type="text"
                        placeholder="Search transactions..."
                        className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                      />
                    </div>
                    
                    <div className="flex gap-2">
                      <select
                        className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent"
                        value={selectedSeverity}
                        onChange={(e) => setSelectedSeverity(e.target.value)}
                      >
                        <option>All Severities</option>
                        <option>High-Risk</option>
                        <option>Suspicious</option>
                        <option>Normal</option>
                      </select>
                      
                      <button className="flex items-center gap-2 px-4 py-2 bg-accent hover:bg-blue-500 rounded-lg transition-colors">
                        <Filter className="w-4 h-4" />
                        <span>Apply Filters</span>
                      </button>
                    </div>
                  </div>
                  
                  {/* Anomaly Details */}
                  <div className="mb-4">
                    <h2 className="text-lg font-semibold mb-2">Anomaly Details</h2>
                    <p className="text-gray-400 text-sm">
                      {selectedTransaction 
                        ? `Viewing details for ${selectedTransaction.userId}`
                        : "Select a transaction to view its anomaly details."}
                    </p>
                  </div>
                </div>

                {/* Live Transaction Feed */}
                <div className="glass-card p-6">
                  <h2 className="text-xl font-bold mb-4">Live Transaction Feed</h2>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-white/10">
                          <th className="text-left py-3 px-4 text-gray-400 font-medium text-[14px]">Timestamp</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium text-[14px]">User ID</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium text-[14px]">Amount</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium text-[14px]">Merchant</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium text-[14px]">Location</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium text-[14px]">Score</th>
                          <th className="text-left py-3 px-4 text-gray-400 font-medium text-[14px]">Severity</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredTransactions.map((tx, index) => (
                          <tr 
                            key={index}
                            className={`border-b border-white/10 hover:bg-white/5 cursor-pointer transition-colors ${
                              selectedTransaction?.userId === tx.userId ? 'bg-white/10' : ''
                            }`}
                            onClick={() => setSelectedTransaction(tx)}
                          >
                            <td className="py-3 px-4 text-sm">{tx.timestamp}</td>
                            <td className="py-3 px-4 font-medium">{tx.userId}</td>
                            <td className="py-3 px-4 font-medium">{tx.amount}</td>
                            <td className="py-3 px-4">{tx.merchant}</td>
                            <td className="py-3 px-4">{tx.location}</td>
                            <td className={`py-3 px-4 font-bold ${getScoreColor(tx.score)}`}>
                              {tx.score.toFixed(2)}
                            </td>
                            <td className="py-3 px-4">
                              <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getSeverityColor(tx.severity)}`}>
                                {tx.severity}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* Right Column - Real-time Graph */}
              <div className="glass-card p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold">Real-time Transaction Activity</h2>
                  <Activity className="w-6 h-6 text-accent" />
                </div>
                
                {/* Chart */}
                <div className="relative h-64">
                  <div className="absolute inset-0 flex items-end">
                    {/* Y-axis labels */}
                    <div className="flex flex-col justify-between h-full pb-8 pr-4 text-right">
                      {[32, 24, 16, 8, 0].map((value) => (
                        <span key={value} className="text-gray-400 text-sm">
                          {value}
                        </span>
                      ))}
                    </div>
                    
                    {/* Chart bars */}
                    <div className="flex-1 flex items-end justify-between px-4 pb-8">
                      {chartData.map((value, index) => (
                        <div key={index} className="flex flex-col items-center" style={{ width: '14%' }}>
                          <div
                            className="w-full bg-gradient-to-t from-accent to-blue-400 rounded-t-lg transition-all hover:opacity-80"
                            style={{ height: `${(value / 32) * 100}%` }}
                          />
                          <span className="text-gray-400 text-sm mt-2">{chartLabels[index]}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  {/* Grid lines */}
                  <div className="absolute inset-0 pointer-events-none">
                    {[0, 25, 50, 75, 100].map((line) => (
                      <div
                        key={line}
                        className="absolute left-0 right-0 border-t border-white/10"
                        style={{ bottom: `${line}%` }}
                      />
                    ))}
                  </div>
                </div>
                
                {/* Chart labels */}
                <div className="flex justify-between mt-12 pt-4 border-t border-white/10">
                  {chartLabels.map((label, index) => (
                    <span key={index} className="text-gray-400 text-sm">
                      {label}
                    </span>
                  ))}
                </div>
                
                {/* Stats summary */}
                <div className="mt-6 grid grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-white/5 rounded-lg">
                    <div className="text-2xl font-bold text-success">42</div>
                    <div className="text-gray-400 text-sm">Normal</div>
                  </div>
                  <div className="text-center p-4 bg-white/5 rounded-lg">
                    <div className="text-2xl font-bold text-warning">18</div>
                    <div className="text-gray-400 text-sm">Suspicious</div>
                  </div>
                  <div className="text-center p-4 bg-white/5 rounded-lg">
                    <div className="text-2xl font-bold text-danger">7</div>
                    <div className="text-gray-400 text-sm">High-Risk</div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
};

export default App;
