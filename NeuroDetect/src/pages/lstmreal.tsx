import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
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
  Brain,
  Target,
  AlertCircle,
  LineChart,
  PieChart,
  Cpu,
  Server,
  Network,
  ZapOff,
  ExternalLink,
  BarChart,
  Eye,
  EyeOff,
  Thermometer,
  GitBranch,
  Layers,
  Hash,
  Percent,
  Timer,
  Database as DbIcon,
  Cloud,
  Cpu as Processor
} from 'lucide-react';
import './css/lstmreal.css';

// Global WebSocket reference to persist across component remounts
declare global {
  interface Window {
    lstmDetectionWS?: WebSocket;
  }
}

interface LSTMFraudDetectionRecord {
  transaction_id: string;
  transaction_data: any;
  fraud_score: number;
  optimal_threshold: number;
  is_fraud: boolean;
  confidence: number;
  risk_level: 'Low' | 'Medium-Low' | 'Medium-High' | 'High';
  processing_time_ms: number;
  timestamp: string;
  sequence_length?: number;
  model_type?: string;
  [key: string]: any;
}

interface ModelStats {
  total_processed: number;
  fraud_detected: number;
  preprocessing_errors: number;
  dataset_loops: number;
  avg_processing_time: number;
  throughput_tps: number;
  fraud_rate: number;
  success_rate: number;
  risk_distribution?: {
    Low: number;
    'Medium-Low': number;
    'Medium-High': number;
    High: number;
  };
  prediction_history?: Array<{
    score: number;
    is_fraud: boolean;
    risk: string;
    timestamp: string;
  }>;
}

interface ModelInfo {
  input_dim: number;
  architecture: string;
  threshold: number;
  device: string;
  expected_features: number;
  feature_names: string[];
  num_features?: number;
  sequence_length?: number;
  hidden_size?: number;
  num_layers?: number;
}

const LSTMFraudDetectionDashboard: React.FC = () => {
  const navigate = useNavigate();
  
  // WebSocket state
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
  // Data state
  const [records, setRecords] = useState<LSTMFraudDetectionRecord[]>([]);
  const [modelStats, setModelStats] = useState<ModelStats>({
    total_processed: 0,
    fraud_detected: 0,
    preprocessing_errors: 0,
    dataset_loops: 0,
    avg_processing_time: 0,
    throughput_tps: 0,
    fraud_rate: 0,
    success_rate: 0,
    risk_distribution: { Low: 0, 'Medium-Low': 0, 'Medium-High': 0, High: 0 },
    prediction_history: []
  });
  
  const [modelInfo, setModelInfo] = useState<ModelInfo>({
    input_dim: 0,
    architecture: 'Enhanced LSTM',
    threshold: 0.5,
    device: 'cpu',
    expected_features: 0,
    feature_names: [],
    num_features: 0,
    sequence_length: 10,
    hidden_size: 128,
    num_layers: 2
  });

  // UI state
  const [streamSpeed, setStreamSpeed] = useState(1);
  const [maxRecords, setMaxRecords] = useState(500);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRisk, setSelectedRisk] = useState('all');
  const [showFraudOnly, setShowFraudOnly] = useState(false);
  const [activeTab, setActiveTab] = useState('lstm');
  const [scoreHistory, setScoreHistory] = useState<number[]>([]);
  const [showScoreChart, setShowScoreChart] = useState(true);

  // Refs
  const recordsRef = useRef<LSTMFraudDetectionRecord[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);

  // Sync streaming state across pages using localStorage
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'lstm_detection_streaming') {
        const newState = e.newValue === 'true';
        setIsStreaming(newState);
        
        // Send the appropriate command to the WebSocket
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          if (newState && !isStreaming) {
            wsRef.current.send(JSON.stringify({ command: 'start_stream' }));
          } else if (!newState && isStreaming) {
            wsRef.current.send(JSON.stringify({ command: 'stop_stream' }));
          }
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    
    // Check initial state from localStorage
    const savedState = localStorage.getItem('lstm_detection_streaming');
    if (savedState !== null) {
      const streamingState = savedState === 'true';
      setIsStreaming(streamingState);
    }

    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [isStreaming]);

  // Initialize WebSocket connection
  useEffect(() => {
    const setupWebSocketHandlers = (ws: WebSocket) => {
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
        localStorage.setItem('lstm_detection_streaming', 'false');
        
        // Clear global reference
        if (window.lstmDetectionWS === ws) {
          window.lstmDetectionWS = undefined;
        }
        
        // Attempt to reconnect after 3 seconds
        setTimeout(() => {
          if (!window.lstmDetectionWS || window.lstmDetectionWS.readyState === WebSocket.CLOSED) {
            connectWebSocket();
          }
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };

    const connectWebSocket = () => {
      // Check if there's already an active WebSocket connection
      if (window.lstmDetectionWS && window.lstmDetectionWS.readyState === WebSocket.OPEN) {
        console.log('Reusing existing WebSocket connection');
        wsRef.current = window.lstmDetectionWS;
        setSocket(window.lstmDetectionWS);
        setIsConnected(true);
        setConnectionStatus('connected');
        
        // Re-attach event handlers for this component instance
        setupWebSocketHandlers(window.lstmDetectionWS);
        return;
      }

      // Close any existing but non-functional WebSocket
      if (window.lstmDetectionWS) {
        try {
          window.lstmDetectionWS.close();
        } catch (e) {
          // Ignore errors
        }
      }

      setConnectionStatus('connecting');
      
      const ws = new WebSocket('ws://localhost:8765');
      wsRef.current = ws;
      window.lstmDetectionWS = ws; // Store globally

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setConnectionStatus('connected');
        
        // Request model info
        sendCommand('get_model_info');
        
        // Request status
        sendCommand('get_status');
      };

      // Set up event handlers
      setupWebSocketHandlers(ws);

      setSocket(ws);
    };

    connectWebSocket();

    return () => {
      // Only close WebSocket if not streaming (check localStorage for current state)
      const currentStreamingState = localStorage.getItem('lstm_detection_streaming');
      if (window.lstmDetectionWS && currentStreamingState !== 'true') {
        console.log('Closing WebSocket - streaming is not active');
        window.lstmDetectionWS.close();
        window.lstmDetectionWS = undefined;
      } else {
        console.log('Keeping WebSocket alive - streaming is active');
      }
    };
  }, []);

  // Periodically request statistics from server
  useEffect(() => {
    if (!isConnected || !isStreaming) return;

    const statsInterval = setInterval(() => {
      sendCommand('get_stats');
    }, 5000); // Request stats every 5 seconds

    return () => clearInterval(statsInterval);
  }, [isConnected, isStreaming]);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((data: any) => {
    // Control messages
    if (data.response === 'pong') {
      console.log('PONG from server', data);
      return;
    }

    if (data.error) {
      console.error('Server error:', data.error);
      return;
    }

    // Model info response
    if (data.model_info) {
      setModelInfo({
        input_dim: data.model_info.input_dim || 0,
        architecture: data.model_info.architecture || 'Enhanced LSTM',
        threshold: data.model_info.threshold || data.model_info.optimal_threshold || 0.5,
        device: data.model_info.device || 'cpu',
        expected_features: data.model_info.expected_features || 0,
        feature_names: data.model_info.feature_names || [],
        num_features: data.model_info.num_features || data.model_info.input_dim || 0,
        sequence_length: data.model_info.sequence_length || 10,
        hidden_size: data.model_info.hidden_size || 128,
        num_layers: data.model_info.num_layers || 2
      });
      return;
    }

    // Status messages
    if (typeof data.streaming === 'boolean' || data.status || data.speed) {
      if (typeof data.streaming === 'boolean') {
        setIsStreaming(data.streaming);
        localStorage.setItem('lstm_detection_streaming', data.streaming.toString());
      }
      if (data.status === 'stopped') {
        setIsStreaming(false);
        localStorage.setItem('lstm_detection_streaming', 'false');
      }
      if (data.status === 'already_streaming') {
        setIsStreaming(true);
        localStorage.setItem('lstm_detection_streaming', 'true');
      }
      if (data.speed) {
        setStreamSpeed(data.speed);
      }
      return;
    }

    // Statistics update
    if (data.stats) {
      setModelStats(prev => ({
        ...prev,
        ...data.stats
      }));
      return;
    }

    // Handle END_OF_DATASET
    if (data.message === 'END_OF_DATASET') {
      console.log('🔄 Dataset completed');
      return;
    }

    // Check if this is an LSTM fraud detection result (has fraud_score or is_fraud field)
    if (data.fraud_score !== undefined || data.is_fraud !== undefined || data.transaction_id) {
      // Log every 10th record to avoid console spam
      if (recordsRef.current.length % 10 === 0) {
        console.log('📊 LSTM fraud detection result:', {
          id: data.transaction_id,
          is_fraud: data.is_fraud,
          score: data.fraud_score,
          risk: data.risk_level,
          total_records: recordsRef.current.length
        });
      }

      // This is an LSTM fraud detection result
      const newRecord: LSTMFraudDetectionRecord = {
        transaction_id: data.transaction_id || `TXN_${Date.now()}`,
        transaction_data: data.transaction_data || data,
        fraud_score: data.fraud_score || 0,
        optimal_threshold: data.optimal_threshold || modelInfo.threshold,
        is_fraud: data.is_fraud === true || data.is_fraud === 1,
        confidence: data.confidence || 0,
        risk_level: data.risk_level || 'Low',
        processing_time_ms: data.processing_time_ms || 0,
        timestamp: data.timestamp || data.stream_timestamp || new Date().toISOString(),
        sequence_length: data.sequence_length || modelInfo.sequence_length,
        model_type: data.model_type || 'LSTM',
        ...data
      };

      // Update records
      const updatedRecords = [newRecord, ...recordsRef.current.slice(0, maxRecords - 1)];
      recordsRef.current = updatedRecords;
      setRecords(updatedRecords);
      
      // Publish data to localStorage for streaming page
      localStorage.setItem('lstm_detection_data', JSON.stringify(updatedRecords));

      // Update score history
      setScoreHistory(prev => [...prev.slice(-49), newRecord.fraud_score]);

      // Update statistics
      updateStats(newRecord);
    }
  }, [maxRecords, modelInfo.threshold, modelInfo.sequence_length]);

  // Update statistics
  const updateStats = useCallback((record: LSTMFraudDetectionRecord) => {
    setModelStats(prev => {
      const newTotal = prev.total_processed + 1;
      const newFraudCount = prev.fraud_detected + (record.is_fraud ? 1 : 0);
      const newFraudRate = (newFraudCount / newTotal) * 100;
      const newSuccessRate = ((newTotal - prev.preprocessing_errors) / newTotal) * 100;
      
      // Calculate average processing time
      const totalProcessingTime = (prev.avg_processing_time * prev.total_processed) + record.processing_time_ms;
      const newAvgProcessingTime = totalProcessingTime / newTotal;

      return {
        ...prev,
        total_processed: newTotal,
        fraud_detected: newFraudCount,
        avg_processing_time: newAvgProcessingTime,
        fraud_rate: newFraudRate,
        success_rate: newSuccessRate
      };
    });
  }, []);

  // WebSocket commands
  const sendCommand = useCallback((command: string, data?: any) => {
    // Use wsRef.current or fall back to global reference
    const ws = wsRef.current || window.lstmDetectionWS;
    
    if (!ws) {
      console.warn('WebSocket not initialized');
      return;
    }

    if (ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not open');
      return;
    }

    ws.send(JSON.stringify({ command, ...data }));
  }, []);

  const startStreaming = () => {
    setIsStreaming(true);
    localStorage.setItem('lstm_detection_streaming', 'true');
    sendCommand('start_stream');
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    localStorage.setItem('lstm_detection_streaming', 'false');
    sendCommand('stop_stream');
  };

  const updateStreamSpeed = (speed: number) => {
    setStreamSpeed(speed);
    sendCommand('set_speed', { speed });
  };

  const resetStats = () => {
    setRecords([]);
    setScoreHistory([]);
    setModelStats({
      total_processed: 0,
      fraud_detected: 0,
      preprocessing_errors: 0,
      dataset_loops: 0,
      avg_processing_time: 0,
      throughput_tps: 0,
      fraud_rate: 0,
      success_rate: 0,
      risk_distribution: { Low: 0, 'Medium-Low': 0, 'Medium-High': 0, High: 0 },
      prediction_history: []
    });
    
    // Clear data in localStorage for streaming page
    localStorage.setItem('lstm_detection_data', JSON.stringify([]));
  };

  // Filter records
  const filteredRecords = records.filter(record => {
    const matchesSearch = searchTerm === '' || 
      record.transaction_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.transaction_data?.category?.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesRisk = selectedRisk === 'all' || 
      record.risk_level === selectedRisk;

    const matchesFraudFilter = !showFraudOnly || record.is_fraud;

    return matchesSearch && matchesRisk && matchesFraudFilter;
  });

  // Get risk distribution - use server data if available, otherwise calculate from local records
  const riskDistribution = modelStats.risk_distribution && 
    (modelStats.risk_distribution.Low > 0 || 
     modelStats.risk_distribution['Medium-Low'] > 0 || 
     modelStats.risk_distribution['Medium-High'] > 0 || 
     modelStats.risk_distribution.High > 0)
    ? modelStats.risk_distribution
    : {
        Low: records.filter(r => r.risk_level === 'Low').length,
        'Medium-Low': records.filter(r => r.risk_level === 'Medium-Low').length,
        'Medium-High': records.filter(r => r.risk_level === 'Medium-High').length,
        High: records.filter(r => r.risk_level === 'High').length
      };

  // Format fraud score
  const formatScore = (score: number) => {
    return score.toFixed(4);
  };

  // Get score severity color
  const getScoreColor = (score: number, threshold: number) => {
    const ratio = score / threshold;
    if (ratio < 0.5) return 'low';
    if (ratio < 1.0) return 'medium';
    return 'high';
  };

  // Risk level color
  const getRiskColor = (riskLevel: string) => {
    switch(riskLevel) {
      case 'Low': return 'var(--success)';
      case 'Medium-Low': return 'var(--info)';
      case 'Medium-High': return 'var(--warning)';
      case 'High': return 'var(--danger)';
      default: return 'var(--text-muted)';
    }
  };

  // Draw fraud score chart
  useEffect(() => {
    if (!chartRef.current || scoreHistory.length === 0) return;

    const canvas = chartRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set canvas dimensions
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;

    // Draw threshold line
    ctx.beginPath();
    const thresholdY = height - padding - (modelInfo.threshold / Math.max(...scoreHistory, modelInfo.threshold, 1)) * chartHeight;
    ctx.moveTo(padding, thresholdY);
    ctx.lineTo(width - padding, thresholdY);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw threshold label
    ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.font = '12px monospace';
    ctx.fillText(`Threshold: ${modelInfo.threshold.toFixed(4)}`, width - padding - 120, thresholdY - 10);

    // Draw score line
    if (scoreHistory.length > 1) {
      ctx.beginPath();
      scoreHistory.forEach((score, index) => {
        const x = padding + (index / (scoreHistory.length - 1)) * chartWidth;
        const y = height - padding - (score / Math.max(...scoreHistory, modelInfo.threshold, 1)) * chartHeight;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.strokeStyle = 'var(--primary)';
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    // Draw points
    scoreHistory.forEach((score, index) => {
      const x = padding + (index / (scoreHistory.length - 1)) * chartWidth;
      const y = height - padding - (score / Math.max(...scoreHistory, modelInfo.threshold, 1)) * chartHeight;
      
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = getScoreColor(score, modelInfo.threshold) === 'high' ? 
        'var(--danger)' : getScoreColor(score, modelInfo.threshold) === 'medium' ? 
        'var(--warning)' : 'var(--success)';
      ctx.fill();
    });

  }, [scoreHistory, modelInfo.threshold]);

  return (
    <div className="fraud-dashboard">
      {/* Sidebar */}
      <aside className="fraud-sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <Brain className="logo-icon" />
            <span className="logo-text">NeuroDetect</span>
          </div>
          <div className="model-badge">
            <div className="model-type">LSTM</div>
            <div className={`connection-dot ${connectionStatus}`}></div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button 
            className="nav-item"
            onClick={() => navigate('/')}
          >
            <BarChart3 className="nav-icon" />
            <span>Autoencoder</span>
          </button>
          <button 
            className={`nav-item active`}
            onClick={() => setActiveTab('lstm')}
          >
            <Target className="nav-icon" />
            <span>LSTM</span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'model' ? 'active' : ''}`}
            onClick={() => setActiveTab('model')}
          >
            <Layers className="nav-icon" />
            <span>Model</span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            <LineChart className="nav-icon" />
            <span>Analytics</span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'system' ? 'active' : ''}`}
            onClick={() => setActiveTab('system')}
          >
            <Cpu className="nav-icon" />
            <span>System</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className="threshold-display">
            <Thermometer className="threshold-icon" />
            <div>
              <div className="threshold-value">
                {modelInfo.threshold.toFixed(4)}
              </div>
              <div className="threshold-label">Detection Threshold</div>
            </div>
          </div>
          <div className="accuracy-display">
            <Percent className="accuracy-icon" />
            <div>
              <div className="accuracy-value">
                {(100 - modelStats.fraud_rate).toFixed(1)}%
              </div>
              <div className="accuracy-label">Accuracy</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="fraud-main">
        {/* Top Bar */}
        <header className="fraud-topbar">
          <div className="topbar-left">
            <h1>LSTM Fraud Detection</h1>
            <p className="subtitle">Real-time sequence-based fraud detection using LSTM networks</p>
          </div>
          
          <div className="topbar-right">
            <div className="model-arch">
              <GitBranch className="arch-icon" />
              <span>{modelInfo.architecture}</span>
            </div>
            
            <div className="control-group">
              <button 
                className="control-btn test-btn"
                onClick={() => navigate('/streaming')}
              >
                <Network className="btn-icon" />
                <span>View</span>
              </button>
              
              <button
                className={`control-btn start-btn ${!isConnected || isStreaming ? 'disabled' : ''}`}
                onClick={startStreaming}
                disabled={!isConnected || isStreaming}
              >
                <Play className="btn-icon" />
                <span>Start</span>
              </button>
              
              <button
                className={`control-btn stop-btn ${!isConnected || !isStreaming ? 'disabled' : ''}`}
                onClick={stopStreaming}
                disabled={!isConnected || !isStreaming}
              >
                <Pause className="btn-icon" />
                <span>Stop</span>
              </button>
              
              <button
                className="control-btn reset-btn"
                onClick={resetStats}
              >
                <RefreshCw className="btn-icon" />
                <span>Reset</span>
              </button>
            </div>
          </div>
        </header>

        {/* Model Performance Stats */}
        <div className="performance-grid">
          <div className="performance-card large">
            <div className="card-header">
              <h3>Fraud Score Chart</h3>
              <button 
                className="chart-toggle"
                onClick={() => setShowScoreChart(!showScoreChart)}
              >
                {showScoreChart ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            <div className={`chart-container ${showScoreChart ? 'visible' : 'hidden'}`}>
              <canvas ref={chartRef} className="error-chart"></canvas>
            </div>
            <div className="chart-footer">
              <div className="chart-stats">
                <div className="chart-stat">
                  <span className="stat-label">Current Score:</span>
                  <span className="stat-value">
                    {records[0]?.fraud_score ? formatScore(records[0].fraud_score) : 'N/A'}
                  </span>
                </div>
                <div className="chart-stat">
                  <span className="stat-label">Threshold:</span>
                  <span className="stat-value">{modelInfo.threshold.toFixed(4)}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="performance-card">
            <div className="card-header">
              <h3>Detection Rate</h3>
              <Target className="card-icon" />
            </div>
            <div className="detection-rate">
              <div className="rate-value">{modelStats.fraud_rate.toFixed(2)}%</div>
              <div className="rate-label">Fraud Detection Rate</div>
              <div className="rate-bar">
                <div 
                  className="rate-fill"
                  style={{ width: `${Math.min(modelStats.fraud_rate, 100)}%` }}
                ></div>
              </div>
              <div className="rate-stats">
                <span>Detected: {modelStats.fraud_detected}</span>
                <span>Total: {modelStats.total_processed}</span>
              </div>
            </div>
          </div>

          <div className="performance-card">
            <div className="card-header">
              <h3>Processing Speed</h3>
              <Gauge className="card-icon" />
            </div>
            <div className="speed-metrics">
              <div className="speed-value">
                {modelStats.avg_processing_time.toFixed(2)} ms
              </div>
              <div className="speed-label">Avg Processing Time</div>
              
              <div className="speed-control">
                <div className="speed-header">
                  <span className="speed-label">Stream Speed</span>
                  <span className="speed-value">{streamSpeed}/sec</span>
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
              </div>
            </div>
          </div>

          <div className="performance-card">
            <div className="card-header">
              <h3>Risk Distribution</h3>
              <PieChart className="card-icon" />
            </div>
            <div className="risk-distribution">
              {Object.entries(riskDistribution).map(([risk, count]) => {
                const totalCount = Object.values(riskDistribution).reduce((sum, c) => sum + c, 0);
                const percentage = totalCount > 0 ? ((count / totalCount) * 100).toFixed(1) : '0.0';
                return (
                  <div key={risk} className="risk-item">
                    <div className="risk-info">
                      <div className="risk-dot" style={{ backgroundColor: getRiskColor(risk) }}></div>
                      <span className="risk-label">{risk}</span>
                      <span className="risk-count">{count} ({percentage}%)</span>
                    </div>
                    <div className="risk-bar">
                      <div 
                        className="risk-fill"
                        style={{ 
                          width: `${percentage}%`,
                          backgroundColor: getRiskColor(risk)
                        }}
                      ></div>
                    </div>
                  </div>
                );
              })}
              {Object.values(riskDistribution).reduce((sum, c) => sum + c, 0) === 0 && (
                <div style={{ padding: '1rem', textAlign: 'center', opacity: 0.5 }}>
                  No predictions yet. Start streaming to see risk distribution.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="content-area">
          {/* Left Column */}
          <div className="content-column">
            {/* Model Architecture Card */}
            <div className="content-card">
              <div className="card-header">
                <h2>Model Architecture</h2>
                <Brain className="card-icon" />
              </div>
              
              <div className="model-architecture">
                <div className="arch-layers">
                  <div className="layer input">
                    <div className="layer-info">
                      <Hash className="layer-icon" />
                      <div>
                        <div className="layer-name">Input Layer</div>
                        <div className="layer-dims">{modelInfo.input_dim} features</div>
                      </div>
                    </div>
                    <div className="layer-nodes">{modelInfo.input_dim}</div>
                  </div>
                  
                  <div className="layer-arrow">→</div>
                  
                  <div className="layer hidden">
                    <div className="layer-info">
                      <Layers className="layer-icon" />
                      <div>
                        <div className="layer-name">LSTM Layer</div>
                        <div className="layer-dims">Sequence Processing</div>
                      </div>
                    </div>
                    <div className="layer-nodes">{modelInfo.hidden_size}</div>
                  </div>
                  
                  <div className="layer-arrow">→</div>
                  
                  <div className="layer hidden">
                    <div className="layer-info">
                      <Layers className="layer-icon" />
                      <div>
                        <div className="layer-name">Attention</div>
                        <div className="layer-dims">Context Weighing</div>
                      </div>
                    </div>
                    <div className="layer-nodes">{modelInfo.hidden_size}</div>
                  </div>
                  
                  <div className="layer-arrow">→</div>
                  
                  <div className="layer latent">
                    <div className="layer-info">
                      <GitBranch className="layer-icon" />
                      <div>
                        <div className="layer-name">Output</div>
                        <div className="layer-dims">Classification</div>
                      </div>
                    </div>
                    <div className="layer-nodes">1</div>
                  </div>
                </div>
                
                <div className="model-info">
                  <div className="info-item">
                    <span className="info-label">Threshold:</span>
                    <span className="info-value">{modelInfo.threshold.toFixed(6)}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Sequence Length:</span>
                    <span className="info-value">{modelInfo.sequence_length}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Hidden Size:</span>
                    <span className="info-value">{modelInfo.hidden_size}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">LSTM Layers:</span>
                    <span className="info-value">{modelInfo.num_layers}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Device:</span>
                    <span className="info-value">{modelInfo.device.toUpperCase()}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Total Features:</span>
                    <span className="info-value">{modelInfo.num_features || modelInfo.expected_features}</span>
                  </div>
                  <div className="info-item" style={{ gridColumn: '1 / -1' }}>
                    <span className="info-label">Feature Names:</span>
                    <span className="info-value" title={modelInfo.feature_names?.join(', ')}>
                      {modelInfo.feature_names?.slice(0, 5).join(', ')}
                      {modelInfo.feature_names?.length > 5 && ` ... +${modelInfo.feature_names.length - 5} more`}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Live Detections Table */}
            <div className="content-card large">
              <div className="card-header">
                <h2>Live Detections</h2>
                <div className="live-badge">
                  <div className={`live-dot ${isStreaming ? 'active' : ''}`}></div>
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
                    value={selectedRisk}
                    onChange={(e) => setSelectedRisk(e.target.value)}
                  >
                    <option value="all">All Risks</option>
                    <option value="Low">Low Risk</option>
                    <option value="Medium-Low">Medium-Low</option>
                    <option value="Medium-High">Medium-High</option>
                    <option value="High">High Risk</option>
                  </select>

                  <button
                    className={`filter-btn ${showFraudOnly ? 'active' : ''}`}
                    onClick={() => setShowFraudOnly(!showFraudOnly)}
                  >
                    <Filter className="btn-icon" />
                    <span>Fraud Only</span>
                  </button>

                  <button
                    className="filter-btn"
                    onClick={() => setMaxRecords(prev => prev === 500 ? 100 : 500)}
                  >
                    <span>Limit: {maxRecords}</span>
                  </button>
                </div>
              </div>

              {/* Detections Table */}
              <div className="detections-table">
                <div className="table-header">
                  <div className="table-col">Transaction ID</div>
                  <div className="table-col">Fraud Score</div>
                  <div className="table-col">Risk Level</div>
                  <div className="table-col">Fraud Status</div>
                  <div className="table-col">Confidence</div>
                </div>
                
                <div className="table-body">
                  {filteredRecords.slice(0, 20).map((record, index) => (
                    <div 
                      key={`${record.transaction_id}-${index}`}
                      className={`table-row ${record.risk_level.toLowerCase().replace('-', '')}`}
                    >
                      <div className="table-col">
                        <div className="transaction-id">{record.transaction_id}</div>
                        <div className="transaction-category">
                          {record.transaction_data?.category || 'Unknown'}
                        </div>
                      </div>
                      <div className="table-col">
                        <div className="error-display">
                          <div className="error-value" title={`Fraud Score: ${record.fraud_score}`}>
                            {formatScore(record.fraud_score)}
                          </div>
                          <div className="error-threshold" style={{ fontSize: '0.7em', opacity: 0.6 }}>
                            vs {formatScore(record.optimal_threshold)}
                          </div>
                          <div className="error-bar">
                            <div 
                              className="error-fill"
                              style={{ 
                                width: `${Math.min((record.fraud_score / record.optimal_threshold) * 100, 100)}%`,
                                backgroundColor: getRiskColor(record.risk_level)
                              }}
                            ></div>
                          </div>
                        </div>
                      </div>
                      <div className="table-col">
                        <div 
                          className="risk-badge"
                          style={{ backgroundColor: getRiskColor(record.risk_level) }}
                        >
                          {record.risk_level}
                        </div>
                        <div style={{ fontSize: '0.7em', marginTop: '4px', opacity: 0.7 }}>
                          {record.processing_time_ms.toFixed(2)} ms
                        </div>
                      </div>
                      <div className="table-col">
                        {record.is_fraud ? (
                          <span className="fraud-indicator fraud">
                            <AlertTriangle className="indicator-icon" />
                            Fraud
                          </span>
                        ) : (
                          <span className="fraud-indicator normal">
                            <CheckCircle className="indicator-icon" />
                            Normal
                          </span>
                        )}
                      </div>
                      <div className="table-col">
                        <div className="fraud-probability">
                          <div className="probability-value">
                            {(record.confidence * 100).toFixed(1)}%
                          </div>
                          <div className="probability-bar">
                            <div 
                              className="probability-fill"
                              style={{ 
                                width: `${Math.min(record.confidence * 100, 100)}%`,
                                backgroundColor: getRiskColor(record.risk_level)
                              }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="table-footer">
                <span>Showing {Math.min(filteredRecords.length, 20)} of {records.length} records (Total processed: {modelStats.total_processed})</span>
                <span>Last updated: {records[0]?.timestamp ? new Date(records[0].timestamp).toLocaleTimeString() : '--:--:--'}</span>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="content-column">
            {/* System Metrics */}
            <div className="content-card">
              <div className="card-header">
                <h2>System Metrics</h2>
                <Server className="card-icon" />
              </div>
              
              <div className="metrics-grid">
                <div className="metric-item">
                  <div className="metric-icon">
                    <Timer />
                  </div>
                  <div className="metric-content">
                    <div className="metric-value">
                      {modelStats.avg_processing_time.toFixed(2)} ms
                    </div>
                    <div className="metric-label">Avg Processing</div>
                  </div>
                </div>

                <div className="metric-item">
                  <div className="metric-icon">
                    <Processor />
                  </div>
                  <div className="metric-content">
                    <div className="metric-value">
                      {modelStats.throughput_tps.toFixed(2)} TPS
                    </div>
                    <div className="metric-label">Throughput</div>
                  </div>
                </div>

                <div className="metric-item">
                  <div className="metric-icon">
                    <Database />
                  </div>
                  <div className="metric-content">
                    <div className="metric-value">
                      {modelStats.total_processed}
                    </div>
                    <div className="metric-label">Total Processed</div>
                  </div>
                </div>

                <div className="metric-item">
                  <div className="metric-icon">
                    <Cloud />
                  </div>
                  <div className="metric-content">
                    <div className="metric-value">
                      {modelStats.dataset_loops}
                    </div>
                    <div className="metric-label">Dataset Loops</div>
                  </div>
                </div>
              </div>

              <div className="system-stats">
                <div className="stat-item">
                  <span className="stat-label">Success Rate:</span>
                  <span className="stat-value">{modelStats.success_rate.toFixed(1)}%</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Preprocessing Errors:</span>
                  <span className="stat-value">{modelStats.preprocessing_errors}</span>
                </div>
              </div>
            </div>

            {/* Feature Importance */}
            <div className="content-card">
              <div className="card-header">
                <h2>Top Features</h2>
                <BarChart className="card-icon" />
              </div>
              
              <div className="features-list">
                {modelInfo.feature_names?.slice(0, 8).map((feature, index) => {
                  // Calculate a weight based on feature position (earlier features tend to be more important)
                  const weight = 100 - (index * 10);
                  return (
                    <div key={feature} className="feature-item">
                      <div className="feature-info">
                        <span className="feature-name">{feature}</span>
                        <span className="feature-weight">
                          {weight}%
                        </span>
                      </div>
                      <div className="feature-bar">
                        <div 
                          className="feature-fill"
                          style={{ 
                            width: `${weight}%`,
                            backgroundColor: `hsl(${220 - index * 15}, 70%, 50%)`
                          }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
                {(!modelInfo.feature_names || modelInfo.feature_names.length === 0) && (
                  <div style={{ padding: '1rem', textAlign: 'center', opacity: 0.5 }}>
                    No feature data available. Start streaming to load model info.
                  </div>
                )}
              </div>
            </div>

            {/* Alert Summary */}
            <div className="content-card">
              <div className="card-header">
                <h2>Alert Summary</h2>
                <AlertCircle className="card-icon" />
              </div>
              
              <div className="alerts-summary">
                <div className="alert-item high">
                  <div className="alert-count">{riskDistribution.High}</div>
                  <div className="alert-info">
                    <div className="alert-title">High Risk Alerts</div>
                    <div className="alert-desc">Immediate action required</div>
                  </div>
                </div>
                
                <div className="alert-item medium">
                  <div className="alert-count">{riskDistribution['Medium-High']}</div>
                  <div className="alert-info">
                    <div className="alert-title">Medium-High Risk Alerts</div>
                    <div className="alert-desc">Review recommended</div>
                  </div>
                </div>
                
                <div className="alert-item medium">
                  <div className="alert-count">{riskDistribution['Medium-Low']}</div>
                  <div className="alert-info">
                    <div className="alert-title">Medium-Low Risk Alerts</div>
                    <div className="alert-desc">Monitor closely</div>
                  </div>
                </div>
                
                <div className="alert-item low">
                  <div className="alert-count">{riskDistribution.Low}</div>
                  <div className="alert-info">
                    <div className="alert-title">Low Risk Alerts</div>
                    <div className="alert-desc">Monitor only</div>
                  </div>
                </div>
              </div>

              <div className="card-actions">
                <button className="action-btn export-btn">
                  <Download className="btn-icon" />
                  Export Alerts
                </button>
                <button className="action-btn settings-btn">
                  <Settings className="btn-icon" />
                  Alert Settings
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="fraud-footer">
          <div className="footer-info">
            <div className="info-item">
              <span className="info-label">LSTM Model:</span>
              <span className="info-value">{modelInfo.architecture} Architecture</span>
            </div>
            <div className="info-item">
              <span className="info-label">Connected to:</span>
              <span className="info-value">ws://localhost:8765</span>
            </div>
            <div className="info-item">
              <span className="info-label">Status:</span>
              <span className={`status-value ${isStreaming ? 'streaming' : 'paused'}`}>
                {isStreaming ? 'Streaming Active' : 'Streaming Paused'}
              </span>
            </div>
          </div>
          <div className="footer-meta">
            <span>LSTM Fraud Detection System v2.0 • Real-time Sequence-Based Detection</span>
          </div>
        </footer>
      </main>
    </div>
  );
};

export default LSTMFraudDetectionDashboard;
