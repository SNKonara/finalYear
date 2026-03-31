import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  AlertTriangle,
  CheckCircle,
  BarChart3,
  Brain,
  Layers,
  LineChart,
  Cpu,
  Thermometer,
  Eye,
  EyeOff,
  AlertCircle,
  BarChart,
  Upload
} from 'lucide-react';
import '../../pages/css/aereal.css';
import { useModelNavbar } from '../layout/ModelNavbarContext';
import { useTheme } from '../theme/ThemeContext';
import { useAuth } from '../auth/AuthContext';

type ModelType = 'autoencoder' | 'lstm' | 'snn';

const getModelFromRoute = (pathname: string, stateModel?: ModelType | null): ModelType => {
  if (stateModel) {
    return stateModel;
  }
  if (pathname === '/lstmreal') {
    return 'lstm';
  }
  if (pathname === '/snnreal') {
    return 'snn';
  }
  return 'snn';
};

const getRouteForModel = (model: ModelType): string => {
  if (model === 'lstm') {
    return '/lstmreal';
  }
  if (model === 'snn') {
    return '/snnreal';
  }
  return '/';
};

interface FraudDetectionRecord {
  transaction_id: string;
  transaction_data: any;
  reconstruction_error?: number;
  fraud_score?: number;
  fraud_probability?: number;
  decision_threshold?: number | null;
  threshold: number;
  optimal_threshold?: number;
  is_fraud: boolean;
  confidence?: number;
  risk_level: string;
  processing_time_ms: number;
  timestamp: string;
  sequence_length?: number;
  time_steps?: number;
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
  risk_distribution?: Record<string, number>;
  prediction_history?: Array<{
    error?: number;
    score?: number;
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
  top_categories?: string[];
  category_columns?: string[];
  hidden_size?: number;
  num_layers?: number;
  output_size?: number;
  time_steps?: number;
  threshold_scale?: number;
  global_threshold?: number;
  unknown_customer_policy?: string;
  performance?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
    auc?: number;
  };
}

interface DashboardSnapshot {
  records: FraudDetectionRecord[];
  errorHistory: number[];
  modelStats: ModelStats;
  modelInfo: ModelInfo;
  streamSpeed: number;
}

const getDashboardSnapshotKey = (model: ModelType) => `fraud_dashboard_snapshot_${model}`;

declare global {
  interface Window {
    unifiedDetectionWS?: WebSocket;
  }
}

const UnifiedModelsDashboard: React.FC = () => {
  const { setNavbarConfig, clearNavbarConfig } = useModelNavbar();
  const { isDarkTheme, currentTheme } = useTheme();
  const { currentUser } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Pre-select the model if batch upload passed one via navigation state
  const stateModel = (location.state as { model?: ModelType } | null)?.model;
  const [selectedModel, setSelectedModel] = useState<ModelType>(() => getModelFromRoute(location.pathname, stateModel ?? null));

  // WebSocket state
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
  // Data state
  const [records, setRecords] = useState<FraudDetectionRecord[]>([]);
  const [modelStats, setModelStats] = useState<ModelStats>({
    total_processed: 0,
    fraud_detected: 0,
    preprocessing_errors: 0,
    dataset_loops: 0,
    avg_processing_time: 0,
    throughput_tps: 0,
    fraud_rate: 0,
    success_rate: 0,
    risk_distribution: {},
    prediction_history: []
  });
  
  const [modelInfo, setModelInfo] = useState<ModelInfo>({
    input_dim: 0,
    architecture: '128-64-16',
    threshold: 0,
    device: 'cpu',
    expected_features: 0,
    feature_names: [],
    num_features: 0,
    top_categories: [],
    category_columns: []
  });

  // UI state
  const [streamSpeed, setStreamSpeed] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRisk, setSelectedRisk] = useState('all');
  const [showFraudOnly, setShowFraudOnly] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [errorHistory, setErrorHistory] = useState<number[]>([]);
  const [showChart, setShowChart] = useState(true);
  // Refs
  const recordsRef = useRef<FraudDetectionRecord[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);

  // Helper function to normalize numbers
  const toNumber = (value: unknown, fallback = 0): number => {
    const parsed = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  // Get model display name
  const getModelDisplayName = (model: ModelType) => {
    const names = { autoencoder: 'Autoencoder', lstm: 'LSTM', snn: 'SNN' };
    return names[model];
  };

  // Get risk levels for current model
  const getRiskLevels = () => {
    if (selectedModel === 'autoencoder') {
      return ['Low', 'Medium', 'High'];
    }
    return ['Low', 'Medium-Low', 'Medium-High', 'High'];
  };

  // Handle model change
  const handleModelChange = (model: ModelType) => {
    setSelectedModel(model);
    const targetRoute = getRouteForModel(model);
    if (location.pathname !== targetRoute) {
      navigate(targetRoute);
    }
    setRecords([]);
    setErrorHistory([]);
    setModelStats({
      total_processed: 0,
      fraud_detected: 0,
      preprocessing_errors: 0,
      dataset_loops: 0,
      avg_processing_time: 0,
      throughput_tps: 0,
      fraud_rate: 0,
      success_rate: 0,
      risk_distribution: {},
      prediction_history: []
    });
    recordsRef.current = [];
    
    // Send model change command to WebSocket
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: 'set_model', model }));
      sendCommand('get_model_info');
      sendCommand('get_status');
    }
  };

  useEffect(() => {
    const routeModel = getModelFromRoute(location.pathname, stateModel ?? null);
    setSelectedModel((previous) => (previous === routeModel ? previous : routeModel));
  }, [location.pathname, stateModel]);

  // Restore cached dashboard state instantly when returning to this page.
  useEffect(() => {
    const snapshotRaw = localStorage.getItem(getDashboardSnapshotKey(selectedModel));
    if (!snapshotRaw) {
      return;
    }

    try {
      const snapshot = JSON.parse(snapshotRaw) as Partial<DashboardSnapshot>;
      const cachedRecords = Array.isArray(snapshot.records) ? snapshot.records.slice(0, 500) : [];
      const cachedErrorHistory = Array.isArray(snapshot.errorHistory)
        ? snapshot.errorHistory.map((value) => toNumber(value, 0)).slice(-50)
        : [];

      if (cachedRecords.length > 0) {
        recordsRef.current = cachedRecords;
        setRecords(cachedRecords);
      }

      if (cachedErrorHistory.length > 0) {
        setErrorHistory(cachedErrorHistory);
      }

      if (snapshot.modelStats) {
        setModelStats((previous) => ({ ...previous, ...snapshot.modelStats }));
      }

      if (snapshot.modelInfo) {
        setModelInfo((previous) => ({ ...previous, ...snapshot.modelInfo }));
      }

      if (typeof snapshot.streamSpeed === 'number' && Number.isFinite(snapshot.streamSpeed)) {
        setStreamSpeed(snapshot.streamSpeed);
      }
    } catch (error) {
      console.warn('Failed to restore dashboard snapshot:', error);
    }
  }, [selectedModel]);

  // Persist a lightweight snapshot so chart/table can render immediately after navigation.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      const snapshot: DashboardSnapshot = {
        records: records.slice(0, 120),
        errorHistory: errorHistory.slice(-50),
        modelStats,
        modelInfo,
        streamSpeed,
      };

      localStorage.setItem(getDashboardSnapshotKey(selectedModel), JSON.stringify(snapshot));
    }, 250);

    return () => window.clearTimeout(timer);
  }, [selectedModel, records, errorHistory, modelStats, modelInfo, streamSpeed]);

  // Sync streaming state across pages using localStorage
  useEffect(() => {
    const storageKey = `fraud_detection_streaming_${selectedModel}`;
    
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === storageKey) {
        const newState = e.newValue === 'true';
        setIsStreaming(newState);
        
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
    
    const savedState = localStorage.getItem(storageKey);
    if (savedState !== null) {
      const streamingState = savedState === 'true';
      setIsStreaming(streamingState);
    }

    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [selectedModel, isStreaming]);

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
        localStorage.setItem(`fraud_detection_streaming_${selectedModel}`, 'false');
        
        if (window.unifiedDetectionWS === ws) {
          window.unifiedDetectionWS = undefined;
        }
        
        setTimeout(() => {
          if (!window.unifiedDetectionWS || window.unifiedDetectionWS.readyState === WebSocket.CLOSED) {
            connectWebSocket();
          }
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };

    const connectWebSocket = () => {
      if (window.unifiedDetectionWS && window.unifiedDetectionWS.readyState === WebSocket.OPEN) {
        console.log('Reusing existing WebSocket connection');
        wsRef.current = window.unifiedDetectionWS;
        setIsConnected(true);
        setConnectionStatus('connected');
        
        setupWebSocketHandlers(window.unifiedDetectionWS);
        return;
      }

      if (window.unifiedDetectionWS) {
        try {
          window.unifiedDetectionWS.close();
        } catch (e) {
          // Ignore
        }
      }

      setConnectionStatus('connecting');
      
      const ws = new WebSocket('ws://localhost:8765');
      wsRef.current = ws;
      window.unifiedDetectionWS = ws;

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setConnectionStatus('connected');
        
        ws.send(JSON.stringify({ command: 'set_model', model: selectedModel }));
        sendCommand('get_model_info');
        sendCommand('get_status');
      };

      setupWebSocketHandlers(ws);
    };

    connectWebSocket();

    return () => {
      const currentStreamingState = localStorage.getItem(`fraud_detection_streaming_${selectedModel}`);
      if (window.unifiedDetectionWS && currentStreamingState !== 'true') {
        console.log('Closing WebSocket - streaming is not active');
        window.unifiedDetectionWS.close();
        window.unifiedDetectionWS = undefined;
      } else {
        console.log('Keeping WebSocket alive - streaming is active');
      }
    };
  }, [selectedModel]);

  // Request statistics periodically
  useEffect(() => {
    if (!isConnected || !isStreaming) return;

    const statsInterval = setInterval(() => {
      sendCommand('get_stats');
    }, 5000);

    return () => clearInterval(statsInterval);
  }, [isConnected, isStreaming]);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((data: any) => {
    if (data.response === 'pong') {
      console.log('PONG from server', data);
      return;
    }

    if (data.error) {
      console.error('Server error:', data.error);
      return;
    }

    if (data.model_info) {
      setModelInfo({
        input_dim: toNumber(data.model_info.input_dim, 0),
        architecture: data.model_info.architecture || '128-64-16',
        threshold: toNumber(data.model_info.threshold, 0),
        device: data.model_info.device || 'cpu',
        expected_features: toNumber(data.model_info.expected_features, 0),
        feature_names: data.model_info.feature_names || [],
        num_features: toNumber(data.model_info.num_features ?? data.model_info.input_dim, 0),
        top_categories: data.model_info.top_categories || [],
        category_columns: data.model_info.category_columns || [],
        hidden_size: toNumber(data.model_info.hidden_size, 64),
        num_layers: toNumber(data.model_info.num_layers, 2),
        output_size: toNumber(data.model_info.output_size, 2),
        time_steps: toNumber(data.model_info.time_steps, 20),
        threshold_scale: toNumber(data.model_info.threshold_scale, 1.0),
        global_threshold: toNumber(data.model_info.global_threshold, 0.5),
        unknown_customer_policy: data.model_info.unknown_customer_policy || 'global',
        performance: data.model_info.performance || undefined
      });
      return;
    }

    if (typeof data.streaming === 'boolean' || data.status || data.speed) {
      if (typeof data.streaming === 'boolean') {
        setIsStreaming(data.streaming);
        localStorage.setItem(`fraud_detection_streaming_${selectedModel}`, data.streaming.toString());
      }
      if (data.status === 'stopped') {
        setIsStreaming(false);
        localStorage.setItem(`fraud_detection_streaming_${selectedModel}`, 'false');
      }
      if (data.status === 'already_streaming') {
        setIsStreaming(true);
        localStorage.setItem(`fraud_detection_streaming_${selectedModel}`, 'true');
      }
      if (data.speed) {
        setStreamSpeed(data.speed);
      }
      return;
    }

    if (data.stats) {
      const riskDist = data.stats.risk_distribution || {};
      setModelStats(prev => ({
        ...prev,
        total_processed: toNumber(data.stats.total_processed, prev.total_processed),
        fraud_detected: toNumber(data.stats.fraud_detected, prev.fraud_detected),
        preprocessing_errors: toNumber(data.stats.preprocessing_errors, prev.preprocessing_errors),
        dataset_loops: toNumber(data.stats.dataset_loops, prev.dataset_loops),
        avg_processing_time: toNumber(data.stats.avg_processing_time, prev.avg_processing_time),
        throughput_tps: toNumber(data.stats.throughput_tps, prev.throughput_tps),
        fraud_rate: toNumber(data.stats.fraud_rate, prev.fraud_rate),
        success_rate: toNumber(data.stats.success_rate, prev.success_rate),
        risk_distribution: Object.fromEntries(
          Object.entries(riskDist).map(([key, val]) => [key, toNumber(val, 0)])
        ),
        prediction_history: data.stats.prediction_history ?? prev.prediction_history
      }));
      return;
    }

    if (data.message === 'END_OF_DATASET') {
      console.log('🔄 Dataset completed');
      return;
    }

    // Handle fraud detection records
    if (data.is_fraud !== undefined || data.transaction_id) {
      const newRecord: FraudDetectionRecord = {
        transaction_id: data.transaction_id || `TXN_${Date.now()}`,
        transaction_data: data.transaction_data || data,
        reconstruction_error: toNumber(data.reconstruction_error, 0),
        fraud_score: toNumber(data.fraud_score, 0),
        fraud_probability: toNumber(data.fraud_probability, 0),
        decision_threshold: data.decision_threshold,
        threshold: toNumber(data.threshold || data.optimal_threshold, modelInfo.threshold),
        optimal_threshold: toNumber(data.optimal_threshold, 0),
        is_fraud: data.is_fraud === true || data.is_fraud === 1,
        confidence: toNumber(data.confidence, 0),
        risk_level: data.risk_level || 'Low',
        processing_time_ms: toNumber(data.processing_time_ms, 0),
        timestamp: data.timestamp || data.stream_timestamp || new Date().toISOString(),
        ...data
      };

      const updatedRecords = [newRecord, ...recordsRef.current.slice(0, 499)];
      recordsRef.current = updatedRecords;
      setRecords(updatedRecords);

      const errorValue = data.reconstruction_error ?? data.fraud_score ?? 0;
      setErrorHistory(prev => [...prev.slice(-49), toNumber(errorValue, 0)]);

      updateStats(newRecord);
    }
  }, [selectedModel]);

  // Update statistics
  const updateStats = useCallback((record: FraudDetectionRecord) => {
    setModelStats(prev => {
      const newTotal = prev.total_processed + 1;
      const newFraudCount = prev.fraud_detected + (record.is_fraud ? 1 : 0);
      const newFraudRate = (newFraudCount / newTotal) * 100;
      const newSuccessRate = ((newTotal - prev.preprocessing_errors) / newTotal) * 100;
      
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
    const ws = wsRef.current || window.unifiedDetectionWS;
    
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
    localStorage.setItem(`fraud_detection_streaming_${selectedModel}`, 'true');
    sendCommand('start_stream');
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    localStorage.setItem(`fraud_detection_streaming_${selectedModel}`, 'false');
    sendCommand('stop_stream');
  };

  const updateStreamSpeed = (speed: number) => {
    setStreamSpeed(speed);
    sendCommand('set_speed', { speed });
  };

  const resetStats = () => {
    setRecords([]);
    setErrorHistory([]);
    setModelStats({
      total_processed: 0,
      fraud_detected: 0,
      preprocessing_errors: 0,
      dataset_loops: 0,
      avg_processing_time: 0,
      throughput_tps: 0,
      fraud_rate: 0,
      success_rate: 0,
      risk_distribution: {},
      prediction_history: []
    });

    localStorage.removeItem(getDashboardSnapshotKey(selectedModel));
    sendCommand('reset_stats');
  };

  const resolvedAccuracy = (() => {
    const accuracy = modelInfo.performance?.accuracy;
    if (typeof accuracy !== 'number' || !Number.isFinite(accuracy)) {
      return null;
    }
    return accuracy <= 1 ? accuracy * 100 : accuracy;
  })();

  const architectureHeadline = (() => {
    if (selectedModel === 'autoencoder') {
      const encodedLayers = modelInfo.architecture
        .split('-')
        .map((value) => value.trim())
        .filter(Boolean);

      if (encodedLayers.length > 0) {
        return `Input (${modelInfo.input_dim}) → ${encodedLayers.join(' → ')} → Output (${modelInfo.input_dim})`;
      }

      return `Input (${modelInfo.input_dim}) → Output (${modelInfo.input_dim})`;
    }

    if (selectedModel === 'lstm') {
      return modelInfo.architecture || 'Bidirectional LSTM + Attention';
    }

    return `${modelInfo.architecture || 'SNN'} • ${modelInfo.time_steps || 20} time steps`;
  })();

  const architectureDescription = (() => {
    if (selectedModel === 'autoencoder') {
      return 'Encoder-decoder reconstruction model loaded from the saved checkpoint weights.';
    }

    if (selectedModel === 'lstm') {
      return `Sequence-aware classifier with ${modelInfo.hidden_size || 0} hidden units and ${modelInfo.num_layers || 0} recurrent layers.`;
    }

    return `Spiking classifier using ${modelInfo.hidden_size || 0} hidden neurons, ${modelInfo.time_steps || 20} temporal steps, and ${modelInfo.unknown_customer_policy || 'global'} threshold fallback.`;
  })();

  useEffect(() => {
    setNavbarConfig({
      selectedModel,
      availableModels: ['autoencoder', 'lstm', 'snn'],
      accuracyText: resolvedAccuracy !== null ? `${resolvedAccuracy.toFixed(1)}% ACC` : 'METRICS PENDING',
      isStreaming,
      isConnected,
      connectionStatus,
      onSelectModel: handleModelChange,
      onStart: startStreaming,
      onStop: stopStreaming,
      onReset: resetStats,
      theme: {
        headerBackground: currentTheme.bgPrimary,
        panelBackground: currentTheme.bgSecondary,
        borderColor: currentTheme.borderColor,
        textPrimary: currentTheme.textPrimary,
        textMuted: currentTheme.textMuted,
        accent: '#8b5cf6',
        success: '#10b981',
        danger: '#ef4444',
        shadow: isDarkTheme ? '0 16px 40px rgba(2, 6, 23, 0.44)' : '0 16px 36px rgba(15, 23, 42, 0.12)',
      },
    });
  }, [connectionStatus, currentTheme.bgPrimary, currentTheme.bgSecondary, currentTheme.borderColor, currentTheme.textMuted, currentTheme.textPrimary, isConnected, isDarkTheme, isStreaming, resolvedAccuracy, selectedModel, setNavbarConfig]);

  useEffect(() => {
    return () => {
      clearNavbarConfig();
    };
  }, [clearNavbarConfig]);

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

  // Format error value based on model
  const formatErrorValue = (record: FraudDetectionRecord) => {
    if (selectedModel === 'autoencoder' && record.reconstruction_error !== undefined) {
      return record.reconstruction_error.toExponential(3);
    }
    if ((selectedModel === 'lstm' || selectedModel === 'snn') && record.fraud_score !== undefined) {
      return record.fraud_score.toFixed(4);
    }
    return '0.0000';
  };

  // Risk level color
  const getRiskColor = (riskLevel: string) => {
    switch(riskLevel) {
      case 'Low': return 'var(--success)';
      case 'Medium-Low': return 'var(--info)';
      case 'Medium': return 'var(--warning)';
      case 'Medium-High': return 'var(--warning)';
      case 'High': return 'var(--danger)';
      default: return 'var(--text-muted)';
    }
  };

  // Draw error/score chart
  useEffect(() => {
    if (!showChart || !chartRef.current || errorHistory.length === 0) return;

    const drawChart = () => {
      const canvas = chartRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.offsetWidth;
      const height = canvas.offsetHeight;
      if (width === 0 || height === 0) {
        return;
      }

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      ctx.clearRect(0, 0, width, height);

    const padding = 40;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;
    const maxValue = Math.max(...errorHistory, modelInfo.threshold, 0.000001);
    const pointsDenominator = Math.max(errorHistory.length - 1, 1);

    // Draw threshold line
    ctx.beginPath();
    const thresholdY = height - padding - (modelInfo.threshold / maxValue) * chartHeight;
    ctx.moveTo(padding, thresholdY);
    ctx.lineTo(width - padding, thresholdY);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw threshold label
    ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.font = '12px monospace';
    const thresholdLabel = selectedModel === 'autoencoder' 
      ? `Threshold: ${modelInfo.threshold.toExponential(3)}`
      : `Threshold: ${modelInfo.threshold.toFixed(4)}`;
    ctx.fillText(thresholdLabel, width - padding - 140, thresholdY - 10);

    // Draw error line
    if (errorHistory.length > 1) {
      ctx.beginPath();
      errorHistory.forEach((error, index) => {
        const x = padding + (index / pointsDenominator) * chartWidth;
        const y = height - padding - (error / maxValue) * chartHeight;
        
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
    errorHistory.forEach((error, index) => {
      const x = padding + (index / pointsDenominator) * chartWidth;
      const y = height - padding - (error / maxValue) * chartHeight;
      
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      
      const color = error > modelInfo.threshold ? 'var(--danger)' : 
                    error > modelInfo.threshold * 0.7 ? 'var(--warning)' : 'var(--success)';
      ctx.fillStyle = color;
      ctx.fill();
    });
    };

    const frameId = window.requestAnimationFrame(drawChart);
    return () => window.cancelAnimationFrame(frameId);
  }, [errorHistory, modelInfo.threshold, selectedModel, showChart]);

  return (
    <div 
      className="fraud-dashboard"
      style={{
        backgroundColor: currentTheme.bgPrimary,
        color: currentTheme.textPrimary,
        transition: 'all 0.3s ease',
      }}
    >
      {/* Sidebar */}
      <aside className="fraud-sidebar" style={{
        backgroundColor: currentTheme.bgSidebar,
        borderRightColor: currentTheme.borderColor,
      }}>
        <div className="sidebar-section-label">Sections</div>

        <nav className="sidebar-nav">
          <button 
            className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            <BarChart3 className="nav-icon" />
            <span>Dashboard</span>
          </button>

          {currentUser?.role === 'admin' ? (
            <>
              <button className="nav-item" onClick={() => navigate('/user-management')}>
                <Layers className="nav-icon" />
                <span>User</span>
              </button>
              <button className="nav-item" onClick={() => navigate('/reports')}>
                <AlertCircle className="nav-icon" />
                <span>Report</span>
              </button>
              <button className="nav-item" onClick={() => navigate('/system')}>
                <Cpu className="nav-icon" />
                <span>System</span>
              </button>
            </>
          ) : (
            <>
              <button className="nav-item" onClick={() => navigate('/investigations')}>
                <LineChart className="nav-icon" />
                <span>Investigation</span>
              </button>
              <button className={`nav-item ${activeTab === 'system' ? 'active' : ''}`} onClick={() => setActiveTab('system')}>
                <Cpu className="nav-icon" />
                <span>System</span>
              </button>
              <button
                className="nav-item"
                onClick={() => navigate('/batch-upload', { state: { model: selectedModel } })}
              >
                <Upload className="nav-icon" />
                <span>Batch Upload</span>
              </button>
              <button className="nav-item" onClick={() => navigate('/reports')}>
                <AlertCircle className="nav-icon" />
                <span>Reports</span>
              </button>
            </>
          )}
        </nav>

        <div className="sidebar-footer">
          <div className="threshold-display">
            <Thermometer className="threshold-icon" />
            <div>
              <div className="threshold-value">
                {selectedModel === 'autoencoder' 
                  ? modelInfo.threshold.toExponential(3)
                  : modelInfo.threshold.toFixed(4)}
              </div>
              <div className="threshold-label">Detection Threshold</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Status Bar */}
        <div style={{
          padding: '16px 24px',
          borderBottom: `1px solid ${currentTheme.borderColor}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '16px',
          backgroundColor: currentTheme.bgSecondary,
          flexWrap: 'wrap'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '16px',
            padding: '12px 16px',
            borderRadius: '16px',
            border: `1px solid ${currentTheme.borderColor}`,
            background: isDarkTheme
              ? 'linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(15, 23, 42, 0.08))'
              : 'linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(255, 255, 255, 0.92))',
            boxShadow: isDarkTheme
              ? '0 10px 30px rgba(15, 23, 42, 0.2)'
              : '0 10px 24px rgba(148, 163, 184, 0.16)',
            minWidth: '280px',
          }}>
            <div style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: '#3b82f6',
              boxShadow: '0 0 0 6px rgba(59, 130, 246, 0.14)',
              flexShrink: 0,
            }} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', minWidth: '220px', flex: 1 }}>
              <div style={{
                fontSize: '11px',
                fontWeight: 700,
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
                color: currentTheme.textMuted,
              }}>
                Streaming Speed
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '13px', color: currentTheme.textSecondary, whiteSpace: 'nowrap' }}>
                  Speed:
                </span>
                <input
                  type="range"
                  min="0.5"
                  max="10"
                  step="0.5"
                  value={streamSpeed}
                  onChange={(e) => updateStreamSpeed(parseFloat(e.target.value))}
                  style={{
                    width: '100%',
                    maxWidth: '180px',
                    accentColor: '#2563eb',
                    cursor: 'pointer',
                  }}
                />
                <span style={{
                  fontSize: '13px',
                  color: currentTheme.textPrimary,
                  minWidth: '56px',
                  fontWeight: 700,
                  padding: '6px 10px',
                  borderRadius: '999px',
                  backgroundColor: isDarkTheme ? 'rgba(15, 23, 42, 0.26)' : 'rgba(255, 255, 255, 0.8)',
                  border: `1px solid ${currentTheme.borderColor}`,
                  textAlign: 'center',
                }}>
                  {streamSpeed}/sec
                </span>
              </div>
            </div>
          </div>

          <div style={{
            marginLeft: 'auto',
            fontSize: '13px',
            color: currentTheme.textMuted,
            display: 'flex',
            alignItems: 'center',
            gap: '12px'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '10px 14px',
              borderRadius: '999px',
              border: `1px solid ${currentTheme.borderColor}`,
              backgroundColor: currentTheme.bgCard,
              color: currentTheme.textSecondary,
              fontWeight: 600,
            }}>
              <span style={{
                display: 'inline-block',
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: isStreaming ? '#10b981' : currentTheme.textMuted,
                animation: isStreaming ? 'pulse 1.5s infinite' : 'none'
              }}></span>
              {isStreaming ? 'Live Stream' : 'Paused'}
            </div>

          </div>
        </div>

        {/* Content Area */}
        <div style={{ flex: 1, overflow: 'auto', padding: '24px', backgroundColor: currentTheme.bgPrimary }}>
          {/* Model Performance Stats */}
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 2fr) minmax(280px, 1fr)', gap: '16px', marginBottom: '24px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {/* Threshold Graph */}
              <div style={{
                background: isDarkTheme
                  ? 'linear-gradient(145deg, rgba(30, 41, 59, 0.98), rgba(15, 23, 42, 0.96))'
                  : 'linear-gradient(145deg, #ffffff, #f8fbff)',
                padding: '16px',
                borderRadius: '14px',
                border: `1px solid ${currentTheme.borderColor}`,
                color: currentTheme.textPrimary,
                boxShadow: isDarkTheme ? '0 18px 38px rgba(15, 23, 42, 0.28)' : '0 16px 32px rgba(148, 163, 184, 0.16)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '600' }}>
                    {selectedModel === 'autoencoder' ? 'Reconstruction Error' : 'Fraud Score'} Chart
                  </h3>
                  <button 
                    onClick={() => setShowChart(!showChart)}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: currentTheme.textMuted }}
                  >
                    {showChart ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
                {showChart && (
                  <canvas 
                    ref={chartRef} 
                    style={{
                      width: '100%',
                      height: '180px',
                      border: `1px solid ${currentTheme.borderColor}`,
                      borderRadius: '10px',
                      marginBottom: '12px',
                      background: isDarkTheme ? 'rgba(15, 23, 42, 0.45)' : 'rgba(239, 246, 255, 0.7)',
                    }}
                  />
                )}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '12px' }}>
                  <div style={{
                    padding: '10px 12px',
                    borderRadius: '10px',
                    background: isDarkTheme ? 'rgba(59, 130, 246, 0.14)' : 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid rgba(59, 130, 246, 0.24)',
                  }}>
                    <div style={{ color: currentTheme.textMuted, marginBottom: '4px' }}>Current</div>
                    <div style={{ fontFamily: 'monospace', fontWeight: '700', color: '#38bdf8' }}>
                      {records[0] ? formatErrorValue(records[0]) : 'N/A'}
                    </div>
                  </div>
                  <div style={{
                    padding: '10px 12px',
                    borderRadius: '10px',
                    background: isDarkTheme ? 'rgba(239, 68, 68, 0.12)' : 'rgba(254, 226, 226, 0.75)',
                    border: '1px solid rgba(239, 68, 68, 0.24)',
                  }}>
                    <div style={{ color: currentTheme.textMuted, marginBottom: '4px' }}>Threshold</div>
                    <div style={{ fontFamily: 'monospace', fontWeight: '700', color: '#ef4444' }}>
                      {selectedModel === 'autoencoder' ? modelInfo.threshold.toExponential(3) : modelInfo.threshold.toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Alert Summary */}
              <div style={{
                background: isDarkTheme
                  ? 'linear-gradient(145deg, rgba(35, 25, 25, 0.98), rgba(30, 41, 59, 0.96))'
                  : 'linear-gradient(145deg, #ffffff, #fff7f7)',
                padding: '16px',
                borderRadius: '14px',
                border: `1px solid ${currentTheme.borderColor}`,
                boxShadow: isDarkTheme ? '0 16px 34px rgba(15, 23, 42, 0.22)' : '0 14px 28px rgba(248, 113, 113, 0.14)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
                  <AlertCircle size={16} color="#f97316" />
                  <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: currentTheme.textPrimary }}>Alert Summary</h3>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '10px' }}>
                  <div style={{ padding: '12px', backgroundColor: isDarkTheme ? 'rgba(239, 68, 68, 0.16)' : 'rgba(254, 226, 226, 0.95)', borderRadius: '10px', border: '1px solid rgba(239, 68, 68, 0.3)', fontSize: '12px' }}>
                    <div style={{ color: '#f87171', fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '6px' }}>High Risk</div>
                    <div style={{ fontWeight: '800', fontSize: '22px', color: '#ef4444' }}>
                      {Object.entries(modelStats.risk_distribution || {}).find(([k]) => k === 'High')?.[1] || 0}
                    </div>
                  </div>
                  <div style={{ padding: '12px', backgroundColor: isDarkTheme ? 'rgba(245, 158, 11, 0.16)' : 'rgba(254, 243, 199, 0.95)', borderRadius: '10px', border: '1px solid rgba(245, 158, 11, 0.3)', fontSize: '12px' }}>
                    <div style={{ color: '#f59e0b', fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '6px' }}>
                      {getRiskLevels().includes('Medium-High') ? 'Medium-High' : getRiskLevels().includes('Medium') ? 'Medium' : 'Medium'}
                    </div>
                    <div style={{ fontWeight: '800', fontSize: '22px', color: '#f59e0b' }}>
                      {getRiskLevels().includes('Medium-High')
                        ? Object.entries(modelStats.risk_distribution || {}).find(([k]) => k === 'Medium-High')?.[1] || 0
                        : Object.entries(modelStats.risk_distribution || {}).find(([k]) => k === 'Medium')?.[1] || 0}
                    </div>
                  </div>
                  <div style={{ padding: '12px', backgroundColor: isDarkTheme ? 'rgba(16, 185, 129, 0.16)' : 'rgba(220, 252, 231, 0.95)', borderRadius: '10px', border: '1px solid rgba(16, 185, 129, 0.3)', fontSize: '12px' }}>
                    <div style={{ color: '#10b981', fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '6px' }}>Low Risk</div>
                    <div style={{ fontWeight: '800', fontSize: '22px', color: '#10b981' }}>
                      {Object.entries(modelStats.risk_distribution || {}).find(([k]) => k === 'Low')?.[1] || 0}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Detection Rate */}
            <div style={{
              background: isDarkTheme
                ? 'linear-gradient(160deg, rgba(127, 29, 29, 0.92), rgba(30, 41, 59, 0.95))'
                : 'linear-gradient(160deg, #fff1f2, #ffffff)',
              padding: '20px',
              borderRadius: '16px',
              border: '1px solid rgba(239, 68, 68, 0.2)',
              color: currentTheme.textPrimary,
              boxShadow: isDarkTheme ? '0 22px 42px rgba(127, 29, 29, 0.18)' : '0 18px 34px rgba(248, 113, 113, 0.16)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              minHeight: '100%',
            }}>
              <div>
                <div style={{ fontSize: '12px', color: isDarkTheme ? '#fca5a5' : '#b91c1c', marginBottom: '12px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                  Fraud Detection Rate
                </div>
                <div style={{ fontSize: '40px', lineHeight: 1, fontWeight: '800', color: '#ef4444', marginBottom: '14px' }}>
                  {modelStats.fraud_rate.toFixed(2)}%
                </div>
                <div style={{ fontSize: '13px', color: currentTheme.textSecondary, marginBottom: '16px' }}>
                  Share of processed transactions currently flagged as fraud by the active model.
                </div>
              </div>
              <div>
                <div style={{ height: '10px', backgroundColor: isDarkTheme ? 'rgba(255,255,255,0.08)' : 'rgba(248, 113, 113, 0.12)', borderRadius: '999px', overflow: 'hidden', marginBottom: '14px' }}>
                  <div 
                    style={{ 
                      height: '100%', 
                      background: 'linear-gradient(90deg, #fb7185, #ef4444)',
                      width: `${Math.min(modelStats.fraud_rate, 100)}%`
                    }}
                  />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div style={{ padding: '12px', borderRadius: '12px', backgroundColor: isDarkTheme ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.85)', border: `1px solid ${currentTheme.borderColor}` }}>
                    <div style={{ fontSize: '11px', color: currentTheme.textMuted, marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Fraud Count</div>
                    <div style={{ fontSize: '20px', fontWeight: '800', color: currentTheme.textPrimary }}>{modelStats.fraud_detected}</div>
                  </div>
                  <div style={{ padding: '12px', borderRadius: '12px', backgroundColor: isDarkTheme ? 'rgba(255,255,255,0.05)' : 'rgba(255,255,255,0.85)', border: `1px solid ${currentTheme.borderColor}` }}>
                    <div style={{ fontSize: '11px', color: currentTheme.textMuted, marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Processed</div>
                    <div style={{ fontSize: '20px', fontWeight: '800', color: currentTheme.textPrimary }}>{modelStats.total_processed}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Layout: Left (Model Info + Architecture) | Right (System Metrics + Features) */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px' }}>
            {/* LEFT COLUMN */}
            <div>
              {/* Model Architecture */}
              <div style={{
                background: isDarkTheme
                  ? 'linear-gradient(145deg, rgba(36, 48, 77, 0.98), rgba(15, 23, 42, 0.94))'
                  : 'linear-gradient(145deg, #ffffff, #f5f9ff)',
                padding: '18px',
                borderRadius: '16px',
                border: `1px solid ${currentTheme.borderColor}`,
                marginBottom: '16px',
                boxShadow: isDarkTheme ? '0 18px 36px rgba(15, 23, 42, 0.22)' : '0 16px 32px rgba(148, 163, 184, 0.14)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
                  <div style={{
                    width: '34px',
                    height: '34px',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: isDarkTheme ? 'rgba(96, 165, 250, 0.14)' : 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid rgba(59, 130, 246, 0.2)',
                  }}>
                    <Brain size={16} color="#3b82f6" />
                  </div>
                  <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: currentTheme.textPrimary }}>Model Architecture</h3>
                </div>

                <div style={{ fontSize: '12px', color: currentTheme.textPrimary }}>
                  <div style={{
                    marginBottom: '12px',
                    padding: '14px 16px',
                    borderRadius: '14px',
                    background: isDarkTheme ? 'rgba(15, 23, 42, 0.4)' : 'rgba(255, 255, 255, 0.9)',
                    border: `1px solid ${currentTheme.borderColor}`,
                  }}>
                    <div style={{ fontWeight: '700', fontSize: '18px', lineHeight: 1.35 }}>{architectureHeadline}</div>
                    <div style={{ color: currentTheme.textMuted, fontSize: '12px', marginTop: '6px' }}>{modelInfo.architecture || getModelDisplayName(selectedModel)}</div>
                  </div>
                  <div style={{
                    padding: '12px 14px',
                    background: isDarkTheme ? 'rgba(37, 99, 235, 0.08)' : 'rgba(239, 246, 255, 0.95)',
                    borderRadius: '12px',
                    fontSize: '12px',
                    color: currentTheme.textSecondary,
                    border: '1px solid rgba(59, 130, 246, 0.14)',
                    lineHeight: 1.6,
                  }}>
                    {architectureDescription}
                  </div>
                </div>
              </div>

              {/* Model Parameters */}
              <div style={{
                background: isDarkTheme
                  ? 'linear-gradient(145deg, rgba(23, 31, 54, 0.98), rgba(15, 23, 42, 0.94))'
                  : 'linear-gradient(145deg, #ffffff, #f8fbff)',
                padding: '18px',
                borderRadius: '16px',
                border: `1px solid ${currentTheme.borderColor}`,
                boxShadow: isDarkTheme ? '0 18px 36px rgba(15, 23, 42, 0.2)' : '0 16px 30px rgba(148, 163, 184, 0.12)',
              }}>
                <h3 style={{ margin: '0 0 14px 0', fontSize: '14px', fontWeight: '600', color: currentTheme.textPrimary }}>Model Configuration</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '12px', color: currentTheme.textPrimary }}>
                  <div style={{ padding: '12px', borderRadius: '12px', background: isDarkTheme ? 'rgba(59, 130, 246, 0.08)' : 'rgba(239, 246, 255, 0.9)', border: '1px solid rgba(59, 130, 246, 0.14)' }}>
                    <div style={{ color: currentTheme.textMuted, marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.06em', fontSize: '11px' }}>Input Dimension</div>
                    <div style={{ fontWeight: '700', fontSize: '22px' }}>{modelInfo.input_dim}</div>
                  </div>
                  <div style={{ padding: '12px', borderRadius: '12px', background: isDarkTheme ? 'rgba(139, 92, 246, 0.08)' : 'rgba(245, 243, 255, 0.9)', border: '1px solid rgba(139, 92, 246, 0.14)' }}>
                    <div style={{ color: currentTheme.textMuted, marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.06em', fontSize: '11px' }}>Expected Features</div>
                    <div style={{ fontWeight: '700', fontSize: '22px' }}>{modelInfo.expected_features}</div>
                  </div>
                  <div style={{ padding: '12px', borderRadius: '12px', background: isDarkTheme ? 'rgba(239, 68, 68, 0.08)' : 'rgba(254, 242, 242, 0.95)', border: '1px solid rgba(239, 68, 68, 0.14)' }}>
                    <div style={{ color: currentTheme.textMuted, marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.06em', fontSize: '11px' }}>Detection Threshold</div>
                    <div style={{ fontWeight: '700', fontFamily: 'monospace', fontSize: '16px', color: '#ef4444' }}>
                      {selectedModel === 'autoencoder' ? modelInfo.threshold.toExponential(4) : modelInfo.threshold.toFixed(4)}
                    </div>
                  </div>
                  <div style={{ padding: '12px', borderRadius: '12px', background: isDarkTheme ? 'rgba(16, 185, 129, 0.08)' : 'rgba(236, 253, 245, 0.95)', border: '1px solid rgba(16, 185, 129, 0.14)' }}>
                    <div style={{ color: currentTheme.textMuted, marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.06em', fontSize: '11px' }}>Device</div>
                    <div style={{ fontWeight: '700', fontSize: '18px', color: '#10b981' }}>{modelInfo.device.toUpperCase()}</div>
                  </div>
                  {selectedModel !== 'autoencoder' && (
                    <>
                      <div style={{ padding: '12px', borderRadius: '12px', background: isDarkTheme ? 'rgba(234, 179, 8, 0.08)' : 'rgba(254, 249, 195, 0.95)', border: '1px solid rgba(234, 179, 8, 0.14)' }}>
                        <div style={{ color: currentTheme.textMuted, marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.06em', fontSize: '11px' }}>Hidden Size</div>
                        <div style={{ fontWeight: '700', fontSize: '18px' }}>{modelInfo.hidden_size}</div>
                      </div>
                      <div style={{ padding: '12px', borderRadius: '12px', background: isDarkTheme ? 'rgba(14, 165, 233, 0.08)' : 'rgba(240, 249, 255, 0.95)', border: '1px solid rgba(14, 165, 233, 0.14)' }}>
                        <div style={{ color: currentTheme.textMuted, marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.06em', fontSize: '11px' }}>Time Steps</div>
                        <div style={{ fontWeight: '700', fontSize: '18px' }}>{modelInfo.time_steps}</div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* RIGHT COLUMN */}
            <div>
              {/* System Metrics */}
              <div style={{
                background: isDarkTheme
                  ? 'linear-gradient(145deg, rgba(23, 31, 54, 0.98), rgba(15, 23, 42, 0.94))'
                  : 'linear-gradient(145deg, #ffffff, #f7fbff)',
                padding: '18px',
                borderRadius: '16px',
                border: `1px solid ${currentTheme.borderColor}`,
                marginBottom: '16px',
                boxShadow: isDarkTheme ? '0 18px 36px rgba(15, 23, 42, 0.2)' : '0 16px 30px rgba(148, 163, 184, 0.12)',
              }}>
                <h3 style={{ margin: '0 0 14px 0', fontSize: '14px', fontWeight: '600', color: currentTheme.textPrimary }}>System Metrics</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '12px', color: currentTheme.textPrimary }}>
                  <div style={{ padding: '12px', borderRadius: '12px', background: isDarkTheme ? 'rgba(59, 130, 246, 0.08)' : 'rgba(239, 246, 255, 0.95)', border: '1px solid rgba(59, 130, 246, 0.14)' }}>
                    <div style={{ color: currentTheme.textMuted, marginBottom: '6px' }}>Throughput</div>
                    <div style={{ fontSize: '22px', fontWeight: '800', color: '#3b82f6' }}>{modelStats.throughput_tps.toFixed(2)}</div>
                    <div style={{ fontSize: '11px', color: currentTheme.textMuted, marginTop: '2px' }}>TPS</div>
                  </div>
                  <div style={{ padding: '12px', borderRadius: '12px', background: isDarkTheme ? 'rgba(249, 115, 22, 0.08)' : 'rgba(255, 237, 213, 0.95)', border: '1px solid rgba(249, 115, 22, 0.14)' }}>
                    <div style={{ color: currentTheme.textMuted, marginBottom: '6px' }}>Avg Processing</div>
                    <div style={{ fontSize: '22px', fontWeight: '800', color: '#f97316' }}>{modelStats.avg_processing_time.toFixed(2)}</div>
                    <div style={{ fontSize: '11px', color: currentTheme.textMuted, marginTop: '2px' }}>ms</div>
                  </div>
                  <div style={{ padding: '12px', borderRadius: '12px', background: isDarkTheme ? 'rgba(16, 185, 129, 0.08)' : 'rgba(236, 253, 245, 0.95)', border: '1px solid rgba(16, 185, 129, 0.14)' }}>
                    <div style={{ color: currentTheme.textMuted, marginBottom: '6px' }}>Success Rate</div>
                    <div style={{ fontSize: '22px', fontWeight: '800', color: '#10b981' }}>{modelStats.success_rate.toFixed(1)}%</div>
                  </div>
                  <div style={{ padding: '12px', borderRadius: '12px', background: isDarkTheme ? 'rgba(239, 68, 68, 0.08)' : 'rgba(254, 242, 242, 0.95)', border: '1px solid rgba(239, 68, 68, 0.14)' }}>
                    <div style={{ color: currentTheme.textMuted, marginBottom: '6px' }}>Errors</div>
                    <div style={{ fontSize: '22px', fontWeight: '800', color: '#ef4444' }}>{modelStats.preprocessing_errors}</div>
                  </div>
                </div>
              </div>

              {/* Top Features */}
              <div style={{
                background: isDarkTheme
                  ? 'linear-gradient(145deg, rgba(23, 31, 54, 0.98), rgba(15, 23, 42, 0.94))'
                  : 'linear-gradient(145deg, #ffffff, #f7fbff)',
                padding: '18px',
                borderRadius: '16px',
                border: `1px solid ${currentTheme.borderColor}`,
                marginBottom: '16px',
                boxShadow: isDarkTheme ? '0 18px 36px rgba(15, 23, 42, 0.2)' : '0 16px 30px rgba(148, 163, 184, 0.12)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
                  <div style={{
                    width: '34px',
                    height: '34px',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: isDarkTheme ? 'rgba(14, 165, 233, 0.12)' : 'rgba(14, 165, 233, 0.1)',
                    border: '1px solid rgba(14, 165, 233, 0.2)',
                  }}>
                    <BarChart size={16} color="#0ea5e9" />
                  </div>
                  <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: currentTheme.textPrimary }}>Top Features</h3>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '12px', color: currentTheme.textPrimary }}>
                  {modelInfo.feature_names?.slice(0, 5).map((feature, idx) => (
                    <div key={idx} style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      padding: '10px 12px',
                      borderRadius: '12px',
                      background: isDarkTheme ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.88)',
                      border: `1px solid ${currentTheme.borderColor}`,
                    }}>
                      <div style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '10px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        backgroundColor: isDarkTheme ? `hsla(${220 - idx * 15}, 80%, 60%, 0.16)` : `hsla(${220 - idx * 15}, 80%, 50%, 0.12)`,
                        color: `hsl(${220 - idx * 15}, 70%, 52%)`,
                        fontSize: '11px',
                        fontWeight: 800,
                        flexShrink: 0,
                      }}>
                        {idx + 1}
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '12px', fontWeight: 600 }}>
                          {feature}
                        </div>
                      </div>
                      <div style={{ width: '74px', height: '6px', backgroundColor: currentTheme.bgSecondary, borderRadius: '999px', overflow: 'hidden', flexShrink: 0 }}>
                        <div 
                          style={{ 
                            height: '100%', 
                            background: `linear-gradient(90deg, hsl(${220 - idx * 15}, 70%, 58%), hsl(${200 - idx * 12}, 75%, 48%))`,
                            width: `${100 - idx * 15}%`
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          </div>

          {/* Live Detections Table */}
          <div style={{ backgroundColor: currentTheme.bgCard, borderRadius: '8px', border: `1px solid ${currentTheme.borderColor}`, overflow: 'hidden' }}>
            <div style={{ padding: '16px', borderBottom: `1px solid ${currentTheme.borderColor}`, display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
              <h3 style={{ margin: 0, flex: 1, color: currentTheme.textPrimary }}>Live Detections</h3>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <input
                  type="text"
                  placeholder="Search..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: currentTheme.bgSecondary,
                    border: `1px solid ${currentTheme.borderColor}`,
                    borderRadius: '4px',
                    color: currentTheme.textPrimary,
                    fontSize: '13px'
                  }}
                />
                <select
                  value={selectedRisk}
                  onChange={(e) => setSelectedRisk(e.target.value)}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: currentTheme.bgSecondary,
                    border: `1px solid ${currentTheme.borderColor}`,
                    borderRadius: '4px',
                    color: currentTheme.textPrimary,
                    fontSize: '13px'
                  }}
                >
                  <option value="all">All Risks</option>
                  {getRiskLevels().map(level => (
                    <option key={level} value={level}>{level}</option>
                  ))}
                </select>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '13px', color: currentTheme.textPrimary }}>
                  <input
                    type="checkbox"
                    checked={showFraudOnly}
                    onChange={(e) => setShowFraudOnly(e.target.checked)}
                  />
                  Fraud Only
                </label>
              </div>
            </div>
            
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', fontSize: '13px', borderCollapse: 'collapse', color: currentTheme.textPrimary }}>
                <thead style={{ backgroundColor: currentTheme.bgSecondary, borderBottom: `1px solid ${currentTheme.borderColor}` }}>
                  <tr>
                    <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: currentTheme.textPrimary }}>Transaction ID</th>
                    <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: currentTheme.textPrimary }}>
                      {selectedModel === 'autoencoder' ? 'Reconstruction Error' : 'Score'}
                    </th>
                    <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: currentTheme.textPrimary }}>Risk Level</th>
                    <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: currentTheme.textPrimary }}>Status</th>
                    <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: currentTheme.textPrimary }}>Processing Time</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRecords.slice(0, 20).map((record, idx) => (
                    <tr key={`${record.transaction_id}-${idx}`} style={{ borderBottom: `1px solid ${currentTheme.borderColor}` }}>
                      <td style={{ padding: '12px' }}>
                        <div style={{ fontWeight: '600' }}>{record.transaction_id}</div>
                        <div style={{ fontSize: '12px', color: currentTheme.textMuted }}>
                          {record.transaction_data?.category || 'Unknown'}
                        </div>
                      </td>
                      <td style={{ padding: '12px', fontFamily: 'monospace' }}>
                        {formatErrorValue(record)}
                      </td>
                      <td style={{ padding: '12px' }}>
                        <span style={{
                          padding: '4px 8px',
                          borderRadius: '4px',
                          backgroundColor: getRiskColor(record.risk_level),
                          color: 'white',
                          fontSize: '12px',
                          fontWeight: '600'
                        }}>
                          {record.risk_level}
                        </span>
                      </td>
                      <td style={{ padding: '12px' }}>
                        <span style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          color: record.is_fraud ? '#ef4444' : '#10b981'
                        }}>
                          {record.is_fraud ? (
                            <>
                              <AlertTriangle size={14} />
                              <span>Fraud</span>
                            </>
                          ) : (
                            <>
                              <CheckCircle size={14} />
                              <span>Normal</span>
                            </>
                          )}
                        </span>
                      </td>
                      <td style={{ padding: '12px' }}>
                        {record.processing_time_ms.toFixed(2)} ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ padding: '12px', borderTop: '1px solid var(--border-color)', fontSize: '12px', color: 'var(--text-muted)', textAlign: 'right' }}>
              Showing {Math.min(filteredRecords.length, 20)} of {records.length} records (Total: {modelStats.total_processed})
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
};

export default UnifiedModelsDashboard;
