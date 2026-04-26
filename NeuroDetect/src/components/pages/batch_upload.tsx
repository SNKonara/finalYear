import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Upload,
  FileText,
  Shield,
  Brain,
  Activity,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Database,
  RefreshCw,
  BarChart3,
  Search,
  Download as DownloadIcon,
  Cpu,
  Server,
  Zap,
  Target,
  Sliders,
  Thermometer,
  GitBranch,
  Layers,
  Percent,
  ChevronRight,
  Sparkles,
  GanttChart,
  FileUp,
  Trash2,
  EyeOff,
  Maximize2,
  Minimize2,
  Check,
  Info
} from 'lucide-react';
import '../../pages/css/batch_upload.css';
import { useAuth } from '../auth/AuthContext';

interface ModelOption {
  id: 'autoencoder' | 'lstm' | 'snn';
  name: string;
  icon: React.ReactNode;
  description: string;
  features: string[];
  color: string;
}

interface BatchResult {
  transaction_id?: string;
  amount?: number;
  category?: string;
  gender?: string;
  reconstruction_error?: number;
  fraud_score?: number;
  fraud_prediction: number;
  fraud_probability: number;
  risk_category: 'Low' | 'Medium' | 'High';
  processing_time_ms?: number;
  [key: string]: any;
}

interface ModelInfo {
  input_dim: number;
  architecture: string;
  threshold: number;
  device: string;
  expected_features: number;
  feature_names: string[];
  performance?: {
    f1_score?: number;
    fraud_f1?: number;
    fraud_recall?: number;
    roc_auc?: number;
  };
}

const getDashboardRouteForModel = (model: ModelOption['id']): string => {
  if (model === 'lstm') {
    return '/lstmreal';
  }
  if (model === 'snn') {
    return '/snnreal';
  }
  return '/';
};

const BatchProcessing: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { currentUser } = useAuth();

  // Pre-select the model if realtime models page passed one via navigation state
  const stateModel = (location.state as { model?: ModelOption['id'] } | null)?.model;
  // Model selection
  const [selectedModel, setSelectedModel] = useState<ModelOption['id']>(stateModel ?? 'autoencoder');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);

  // File upload
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [filePreview, setFilePreview] = useState<any[]>([]);
  const [fileHeaders, setFileHeaders] = useState<string[]>([]);

  // Processing
  const [isProcessing, setIsProcessing] = useState(false);
  const [processProgress, setProcessProgress] = useState(0);
  const [results, setResults] = useState<BatchResult[]>([]);
  const [processingError, setProcessingError] = useState<string | null>(null);

  // Threshold tuning
  const [threshold, setThreshold] = useState<number>(0.5);
  const [customThreshold, setCustomThreshold] = useState<boolean>(false);
  const [persistThresholdOverride, setPersistThresholdOverride] = useState<boolean>(false);

  // UI state
  const [activeTab, setActiveTab] = useState<'upload' | 'results' | 'tuning'>('upload');
  const [showPreview, setShowPreview] = useState(true);
  const [selectedRiskFilter, setSelectedRiskFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());
  const [displayLimit, setDisplayLimit] = useState<number>(25);

  // File input ref
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (params.get('tuning') === '1') {
      setActiveTab('tuning');
    }
  }, [location.search]);

  // Model options
  const modelOptions: ModelOption[] = [
    {
      id: 'autoencoder',
      name: 'Autoencoder',
      icon: <Layers className="batch-model-icon" />,
      description: 'Use when you need quick baseline anomaly screening on large mixed transaction batches.',
      features: ['Best for broad anomaly spotting', 'Fast first-pass triage', 'No sequence history required'],
      color: '#8b5cf6'
    },
    {
      id: 'lstm',
      name: 'LSTM with Attention',
      icon: <GanttChart className="batch-model-icon" />,
      description: 'Use when temporal behavior matters and you want strong accuracy from transaction sequence patterns.',
      features: ['Best for time-series fraud signals', 'Captures evolving behavior over time', 'Strong for high-volume production runs'],
      color: '#3b82f6'
    },
    {
      id: 'snn',
      name: 'Spiking Neural Network',
      icon: <Sparkles className="batch-model-icon" />,
      description: 'Use when you need customer-personalized detection and sensitivity to individual behavior shifts.',
      features: ['Best for personalized customer risk', 'Adapts to subtle pattern changes', 'Useful for targeted investigation workflows'],
      color: '#10b981'
    }
  ];

  // Load model info when model changes
  useEffect(() => {
    if (!selectedModel) return;

    const loadModelInfo = async () => {
      setIsLoadingModel(true);
      setModelLoadError(null);

      const fallbackModelInfo: Record<string, ModelInfo> = {
        autoencoder: {
          input_dim: 24,
          architecture: '128-64-16',
          threshold: 0.0004117581993341446,
          device: 'cpu',
          expected_features: 24,
          feature_names: ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'hour', 'day_of_week', 'day_of_month', 'month', 'distance', 'log_amt', 'amt_per_pop', 'hour_sin', 'hour_cos', 'cat_food_dining', 'cat_gas_transport', 'cat_grocery_pos', 'cat_home', 'cat_kids_pets', 'cat_other', 'cat_shopping_net', 'cat_shopping_pos', 'gender_M'],
          performance: {
            f1_score: 0.892,
            fraud_f1: 0.856,
            fraud_recall: 0.834,
            roc_auc: 0.945
          }
        },
        lstm: {
          input_dim: 24,
          architecture: 'Bidirectional LSTM with Attention (Hidden: 256, Layers: 2)',
          threshold: 0.86851567029953,
          device: 'cpu',
          expected_features: 24,
          feature_names: ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'hour', 'day_of_week', 'day_of_month', 'month', 'distance', 'log_amt', 'amt_per_pop', 'hour_sin', 'hour_cos', 'cat_food_dining', 'cat_gas_transport', 'cat_grocery_pos', 'cat_home', 'cat_kids_pets', 'cat_other', 'cat_shopping_net', 'cat_shopping_pos', 'gender_M'],
          performance: {
            f1_score: 0.9925,
            fraud_f1: 0.9925,
            fraud_recall: 0.998,
            roc_auc: 0.9992
          }
        },
        snn: {
          input_dim: 24,
          architecture: 'SNN-FC24-64-64-2',
          threshold: 0.5,
          device: 'cpu',
          expected_features: 24,
          feature_names: ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'hour', 'day_of_week', 'day_of_month', 'month', 'distance', 'log_amt', 'amt_per_pop', 'hour_sin', 'hour_cos', 'cat_food_dining', 'cat_gas_transport', 'cat_grocery_pos', 'cat_home', 'cat_kids_pets', 'cat_other', 'cat_shopping_net', 'cat_shopping_pos', 'gender_M'],
          performance: {
            f1_score: 0.0,
            fraud_f1: 0.0,
            fraud_recall: 0.0,
            roc_auc: 0.0
          }
        }
      };

      try {
        const response = await fetch('http://localhost:8000/models');

        if (!response.ok) {
          throw new Error('Failed to fetch model info from backend');
        }

        const modelsData = await response.json();
        const backendModel = modelsData?.[selectedModel];

        if (!backendModel?.loaded) {
          throw new Error(`${selectedModel.toUpperCase()} model is not loaded on backend`);
        }

        const normalizedInfo: ModelInfo = {
          input_dim: backendModel.input_dim ?? backendModel.num_features ?? backendModel.expected_features ?? 0,
          architecture: backendModel.architecture ?? 'N/A',
          threshold: backendModel.threshold ?? 0.5,
          device: backendModel.device ?? 'cpu',
          expected_features: backendModel.expected_features ?? backendModel.num_features ?? 0,
          feature_names: backendModel.feature_names ?? [],
          performance: {
            f1_score: backendModel.performance?.f1_score ?? backendModel.performance?.f1,
            fraud_f1: backendModel.performance?.fraud_f1 ?? backendModel.performance?.f1,
            fraud_recall: backendModel.performance?.fraud_recall ?? backendModel.performance?.recall,
            roc_auc: backendModel.performance?.roc_auc ?? backendModel.performance?.auc,
          }
        };

        setModelInfo(normalizedInfo);
        setThreshold(normalizedInfo.threshold);
      } catch (error) {
        const fallback = fallbackModelInfo[selectedModel];
        setModelInfo(fallback);
        setThreshold(fallback.threshold);
        setModelLoadError('Using fallback model info because backend model metadata could not be loaded.');
      } finally {
        setIsLoadingModel(false);
      }
    };

    loadModelInfo();
  }, [selectedModel]);

  // Handle file selection
  const handleFileSelect = (selectedFile: File | null) => {
    if (!selectedFile) return;

    setFile(selectedFile);
    setUploadProgress(0);
    
    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 10;
      });
    }, 200);

    // Parse CSV preview
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const lines = text.split('\n').slice(0, 6);
      const headers = lines[0].split(',');
      setFileHeaders(headers);
      
      const preview = lines.slice(1, 6).map(line => {
        const values = line.split(',');
        return headers.reduce((obj: any, header, index) => {
          obj[header.trim()] = values[index]?.trim() || '';
          return obj;
        }, {});
      });
      
      setFilePreview(preview);
    };
    reader.readAsText(selectedFile);
  };

  // Handle file drop
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && (droppedFile.type === 'text/csv' || droppedFile.name.endsWith('.csv'))) {
      handleFileSelect(droppedFile);
    }
  };

  // Process file
  const processFile = async (options?: { forceCustomThreshold?: boolean }) => {
    if (!file || !modelInfo) return;

    setIsProcessing(true);
    setProcessingError(null);
    setProcessProgress(0);
    setResults([]);

    const shouldUseCustomThreshold = options?.forceCustomThreshold ?? customThreshold;

    try {
      // Prepare form data
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model_type', selectedModel);
      if (shouldUseCustomThreshold) {
        formData.append('threshold', threshold.toString());
      }

      // Send to backend with timeout handling
      setProcessProgress(20);
      console.log(`Processing with ${selectedModel} model...`);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
      
      const response = await fetch('http://localhost:8000/batch/process', {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Processing failed');
      }

      setProcessProgress(60);
      const data = await response.json();

      if (!data.success) {
        throw new Error('Processing failed');
      }

      console.log('Processing complete:', data.statistics);
      console.log('Total results received:', data.results?.length || data.preview?.length);

      // Transform results for display - using data.results to get ALL transactions
      const processedResults: BatchResult[] = (data.results || data.preview).map((item: any) => ({
        transaction_id: item.trans_num || `TXN${Math.random().toString(36).substr(2, 9)}`,
        amount: item.amt || 0,
        category: item.category || 'N/A',
        gender: item.gender || 'N/A',
        reconstruction_error: selectedModel === 'autoencoder' ? item.fraud_score : undefined,
        fraud_score: item.fraud_score || 0,
        fraud_prediction: item.prediction || 0,
        fraud_probability: item.fraud_score || 0,
        risk_category:
          item.risk_level ||
          ((item.prediction || 0) === 1
            ? 'High'
            : (item.fraud_score || 0) > ((data.statistics?.threshold || threshold) * 0.7)
              ? 'Medium'
              : 'Low'),
        processing_time_ms: 25
      }));

      setResults(processedResults);
      setProcessProgress(100);
      setIsProcessing(false);
      setActiveTab('results');

      // Store batch info for download
      (window as any).currentBatchId = data.batch_id;
      (window as any).batchStats = data.statistics;

      console.log('Batch processing complete:', data.statistics);

    } catch (error: any) {
      console.error('Processing error:', error);
      if (error.name === 'AbortError') {
        setProcessingError('Processing timeout - file may be too large. Try with a smaller file.');
      } else {
        setProcessingError(error.message || 'Failed to process file');
      }
      setIsProcessing(false);
      setProcessProgress(0);
    }
  };

  const persistThreshold = async (): Promise<boolean> => {
    if (!persistThresholdOverride) {
      return true;
    }

    try {
      const response = await fetch('http://localhost:8000/batch/threshold', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_type: selectedModel,
          threshold,
          persist: true,
        }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Failed to persist threshold override');
      }

      return true;
    } catch (error: any) {
      setProcessingError(error.message || 'Failed to persist threshold override');
      return false;
    }
  };

  const applyThresholdAndReprocess = async () => {
    if (isProcessing || !file || !modelInfo) {
      return;
    }

    setCustomThreshold(true);
    const persistedOk = await persistThreshold();
    if (!persistedOk) {
      return;
    }

    await processFile({ forceCustomThreshold: true });
  };

  // Export results
  const exportResults = async (format: 'csv' | 'pdf' = 'csv') => {
    const batchId = (window as any).currentBatchId;
    
    if (!batchId) {
      // Fallback to local export
      const csvContent = [
        Object.keys(results[0]).join(','),
        ...results.map(row => Object.values(row).join(','))
      ].join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `fraud_predictions_${selectedModel}_${new Date().toISOString().slice(0,10)}.csv`;
      a.click();
      return;
    }

    // Download from backend
    try {
      const response = await fetch(`http://localhost:8000/batch/download/${batchId}?format=${format}`);
      if (!response.ok) throw new Error('Download failed');

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${batchId}_report.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download error:', error);
      alert('Failed to download report');
    }
  };

  // Filter results
  const filteredResults = results.filter(result => {
    const matchesSearch = searchTerm === '' || 
      result.transaction_id?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      result.category?.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesRisk = selectedRiskFilter === 'all' || 
      result.risk_category === selectedRiskFilter;

    return matchesSearch && matchesRisk;
  });

  // Paginate results
  const displayedResults = filteredResults.slice(0, displayLimit);
  const hasMoreResults = filteredResults.length > displayLimit;

  // Load more results
  const loadMoreResults = () => {
    setDisplayLimit(prev => prev + 25);
  };

  // Reset display limit when filters change
  useEffect(() => {
    setDisplayLimit(25);
  }, [searchTerm, selectedRiskFilter]);

  // Calculate statistics
  const statistics = {
    total: results.length,
    fraudCount: results.filter(r => r.fraud_prediction === 1).length,
    fraudRate: results.length > 0 ? (results.filter(r => r.fraud_prediction === 1).length / results.length) * 100 : 0,
    highRisk: results.filter(r => r.risk_category === 'High').length,
    mediumRisk: results.filter(r => r.risk_category === 'Medium').length,
    lowRisk: results.filter(r => r.risk_category === 'Low').length,
    totalAmount: results.reduce((sum, r) => sum + (r.amount || 0), 0),
    avgFraudAmount: results.filter(r => r.fraud_prediction === 1).reduce((sum, r) => sum + (r.amount || 0), 0) / 
                   (results.filter(r => r.fraud_prediction === 1).length || 1)
  };

  const thresholdConfig = selectedModel === 'autoencoder'
    ? {
        min: 0,
        max: Math.max(0.01, (modelInfo?.threshold || 0.0005) * 20),
        step: 0.00001,
        recommendations: [0.0002, 0.0005, 0.001],
      }
    : {
        min: 0,
        max: 1,
        step: 0.001,
        recommendations: [0.15, 0.35, 0.55],
      };

  const formatThreshold = (value: number) => (
    selectedModel === 'autoencoder' ? value.toExponential(4) : value.toFixed(4)
  );

  const thresholdSamples = Array.from({ length: 20 }, (_, index) => {
    const ratio = index / 19;
    return thresholdConfig.min + ((thresholdConfig.max - thresholdConfig.min) * ratio);
  });

  const isFlaggedAtThreshold = (score: number, candidate: number) => (
    selectedModel === 'autoencoder' ? score > candidate : score >= candidate
  );

  // Toggle row expansion
  const toggleRow = (index: number) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedRows(newExpanded);
  };

  // Reset all data for new batch
  const resetBatch = () => {
    setFile(null);
    setResults([]);
    setProcessProgress(0);
    setUploadProgress(0);
    setFilePreview([]);
    setFileHeaders([]);
    setSearchTerm('');
    setSelectedRiskFilter('all');
    setExpandedRows(new Set());
    setDisplayLimit(25);
    setActiveTab('upload');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="batch-dashboard">
      {/* Sidebar */}
      <aside className="batch-sidebar">
        <nav className="sidebar-nav">
          <button 
            className="nav-item"
            onClick={() => navigate(getDashboardRouteForModel(selectedModel), { state: { model: selectedModel } })}
          >
            <Brain className="nav-icon" />
            <span>Dashboard</span>
          </button>
          <button 
            className="nav-item"
            onClick={() => navigate(currentUser?.role === 'admin' ? '/reports' : '/investigations')}
          >
            <Search className="nav-icon" />
            <span>{currentUser?.role === 'admin' ? 'Report' : 'Investigation'}</span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <Upload className="nav-icon" />
            <span>Upload & Process</span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => setActiveTab('results')}
            disabled={results.length === 0}
          >
            <BarChart3 className="nav-icon" />
            <span>Results</span>
            {results.length > 0 && <span className="nav-badge">{results.length}</span>}
          </button>
          <button 
            className={`nav-item ${activeTab === 'tuning' ? 'active' : ''}`}
            onClick={() => setActiveTab('tuning')}
            disabled={results.length === 0}
          >
            <Sliders className="nav-icon" />
            <span>Model Tune</span>
          </button>
          <button 
            className="nav-item"
            onClick={() => navigate('/reports')}
          >
            <FileText className="nav-icon" />
            <span>Reports</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className="selected-model">
            <div className="model-indicator" style={{ backgroundColor: modelOptions.find(m => m.id === selectedModel)?.color }}></div>
            <div>
              <div className="model-name">{modelOptions.find(m => m.id === selectedModel)?.name}</div>
              <div className="model-threshold">Threshold: {threshold.toExponential(3)}</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="batch-main">
        {isLoadingModel && (
          <div className="processing-note" style={{ marginBottom: '12px' }}>
            <Info size={14} />
            <span>Loading saved model metadata...</span>
          </div>
        )}

        {modelLoadError && (
          <div className="error-message" style={{ marginBottom: '12px' }}>
            <AlertTriangle size={16} />
            <span>{modelLoadError}</span>
            <button onClick={() => setModelLoadError(null)} className="error-close">
              ×
            </button>
          </div>
        )}

        {/* Model Info Banner */}
        {modelInfo && (
          <div className="model-info-banner">
            <div className="model-info-item">
              <Cpu size={16} />
              <span>Input Dim: {modelInfo.input_dim}</span>
            </div>
            <div className="model-info-item">
              <GitBranch size={16} />
              <span>Architecture: {modelInfo.architecture}</span>
            </div>
            <div className="model-info-item">
              <Thermometer size={16} />
              <span>Threshold: {modelInfo.threshold.toExponential(4)}</span>
            </div>
            <div className="model-info-item">
              <Server size={16} />
              <span>Device: {modelInfo.device}</span>
            </div>
            {modelInfo.performance && (
              <>
                <div className="model-info-item">
                  <Target size={16} />
                  <span>F1: {modelInfo.performance.f1_score?.toFixed(4)}</span>
                </div>
                <div className="model-info-item">
                  <Activity size={16} />
                  <span>AUC: {modelInfo.performance.roc_auc?.toFixed(4)}</span>
                </div>
              </>
            )}
          </div>
        )}

        {/* Content Area */}
        <div className="batch-content-area">
          <div className="batch-page-header">
            <div className="batch-page-header-text">
              <h1>Batch Fraud Detection</h1>
              <p className="batch-subtitle">Process multiple transactions through trained models</p>
            </div>
            <div className="batch-model-selector">
              {modelOptions.map(model => (
                <button
                  key={model.id}
                  className={`batch-model-btn ${selectedModel === model.id ? 'active' : ''}`}
                  onClick={() => setSelectedModel(model.id)}
                  style={{ '--model-color': model.color } as React.CSSProperties}
                >
                  {model.icon}
                  <span>{model.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <div className="upload-container">
              {/* Model Selection Cards */}
              <div className="model-cards">
                {modelOptions.map(model => (
                  <div
                    key={model.id}
                    className={`model-card ${selectedModel === model.id ? 'selected' : ''}`}
                    onClick={() => setSelectedModel(model.id)}
                    style={{ '--card-color': model.color } as React.CSSProperties}
                  >
                    <div className="card-header">
                      {model.icon}
                      <h3>{model.name}</h3>
                      {selectedModel === model.id && (
                        <div className="selected-badge">
                          <Check size={12} />
                        </div>
                      )}
                    </div>
                    <p className="model-description">{model.description}</p>
                    <ul className="model-features">
                      {model.features.map((feature, idx) => (
                        <li key={idx}>{feature}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>

              {/* File Upload Area */}
              <div className="upload-area">
                <div className="upload-header">
                  <h2>Upload Transaction File</h2>
                  <span className="file-format">CSV only</span>
                </div>

                <div
                  className={`drop-zone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    accept=".csv"
                    onChange={(e) => handleFileSelect(e.target.files?.[0] || null)}
                    style={{ display: 'none' }}
                  />
                  
                  {!file ? (
                    <>
                      <FileUp className="drop-icon" />
                      <p className="drop-text">Drag & drop your CSV file here</p>
                      <p className="drop-subtext">or click to browse</p>
                    </>
                  ) : (
                    <div className="file-info">
                      <FileText className="file-icon" />
                      <div className="file-details">
                        <div className="file-name">{file.name}</div>
                        <div className="file-size">{(file.size / 1024).toFixed(2)} KB</div>
                      </div>
                      <button
                        className="file-remove"
                        onClick={(e) => {
                          e.stopPropagation();
                          setFile(null);
                          setFilePreview([]);
                          setUploadProgress(0);
                        }}
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  )}
                </div>

                {/* Upload Progress */}
                {file && uploadProgress < 100 && (
                  <div className="upload-progress">
                    <div className="progress-bar">
                      <div className="progress-fill" style={{ width: `${uploadProgress}%` }}></div>
                    </div>
                    <span className="progress-text">{uploadProgress}% uploaded</span>
                  </div>
                )}

                {/* File Preview */}
                {file && filePreview.length > 0 && showPreview && (
                  <div className="file-preview">
                    <div className="preview-header">
                      <h3>File Preview</h3>
                      <button className="preview-toggle" onClick={() => setShowPreview(false)}>
                        <EyeOff size={14} />
                      </button>
                    </div>
                    <div className="preview-table-container">
                      <table className="preview-table">
                        <thead>
                          <tr>
                            {fileHeaders.slice(0, 6).map((header, idx) => (
                              <th key={idx}>{header}</th>
                            ))}
                            {fileHeaders.length > 6 && <th>...</th>}
                          </tr>
                        </thead>
                        <tbody>
                          {filePreview.map((row, rowIdx) => (
                            <tr key={rowIdx}>
                              {fileHeaders.slice(0, 6).map((header, colIdx) => (
                                <td key={colIdx}>{row[header] || ''}</td>
                              ))}
                              {fileHeaders.length > 6 && <td>...</td>}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="preview-footer">
                      <span>Showing first 5 rows of {fileHeaders.length} columns</span>
                    </div>
                  </div>
                )}

                {/* Process Button */}
                <button
                  className={`process-btn ${isProcessing ? 'processing' : ''}`}
                  onClick={() => void processFile()}
                  disabled={!file || !modelInfo || isProcessing}
                >
                  {isProcessing ? (
                    <>
                      <RefreshCw className="spin" />
                      <span>Processing with {selectedModel.toUpperCase()}... {processProgress}%</span>
                    </>
                  ) : (
                    <>
                      <Zap />
                      <span>Process File</span>
                    </>
                  )}
                </button>

                {/* Processing Status Message */}
                {isProcessing && selectedModel === 'lstm' && (
                  <div className="processing-note">
                    <Info size={14} />
                    <span>LSTM processing may take longer for large files...</span>
                  </div>
                )}

                {/* Error Display */}
                {processingError && (
                  <div className="error-message">
                    <AlertTriangle size={16} />
                    <span>{processingError}</span>
                    <button onClick={() => setProcessingError(null)} className="error-close">
                      ×
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && results.length > 0 && (
            <div className="results-container">
              {/* Statistics Cards */}
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-icon total">
                    <Database />
                  </div>
                  <div className="stat-content">
                    <div className="stat-value">{statistics.total}</div>
                    <div className="stat-label">Total Transactions</div>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon fraud">
                    <AlertTriangle />
                  </div>
                  <div className="stat-content">
                    <div className="stat-value">{statistics.fraudCount}</div>
                    <div className="stat-label">Fraud Detected</div>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon rate">
                    <Percent />
                  </div>
                  <div className="stat-content">
                    <div className="stat-value">{statistics.fraudRate.toFixed(2)}%</div>
                    <div className="stat-label">Fraud Rate</div>
                  </div>
                </div>

                <div className="stat-card">
                  <div className="stat-icon amount">
                    <TrendingUp />
                  </div>
                  <div className="stat-content">
                    <div className="stat-value">${statistics.totalAmount.toFixed(2)}</div>
                    <div className="stat-label">Total Amount</div>
                  </div>
                </div>
              </div>

              {/* Risk Distribution */}
              <div className="risk-distribution-card">
                <h3>Risk Distribution</h3>
                <div className="risk-bars">
                  <div className="risk-bar-item">
                    <span className="risk-label">Low Risk</span>
                    <div className="bar-container">
                      <div 
                        className="bar-fill low"
                        style={{ width: `${(statistics.lowRisk / statistics.total) * 100}%` }}
                      ></div>
                    </div>
                    <span className="risk-count">{statistics.lowRisk}</span>
                  </div>
                  <div className="risk-bar-item">
                    <span className="risk-label">Medium Risk</span>
                    <div className="bar-container">
                      <div 
                        className="bar-fill medium"
                        style={{ width: `${(statistics.mediumRisk / statistics.total) * 100}%` }}
                      ></div>
                    </div>
                    <span className="risk-count">{statistics.mediumRisk}</span>
                  </div>
                  <div className="risk-bar-item">
                    <span className="risk-label">High Risk</span>
                    <div className="bar-container">
                      <div 
                        className="bar-fill high"
                        style={{ width: `${(statistics.highRisk / statistics.total) * 100}%` }}
                      ></div>
                    </div>
                    <span className="risk-count">{statistics.highRisk}</span>
                  </div>
                </div>
              </div>

              {/* Filters and Export */}
              <div className="results-header">
                <div className="search-filter">
                  <div className="search-box">
                    <Search className="search-icon" />
                    <input
                      type="text"
                      placeholder="Search transactions..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                    />
                  </div>
                  <select
                    className="risk-filter"
                    value={selectedRiskFilter}
                    onChange={(e) => setSelectedRiskFilter(e.target.value)}
                  >
                    <option value="all">All Risks</option>
                    <option value="Low">Low Risk</option>
                    <option value="Medium">Medium Risk</option>
                    <option value="High">High Risk</option>
                  </select>
                </div>
                <div className="export-buttons">
                  <button className="export-btn" onClick={() => exportResults('csv')}>
                    <DownloadIcon size={16} />
                    Export CSV
                  </button>
                  <button className="export-btn pdf" onClick={() => exportResults('pdf')}>
                    <FileText size={16} />
                    Download PDF Report
                  </button>
                  <button className="export-btn reset" onClick={resetBatch} title="Start New Batch">
                    <RefreshCw size={16} />
                    New Batch
                  </button>
                </div>
              </div>

              {/* Results Table */}
              <div className="results-table-container">
                <table className="results-table">
                  <thead>
                    <tr>
                      <th></th>
                      <th>Transaction ID</th>
                      <th>Amount</th>
                      <th>Category</th>
                      <th>Fraud Score</th>
                      <th>Risk Level</th>
                      <th>Prediction</th>
                      <th>Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayedResults.map((result, idx) => (
                      <React.Fragment key={idx}>
                        <tr className={`result-row ${result.risk_category.toLowerCase()}`}>
                          <td>
                            <button className="expand-btn" onClick={() => toggleRow(idx)}>
                              {expandedRows.has(idx) ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                            </button>
                          </td>
                          <td className="transaction-id">{result.transaction_id}</td>
                          <td className="amount">${result.amount?.toFixed(2)}</td>
                          <td>{result.category}</td>
                          <td>
                            <div className="score-cell">
                              <span className="score-value">{(result.fraud_probability * 100).toFixed(1)}%</span>
                              <div className="score-bar">
                                <div 
                                  className="score-fill"
                                  style={{ 
                                    width: `${result.fraud_probability * 100}%`,
                                    backgroundColor: result.fraud_prediction === 1 ? 'var(--danger)' : 'var(--success)'
                                  }}
                                ></div>
                              </div>
                            </div>
                          </td>
                          <td>
                            <span className={`risk-badge ${result.risk_category.toLowerCase()}`}>
                              {result.risk_category}
                            </span>
                          </td>
                          <td>
                            {result.fraud_prediction === 1 ? (
                              <span className="prediction fraud">
                                <AlertTriangle size={12} />
                                Fraud
                              </span>
                            ) : (
                              <span className="prediction normal">
                                <CheckCircle size={12} />
                                Normal
                              </span>
                            )}
                          </td>
                          <td>
                            <button className="details-btn" onClick={() => toggleRow(idx)}>
                              View
                              <ChevronRight size={12} />
                            </button>
                          </td>
                        </tr>
                        {expandedRows.has(idx) && (
                          <tr className="expanded-row">
                            <td colSpan={8}>
                              <div className="expanded-content">
                                <div className="expanded-section">
                                  <h4>Transaction Details</h4>
                                  <div className="details-grid">
                                    {Object.entries(result).map(([key, value]) => (
                                      <div key={key} className="detail-item">
                                        <span className="detail-label">{key}:</span>
                                        <span className="detail-value">
                                          {typeof value === 'number' ? 
                                            (key.includes('amount') ? `$${value.toFixed(2)}` : 
                                             key.includes('score') || key.includes('probability') ? `${(value * 100).toFixed(2)}%` :
                                             value.toExponential?.(4) || value) : 
                                            String(value)}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* See More Button */}
              {hasMoreResults && (
                <div className="see-more-container">
                  <button className="see-more-btn" onClick={loadMoreResults}>
                    <ChevronRight size={16} />
                    Show More Results ({displayedResults.length} of {filteredResults.length})
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Threshold Tuning Tab */}
          {activeTab === 'tuning' && results.length > 0 && (
            <div className="tuning-container">
              <div className="tuning-header">
                <h2>Threshold Tuning</h2>
                <p className="tuning-description">
                  Adjust the detection threshold to balance between precision and recall
                </p>
              </div>

              <label className="checkbox-label" style={{ marginBottom: '16px', display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="checkbox"
                  className="checkbox-input"
                  checked={customThreshold}
                  onChange={(event) => setCustomThreshold(event.target.checked)}
                />
                <span className="checkbox-text">Use custom threshold for the next batch run</span>
              </label>

              <div className="tuning-controls">
                <div className="threshold-control">
                  <label>
                    <span>Detection Threshold</span>
                    <span className="threshold-value">{formatThreshold(threshold)}</span>
                  </label>
                  <input
                    type="range"
                    min={thresholdConfig.min}
                    max={thresholdConfig.max}
                    step={thresholdConfig.step}
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="threshold-slider"
                  />
                  <div className="threshold-info">
                    <span>Lower = More Sensitive</span>
                    <span>Higher = More Specific</span>
                  </div>
                </div>

                <div className="threshold-stats">
                  <div className="stat-box">
                    <span className="stat-label">Fraud Detected</span>
                    <span className="stat-value">
                      {results.filter(r => isFlaggedAtThreshold(r.fraud_probability, threshold)).length}
                    </span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Fraud Rate</span>
                    <span className="stat-value">
                      {((results.filter(r => isFlaggedAtThreshold(r.fraud_probability, threshold)).length / results.length) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Avg Score</span>
                    <span className="stat-value">
                      {(results.reduce((sum, r) => sum + r.fraud_probability, 0) / results.length * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Threshold Impact Chart */}
              <div className="threshold-chart">
                <h3>Threshold Impact Analysis</h3>
                <div className="chart-container">
                  {thresholdSamples.map((thresh, i) => {
                    const fraudCount = results.filter(r => isFlaggedAtThreshold(r.fraud_probability, thresh)).length;
                    const height = (fraudCount / results.length) * 100;
                    return (
                      <div key={i} className="chart-bar-container">
                        <div 
                          className="chart-bar"
                          style={{ 
                            height: `${height}%`,
                            backgroundColor: thresh <= threshold ? 'var(--primary)' : 'var(--bg-tertiary)'
                          }}
                        ></div>
                        <span className="chart-label">{formatThreshold(thresh)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Threshold Recommendations */}
              <div className="threshold-recommendations">
                <h3>Recommendations</h3>
                <div className="rec-cards">
                  <div className="rec-card">
                    <div className="rec-icon conservative">
                      <Shield />
                    </div>
                    <div className="rec-content">
                      <h4>Conservative ({formatThreshold(thresholdConfig.recommendations[0])})</h4>
                      <p>Minimize false positives, catch only high-confidence fraud</p>
                      <button 
                        className="apply-btn"
                        onClick={() => setThreshold(thresholdConfig.recommendations[0])}
                      >
                        Apply
                      </button>
                    </div>
                  </div>
                  <div className="rec-card">
                    <div className="rec-icon balanced">
                      <Target />
                    </div>
                    <div className="rec-content">
                      <h4>Balanced ({formatThreshold(thresholdConfig.recommendations[1])})</h4>
                      <p>Balance between precision and recall</p>
                      <button 
                        className="apply-btn"
                        onClick={() => setThreshold(thresholdConfig.recommendations[1])}
                      >
                        Apply
                      </button>
                    </div>
                  </div>
                  <div className="rec-card">
                    <div className="rec-icon aggressive">
                      <Activity />
                    </div>
                    <div className="rec-content">
                      <h4>Aggressive ({formatThreshold(thresholdConfig.recommendations[2])})</h4>
                      <p>Maximize fraud detection, accept more false positives</p>
                      <button 
                        className="apply-btn"
                        onClick={() => setThreshold(thresholdConfig.recommendations[2])}
                      >
                        Apply
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              <label className="checkbox-label" style={{ marginBottom: '16px', display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="checkbox"
                  className="checkbox-input"
                  checked={persistThresholdOverride}
                  onChange={(event) => setPersistThresholdOverride(event.target.checked)}
                />
                <span className="checkbox-text">Persist this threshold as default for future runs</span>
              </label>

              {/* Apply Threshold Button */}
              <button
                className="apply-threshold-btn"
                onClick={applyThresholdAndReprocess}
                disabled={isProcessing || !file}
              >
                {isProcessing ? 'Reprocessing...' : 'Apply New Threshold & Reprocess'}
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default BatchProcessing;
