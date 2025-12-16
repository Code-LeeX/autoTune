import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  FormControlLabel,
  Switch,
  Grid,
  Alert,
  CircularProgress,
  Divider,
  Chip,
  Card,
  CardContent,
  Tooltip,
  LinearProgress
} from '@mui/material';
import TuneIcon from '@mui/icons-material/Tune';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import RefreshIcon from '@mui/icons-material/Refresh';
import MusicNoteIcon from '@mui/icons-material/MusicNote';
import GraphicEqIcon from '@mui/icons-material/GraphicEq';
import SettingsIcon from '@mui/icons-material/Settings';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import InfoIcon from '@mui/icons-material/Info';

import {
  AudioAPI,
  PitchAnalysisData,
  CorrectionResult,
  AvailableScales,
  ProcessingStatus,
  AudioProcessingWebSocket
} from '../../api/audioApi';

interface ParameterPanelProps {
  sessionId?: string;
  analysisResult?: PitchAnalysisData;
  onCorrectionComplete?: (sessionId: string, correctionResult: CorrectionResult) => void;
  onError?: (error: string) => void;
  onReset?: () => void;
  disabled?: boolean;
}

interface CorrectionParams {
  key: string;
  scale_type: string;
  correction_strength: number;
  preserve_vibrato: boolean;
  preserve_formants: boolean;
  smoothing_factor: number;
}

interface CorrectionState {
  status: 'idle' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
  result?: CorrectionResult;
}

export const ParameterPanel: React.FC<ParameterPanelProps> = ({
  sessionId,
  analysisResult,
  onCorrectionComplete,
  onError,
  onReset,
  disabled = false
}) => {
  const [availableScales, setAvailableScales] = useState<AvailableScales | null>(null);
  const [scalesLoading, setScalesLoading] = useState(false);
  const [correctionState, setCorrectionState] = useState<CorrectionState>({
    status: 'idle',
    progress: 0
  });
  const [websocket, setWebsocket] = useState<AudioProcessingWebSocket | null>(null);

  // Key detection state
  const [keyDetection, setKeyDetection] = useState<{
    detectedKey: string | null;
    confidence: number;
    isAutoDetected: boolean;
  }>({
    detectedKey: null,
    confidence: 0,
    isAutoDetected: false
  });

  // Correction parameters with sensible defaults
  const [params, setParams] = useState<CorrectionParams>({
    key: 'C',
    scale_type: 'major',
    correction_strength: 0.8,
    preserve_vibrato: true,
    preserve_formants: true,
    smoothing_factor: 0.1
  });

  const loadAvailableScales = useCallback(async () => {
    try {
      setScalesLoading(true);
      const scales = await AudioAPI.getAvailableScales();
      setAvailableScales(scales);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '加载音阶失败';
      console.error('Failed to load available scales:', error);
      onError?.(errorMessage);
    } finally {
      setScalesLoading(false);
    }
  }, [onError]);

  // Load available scales on mount
  useEffect(() => {
    loadAvailableScales();
  }, [loadAvailableScales]);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (websocket) {
        websocket.disconnect();
      }
    };
  }, [websocket]);

  // Improved key detection with confidence calculation
  const detectMostLikelyKey = useCallback((data: PitchAnalysisData): { key: string; confidence: number } => {
    if (!data.frequency || data.frequency.length === 0) {
      return { key: 'C', confidence: 0 };
    }

    // Convert frequencies to note classes (0=C, 1=C#, 2=D, etc.)
    const noteClasses = data.frequency
      .filter((freq, i) => data.voiced_mask[i] && freq > 0)
      .map(freq => Math.round(12 * Math.log2(freq / 440)) % 12)
      .map(note => (note + 9) % 12); // Convert to C=0 system

    if (noteClasses.length === 0) {
      return { key: 'C', confidence: 0 };
    }

    // Count note class occurrences
    const counts = new Array(12).fill(0);
    noteClasses.forEach(note => counts[note]++);

    // Find most common notes
    const maxCount = Math.max(...counts);
    const secondMaxCount = Math.max(...counts.filter(c => c < maxCount));
    const likelyTonic = counts.indexOf(maxCount);

    // Calculate confidence based on how dominant the top note is
    const totalNotes = noteClasses.length;
    const dominance = maxCount / totalNotes;
    const separation = secondMaxCount > 0 ? (maxCount - secondMaxCount) / totalNotes : dominance;
    const confidence = Math.min(0.95, dominance * 0.7 + separation * 0.3);

    // Convert back to note names
    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    return {
      key: noteNames[likelyTonic],
      confidence: confidence
    };
  }, []);

  // Auto-detect key from analysis result
  useEffect(() => {
    if (analysisResult && availableScales) {
      const detection = detectMostLikelyKey(analysisResult);
      if (detection.key && availableScales.keys.includes(detection.key)) {
        setKeyDetection({
          detectedKey: detection.key,
          confidence: detection.confidence,
          isAutoDetected: true
        });
        setParams(prev => ({ ...prev, key: detection.key }));
      }
    }
  }, [analysisResult, availableScales, detectMostLikelyKey]);

  const handleParameterChange = (param: keyof CorrectionParams, value: any) => {
    setParams(prev => ({ ...prev, [param]: value }));

    // If user manually changes the key, mark as not auto-detected
    if (param === 'key') {
      setKeyDetection(prev => ({ ...prev, isAutoDetected: false }));
    }
  };

  // Manual key re-detection
  const handleRedetectKey = () => {
    if (analysisResult && availableScales) {
      const detection = detectMostLikelyKey(analysisResult);
      if (detection.key && availableScales.keys.includes(detection.key)) {
        setKeyDetection({
          detectedKey: detection.key,
          confidence: detection.confidence,
          isAutoDetected: true
        });
        setParams(prev => ({ ...prev, key: detection.key }));
      }
    }
  };

  const startCorrection = async () => {
    if (!sessionId) {
      onError?.('没有可用的会话进行修正');
      return;
    }

    try {
      setCorrectionState({
        status: 'processing',
        progress: 0
      });

      // Setup WebSocket for real-time updates
      const ws = new AudioProcessingWebSocket();
      setWebsocket(ws);

      ws.onStatusUpdate((status: ProcessingStatus) => {
        setCorrectionState(prev => ({
          ...prev,
          progress: status.progress * 100
        }));

        if (status.status === 'corrected') {
          setCorrectionState({
            status: 'completed',
            progress: 100,
            result: status.data
          });

          // Notify parent component
          if (onCorrectionComplete && status.data) {
            onCorrectionComplete(sessionId, status.data);
          }

          // Disconnect WebSocket
          ws.disconnect();
          setWebsocket(null);
        }
      });

      ws.onError((error) => {
        setCorrectionState({
          status: 'error',
          progress: 0,
          error
        });
        onError?.(error);
        ws.disconnect();
        setWebsocket(null);
      });

      // Connect WebSocket
      await ws.connect(sessionId);

      // Start correction process
      await AudioAPI.correctPitch(sessionId, params);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '修正失败';
      setCorrectionState({
        status: 'error',
        progress: 0,
        error: errorMessage
      });
      onError?.(errorMessage);

      if (websocket) {
        websocket.disconnect();
        setWebsocket(null);
      }
    }
  };

  const handleReset = () => {
    setCorrectionState({
      status: 'idle',
      progress: 0
    });
    setParams({
      key: 'C',
      scale_type: 'major',
      correction_strength: 0.8,
      preserve_vibrato: true,
      preserve_formants: true,
      smoothing_factor: 0.1
    });
    setKeyDetection({
      detectedKey: null,
      confidence: 0,
      isAutoDetected: false
    });
    onReset?.();
  };

  const canStartCorrection = sessionId && analysisResult && !disabled && correctionState.status === 'idle';
  const isProcessing = correctionState.status === 'processing';

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
        <TuneIcon sx={{ color: 'primary.main' }} />
        <Typography variant="h6">
          音调修正参数
        </Typography>
      </Box>

      {/* Error Display */}
      {correctionState.error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {correctionState.error}
        </Alert>
      )}

      {/* Analysis Summary */}
      {analysisResult && (
        <Card variant="outlined" sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <GraphicEqIcon fontSize="small" />
              分析摘要
            </Typography>
            <Grid container spacing={1}>
              <Grid item>
                <Chip
                  size="small"
                  label={`${(analysisResult.stats.voicing_ratio * 100).toFixed(1)}% 有声`}
                  color="success"
                  variant="outlined"
                />
              </Grid>
              <Grid item>
                <Chip
                  size="small"
                  label={`${analysisResult.stats.pitch_range.min.toFixed(0)}-${analysisResult.stats.pitch_range.max.toFixed(0)}Hz`}
                  color="info"
                  variant="outlined"
                />
              </Grid>
              {analysisResult.stats.vibrato && (
                <Grid item>
                  <Chip
                    size="small"
                    label={`${(analysisResult.stats.vibrato.vibrato_ratio * 100).toFixed(1)}% 颤音`}
                    color="secondary"
                    variant="outlined"
                  />
                </Grid>
              )}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Musical Parameters */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <MusicNoteIcon fontSize="small" />
          音阶设置
        </Typography>

        <Grid container spacing={2}>
          <Grid item xs={6}>
            <FormControl fullWidth disabled={disabled || isProcessing || scalesLoading}>
              <InputLabel>调性</InputLabel>
              <Select
                value={params.key}
                label="调性"
                onChange={(e) => handleParameterChange('key', e.target.value)}
              >
                {availableScales?.keys.map((key) => (
                  <MenuItem key={key} value={key}>{key}</MenuItem>
                )) || []}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6}>
            <FormControl fullWidth disabled={disabled || isProcessing || scalesLoading}>
              <InputLabel>音阶类型</InputLabel>
              <Select
                value={params.scale_type}
                label="音阶类型"
                onChange={(e) => handleParameterChange('scale_type', e.target.value)}
              >
                {availableScales?.scale_types.map((scale) => (
                  <MenuItem key={scale} value={scale}>
                    {scale.charAt(0).toUpperCase() + scale.slice(1)}
                  </MenuItem>
                )) || []}
              </Select>
            </FormControl>
          </Grid>
        </Grid>

        {/* Key Detection Info */}
        {keyDetection.detectedKey && (
          <Box sx={{ mt: 2, p: 2, backgroundColor: 'action.hover', borderRadius: 1 }}>
            <Grid container alignItems="center" spacing={1}>
              <Grid item>
                <AutoFixHighIcon sx={{ color: keyDetection.isAutoDetected ? 'success.main' : 'text.secondary', fontSize: 20 }} />
              </Grid>
              <Grid item xs>
                <Typography variant="body2" color="text.secondary">
                  {keyDetection.isAutoDetected ? (
                    <>
                      自动检测到调性: <strong>{keyDetection.detectedKey}</strong>
                      <Chip
                        size="small"
                        label={`置信度: ${(keyDetection.confidence * 100).toFixed(0)}%`}
                        color={keyDetection.confidence > 0.7 ? 'success' : keyDetection.confidence > 0.4 ? 'warning' : 'error'}
                        sx={{ ml: 1 }}
                      />
                    </>
                  ) : (
                    <>已手动设置调性，自动检测到: <strong>{keyDetection.detectedKey}</strong></>
                  )}
                </Typography>
              </Grid>
              {analysisResult && (
                <Grid item>
                  <Tooltip title="重新自动检测调性">
                    <Button
                      size="small"
                      variant="outlined"
                      startIcon={<RefreshIcon />}
                      onClick={handleRedetectKey}
                      disabled={disabled || isProcessing}
                    >
                      重新检测
                    </Button>
                  </Tooltip>
                </Grid>
              )}
            </Grid>

            {keyDetection.confidence < 0.4 && (
              <Box sx={{ mt: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                <InfoIcon sx={{ color: 'warning.main', fontSize: 16 }} />
                <Typography variant="caption" color="warning.main">
                  检测置信度较低，建议手动选择或确认调性
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* Processing Parameters */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SettingsIcon fontSize="small" />
          处理设置
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Typography gutterBottom>
              修正强度: {(params.correction_strength * 100).toFixed(0)}%
            </Typography>
            <Tooltip title="较高的值会应用更强的音调修正。较低的值保留更多自然变化。">
              <Slider
                value={params.correction_strength}
                onChange={(_, value) => handleParameterChange('correction_strength', value)}
                min={0.0}
                max={1.0}
                step={0.05}
                disabled={disabled || isProcessing}
                marks={[
                  { value: 0.0, label: '0%' },
                  { value: 0.5, label: '50%' },
                  { value: 1.0, label: '100%' }
                ]}
              />
            </Tooltip>
          </Grid>
          <Grid item xs={12}>
            <Typography gutterBottom>
              平滑因子: {(params.smoothing_factor * 100).toFixed(0)}%
            </Typography>
            <Tooltip title="控制音调修正的时间平滑。较高的值产生更平滑的过渡。">
              <Slider
                value={params.smoothing_factor}
                onChange={(_, value) => handleParameterChange('smoothing_factor', value)}
                min={0.0}
                max={0.5}
                step={0.01}
                disabled={disabled || isProcessing}
                marks={[
                  { value: 0.0, label: '0%' },
                  { value: 0.25, label: '25%' },
                  { value: 0.5, label: '50%' }
                ]}
              />
            </Tooltip>
          </Grid>

          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={params.preserve_vibrato}
                  onChange={(e) => handleParameterChange('preserve_vibrato', e.target.checked)}
                  disabled={disabled || isProcessing}
                />
              }
              label="保持颤音"
            />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
              在修正过程中保持自然的颤音振荡
            </Typography>
          </Grid>

          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={params.preserve_formants}
                  onChange={(e) => handleParameterChange('preserve_formants', e.target.checked)}
                  disabled={disabled || isProcessing}
                />
              }
              label="保持共振峰"
            />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
              通过保留共振峰频率来维持声音特性
            </Typography>
          </Grid>
        </Grid>
      </Box>

      {/* Progress Display */}
      {isProcessing && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            处理中: {correctionState.progress.toFixed(0)}%
          </Typography>
          <LinearProgress
            variant="determinate"
            value={correctionState.progress}
            sx={{ borderRadius: 1 }}
          />
        </Box>
      )}

      {/* Correction Results */}
      {correctionState.status === 'completed' && correctionState.result && (
        <Card variant="outlined" sx={{ mb: 3, backgroundColor: 'success.dark', color: 'success.contrastText' }}>
          <CardContent>
            <Typography variant="subtitle2" gutterBottom>
              ✅ 修正结果
            </Typography>
            <Grid container spacing={1}>
              <Grid item>
                <Chip
                  size="small"
                  label={`${correctionState.result.correction_stats.frames_corrected} 帧已修正`}
                  color="success"
                />
              </Grid>
              <Grid item>
                <Chip
                  size="small"
                  label={`${(correctionState.result.correction_stats.correction_ratio * 100).toFixed(1)}% 修正比例`}
                  color="info"
                />
              </Grid>
              {correctionState.result.correction_stats.pitch_accuracy_improvement_cents && (
                <Grid item>
                  <Chip
                    size="small"
                    label={`${correctionState.result.correction_stats.pitch_accuracy_improvement_cents.toFixed(1)} 音分改善`}
                    color="secondary"
                  />
                </Grid>
              )}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Action Buttons */}
      <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
        <Button
          variant="contained"
          startIcon={isProcessing ? <CircularProgress size={16} /> : <PlayArrowIcon />}
          onClick={startCorrection}
          disabled={!canStartCorrection || isProcessing}
          size="large"
        >
          {isProcessing ? '处理中...' : '应用修正'}
        </Button>

        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleReset}
          disabled={isProcessing}
        >
          重置
        </Button>
      </Box>

      {/* Help Text */}
      <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block', textAlign: 'center' }}>
        调整上面的参数并点击"应用修正"来处理您的音频
      </Typography>
    </Paper>
  );
};