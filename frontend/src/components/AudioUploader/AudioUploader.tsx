import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Paper,
  Typography,
  Button,
  LinearProgress,
  Alert,
  Chip,
  Grid,
  Card,
  CardContent
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AudioFileIcon from '@mui/icons-material/AudioFile';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import AnalyticsIcon from '@mui/icons-material/Analytics';

import {
  AudioAPI,
  AudioUploadResponse,
  AudioAnalysisParams,
  ProcessingStatus,
  AudioProcessingWebSocket
} from '../../api/audioApi';

interface AudioUploaderProps {
  onUploadComplete?: (sessionId: string, uploadResult: AudioUploadResponse) => void;
  onAnalysisComplete?: (sessionId: string, analysisResult: ProcessingStatus) => void;
  onError?: (error: string) => void;
  analysisParams?: AudioAnalysisParams;
  autoStartAnalysis?: boolean;
}

interface UploadState {
  status: 'idle' | 'uploading' | 'uploaded' | 'analyzing' | 'analyzed' | 'error';
  progress: number;
  sessionId?: string;
  uploadResult?: AudioUploadResponse;
  analysisResult?: ProcessingStatus;
  error?: string;
}

export const AudioUploader: React.FC<AudioUploaderProps> = ({
  onUploadComplete,
  onAnalysisComplete,
  onError,
  analysisParams = { confidence_threshold: 0.85, analyze_vibrato: true },
  autoStartAnalysis = true,
}) => {
  const [uploadState, setUploadState] = useState<UploadState>({
    status: 'idle',
    progress: 0,
  });

  const [websocket, setWebsocket] = useState<AudioProcessingWebSocket | null>(null);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (websocket) {
        websocket.disconnect();
      }
    };
  }, [websocket]);

  const startAnalysis = useCallback(async (sessionId: string) => {
    try {
      setUploadState(prev => ({
        ...prev,
        status: 'analyzing',
        progress: 0,
      }));

      // Setup WebSocket for real-time updates
      const ws = new AudioProcessingWebSocket();
      setWebsocket(ws);

      let isCompleted = false;

      ws.onStatusUpdate((status) => {
        setUploadState(prev => ({
          ...prev,
          progress: status.progress * 100,
        }));

        if (status.status === 'analyzed' && !isCompleted) {
          isCompleted = true;
          setUploadState(prev => ({
            ...prev,
            status: 'analyzed',
            progress: 100,
            analysisResult: status,
          }));

          // Notify parent component
          if (onAnalysisComplete) {
            onAnalysisComplete(sessionId, status);
          }

          // Disconnect WebSocket
          ws.disconnect();
          setWebsocket(null);
        }
      });

      ws.onError((error) => {
        console.warn('WebSocket error, will fallback to polling:', error);
        // Don't set error state immediately, try fallback polling first
      });

      // Connect WebSocket
      try {
        await ws.connect(sessionId);
      } catch (wsError) {
        console.warn('WebSocket connection failed, using polling fallback');
      }

      // Start analysis
      await AudioAPI.startAnalysis(sessionId);

      // Fallback polling mechanism in case WebSocket fails
      const pollInterval = setInterval(async () => {
        try {
          if (!isCompleted) {
            const status = await AudioAPI.getStatus(sessionId);

            setUploadState(prev => ({
              ...prev,
              progress: status.progress * 100,
            }));

            if (status.status === 'analyzed' && !isCompleted) {
              isCompleted = true;
              clearInterval(pollInterval);

              setUploadState(prev => ({
                ...prev,
                status: 'analyzed',
                progress: 100,
                analysisResult: status,
              }));

              // Notify parent component
              if (onAnalysisComplete) {
                onAnalysisComplete(sessionId, status);
              }

              // Disconnect WebSocket
              ws.disconnect();
              setWebsocket(null);
            } else if (status.status === 'error') {
              isCompleted = true;
              clearInterval(pollInterval);
              const errorMsg = status.message || 'Analysis failed';
              setUploadState(prev => ({
                ...prev,
                status: 'error',
                error: errorMsg,
              }));
              onError?.(errorMsg);
              ws.disconnect();
              setWebsocket(null);
            }
          }
        } catch (pollError) {
          console.error('Polling error:', pollError);
        }
      }, 1000); // Poll every second

      // Cleanup polling after 5 minutes max
      setTimeout(() => {
        if (!isCompleted) {
          clearInterval(pollInterval);
          const timeoutError = 'Analysis timeout after 5 minutes';
          setUploadState(prev => ({
            ...prev,
            status: 'error',
            error: timeoutError,
          }));
          onError?.(timeoutError);
          ws.disconnect();
          setWebsocket(null);
        }
      }, 5 * 60 * 1000);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Analysis failed';
      setUploadState(prev => ({
        ...prev,
        status: 'error',
        error: errorMessage,
      }));
      onError?.(errorMessage);

      if (websocket) {
        websocket.disconnect();
        setWebsocket(null);
      }
    }
  }, [onAnalysisComplete, onError, websocket]);

  const handleFileUpload = useCallback(async (file: File) => {
    try {
      setUploadState({
        status: 'uploading',
        progress: 0,
      });

      // Upload file
      const uploadResult = await AudioAPI.uploadAudio(file, analysisParams);

      setUploadState(prev => ({
        ...prev,
        status: 'uploaded',
        progress: 100,
        sessionId: uploadResult.session_id,
        uploadResult,
      }));

      // Notify parent component
      if (onUploadComplete) {
        onUploadComplete(uploadResult.session_id, uploadResult);
      }

      // Auto-start analysis if enabled
      if (autoStartAnalysis) {
        await startAnalysis(uploadResult.session_id);
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      setUploadState(prev => ({
        ...prev,
        status: 'error',
        error: errorMessage,
      }));
      onError?.(errorMessage);
    }
  }, [analysisParams, autoStartAnalysis, onUploadComplete, onError, startAnalysis]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      handleFileUpload(file);
    }
  }, [handleFileUpload]);

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
    },
    multiple: false,
    disabled: ['uploading', 'analyzing'].includes(uploadState.status),
    maxSize: 50 * 1024 * 1024, // 50MB limit
  });

  const hasFile = acceptedFiles.length > 0 || uploadState.sessionId;

  const getStatusIcon = () => {
    switch (uploadState.status) {
      case 'uploaded':
        return <CheckCircleIcon sx={{ color: 'success.main', fontSize: 24 }} />;
      case 'analyzing':
        return <AnalyticsIcon sx={{ color: 'info.main', fontSize: 24 }} />;
      case 'analyzed':
        return <CheckCircleIcon sx={{ color: 'success.main', fontSize: 24 }} />;
      case 'error':
        return <ErrorIcon sx={{ color: 'error.main', fontSize: 24 }} />;
      default:
        return null;
    }
  };

  const getStatusText = () => {
    switch (uploadState.status) {
      case 'uploading':
        return 'Uploading file...';
      case 'uploaded':
        return 'File uploaded successfully';
      case 'analyzing':
        return 'Analyzing pitch with CREPE...';
      case 'analyzed':
        return 'Analysis completed';
      case 'error':
        return uploadState.error || 'An error occurred';
      default:
        return '';
    }
  };

  const getStatusColor = () => {
    switch (uploadState.status) {
      case 'uploaded':
      case 'analyzed':
        return 'success';
      case 'analyzing':
      case 'uploading':
        return 'info';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Audio Upload & Analysis
      </Typography>

      {uploadState.error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {uploadState.error}
        </Alert>
      )}

      {/* Upload Area */}
      {!hasFile && (
        <Box
          {...getRootProps()}
          sx={{
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'divider',
            borderRadius: 2,
            p: 4,
            textAlign: 'center',
            cursor: ['uploading', 'analyzing'].includes(uploadState.status) ? 'not-allowed' : 'pointer',
            backgroundColor: isDragActive ? 'action.hover' : 'transparent',
            transition: 'all 0.2s ease-in-out',
            '&:hover': {
              borderColor: ['uploading', 'analyzing'].includes(uploadState.status) ? 'divider' : 'primary.main',
              backgroundColor: ['uploading', 'analyzing'].includes(uploadState.status) ? 'transparent' : 'action.hover',
            }
          }}
        >
          <input {...getInputProps()} />
          <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          {isDragActive ? (
            <Typography variant="h6">
              Drop the audio file here...
            </Typography>
          ) : (
            <>
              <Typography variant="h6" gutterBottom>
                Drag & drop audio file here
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                or click to select file
              </Typography>
              <Button variant="outlined" disabled={['uploading', 'analyzing'].includes(uploadState.status)}>
                Browse Files
              </Button>
            </>
          )}
        </Box>
      )}

      {/* File Info & Status */}
      {hasFile && (
        <Card variant="outlined" sx={{ mb: 2 }}>
          <CardContent>
            <Grid container spacing={2} alignItems="center">
              <Grid item>
                <AudioFileIcon sx={{ fontSize: 40, color: 'primary.main' }} />
              </Grid>
              <Grid item xs>
                <Typography variant="h6" gutterBottom>
                  {acceptedFiles[0]?.name || 'Uploaded File'}
                </Typography>
                {uploadState.uploadResult && (
                  <Grid container spacing={1}>
                    <Grid item>
                      <Chip
                        label={`${uploadState.uploadResult.audio_info.duration.toFixed(1)}s`}
                        size="small"
                        variant="outlined"
                      />
                    </Grid>
                    <Grid item>
                      <Chip
                        label={`${uploadState.uploadResult.audio_info.sample_rate}Hz`}
                        size="small"
                        variant="outlined"
                      />
                    </Grid>
                    <Grid item>
                      <Chip
                        label={`${(uploadState.uploadResult.audio_info.file_size / 1024 / 1024).toFixed(1)}MB`}
                        size="small"
                        variant="outlined"
                      />
                    </Grid>
                  </Grid>
                )}
              </Grid>
              <Grid item>
                {getStatusIcon()}
              </Grid>
            </Grid>

            {/* Status Text */}
            {uploadState.status !== 'idle' && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  <Box component="span" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    Status:
                    <Chip
                      label={getStatusText()}
                      color={getStatusColor() as any}
                      size="small"
                    />
                  </Box>
                </Typography>
              </Box>
            )}

            {/* Progress Bar */}
            {(['uploading', 'analyzing'].includes(uploadState.status)) && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Progress: {uploadState.progress.toFixed(0)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={uploadState.progress}
                  sx={{ borderRadius: 1 }}
                />
              </Box>
            )}

            {/* Analysis Results */}
            {uploadState.status === 'analyzed' && uploadState.analysisResult?.data && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Analysis Results:
                </Typography>
                <Grid container spacing={1}>
                  <Grid item>
                    <Chip
                      label={`Voiced: ${(uploadState.analysisResult.data.stats.voicing_ratio * 100).toFixed(1)}%`}
                      size="small"
                      color="success"
                    />
                  </Grid>
                  <Grid item>
                    <Chip
                      label={`Confidence: ${uploadState.analysisResult.data.stats.avg_confidence.toFixed(2)}`}
                      size="small"
                      color="info"
                    />
                  </Grid>
                  <Grid item>
                    <Chip
                      label={`Range: ${uploadState.analysisResult.data.stats.pitch_range.min.toFixed(0)}-${uploadState.analysisResult.data.stats.pitch_range.max.toFixed(0)}Hz`}
                      size="small"
                      variant="outlined"
                    />
                  </Grid>
                  {uploadState.analysisResult.data.stats.vibrato && (
                    <Grid item>
                      <Chip
                        label={`Vibrato: ${(uploadState.analysisResult.data.stats.vibrato.vibrato_ratio * 100).toFixed(1)}%`}
                        size="small"
                        color="secondary"
                      />
                    </Grid>
                  )}
                </Grid>
              </Box>
            )}

            {/* Session ID (for debugging) */}
            {uploadState.sessionId && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Session ID: {uploadState.sessionId}
              </Typography>
            )}
          </CardContent>
        </Card>
      )}

      {/* Manual Analysis Button */}
      {uploadState.status === 'uploaded' && !autoStartAnalysis && (
        <Box sx={{ textAlign: 'center', mb: 2 }}>
          <Button
            variant="contained"
            startIcon={<AnalyticsIcon />}
            onClick={() => uploadState.sessionId && startAnalysis(uploadState.sessionId)}
          >
            Start Analysis
          </Button>
        </Box>
      )}

      {/* Reset Button */}
      {['analyzed', 'error'].includes(uploadState.status) && (
        <Box sx={{ textAlign: 'center', mb: 2 }}>
          <Button
            variant="outlined"
            onClick={() => {
              setUploadState({ status: 'idle', progress: 0 });
              // Files will be cleared when component resets
            }}
          >
            Upload Another File
          </Button>
        </Box>
      )}

      <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
        Supported formats: WAV, MP3, FLAC, M4A, AAC, OGG (Max size: 50MB)
      </Typography>
    </Paper>
  );
};