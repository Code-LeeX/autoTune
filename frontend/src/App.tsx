import React, { useState, useCallback } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import { Alert, Snackbar } from '@mui/material';
import { AudioUploader } from './components/AudioUploader/AudioUploader';
import { WaveformViewer } from './components/WaveformViewer/WaveformViewer';
import { ParameterPanel } from './components/ParameterPanel/ParameterPanel';
import { AudioPlayer } from './components/AudioPlayer/AudioPlayer';
import { AudioExporter } from './components/AudioExporter/AudioExporter';

import {
  AudioUploadResponse,
  ProcessingStatus,
  PitchAnalysisData,
  CorrectionResult
} from './api/audioApi';

// Create dark theme for professional audio app look
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

// App state interface
interface AppState {
  sessionId?: string;
  uploadResult?: AudioUploadResponse;
  analysisResult?: PitchAnalysisData;
  correctionResult?: CorrectionResult;
  currentStep: 'upload' | 'analyze' | 'correct' | 'complete';
}

// Playback state interface
interface PlaybackState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
}

function App() {
  const [appState, setAppState] = useState<AppState>({
    currentStep: 'upload'
  });

  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  const [playbackState, setPlaybackState] = useState<PlaybackState>({
    isPlaying: false,
    currentTime: 0,
    duration: 0
  });

  // Handle successful upload
  const handleUploadComplete = useCallback((sessionId: string, uploadResult: AudioUploadResponse) => {
    console.log('Upload completed:', sessionId, uploadResult);
    setAppState(prev => ({
      ...prev,
      sessionId,
      uploadResult,
      currentStep: 'analyze'
    }));

    setNotification({
      open: true,
      message: `File uploaded successfully: ${uploadResult.audio_info.duration.toFixed(1)}s audio`,
      severity: 'success'
    });
  }, []);

  // Handle successful analysis
  const handleAnalysisComplete = useCallback((sessionId: string, analysisResult: ProcessingStatus) => {
    console.log('Analysis completed:', sessionId, analysisResult);

    if (analysisResult.data) {
      setAppState(prev => ({
        ...prev,
        analysisResult: analysisResult.data,
        currentStep: 'correct'
      }));

      const stats = analysisResult.data.stats;
      setNotification({
        open: true,
        message: `Pitch analysis completed: ${(stats.voicing_ratio * 100).toFixed(1)}% voiced frames`,
        severity: 'success'
      });
    }
  }, []);

  // Handle correction completion
  const handleCorrectionComplete = useCallback((sessionId: string, correctionResult: CorrectionResult) => {
    console.log('Correction completed:', sessionId, correctionResult);
    setAppState(prev => ({
      ...prev,
      correctionResult,
      currentStep: 'complete'
    }));

    const improvement = correctionResult.correction_stats.pitch_accuracy_improvement_cents;
    setNotification({
      open: true,
      message: `Pitch correction completed${improvement ? `: ${improvement.toFixed(1)} cents improvement` : ''}`,
      severity: 'success'
    });
  }, []);

  // Handle errors
  const handleError = useCallback((error: string) => {
    console.error('App error:', error);
    setNotification({
      open: true,
      message: error,
      severity: 'error'
    });
  }, []);

  // Handle notification close
  const handleNotificationClose = () => {
    setNotification(prev => ({ ...prev, open: false }));
  };

  // Reset to start over
  const handleReset = useCallback(() => {
    setAppState({
      currentStep: 'upload'
    });
    setPlaybackState({
      isPlaying: false,
      currentTime: 0,
      duration: 0
    });
  }, []);

  // Playback control handlers
  const handlePlaybackStateChange = useCallback((newState: Partial<PlaybackState>) => {
    setPlaybackState(prev => ({ ...prev, ...newState }));
  }, []);

  const handleSeek = useCallback((time: number) => {
    setPlaybackState(prev => ({ ...prev, currentTime: time }));
  }, []);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Container maxWidth="xl" sx={{ py: 2 }}>
          {/* Header */}
          <Box sx={{ mb: 4, textAlign: 'center' }}>
            <Typography variant="h3" component="h1" gutterBottom>
              AI Pitch Correction Tool
            </Typography>
            <Typography variant="h6" color="text.secondary">
              Professional pitch correction powered by CREPE + PyWorld
            </Typography>
          </Box>

          {/* Main Interface Grid */}
          <Box sx={{ display: 'grid', gap: 3, gridTemplateColumns: { xs: '1fr', lg: '2fr 1fr' } }}>

            {/* Left Panel - Audio Processing */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {/* Audio Upload */}
              <AudioUploader
                onUploadComplete={handleUploadComplete}
                onAnalysisComplete={handleAnalysisComplete}
                onError={handleError}
                autoStartAnalysis={true}
              />

              {/* Waveform Viewer - Show when analysis is complete */}
              {appState.analysisResult && (
                <WaveformViewer
                  sessionId={appState.sessionId}
                  pitchData={appState.analysisResult}
                  correctionResult={appState.correctionResult}
                  isPlaying={playbackState.isPlaying}
                  currentTime={playbackState.currentTime}
                  duration={playbackState.duration}
                  onSeek={handleSeek}
                />
              )}

              {/* Audio Player - Show when upload is complete */}
              {appState.sessionId && (
                <AudioPlayer
                  sessionId={appState.sessionId}
                  hasOriginal={!!appState.uploadResult}
                  hasCorrected={!!appState.correctionResult}
                  playbackState={playbackState}
                  onPlaybackStateChange={handlePlaybackStateChange}
                />
              )}
            </Box>

            {/* Right Panel - Parameters & Controls */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <ParameterPanel
                sessionId={appState.sessionId}
                analysisResult={appState.analysisResult}
                onCorrectionComplete={handleCorrectionComplete}
                onError={handleError}
                onReset={handleReset}
                disabled={appState.currentStep !== 'correct'}
              />

              {/* Audio Export */}
              <AudioExporter
                sessionId={appState.sessionId}
                hasOriginal={!!appState.uploadResult}
                hasCorrected={!!appState.correctionResult}
                onError={handleError}
              />
            </Box>

          </Box>
        </Container>

        {/* Notification Snackbar */}
        <Snackbar
          open={notification.open}
          autoHideDuration={6000}
          onClose={handleNotificationClose}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={handleNotificationClose}
            severity={notification.severity}
            variant="filled"
            sx={{ width: '100%' }}
          >
            {notification.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default App;