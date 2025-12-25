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
import { MixingPanel } from './components/MixingPanel/MixingPanel';
import { ProcessingModeSelector } from './components/ProcessingModeSelector/ProcessingModeSelector';

import {
  AudioUploadResponse,
  ProcessingStatus,
  PitchAnalysisData,
  CorrectionResult,
  ProcessingMode,
  MixingParams
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
  mixingResult?: any; // Mixed audio processing result
  processingResult?: any; // Full processing result (pitch + mixing)
  processingMode: ProcessingMode;
  currentStep: 'upload' | 'mode_select' | 'analyze' | 'correct' | 'mixing' | 'processing' | 'complete';
}

// Playback state interface
interface PlaybackState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
}

function App() {
  const [appState, setAppState] = useState<AppState>({
    currentStep: 'upload',
    processingMode: {
      pitchCorrection: true,
      mixing: false,
      processingOrder: 'pitch_first'
    }
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
      currentStep: 'mode_select'
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

  // Handle processing mode selection
  const handleProcessingModeChange = useCallback(async (mode: ProcessingMode) => {
    console.log('Processing mode changed:', mode);
    setAppState(prev => ({
      ...prev,
      processingMode: mode
    }));

    // Determine next step based on processing mode
    if (mode.pitchCorrection && !appState.analysisResult) {
      // Need pitch analysis - user will need to manually start analysis
      setAppState(prev => ({ ...prev, currentStep: 'analyze' }));
    } else if (mode.mixing && !mode.pitchCorrection) {
      // Mixing only - go directly to mixing
      setAppState(prev => ({ ...prev, currentStep: 'mixing' }));
    } else if (mode.pitchCorrection && mode.mixing && appState.analysisResult) {
      // Complete processing with existing analysis - go to correct step
      setAppState(prev => ({ ...prev, currentStep: 'correct' }));
    } else if (mode.pitchCorrection && appState.analysisResult) {
      // Pitch correction only with existing analysis
      setAppState(prev => ({ ...prev, currentStep: 'correct' }));
    } else {
      // Default case - need analysis
      setAppState(prev => ({ ...prev, currentStep: 'analyze' }));
    }

    setNotification({
      open: true,
      message: `Processing mode set: ${mode.pitchCorrection && mode.mixing ? 'Complete Processing' : mode.pitchCorrection ? 'Pitch Correction Only' : 'Mixing Only'}`,
      severity: 'info'
    });
  }, [appState.sessionId, appState.analysisResult]);

  // Handle mixing completion (mixing-only mode)
  const handleMixingComplete = useCallback((sessionId: string, mixingResult: any) => {
    console.log('Mixing completed:', sessionId, mixingResult);
    setAppState(prev => ({
      ...prev,
      mixingResult,
      currentStep: 'complete'
    }));

    setNotification({
      open: true,
      message: 'Audio mixing completed successfully',
      severity: 'success'
    });
  }, []);

  // Handle full processing completion (pitch + mixing)
  const handleFullProcessingComplete = useCallback((sessionId: string, processingResult: any) => {
    console.log('Full processing completed:', sessionId, processingResult);
    setAppState(prev => ({
      ...prev,
      processingResult,
      currentStep: 'complete'
    }));

    setNotification({
      open: true,
      message: 'Complete audio processing finished successfully',
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
      currentStep: 'upload',
      processingMode: {
        pitchCorrection: true,
        mixing: false,
        processingOrder: 'pitch_first'
      }
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

          {/* Main Interface Grid */}
          <Box sx={{ display: 'grid', gap: 3, gridTemplateColumns: { xs: '1fr', lg: '2fr 1fr' } }}>

            {/* Left Panel - Audio Processing */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {/* Audio Upload */}
              <AudioUploader
                onUploadComplete={handleUploadComplete}
                onAnalysisComplete={handleAnalysisComplete}
                onError={handleError}
                autoStartAnalysis={false}
              />

              {/* Processing Mode Selector - Show after upload */}
              {appState.currentStep === 'mode_select' && appState.sessionId && (
                <ProcessingModeSelector
                  value={appState.processingMode}
                  onChange={handleProcessingModeChange}
                  hasAnalyzedAudio={!!appState.analysisResult}
                />
              )}

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
                  hasMixed={!!appState.mixingResult}
                  hasProcessed={!!appState.processingResult}
                  playbackState={playbackState}
                  onPlaybackStateChange={handlePlaybackStateChange}
                />
              )}
            </Box>

            {/* Right Panel - Parameters & Controls */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {/* Pitch Correction Panel - Show when pitch correction is enabled */}
              {appState.processingMode.pitchCorrection && (
                <ParameterPanel
                  sessionId={appState.sessionId}
                  analysisResult={appState.analysisResult}
                  onCorrectionComplete={handleCorrectionComplete}
                  onError={handleError}
                  onReset={handleReset}
                  disabled={appState.currentStep !== 'correct'}
                />
              )}

              {/* Mixing Panel - Show when mixing is enabled */}
              {appState.processingMode.mixing && appState.sessionId && (
                <MixingPanel
                  sessionId={appState.sessionId}
                  processingMode={appState.processingMode}
                  onMixingComplete={handleMixingComplete}
                  onFullProcessingComplete={handleFullProcessingComplete}
                  onError={handleError}
                  disabled={
                    appState.currentStep === 'mixing' ? false :
                    appState.currentStep === 'processing' ? false :
                    (appState.processingMode.pitchCorrection && appState.processingMode.mixing) ? !appState.correctionResult :
                    true
                  }
                />
              )}

              {/* Audio Export */}
              <AudioExporter
                sessionId={appState.sessionId}
                hasOriginal={!!appState.uploadResult}
                hasCorrected={!!appState.correctionResult}
                hasMixed={!!appState.mixingResult}
                hasProcessed={!!appState.processingResult}
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