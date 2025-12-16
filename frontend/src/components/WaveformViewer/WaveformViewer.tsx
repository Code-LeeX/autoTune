import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Tabs,
  Tab,
  IconButton,
  Toolbar,
  Chip,
  Grid,
  Alert,
  CircularProgress,
  Tooltip
} from '@mui/material';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import FitScreenIcon from '@mui/icons-material/FitScreen';
import GraphicEqIcon from '@mui/icons-material/GraphicEq';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import MusicNoteIcon from '@mui/icons-material/MusicNote';

import {
  PitchAnalysisData,
  CorrectionResult,
  AudioAPI,
  audioUtils
} from '../../api/audioApi';

interface WaveformViewerProps {
  pitchData?: PitchAnalysisData;
  correctionResult?: CorrectionResult;
  sessionId?: string;
  isProcessing?: boolean;
  // Playback position synchronization
  isPlaying?: boolean;
  currentTime?: number;
  duration?: number;
  onSeek?: (time: number) => void;
}

interface CanvasDrawingState {
  zoomLevel: number;
  panPosition: number;
  viewportStart: number;
  viewportEnd: number;
}

export const WaveformViewer: React.FC<WaveformViewerProps> = ({
  pitchData,
  correctionResult,
  sessionId,
  isProcessing = false,
  isPlaying = false,
  currentTime = 0,
  duration = 0,
  onSeek
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [activeTab, setActiveTab] = useState(0);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panPosition, setPanPosition] = useState(0);
  const [audioWaveform, setAudioWaveform] = useState<Float32Array | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 300 });
  const [cursorStyle, setCursorStyle] = useState<'default' | 'grab' | 'grabbing' | 'col-resize' | 'pointer'>('default');

  // Load audio waveform for visualization
  const loadAudioWaveform = useCallback(async () => {
    if (!sessionId) return;

    try {
      setLoading(true);
      setError(null);

      // Download original audio
      const audioBlob = await AudioAPI.downloadAudio(sessionId, 'original');
      const audioArrayBuffer = await audioBlob.arrayBuffer();

      // Create audio context and decode
      const { audioBuffer } = await audioUtils.createAudioContext(audioArrayBuffer);

      // Extract waveform data for visualization
      const waveformData = audioUtils.extractWaveformData(audioBuffer, 2000);
      setAudioWaveform(waveformData);

    } catch (err) {
      console.error('Failed to load audio waveform:', err);
      setError('Failed to load audio for visualization');
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  // Load waveform when sessionId changes
  useEffect(() => {
    if (sessionId && !audioWaveform) {
      loadAudioWaveform();
    }
  }, [sessionId, audioWaveform, loadAudioWaveform]);

  // Calculate drawing parameters
  const getDrawingState = useCallback((): CanvasDrawingState => {
    const audioDuration = pitchData ? Math.max(...pitchData.time) : (duration || 5.0);
    const viewportStart = panPosition * audioDuration;
    const viewportEnd = viewportStart + (audioDuration / zoomLevel);

    return {
      zoomLevel,
      panPosition,
      viewportStart,
      viewportEnd
    };
  }, [zoomLevel, panPosition, pitchData, duration]);

  // Draw waveform
  const drawWaveform = useCallback((canvas: HTMLCanvasElement, data: Float32Array) => {
    const ctx = canvas.getContext('2d');
    if (!ctx || !data || data.length === 0) return;

    const { width, height } = canvas;
    const drawingState = getDrawingState();

    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, width, height);

    // Calculate sample range to draw
    const totalSamples = data.length;
    const samplesPerPixel = totalSamples / width * zoomLevel;
    const startSample = Math.max(0, Math.floor(panPosition * totalSamples));
    const endSample = Math.min(totalSamples, startSample + Math.ceil(width * samplesPerPixel));

    // Draw waveform
    ctx.strokeStyle = '#2196f3';
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (let x = 0; x < width; x++) {
      const sampleIndex = Math.floor(startSample + x * samplesPerPixel);
      if (sampleIndex >= endSample) break;

      const amplitude = data[sampleIndex] || 0;
      const y = (height / 2) + (amplitude * height * 0.4);

      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw center line
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw playhead (playback position indicator)
    if (duration > 0 && currentTime >= 0) {
      const drawingState = getDrawingState();

      // Calculate playhead position relative to current viewport
      const playheadTime = currentTime;
      if (playheadTime >= drawingState.viewportStart && playheadTime <= drawingState.viewportEnd) {
        const viewportProgress = (playheadTime - drawingState.viewportStart) /
                               (drawingState.viewportEnd - drawingState.viewportStart);
        const playheadX = viewportProgress * width;

        // Draw playhead line
        ctx.strokeStyle = isPlaying ? '#ff5722' : '#ff9800'; // Orange when playing, amber when paused
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, height);
        ctx.stroke();

        // Draw playhead handle
        ctx.fillStyle = isPlaying ? '#ff5722' : '#ff9800';
        ctx.beginPath();
        ctx.arc(playheadX, 10, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

  }, [getDrawingState, currentTime, duration, isPlaying]);

  // Draw pitch curves
  const drawPitchCurves = useCallback((canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d');
    if (!ctx || !pitchData) return;

    const { width, height } = canvas;
    const drawingState = getDrawingState();

    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, width, height);

    // Frequency range for visualization (logarithmic scale)
    const minFreq = 80; // Hz
    const maxFreq = 800; // Hz
    const logMin = Math.log(minFreq);
    const logMax = Math.log(maxFreq);

    // Convert frequency to Y coordinate
    const freqToY = (freq: number) => {
      if (freq <= 0) return height;
      const logFreq = Math.log(Math.max(freq, minFreq));
      return height - ((logFreq - logMin) / (logMax - logMin)) * height * 0.9;
    };

    // Convert time to X coordinate
    const timeToX = (time: number) => {
      const duration = Math.max(...pitchData.time);
      const normalizedTime = (time - drawingState.viewportStart) / (drawingState.viewportEnd - drawingState.viewportStart);
      return normalizedTime * width;
    };

    // Draw frequency grid lines
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    const gridFreqs = [100, 200, 300, 400, 600, 800];
    gridFreqs.forEach(freq => {
      const y = freqToY(freq);
      if (y >= 0 && y <= height) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();

        // Frequency labels
        ctx.fillStyle = '#666';
        ctx.font = '10px Arial';
        ctx.fillText(`${freq}Hz`, 5, y - 2);
      }
    });

    // Draw original pitch curve
    if (pitchData.frequency && pitchData.time) {
      ctx.strokeStyle = '#ff4081'; // Pink for original
      ctx.lineWidth = 2;
      ctx.beginPath();

      let firstPoint = true;
      for (let i = 0; i < pitchData.frequency.length; i++) {
        const freq = pitchData.frequency[i];
        const time = pitchData.time[i];
        const confidence = pitchData.confidence[i];
        const isVoiced = pitchData.voiced_mask[i];

        if (isVoiced && freq > 0 && confidence > 0.5) {
          const x = timeToX(time);
          const y = freqToY(freq);

          // Only draw points within viewport
          if (x >= -50 && x <= width + 50) {
            if (firstPoint) {
              ctx.moveTo(x, y);
              firstPoint = false;
            } else {
              ctx.lineTo(x, y);
            }
          }
        } else if (!firstPoint) {
          // Break the line for unvoiced segments
          ctx.stroke();
          ctx.beginPath();
          firstPoint = true;
        }
      }
      ctx.stroke();

      // Draw vibrato highlights
      if (pitchData.vibrato_presence) {
        ctx.fillStyle = 'rgba(255, 193, 7, 0.3)'; // Semi-transparent amber
        for (let i = 0; i < pitchData.vibrato_presence.length; i++) {
          if (pitchData.vibrato_presence[i] && pitchData.frequency[i] > 0) {
            const x = timeToX(pitchData.time[i]);
            const y = freqToY(pitchData.frequency[i]);

            if (x >= 0 && x <= width) {
              ctx.beginPath();
              ctx.arc(x, y, 3, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        }
      }
    }

    // Draw corrected pitch curve
    if (correctionResult?.quantized_frequency) {
      ctx.strokeStyle = '#4caf50'; // Green for corrected
      ctx.lineWidth = 2;
      ctx.beginPath();

      let firstPoint = true;
      for (let i = 0; i < correctionResult.quantized_frequency.length; i++) {
        const freq = correctionResult.quantized_frequency[i];
        const time = pitchData?.time[i];
        const wasCorrected = correctionResult.correction_applied[i];

        if (freq > 0 && time !== undefined) {
          const x = timeToX(time);
          const y = freqToY(freq);

          // Only draw points within viewport
          if (x >= -50 && x <= width + 50) {
            if (firstPoint) {
              ctx.moveTo(x, y);
              firstPoint = false;
            } else {
              ctx.lineTo(x, y);
            }

            // Mark corrected points
            if (wasCorrected) {
              ctx.save();
              ctx.fillStyle = '#4caf50';
              ctx.beginPath();
              ctx.arc(x, y, 2, 0, Math.PI * 2);
              ctx.fill();
              ctx.restore();
            }
          }
        } else if (!firstPoint) {
          ctx.stroke();
          ctx.beginPath();
          firstPoint = true;
        }
      }
      ctx.stroke();
    }

    // Draw playhead (playback position indicator) on pitch view
    if (duration > 0 && currentTime >= 0 && pitchData) {
      const audioDuration = Math.max(...pitchData.time);
      const drawingState = getDrawingState();

      // Calculate playhead position relative to current viewport
      const playheadTime = currentTime;
      if (playheadTime >= drawingState.viewportStart && playheadTime <= drawingState.viewportEnd) {
        const viewportProgress = (playheadTime - drawingState.viewportStart) /
                               (drawingState.viewportEnd - drawingState.viewportStart);
        const playheadX = viewportProgress * width;

        // Draw playhead line
        ctx.strokeStyle = isPlaying ? '#ff5722' : '#ff9800'; // Orange when playing, amber when paused
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, height);
        ctx.stroke();

        // Draw playhead handle
        ctx.fillStyle = isPlaying ? '#ff5722' : '#ff9800';
        ctx.beginPath();
        ctx.arc(playheadX, 10, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

  }, [pitchData, correctionResult, getDrawingState, currentTime, duration, isPlaying]);

  // Main canvas drawing effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Update canvas size
    const rect = canvas.getBoundingClientRect();
    const newSize = { width: rect.width, height: rect.height };
    if (newSize.width !== canvasSize.width || newSize.height !== canvasSize.height) {
      setCanvasSize(newSize);
      canvas.width = newSize.width;
      canvas.height = newSize.height;
    }

    // Draw based on active tab
    if (activeTab === 0 && audioWaveform) {
      drawWaveform(canvas, audioWaveform);
    } else if (activeTab === 1) {
      drawPitchCurves(canvas);
    }

  }, [activeTab, audioWaveform, pitchData, correctionResult, zoomLevel, panPosition, canvasSize, drawWaveform, drawPitchCurves]);

  // Zoom and pan controls
  const handleZoomIn = () => setZoomLevel(prev => Math.min(prev * 1.5, 20));
  const handleZoomOut = () => setZoomLevel(prev => Math.max(prev / 1.5, 0.1));
  const handleFitScreen = () => {
    setZoomLevel(1);
    setPanPosition(0);
  };

  // Mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      handleZoomIn();
    } else {
      handleZoomOut();
    }
  }, []);

  // Enhanced mouse handling for pan and seek
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const startX = e.clientX;
    const startPan = panPosition;

    // Use the consistent drawing state calculation
    const initialState = getDrawingState();
    const audioDuration = pitchData ? Math.max(...pitchData.time) : (duration || 5.0);

    if (audioDuration > 0) {
      // Calculate current playhead position
      const playheadTime = currentTime;
      if (playheadTime >= initialState.viewportStart && playheadTime <= initialState.viewportEnd) {
        const viewportProgress = (playheadTime - initialState.viewportStart) /
                               (initialState.viewportEnd - initialState.viewportStart);
        const playheadX = viewportProgress * canvas.width;

        // Check if click is within 15 pixels of playhead
        if (Math.abs(clickX - playheadX) <= 15) {
          // Playhead drag mode
          const handlePlayheadMove = (e: MouseEvent) => {
            const rect = canvas.getBoundingClientRect();
            const newClickX = e.clientX - rect.left;
            const newProgress = Math.max(0, Math.min(1, newClickX / canvas.width));

            // Recalculate current viewport state for accurate dragging
            const currentState = getDrawingState();
            const newTime = currentState.viewportStart +
                           newProgress * (currentState.viewportEnd - currentState.viewportStart);

            // Clamp to valid time range
            const clampedTime = Math.max(0, Math.min(audioDuration, newTime));
            if (onSeek) {
              onSeek(clampedTime);
            }
          };

          const handlePlayheadUp = () => {
            document.removeEventListener('mousemove', handlePlayheadMove);
            document.removeEventListener('mouseup', handlePlayheadUp);
          };

          document.addEventListener('mousemove', handlePlayheadMove);
          document.addEventListener('mouseup', handlePlayheadUp);
          return; // Don't proceed with pan logic
        }
      }

      // If not near playhead, check for click-to-seek
      const clickProgress = Math.max(0, Math.min(1, clickX / canvas.width));
      const clickTime = initialState.viewportStart +
                       clickProgress * (initialState.viewportEnd - initialState.viewportStart);
      const clampedTime = Math.max(0, Math.min(audioDuration, clickTime));

      if (onSeek) {
        onSeek(clampedTime);
      }
    }

    // Pan handling (only if zoom level > 1)
    if (zoomLevel > 1) {
      const handleMouseMove = (e: MouseEvent) => {
        const deltaX = (e.clientX - startX) / canvasSize.width;
        const maxPanPosition = Math.max(0, 1 - 1/zoomLevel);
        setPanPosition(Math.max(0, Math.min(maxPanPosition, startPan - deltaX / zoomLevel)));
      };

      const handleMouseUp = () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
  }, [panPosition, canvasSize.width, zoomLevel, onSeek, currentTime, duration, pitchData, getDrawingState]);

  // Mouse move handling for cursor updates
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const audioDuration = pitchData ? Math.max(...pitchData.time) : duration;
    const drawingState = getDrawingState();

    if (audioDuration > 0) {
      // Check if mouse is near playhead
      const playheadTime = currentTime;
      if (playheadTime >= drawingState.viewportStart && playheadTime <= drawingState.viewportEnd) {
        const viewportProgress = (playheadTime - drawingState.viewportStart) /
                               (drawingState.viewportEnd - drawingState.viewportStart);
        const playheadX = viewportProgress * canvas.width;

        if (Math.abs(mouseX - playheadX) <= 15) {
          setCursorStyle('col-resize');
          return;
        }
      }
    }

    // Default cursor based on zoom level
    if (zoomLevel > 1) {
      setCursorStyle('grab');
    } else {
      setCursorStyle('pointer');
    }
  }, [currentTime, duration, pitchData, getDrawingState, zoomLevel]);

  // Show loading state if no data available yet
  if (!pitchData && !audioWaveform && !loading) {
    return (
      <Paper elevation={2} sx={{ p: 3, textAlign: 'center' }}>
        <GraphicEqIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          Upload and analyze audio to view waveform and pitch data
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={2} sx={{ overflow: 'hidden' }}>
      {/* Toolbar */}
      <Toolbar variant="dense" sx={{ minHeight: 48, px: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ShowChartIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h6">
            Audio Analyzer
          </Typography>
        </Box>

        <Box sx={{ flexGrow: 1 }} />

        {/* Statistics */}
        {pitchData && (
          <Box sx={{ display: 'flex', gap: 1, mr: 2 }}>
            <Chip
              size="small"
              icon={<MusicNoteIcon />}
              label={`${(pitchData.stats.voicing_ratio * 100).toFixed(1)}% voiced`}
              variant="outlined"
            />
            {pitchData.stats.vibrato && (
              <Chip
                size="small"
                label={`${(pitchData.stats.vibrato.vibrato_ratio * 100).toFixed(1)}% vibrato`}
                color="secondary"
                variant="outlined"
              />
            )}
          </Box>
        )}

        {/* Controls */}
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          <Tooltip title="Zoom In">
            <IconButton size="small" onClick={handleZoomIn} disabled={zoomLevel >= 20}>
              <ZoomInIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom Out">
            <IconButton size="small" onClick={handleZoomOut} disabled={zoomLevel <= 0.1}>
              <ZoomOutIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Fit to Screen">
            <IconButton size="small" onClick={handleFitScreen}>
              <FitScreenIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Toolbar>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
          <Tab
            icon={<GraphicEqIcon />}
            label="Waveform"
            disabled={!audioWaveform && !loading}
          />
          <Tab
            icon={<ShowChartIcon />}
            label="Pitch Analysis"
            disabled={!pitchData}
          />
        </Tabs>
      </Box>

      {/* Error display */}
      {error && (
        <Box sx={{ p: 2 }}>
          <Alert severity="error">{error}</Alert>
        </Box>
      )}

      {/* Canvas container */}
      <Box sx={{ position: 'relative', height: 350, backgroundColor: '#1a1a1a' }}>
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            display: 'block',
            cursor: cursorStyle
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onWheel={handleWheel}
        />

        {/* Loading overlay */}
        {(isProcessing || loading) && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'rgba(0,0,0,0.7)',
              backdropFilter: 'blur(2px)'
            }}
          >
            <Box sx={{ textAlign: 'center' }}>
              <CircularProgress sx={{ mb: 2 }} />
              <Typography color="text.secondary">
                {loading ? 'Loading audio data...' : 'Processing...'}
              </Typography>
            </Box>
          </Box>
        )}

        {/* Zoom level indicator */}
        {zoomLevel !== 1 && (
          <Box
            sx={{
              position: 'absolute',
              top: 8,
              right: 8,
              backgroundColor: 'rgba(0,0,0,0.8)',
              px: 1,
              py: 0.5,
              borderRadius: 1
            }}
          >
            <Typography variant="caption" color="text.secondary">
              {zoomLevel.toFixed(1)}x
            </Typography>
          </Box>
        )}
      </Box>

      {/* Legend */}
      {activeTab === 1 && pitchData && (
        <Box sx={{ p: 2, backgroundColor: 'background.paper', borderTop: 1, borderColor: 'divider' }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item>
              <Typography variant="caption" color="text.secondary">
                Legend:
              </Typography>
            </Grid>
            <Grid item>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Box sx={{ width: 16, height: 2, backgroundColor: '#ff4081' }} />
                <Typography variant="caption">Original Pitch</Typography>
              </Box>
            </Grid>
            {correctionResult && (
              <Grid item>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 16, height: 2, backgroundColor: '#4caf50' }} />
                  <Typography variant="caption">Corrected Pitch</Typography>
                </Box>
              </Grid>
            )}
            {pitchData.vibrato_presence && (
              <Grid item>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: 'rgba(255, 193, 7, 0.7)' }} />
                  <Typography variant="caption">Vibrato</Typography>
                </Box>
              </Grid>
            )}
            <Grid item>
              <Typography variant="caption" color="text.secondary">
                Use mouse wheel to zoom • Click and drag to pan
              </Typography>
            </Grid>
          </Grid>
        </Box>
      )}
    </Paper>
  );
};