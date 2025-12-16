import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  ButtonGroup,
  Slider,
  Grid,
  Alert,
  IconButton,
  Chip,
  LinearProgress,
  Tooltip
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import StopIcon from '@mui/icons-material/Stop';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import VolumeOffIcon from '@mui/icons-material/VolumeOff';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import MusicNoteIcon from '@mui/icons-material/MusicNote';

import { AudioAPI } from '../../api/audioApi';

// Use local PlaybackState interface for internal state
interface LocalPlaybackState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
}

// Import PlaybackState type from App.tsx for props
interface PlaybackState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
}

interface AudioPlayerProps {
  sessionId?: string;
  hasOriginal?: boolean;
  hasCorrected?: boolean;
  playbackState?: PlaybackState;
  onPlaybackStateChange?: (newState: Partial<PlaybackState>) => void;
}

interface AudioState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  muted: boolean;
  loading: boolean;
  error?: string;
}

type AudioType = 'original' | 'corrected';

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
  sessionId,
  hasOriginal = false,
  hasCorrected = false,
  playbackState,
  onPlaybackStateChange
}) => {
  const originalAudioRef = useRef<HTMLAudioElement>(null);
  const correctedAudioRef = useRef<HTMLAudioElement>(null);

  const [audioState, setAudioState] = useState<AudioState>({
    isPlaying: false,
    currentTime: 0,
    duration: 0,
    volume: 1.0,
    muted: false,
    loading: false
  });

  const [activeAudio, setActiveAudio] = useState<AudioType>('original');
  const [compareMode, setCompareMode] = useState(false);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [correctedUrl, setCorrectedUrl] = useState<string | null>(null);

  // Sync external playback state to internal state
  useEffect(() => {
    if (playbackState) {
      setAudioState(prev => ({
        ...prev,
        isPlaying: playbackState.isPlaying,
        currentTime: playbackState.currentTime,
        duration: playbackState.duration
      }));

      // Sync audio element times when seeking from external source
      const currentAudio = getCurrentAudio();
      if (currentAudio && Math.abs(currentAudio.currentTime - playbackState.currentTime) > 0.1) {
        currentAudio.currentTime = playbackState.currentTime;
      }

      // Sync playback state
      if (playbackState.isPlaying && currentAudio?.paused) {
        currentAudio.play();
      } else if (!playbackState.isPlaying && !currentAudio?.paused) {
        currentAudio?.pause();
      }
    }
  }, [playbackState]);

  const loadAudioUrl = useCallback(async (type: AudioType) => {
    if (!sessionId) return;

    try {
      setAudioState(prev => ({ ...prev, loading: true, error: undefined }));

      const audioBlob = await AudioAPI.downloadAudio(sessionId, type);
      const url = URL.createObjectURL(audioBlob);

      if (type === 'original') {
        setOriginalUrl(url);
      } else {
        setCorrectedUrl(url);
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : `Failed to load ${type} audio`;
      setAudioState(prev => ({ ...prev, error: errorMessage }));
      console.error(`Failed to load ${type} audio:`, error);
    } finally {
      setAudioState(prev => ({ ...prev, loading: false }));
    }
  }, [sessionId]);

  // Load audio URLs when session changes
  useEffect(() => {
    if (sessionId && hasOriginal) {
      loadAudioUrl('original');
    }
  }, [sessionId, hasOriginal, loadAudioUrl]);

  useEffect(() => {
    if (sessionId && hasCorrected) {
      loadAudioUrl('corrected');
    }
  }, [sessionId, hasCorrected, loadAudioUrl]);

  // Cleanup URLs on unmount
  useEffect(() => {
    return () => {
      if (originalUrl) URL.revokeObjectURL(originalUrl);
      if (correctedUrl) URL.revokeObjectURL(correctedUrl);
    };
  }, [originalUrl, correctedUrl]);

  const getCurrentAudio = useCallback((): HTMLAudioElement | null => {
    if (compareMode) {
      return activeAudio === 'original' ? originalAudioRef.current : correctedAudioRef.current;
    }
    return activeAudio === 'original' ? originalAudioRef.current : correctedAudioRef.current;
  }, [activeAudio, compareMode]);

  const handlePlay = () => {
    const audio = getCurrentAudio();
    if (!audio) return;

    if (compareMode) {
      if (originalAudioRef.current && correctedAudioRef.current) {
        originalAudioRef.current.currentTime = audioState.currentTime;
        correctedAudioRef.current.currentTime = audioState.currentTime;
        originalAudioRef.current.play();
        correctedAudioRef.current.play();
      }
    } else {
      audio.currentTime = audioState.currentTime;
      audio.play();
    }

    const newState = { isPlaying: true };
    setAudioState(prev => ({ ...prev, ...newState }));
    onPlaybackStateChange?.(newState);
  };

  const handlePause = () => {
    if (compareMode) {
      originalAudioRef.current?.pause();
      correctedAudioRef.current?.pause();
    } else {
      getCurrentAudio()?.pause();
    }

    const newState = { isPlaying: false };
    setAudioState(prev => ({ ...prev, ...newState }));
    onPlaybackStateChange?.(newState);
  };

  const handleStop = () => {
    if (compareMode) {
      if (originalAudioRef.current) originalAudioRef.current.currentTime = 0;
      if (correctedAudioRef.current) correctedAudioRef.current.currentTime = 0;
      originalAudioRef.current?.pause();
      correctedAudioRef.current?.pause();
    } else {
      const audio = getCurrentAudio();
      if (audio) {
        audio.currentTime = 0;
        audio.pause();
      }
    }

    const newState = { isPlaying: false, currentTime: 0 };
    setAudioState(prev => ({ ...prev, ...newState }));
    onPlaybackStateChange?.(newState);
  };

  const handleSeek = (value: number) => {
    const newTime = (value / 100) * audioState.duration;

    if (compareMode) {
      if (originalAudioRef.current) originalAudioRef.current.currentTime = newTime;
      if (correctedAudioRef.current) correctedAudioRef.current.currentTime = newTime;
    } else {
      const audio = getCurrentAudio();
      if (audio) audio.currentTime = newTime;
    }

    const newState = { currentTime: newTime };
    setAudioState(prev => ({ ...prev, ...newState }));
    onPlaybackStateChange?.(newState);
  };

  const handleVolumeChange = (value: number) => {
    const newVolume = value / 100;

    if (originalAudioRef.current) originalAudioRef.current.volume = newVolume;
    if (correctedAudioRef.current) correctedAudioRef.current.volume = newVolume;

    setAudioState(prev => ({ ...prev, volume: newVolume, muted: newVolume === 0 }));
  };

  const toggleMute = () => {
    const newMuted = !audioState.muted;
    const newVolume = newMuted ? 0 : (audioState.volume > 0 ? audioState.volume : 0.5);

    if (originalAudioRef.current) originalAudioRef.current.volume = newVolume;
    if (correctedAudioRef.current) correctedAudioRef.current.volume = newVolume;

    setAudioState(prev => ({ ...prev, muted: newMuted, volume: newVolume }));
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const hasAnyAudio = hasOriginal || hasCorrected;
  const canCompare = hasOriginal && hasCorrected;

  if (!hasAnyAudio) {
    return (
      <Paper elevation={2} sx={{ p: 3, textAlign: 'center' }}>
        <MusicNoteIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          No audio available for playback
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Upload and process audio to enable playback controls
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      {/* Audio Elements */}
      {originalUrl && (
        <audio
          ref={originalAudioRef}
          src={originalUrl}
          preload="metadata"
          onLoadedMetadata={() => {
            if (originalAudioRef.current) {
              const newDuration = originalAudioRef.current.duration;
              setAudioState(prev => ({ ...prev, duration: newDuration }));
              onPlaybackStateChange?.({ duration: newDuration });
            }
          }}
          onTimeUpdate={() => {
            if (originalAudioRef.current && (!compareMode || activeAudio === 'original')) {
              const newTime = originalAudioRef.current.currentTime;
              setAudioState(prev => ({ ...prev, currentTime: newTime }));
              onPlaybackStateChange?.({ currentTime: newTime });
            }
          }}
          onEnded={() => {
            const newState = { isPlaying: false, currentTime: 0 };
            setAudioState(prev => ({ ...prev, ...newState }));
            onPlaybackStateChange?.(newState);
          }}
        />
      )}
      {correctedUrl && (
        <audio
          ref={correctedAudioRef}
          src={correctedUrl}
          preload="metadata"
          onLoadedMetadata={() => {
            if (correctedAudioRef.current) {
              const newDuration = correctedAudioRef.current.duration;
              setAudioState(prev => ({ ...prev, duration: newDuration }));
              onPlaybackStateChange?.({ duration: newDuration });
            }
          }}
          onTimeUpdate={() => {
            if (correctedAudioRef.current && (!compareMode || activeAudio === 'corrected')) {
              const newTime = correctedAudioRef.current.currentTime;
              setAudioState(prev => ({ ...prev, currentTime: newTime }));
              onPlaybackStateChange?.({ currentTime: newTime });
            }
          }}
          onEnded={() => {
            const newState = { isPlaying: false, currentTime: 0 };
            setAudioState(prev => ({ ...prev, ...newState }));
            onPlaybackStateChange?.(newState);
          }}
        />
      )}

      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
        <MusicNoteIcon sx={{ color: 'primary.main' }} />
        <Typography variant="h6">
          Audio Playback
        </Typography>
        {compareMode && (
          <Chip label="Compare Mode" color="secondary" size="small" />
        )}
      </Box>

      {/* Error Display */}
      {audioState.error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {audioState.error}
        </Alert>
      )}

      {/* Loading */}
      {audioState.loading && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Loading audio...
          </Typography>
          <LinearProgress />
        </Box>
      )}

      {/* Audio Selection */}
      {!compareMode && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Select Audio:
          </Typography>
          <ButtonGroup variant="outlined" size="small">
            <Button
              onClick={() => setActiveAudio('original')}
              variant={activeAudio === 'original' ? 'contained' : 'outlined'}
              disabled={!hasOriginal}
            >
              Original
            </Button>
            <Button
              onClick={() => setActiveAudio('corrected')}
              variant={activeAudio === 'corrected' ? 'contained' : 'outlined'}
              disabled={!hasCorrected}
            >
              Corrected
            </Button>
          </ButtonGroup>
        </Box>
      )}

      {/* Playback Controls */}
      <Box sx={{ mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <IconButton
                onClick={audioState.isPlaying ? handlePause : handlePlay}
                disabled={audioState.loading}
                size="large"
                color="primary"
              >
                {audioState.isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
              </IconButton>
              <IconButton onClick={handleStop} disabled={audioState.loading}>
                <StopIcon />
              </IconButton>
            </Box>
          </Grid>

          <Grid item xs>
            <Typography variant="caption" color="text.secondary">
              {formatTime(audioState.currentTime)} / {formatTime(audioState.duration)}
            </Typography>
            <Slider
              value={(audioState.currentTime / audioState.duration) * 100 || 0}
              onChange={(_, value) => handleSeek(value as number)}
              disabled={audioState.loading || audioState.duration === 0}
              size="small"
            />
          </Grid>

          <Grid item>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 120 }}>
              <IconButton onClick={toggleMute} size="small">
                {audioState.muted ? <VolumeOffIcon /> : <VolumeUpIcon />}
              </IconButton>
              <Slider
                value={audioState.volume * 100}
                onChange={(_, value) => handleVolumeChange(value as number)}
                size="small"
                sx={{ width: 60 }}
              />
            </Box>
          </Grid>
        </Grid>
      </Box>

      {/* Action Buttons */}
      <Grid container spacing={2}>
        <Grid item>
          <Tooltip title={canCompare ? "Toggle A/B comparison" : "Need both original and corrected audio"}>
            <span>
              <Button
                startIcon={<CompareArrowsIcon />}
                onClick={() => setCompareMode(prev => !prev)}
                variant={compareMode ? "contained" : "outlined"}
                disabled={!canCompare}
              >
                Compare
              </Button>
            </span>
          </Tooltip>
        </Grid>
      </Grid>

      {/* Compare Mode Info */}
      {compareMode && (
        <Box sx={{ mt: 2, p: 2, backgroundColor: 'action.hover', borderRadius: 1 }}>
          <Typography variant="caption" color="text.secondary">
            📊 Compare Mode: Both original and corrected audio will play simultaneously for direct comparison
          </Typography>
        </Box>
      )}
    </Paper>
  );
};