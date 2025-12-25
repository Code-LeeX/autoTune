import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  Chip,
  Grid,
  Alert,
  Divider
} from '@mui/material';

import TuneIcon from '@mui/icons-material/Tune';
import MusicNoteIcon from '@mui/icons-material/MusicNote';
import AllInclusiveIcon from '@mui/icons-material/AllInclusive';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';

export interface ProcessingMode {
  pitchCorrection: boolean;
  mixing: boolean;
  processingOrder: 'pitch_first' | 'mix_first';
}

interface ProcessingModeSelectorProps {
  value: ProcessingMode;
  onChange: (mode: ProcessingMode) => void;
  disabled?: boolean;
  hasAnalyzedAudio?: boolean;
}

const PROCESSING_MODES = {
  pitch_only: {
    id: 'pitch_only',
    name: 'Pitch Correction Only',
    description: 'Apply only pitch correction and musical scale quantization',
    icon: MusicNoteIcon,
    color: 'primary',
    requirements: ['Analyzed audio required'],
    benefits: ['Natural sound preservation', 'Fast processing', 'Minimal artifacts']
  },
  mixing_only: {
    id: 'mixing_only',
    name: 'Mixing Only',
    description: 'Apply professional mixing effects without pitch correction',
    icon: TuneIcon,
    color: 'secondary',
    requirements: ['Uploaded audio'],
    benefits: ['Works with any audio', 'Professional sound', 'Multiple presets']
  },
  full_processing: {
    id: 'full_processing',
    name: 'Complete Processing',
    description: 'Combine pitch correction with professional mixing effects',
    icon: AllInclusiveIcon,
    color: 'success',
    requirements: ['Analyzed audio required'],
    benefits: ['Best quality', 'Radio-ready sound', 'Customizable order']
  }
} as const;

export const ProcessingModeSelector: React.FC<ProcessingModeSelectorProps> = ({
  value,
  onChange,
  disabled = false,
  hasAnalyzedAudio = false
}) => {
  // Determine current mode based on value
  const getCurrentModeId = () => {
    if (value.pitchCorrection && value.mixing) return 'full_processing';
    if (value.pitchCorrection && !value.mixing) return 'pitch_only';
    if (!value.pitchCorrection && value.mixing) return 'mixing_only';
    return 'pitch_only'; // default
  };

  const [selectedMode, setSelectedMode] = useState(getCurrentModeId());

  const handleModeChange = useCallback((modeId: string) => {
    setSelectedMode(modeId);

    let newMode: ProcessingMode;

    switch (modeId) {
      case 'pitch_only':
        newMode = {
          pitchCorrection: true,
          mixing: false,
          processingOrder: 'pitch_first'
        };
        break;
      case 'mixing_only':
        newMode = {
          pitchCorrection: false,
          mixing: true,
          processingOrder: 'pitch_first'
        };
        break;
      case 'full_processing':
        newMode = {
          pitchCorrection: true,
          mixing: true,
          processingOrder: value.processingOrder
        };
        break;
      default:
        return;
    }

    onChange(newMode);
  }, [onChange, value.processingOrder]);

  const handleProcessingOrderChange = useCallback((order: 'pitch_first' | 'mix_first') => {
    onChange({
      ...value,
      processingOrder: order
    });
  }, [onChange, value]);

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" component="h3" gutterBottom>
          Processing Mode
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Choose how you want to process your audio
        </Typography>
      </Box>

      {/* Mode Selection */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {Object.entries(PROCESSING_MODES).map(([modeId, mode]) => {
          const IconComponent = mode.icon;
          const isSelected = selectedMode === modeId;
          const isDisabled = disabled || (modeId !== 'mixing_only' && !hasAnalyzedAudio);

          return (
            <Grid item xs={12} md={4} key={modeId}>
              <Card
                sx={{
                  cursor: isDisabled ? 'not-allowed' : 'pointer',
                  border: isSelected ? 2 : 1,
                  borderColor: isSelected ? `${mode.color}.main` : 'divider',
                  backgroundColor: isSelected ? `${mode.color}.50` : 'background.paper',
                  opacity: isDisabled ? 0.5 : 1,
                  transition: 'all 0.2s',
                  '&:hover': !isDisabled ? {
                    borderColor: `${mode.color}.main`,
                    transform: 'translateY(-2px)',
                    boxShadow: 2
                  } : {}
                }}
                onClick={() => !isDisabled && handleModeChange(modeId)}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <IconComponent sx={{ color: `${mode.color}.main`, fontSize: 24 }} />
                    <Typography variant="h6" component="h4">
                      {mode.name}
                    </Typography>
                  </Box>

                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {mode.description}
                  </Typography>

                  {/* Requirements */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1 }}>
                      Requirements:
                    </Typography>
                    {mode.requirements.map((req, index) => (
                      <Chip
                        key={index}
                        label={req}
                        size="small"
                        color={modeId !== 'mixing_only' && !hasAnalyzedAudio ? 'error' : 'default'}
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                  </Box>

                  {/* Benefits */}
                  <Box>
                    <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1 }}>
                      Benefits:
                    </Typography>
                    {mode.benefits.map((benefit, index) => (
                      <Typography key={index} variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                        • {benefit}
                      </Typography>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Processing Order Selection (only for full processing) */}
      {selectedMode === 'full_processing' && (
        <>
          <Divider sx={{ mb: 3 }} />

          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <SwapHorizIcon />
              Processing Order
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Choose the order in which effects are applied. Different orders can produce different results.
            </Typography>

            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Card
                  sx={{
                    cursor: 'pointer',
                    border: value.processingOrder === 'pitch_first' ? 2 : 1,
                    borderColor: value.processingOrder === 'pitch_first' ? 'primary.main' : 'divider',
                    backgroundColor: value.processingOrder === 'pitch_first' ? 'primary.50' : 'background.paper',
                    '&:hover': {
                      borderColor: 'primary.main'
                    }
                  }}
                  onClick={() => handleProcessingOrderChange('pitch_first')}
                >
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                      <MusicNoteIcon sx={{ color: 'primary.main' }} />
                      <ArrowForwardIcon sx={{ color: 'text.secondary', fontSize: 16 }} />
                      <TuneIcon sx={{ color: 'secondary.main' }} />
                      <Typography variant="h6">
                        Pitch First
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Apply pitch correction first, then mixing effects
                    </Typography>
                    <Typography variant="caption" color="primary.main">
                      ✓ Recommended for most vocals
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card
                  sx={{
                    cursor: 'pointer',
                    border: value.processingOrder === 'mix_first' ? 2 : 1,
                    borderColor: value.processingOrder === 'mix_first' ? 'secondary.main' : 'divider',
                    backgroundColor: value.processingOrder === 'mix_first' ? 'secondary.50' : 'background.paper',
                    '&:hover': {
                      borderColor: 'secondary.main'
                    }
                  }}
                  onClick={() => handleProcessingOrderChange('mix_first')}
                >
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                      <TuneIcon sx={{ color: 'secondary.main' }} />
                      <ArrowForwardIcon sx={{ color: 'text.secondary', fontSize: 16 }} />
                      <MusicNoteIcon sx={{ color: 'primary.main' }} />
                      <Typography variant="h6">
                        Mixing First
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Apply mixing effects first, then pitch correction
                    </Typography>
                    <Typography variant="caption" color="secondary.main">
                      ⚡ Experimental processing order
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        </>
      )}

      {/* Status Alert */}
      {!hasAnalyzedAudio && selectedMode !== 'mixing_only' && (
        <Alert severity="warning">
          <Typography variant="body2">
            Pitch correction requires analyzed audio. Please upload and analyze your audio first.
          </Typography>
        </Alert>
      )}

      {selectedMode === 'mixing_only' && (
        <Alert severity="info">
          <Typography variant="body2">
            Mixing mode works with any uploaded audio and doesn't require pitch analysis.
          </Typography>
        </Alert>
      )}

      {selectedMode === 'full_processing' && hasAnalyzedAudio && (
        <Alert severity="success">
          <Typography variant="body2">
            Ready for complete processing! This will give you the best quality results.
          </Typography>
        </Alert>
      )}
    </Paper>
  );
};