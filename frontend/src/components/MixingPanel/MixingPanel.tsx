import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Slider,
  Grid,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
  Card,
  CardContent
} from '@mui/material';

import { AudioAPI, MixingParams, MixingPreset } from '../../api/audioApi';

import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import TuneIcon from '@mui/icons-material/Tune';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import GraphicEqIcon from '@mui/icons-material/GraphicEq';
import AudiotrackIcon from '@mui/icons-material/Audiotrack';
import SettingsIcon from '@mui/icons-material/Settings';
import PresetIcon from '@mui/icons-material/LibraryMusic';
import RestoreIcon from '@mui/icons-material/Restore';

interface ProcessingMode {
  pitchCorrection: boolean;
  mixing: boolean;
  processingOrder: 'pitch_first' | 'mix_first';
}

interface MixingPanelProps {
  sessionId?: string;
  processingMode?: ProcessingMode;
  onMixingComplete?: (sessionId: string, result: any) => void;
  onFullProcessingComplete?: (sessionId: string, result: any) => void;
  onError?: (error: string) => void;
  disabled?: boolean;
  initialParams?: Partial<MixingParams>;
}

const DEFAULT_MIXING_PARAMS: MixingParams = {
  noise_gate_enabled: true,
  noise_gate_threshold_db: -35.0,
  noise_gate_ratio: 8.0,
  noise_gate_attack_ms: 1.0,
  noise_gate_release_ms: 150.0,
  highpass_enabled: true,
  highpass_frequency_hz: 80.0,
  compressor_enabled: true,
  compressor_threshold_db: -18.0,
  compressor_ratio: 3.0,
  compressor_attack_ms: 10.0,
  compressor_release_ms: 100.0,
  eq_enabled: true,
  eq_low_enabled: true,
  eq_low_frequency_hz: 150.0,
  eq_low_gain_db: 1.0,
  eq_low_q: 1.0,
  eq_low_mid_enabled: true,
  eq_low_mid_frequency_hz: 400.0,
  eq_low_mid_gain_db: 1.5,
  eq_low_mid_q: 1.2,
  eq_presence_frequency_hz: 2800.0,
  eq_presence_gain_db: 1.5,
  eq_presence_q: 1.2,
  eq_high_enabled: true,
  eq_high_frequency_hz: 8000.0,
  eq_high_gain_db: 0.8,
  eq_high_q: 0.7,
  reverb_enabled: true,
  reverb_type: "algorithm",
  reverb_room_size: 0.2,
  reverb_damping: 0.6,
  reverb_wet_level: 0.12,
  reverb_width: 1.0
};

export const MixingPanel: React.FC<MixingPanelProps> = ({
  sessionId,
  processingMode,
  onMixingComplete,
  onFullProcessingComplete,
  onError,
  disabled = false,
  initialParams = {}
}) => {
  const [mixingParams, setMixingParams] = useState<MixingParams>({
    ...DEFAULT_MIXING_PARAMS,
    ...initialParams
  });

  const [presets, setPresets] = useState<MixingPreset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    noise_gate: true,
    highpass: false,
    compressor: false,
    eq: false,
    reverb: false
  });

  // Load mixing presets
  useEffect(() => {
    loadPresets();
  }, []);

  const loadPresets = async () => {
    try {
      const response = await fetch('/api/audio/mixing/presets');
      if (response.ok) {
        const data = await response.json();
        setPresets(data.presets);
      }
    } catch (error) {
      console.error('Failed to load presets:', error);
    }
  };

  const handleParameterChange = useCallback((section: string, parameter: string, value: any) => {
    setMixingParams(prev => ({
      ...prev,
      [`${section}_${parameter}`]: value
    }));
    // Clear selected preset when manually changing parameters
    if (selectedPreset) {
      setSelectedPreset('');
    }
  }, [selectedPreset]);

  const handlePresetChange = useCallback((presetId: string) => {
    if (!presetId) return;

    const preset = presets.find(p => p.id === presetId);
    if (preset) {
      setMixingParams(preset.params);
      setSelectedPreset(presetId);
    }
  }, [presets]);

  const handleSectionToggle = useCallback((section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  }, []);

  const resetToDefaults = useCallback(() => {
    setMixingParams(DEFAULT_MIXING_PARAMS);
    setSelectedPreset('');
  }, []);

  const applyMixing = async () => {
    if (!sessionId) return;

    setLoading(true);
    setError(null);

    try {
      const result = await AudioAPI.mixAudio(sessionId, mixingParams);
      onMixingComplete?.(sessionId!, result);
    } catch (error) {
      console.error('Mixing error:', error);
      setError('Failed to apply mixing effects');
    } finally {
      setLoading(false);
    }
  };

  const formatDbValue = (value: number) => `${value.toFixed(1)} dB`;
  const formatHzValue = (value: number) => `${value.toFixed(0)} Hz`;
  const formatMsValue = (value: number) => `${value.toFixed(1)} ms`;
  const formatRatioValue = (value: number) => `${value.toFixed(1)}:1`;
  const formatPercentValue = (value: number) => `${(value * 100).toFixed(0)}%`;

  return (
    <Paper elevation={2} sx={{ p: 3, maxWidth: 800, mx: 'auto' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <TuneIcon sx={{ color: 'primary.main', fontSize: 32 }} />
        <Typography variant="h5" component="h2" sx={{ flexGrow: 1 }}>
          Professional Audio Mixing
        </Typography>
        <Tooltip title="Reset to defaults">
          <IconButton onClick={resetToDefaults} disabled={disabled}>
            <RestoreIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Preset Selection */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <PresetIcon sx={{ color: 'secondary.main' }} />
            <Typography variant="h6">Quick Presets</Typography>
          </Box>

          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={8}>
              <FormControl fullWidth size="small">
                <InputLabel>Select Preset</InputLabel>
                <Select
                  value={selectedPreset}
                  label="Select Preset"
                  onChange={(e) => handlePresetChange(e.target.value)}
                  disabled={disabled}
                >
                  <MenuItem value="">
                    <em>Custom Settings</em>
                  </MenuItem>
                  {presets.map((preset) => (
                    <MenuItem key={preset.id} value={preset.id}>
                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                          {preset.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {preset.description}
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              {selectedPreset && (
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {presets.find(p => p.id === selectedPreset)?.use_cases.slice(0, 2).map(useCase => (
                    <Chip key={useCase} label={useCase.replace('_', ' ')} size="small" />
                  ))}
                </Box>
              )}
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Loading */}
      {loading && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Applying mixing effects...
          </Typography>
          <LinearProgress />
        </Box>
      )}

      {/* Mixing Controls */}
      <Box sx={{ mb: 3 }}>
        {/* Noise Gate */}
        <Accordion
          expanded={expandedSections.noise_gate}
          onChange={() => handleSectionToggle('noise_gate')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
              <VolumeUpIcon sx={{ color: 'primary.main' }} />
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Noise Gate / De-breath
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={mixingParams.noise_gate_enabled}
                    onChange={(e) => handleParameterChange('noise_gate', 'enabled', e.target.checked)}
                    disabled={disabled}
                    onClick={(e) => e.stopPropagation()}
                  />
                }
                label=""
                sx={{ mr: 2 }}
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Threshold: {formatDbValue(mixingParams.noise_gate_threshold_db)}</Typography>
                <Slider
                  value={mixingParams.noise_gate_threshold_db}
                  onChange={(_, value) => handleParameterChange('noise_gate', 'threshold_db', value)}
                  min={-60}
                  max={0}
                  step={1}
                  disabled={disabled || !mixingParams.noise_gate_enabled}
                  marks={[
                    { value: -60, label: '-60dB' },
                    { value: -30, label: '-30dB' },
                    { value: 0, label: '0dB' }
                  ]}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Ratio: {formatRatioValue(mixingParams.noise_gate_ratio)}</Typography>
                <Slider
                  value={mixingParams.noise_gate_ratio}
                  onChange={(_, value) => handleParameterChange('noise_gate', 'ratio', value)}
                  min={1}
                  max={20}
                  step={0.1}
                  disabled={disabled || !mixingParams.noise_gate_enabled}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Attack: {formatMsValue(mixingParams.noise_gate_attack_ms)}</Typography>
                <Slider
                  value={mixingParams.noise_gate_attack_ms}
                  onChange={(_, value) => handleParameterChange('noise_gate', 'attack_ms', value)}
                  min={0.1}
                  max={100}
                  step={0.1}
                  disabled={disabled || !mixingParams.noise_gate_enabled}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Release: {formatMsValue(mixingParams.noise_gate_release_ms)}</Typography>
                <Slider
                  value={mixingParams.noise_gate_release_ms}
                  onChange={(_, value) => handleParameterChange('noise_gate', 'release_ms', value)}
                  min={10}
                  max={1000}
                  step={1}
                  disabled={disabled || !mixingParams.noise_gate_enabled}
                />
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* High-pass Filter */}
        <Accordion
          expanded={expandedSections.highpass}
          onChange={() => handleSectionToggle('highpass')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
              <GraphicEqIcon sx={{ color: 'warning.main' }} />
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                High-pass Filter (Subtractive EQ)
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={mixingParams.highpass_enabled}
                    onChange={(e) => handleParameterChange('highpass', 'enabled', e.target.checked)}
                    disabled={disabled}
                    onClick={(e) => e.stopPropagation()}
                  />
                }
                label=""
                sx={{ mr: 2 }}
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Typography gutterBottom>Cutoff Frequency: {formatHzValue(mixingParams.highpass_frequency_hz)}</Typography>
                <Slider
                  value={mixingParams.highpass_frequency_hz}
                  onChange={(_, value) => handleParameterChange('highpass', 'frequency_hz', value)}
                  min={20}
                  max={500}
                  step={1}
                  disabled={disabled || !mixingParams.highpass_enabled}
                  marks={[
                    { value: 20, label: '20Hz' },
                    { value: 80, label: '80Hz' },
                    { value: 150, label: '150Hz' },
                    { value: 500, label: '500Hz' }
                  ]}
                />
                <Typography variant="caption" color="text.secondary">
                  Removes low-frequency rumble and unwanted bass content
                </Typography>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* Compressor */}
        <Accordion
          expanded={expandedSections.compressor}
          onChange={() => handleSectionToggle('compressor')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
              <SettingsIcon sx={{ color: 'info.main' }} />
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Compressor (Dynamic Control)
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={mixingParams.compressor_enabled}
                    onChange={(e) => handleParameterChange('compressor', 'enabled', e.target.checked)}
                    disabled={disabled}
                    onClick={(e) => e.stopPropagation()}
                  />
                }
                label=""
                sx={{ mr: 2 }}
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Threshold: {formatDbValue(mixingParams.compressor_threshold_db)}</Typography>
                <Slider
                  value={mixingParams.compressor_threshold_db}
                  onChange={(_, value) => handleParameterChange('compressor', 'threshold_db', value)}
                  min={-40}
                  max={0}
                  step={1}
                  disabled={disabled || !mixingParams.compressor_enabled}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Ratio: {formatRatioValue(mixingParams.compressor_ratio)}</Typography>
                <Slider
                  value={mixingParams.compressor_ratio}
                  onChange={(_, value) => handleParameterChange('compressor', 'ratio', value)}
                  min={1}
                  max={10}
                  step={0.1}
                  disabled={disabled || !mixingParams.compressor_enabled}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Attack: {formatMsValue(mixingParams.compressor_attack_ms)}</Typography>
                <Slider
                  value={mixingParams.compressor_attack_ms}
                  onChange={(_, value) => handleParameterChange('compressor', 'attack_ms', value)}
                  min={0.1}
                  max={100}
                  step={0.1}
                  disabled={disabled || !mixingParams.compressor_enabled}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Release: {formatMsValue(mixingParams.compressor_release_ms)}</Typography>
                <Slider
                  value={mixingParams.compressor_release_ms}
                  onChange={(_, value) => handleParameterChange('compressor', 'release_ms', value)}
                  min={10}
                  max={1000}
                  step={1}
                  disabled={disabled || !mixingParams.compressor_enabled}
                />
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* EQ */}
        <Accordion
          expanded={expandedSections.eq}
          onChange={() => handleSectionToggle('eq')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
              <AudiotrackIcon sx={{ color: 'success.main' }} />
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Additive EQ (Presence & Air)
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={mixingParams.eq_enabled}
                    onChange={(e) => handleParameterChange('eq', 'enabled', e.target.checked)}
                    disabled={disabled}
                    onClick={(e) => e.stopPropagation()}
                  />
                }
                label=""
                sx={{ mr: 2 }}
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Multi-band EQ for richer, fuller sound processing
            </Typography>

            {/* Low Frequency EQ (100-200Hz) - Body/Warmth */}
            <Box sx={{ mb: 3, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="subtitle2" sx={{ flex: 1 }}>
                  Low EQ (100-200Hz) - Body & Warmth
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      size="small"
                      checked={mixingParams.eq_low_enabled}
                      onChange={(e) => handleParameterChange('eq_low', 'enabled', e.target.checked)}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  }
                  label=""
                />
              </Box>
              {mixingParams.eq_low_enabled && (
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <Typography gutterBottom>Frequency: {formatHzValue(mixingParams.eq_low_frequency_hz)}</Typography>
                    <Slider
                      value={mixingParams.eq_low_frequency_hz}
                      onChange={(_, value) => handleParameterChange('eq_low', 'frequency_hz', value)}
                      min={80}
                      max={250}
                      step={10}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography gutterBottom>Gain: {formatDbValue(mixingParams.eq_low_gain_db)}</Typography>
                    <Slider
                      value={mixingParams.eq_low_gain_db}
                      onChange={(_, value) => handleParameterChange('eq_low', 'gain_db', value)}
                      min={-6}
                      max={6}
                      step={0.1}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography gutterBottom>Q Factor: {mixingParams.eq_low_q.toFixed(1)}</Typography>
                    <Slider
                      value={mixingParams.eq_low_q}
                      onChange={(_, value) => handleParameterChange('eq_low', 'q', value)}
                      min={0.5}
                      max={3.0}
                      step={0.1}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  </Grid>
                </Grid>
              )}
            </Box>

            {/* Low-Mid Frequency EQ (300-500Hz) - Fullness */}
            <Box sx={{ mb: 3, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="subtitle2" sx={{ flex: 1 }}>
                  Low-Mid EQ (300-500Hz) - Fullness
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      size="small"
                      checked={mixingParams.eq_low_mid_enabled}
                      onChange={(e) => handleParameterChange('eq_low_mid', 'enabled', e.target.checked)}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  }
                  label=""
                />
              </Box>
              {mixingParams.eq_low_mid_enabled && (
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <Typography gutterBottom>Frequency: {formatHzValue(mixingParams.eq_low_mid_frequency_hz)}</Typography>
                    <Slider
                      value={mixingParams.eq_low_mid_frequency_hz}
                      onChange={(_, value) => handleParameterChange('eq_low_mid', 'frequency_hz', value)}
                      min={250}
                      max={600}
                      step={25}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography gutterBottom>Gain: {formatDbValue(mixingParams.eq_low_mid_gain_db)}</Typography>
                    <Slider
                      value={mixingParams.eq_low_mid_gain_db}
                      onChange={(_, value) => handleParameterChange('eq_low_mid', 'gain_db', value)}
                      min={-6}
                      max={6}
                      step={0.1}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography gutterBottom>Q Factor: {mixingParams.eq_low_mid_q.toFixed(1)}</Typography>
                    <Slider
                      value={mixingParams.eq_low_mid_q}
                      onChange={(_, value) => handleParameterChange('eq_low_mid', 'q', value)}
                      min={0.5}
                      max={3.0}
                      step={0.1}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  </Grid>
                </Grid>
              )}
            </Box>

            {/* Presence EQ (2-4kHz) - Clarity */}
            <Box sx={{ mb: 3, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ mb: 2 }}>
                Presence EQ (2-4kHz) - Clarity (Always Enabled)
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <Typography gutterBottom>Frequency: {formatHzValue(mixingParams.eq_presence_frequency_hz)}</Typography>
                  <Slider
                    value={mixingParams.eq_presence_frequency_hz}
                    onChange={(_, value) => handleParameterChange('eq_presence', 'frequency_hz', value)}
                    min={1500}
                    max={5000}
                    step={100}
                    disabled={disabled || !mixingParams.eq_enabled}
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Typography gutterBottom>Gain: {formatDbValue(mixingParams.eq_presence_gain_db)}</Typography>
                  <Slider
                    value={mixingParams.eq_presence_gain_db}
                    onChange={(_, value) => handleParameterChange('eq_presence', 'gain_db', value)}
                    min={-6}
                    max={6}
                    step={0.1}
                    disabled={disabled || !mixingParams.eq_enabled}
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Typography gutterBottom>Q Factor: {mixingParams.eq_presence_q.toFixed(1)}</Typography>
                  <Slider
                    value={mixingParams.eq_presence_q}
                    onChange={(_, value) => handleParameterChange('eq_presence', 'q', value)}
                    min={0.5}
                    max={5.0}
                    step={0.1}
                    disabled={disabled || !mixingParams.eq_enabled}
                  />
                </Grid>
              </Grid>
            </Box>

            {/* High Frequency EQ (8kHz+) - Air/Sparkle */}
            <Box sx={{ mb: 3, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="subtitle2" sx={{ flex: 1 }}>
                  High EQ (8kHz+) - Air & Sparkle
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      size="small"
                      checked={mixingParams.eq_high_enabled}
                      onChange={(e) => handleParameterChange('eq_high', 'enabled', e.target.checked)}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  }
                  label=""
                />
              </Box>
              {mixingParams.eq_high_enabled && (
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <Typography gutterBottom>Frequency: {formatHzValue(mixingParams.eq_high_frequency_hz)}</Typography>
                    <Slider
                      value={mixingParams.eq_high_frequency_hz}
                      onChange={(_, value) => handleParameterChange('eq_high', 'frequency_hz', value)}
                      min={6000}
                      max={15000}
                      step={500}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography gutterBottom>Gain: {formatDbValue(mixingParams.eq_high_gain_db)}</Typography>
                    <Slider
                      value={mixingParams.eq_high_gain_db}
                      onChange={(_, value) => handleParameterChange('eq_high', 'gain_db', value)}
                      min={-6}
                      max={6}
                      step={0.1}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography gutterBottom>Q Factor: {mixingParams.eq_high_q.toFixed(1)}</Typography>
                    <Slider
                      value={mixingParams.eq_high_q}
                      onChange={(_, value) => handleParameterChange('eq_high', 'q', value)}
                      min={0.3}
                      max={2.0}
                      step={0.1}
                      disabled={disabled || !mixingParams.eq_enabled}
                    />
                  </Grid>
                </Grid>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Reverb */}
        <Accordion
          expanded={expandedSections.reverb}
          onChange={() => handleSectionToggle('reverb')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
              <AudiotrackIcon sx={{ color: 'secondary.main' }} />
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Reverb (Space & Depth)
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={mixingParams.reverb_enabled}
                    onChange={(e) => handleParameterChange('reverb', 'enabled', e.target.checked)}
                    disabled={disabled}
                    onClick={(e) => e.stopPropagation()}
                  />
                }
                label=""
                sx={{ mr: 2 }}
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Room Size: {formatPercentValue(mixingParams.reverb_room_size)}</Typography>
                <Slider
                  value={mixingParams.reverb_room_size}
                  onChange={(_, value) => handleParameterChange('reverb', 'room_size', value)}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={disabled || !mixingParams.reverb_enabled}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Damping: {formatPercentValue(mixingParams.reverb_damping)}</Typography>
                <Slider
                  value={mixingParams.reverb_damping}
                  onChange={(_, value) => handleParameterChange('reverb', 'damping', value)}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={disabled || !mixingParams.reverb_enabled}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Wet Level: {formatPercentValue(mixingParams.reverb_wet_level)}</Typography>
                <Slider
                  value={mixingParams.reverb_wet_level}
                  onChange={(_, value) => handleParameterChange('reverb', 'wet_level', value)}
                  min={0}
                  max={0.5}
                  step={0.01}
                  disabled={disabled || !mixingParams.reverb_enabled}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Width: {formatPercentValue(mixingParams.reverb_width)}</Typography>
                <Slider
                  value={mixingParams.reverb_width}
                  onChange={(_, value) => handleParameterChange('reverb', 'width', value)}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={disabled || !mixingParams.reverb_enabled}
                />
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>
      </Box>

      {/* Action Buttons */}
      <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
        <Button
          variant="contained"
          size="large"
          onClick={applyMixing}
          disabled={disabled || loading || !sessionId}
          startIcon={<TuneIcon />}
          sx={{ minWidth: 200 }}
        >
          {loading ? 'Processing...' : 'Apply Mixing Effects'}
        </Button>
      </Box>

      {/* Processing Info */}
      <Box sx={{ mt: 3, p: 2, backgroundColor: 'action.hover', borderRadius: 1 }}>
        <Typography variant="caption" color="text.secondary">
          🎚️ Professional mixing chain: {[
            mixingParams.noise_gate_enabled && 'Noise Gate',
            mixingParams.highpass_enabled && 'High-pass Filter',
            mixingParams.compressor_enabled && 'Compressor',
            mixingParams.eq_enabled && 'Presence EQ',
            mixingParams.reverb_enabled && 'Reverb'
          ].filter(Boolean).join(' → ') || 'No effects enabled'}
        </Typography>
      </Box>
    </Paper>
  );
};