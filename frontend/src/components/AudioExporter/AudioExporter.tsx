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
  Grid,
  Alert,
  CircularProgress,
  Chip,
  Card,
  CardContent,
  LinearProgress,
  Tooltip
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import AudioFileIcon from '@mui/icons-material/AudioFile';
import HighQualityIcon from '@mui/icons-material/HighQuality';
import CompressIcon from '@mui/icons-material/Compress';

import { AudioAPI, AudioExportFormat } from '../../api/audioApi';

interface AudioExporterProps {
  sessionId?: string;
  hasOriginal?: boolean;
  hasCorrected?: boolean;
  hasMixed?: boolean;
  hasProcessed?: boolean;
  onError?: (error: string) => void;
}

interface ExportState {
  formats: AudioExportFormat[];
  selectedFormat: string;
  fileType: 'original' | 'corrected' | 'mixed' | 'processed';
  isLoading: boolean;
  isDownloading: boolean;
  error?: string;
}

export const AudioExporter: React.FC<AudioExporterProps> = ({
  sessionId,
  hasOriginal = false,
  hasCorrected = false,
  hasMixed = false,
  hasProcessed = false,
  onError
}) => {
  const [exportState, setExportState] = useState<ExportState>({
    formats: [],
    selectedFormat: 'wav',
    fileType: 'corrected',
    isLoading: false,
    isDownloading: false
  });

  // Load available formats on mount
  useEffect(() => {
    loadAvailableFormats();
  }, []);

  const loadAvailableFormats = useCallback(async () => {
    try {
      setExportState(prev => ({ ...prev, isLoading: true, error: undefined }));
      const formats = await AudioAPI.getAvailableFormats();
      setExportState(prev => ({
        ...prev,
        formats,
        isLoading: false
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load formats';
      setExportState(prev => ({ ...prev, error: errorMessage, isLoading: false }));
      onError?.(errorMessage);
    }
  }, [onError]);

  const handleDownload = async () => {
    if (!sessionId) {
      const error = 'No session available for download';
      setExportState(prev => ({ ...prev, error }));
      onError?.(error);
      return;
    }

    try {
      setExportState(prev => ({ ...prev, isDownloading: true, error: undefined }));

      // Get the audio blob
      const blob = await AudioAPI.downloadAudio(sessionId, exportState.fileType, exportState.selectedFormat);

      // Create download link
      const url = URL.createObjectURL(blob);
      const format = exportState.formats.find(f => f.id === exportState.selectedFormat);
      const filename = `${exportState.fileType}_${sessionId}${format?.extension || '.wav'}`;

      // Trigger download
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Clean up
      URL.revokeObjectURL(url);

      setExportState(prev => ({ ...prev, isDownloading: false }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Download failed';
      setExportState(prev => ({ ...prev, error: errorMessage, isDownloading: false }));
      onError?.(errorMessage);
    }
  };

  const selectedFormatInfo = exportState.formats.find(f => f.id === exportState.selectedFormat);
  const canDownload = sessionId && ((hasOriginal && exportState.fileType === 'original') ||
                                   (hasCorrected && exportState.fileType === 'corrected') ||
                                   (hasMixed && exportState.fileType === 'mixed') ||
                                   (hasProcessed && exportState.fileType === 'processed'));

  if (!hasOriginal && !hasCorrected && !hasMixed && !hasProcessed) {
    return (
      <Paper elevation={2} sx={{ p: 3, textAlign: 'center' }}>
        <AudioFileIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          No audio available for export
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Upload and process audio to enable export functionality
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
        <FileDownloadIcon sx={{ color: 'primary.main' }} />
        <Typography variant="h6">
          Audio Export
        </Typography>
      </Box>

      {/* Error Display */}
      {exportState.error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {exportState.error}
        </Alert>
      )}

      {/* Loading */}
      {exportState.isLoading && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Loading export formats...
          </Typography>
          <LinearProgress />
        </Box>
      )}

      {/* Export Configuration */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* File Type Selection */}
        <Grid item xs={12} md={6}>
          <FormControl fullWidth disabled={exportState.isDownloading}>
            <InputLabel>Audio Type</InputLabel>
            <Select
              value={exportState.fileType}
              label="Audio Type"
              onChange={(e) => setExportState(prev => ({
                ...prev,
                fileType: e.target.value as 'original' | 'corrected' | 'mixed' | 'processed'
              }))}
            >
              {hasOriginal && (
                <MenuItem value="original">Original Audio</MenuItem>
              )}
              {hasCorrected && (
                <MenuItem value="corrected">Corrected Audio</MenuItem>
              )}
              {hasMixed && (
                <MenuItem value="mixed">Mixed Audio</MenuItem>
              )}
              {hasProcessed && (
                <MenuItem value="processed">Processed Audio</MenuItem>
              )}
            </Select>
          </FormControl>
        </Grid>

        {/* Format Selection */}
        <Grid item xs={12} md={6}>
          <FormControl fullWidth disabled={exportState.isDownloading || exportState.isLoading}>
            <InputLabel>Export Format</InputLabel>
            <Select
              value={exportState.selectedFormat}
              label="Export Format"
              onChange={(e) => setExportState(prev => ({
                ...prev,
                selectedFormat: e.target.value
              }))}
            >
              {exportState.formats.map((format) => (
                <MenuItem key={format.id} value={format.id}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {format.is_lossless ? <HighQualityIcon fontSize="small" /> : <CompressIcon fontSize="small" />}
                    <Box>
                      <Typography variant="body2">
                        {format.name} ({format.extension})
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
      </Grid>

      {/* Format Information */}
      {selectedFormatInfo && (
        <Card variant="outlined" sx={{ mb: 3 }}>
          <CardContent>
            <Grid container spacing={2} alignItems="center">
              <Grid item>
                {selectedFormatInfo.is_lossless ? (
                  <HighQualityIcon sx={{ fontSize: 40, color: 'success.main' }} />
                ) : (
                  <CompressIcon sx={{ fontSize: 40, color: 'info.main' }} />
                )}
              </Grid>
              <Grid item xs>
                <Typography variant="h6" gutterBottom>
                  {selectedFormatInfo.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {selectedFormatInfo.description}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                  <Chip
                    size="small"
                    label={selectedFormatInfo.is_lossless ? "Lossless" : "Compressed"}
                    color={selectedFormatInfo.is_lossless ? "success" : "info"}
                    variant="outlined"
                  />
                  <Chip
                    size="small"
                    label={selectedFormatInfo.extension}
                    variant="outlined"
                  />
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Download Progress */}
      {exportState.isDownloading && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Preparing download...
          </Typography>
          <LinearProgress />
        </Box>
      )}

      {/* Download Button */}
      <Box sx={{ textAlign: 'center' }}>
        <Tooltip
          title={
            !canDownload
              ? `No ${exportState.fileType} audio available`
              : `Download ${exportState.fileType} audio as ${selectedFormatInfo?.name || exportState.selectedFormat.toUpperCase()}`
          }
        >
          <span>
            <Button
              variant="contained"
              size="large"
              startIcon={
                exportState.isDownloading ? (
                  <CircularProgress size={16} color="inherit" />
                ) : (
                  <DownloadIcon />
                )
              }
              onClick={handleDownload}
              disabled={!canDownload || exportState.isDownloading || exportState.isLoading}
              sx={{ minWidth: 200 }}
            >
              {exportState.isDownloading ? 'Preparing...' : 'Download Audio'}
            </Button>
          </span>
        </Tooltip>
      </Box>

      {/* Help Text */}
      <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block', textAlign: 'center' }}>
        Select your preferred format and click download to export the processed audio
      </Typography>
    </Paper>
  );
};