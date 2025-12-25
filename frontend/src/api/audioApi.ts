/**
 * API client for audio processing backend
 */

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export interface AudioAnalysisParams {
  confidence_threshold: number;
  analyze_vibrato: boolean;
}

export interface PitchCorrectionParams {
  session_id: string;
  key: string;
  scale_type: string;
  correction_strength: number;
  preserve_vibrato: boolean;
  preserve_formants: boolean;
  smoothing_factor: number;
}

export interface AudioUploadResponse {
  session_id: string;
  status: string;
  audio_info: {
    duration: number;
    sample_rate: number;
    channels: number;
    samples: number;
    bit_depth: number;
    file_size: number;
  };
  message: string;
}

export interface ProcessingStatus {
  session_id: string;
  status: 'uploaded' | 'analyzing' | 'analyzed' | 'correcting' | 'corrected' | 'mixing' | 'mixed' | 'processing' | 'processed' | 'error';
  progress: number;
  message: string;
  data?: any;
}

export interface PitchAnalysisData {
  time: number[];
  frequency: number[];
  confidence: number[];
  voiced_mask: boolean[];
  stats: {
    total_frames: number;
    voiced_frames: number;
    voicing_ratio: number;
    avg_confidence: number;
    pitch_range: {
      min: number;
      max: number;
      mean: number;
      std: number;
    };
    vibrato?: {
      frames_with_vibrato: number;
      vibrato_ratio: number;
      avg_vibrato_rate: number;
      avg_vibrato_extent: number;
    };
  };
  vibrato_presence?: boolean[];
  vibrato_rate?: number[];
  vibrato_extent?: number[];
}

export interface CorrectionResult {
  correction_stats: {
    frames_corrected: number;
    frames_with_vibrato_preserved: number;
    correction_ratio: number;
    pitch_accuracy_improvement_cents?: number;
    avg_original_deviation_cents?: number;
    avg_corrected_deviation_cents?: number;
  };
  quantized_frequency: number[];
  correction_applied: boolean[];
  vibrato_preserved: boolean[];
  pitch_deviation: number[];
}

export interface AvailableScales {
  keys: string[];
  scale_types: string[];
}

export interface AudioExportFormat {
  id: string;
  name: string;
  description: string;
  extension: string;
  mime_type: string;
  is_lossless: boolean;
}

// Mixing functionality interfaces
export interface MixingParams {
  // Noise Gate / De-breath
  noise_gate_enabled: boolean;
  noise_gate_threshold_db: number;
  noise_gate_ratio: number;
  noise_gate_attack_ms: number;
  noise_gate_release_ms: number;

  // High-pass filter (Subtractive EQ)
  highpass_enabled: boolean;
  highpass_frequency_hz: number;

  // Compressor (Dynamic control)
  compressor_enabled: boolean;
  compressor_threshold_db: number;
  compressor_ratio: number;
  compressor_attack_ms: number;
  compressor_release_ms: number;

  // EQ - Multi-band for fuller sound
  eq_enabled: boolean;

  // Low-frequency EQ (Warmth/Body) - 100-200Hz
  eq_low_enabled: boolean;
  eq_low_frequency_hz: number;
  eq_low_gain_db: number;
  eq_low_q: number;

  // Low-mid EQ (Fullness) - 300-500Hz
  eq_low_mid_enabled: boolean;
  eq_low_mid_frequency_hz: number;
  eq_low_mid_gain_db: number;
  eq_low_mid_q: number;

  // Presence EQ (Clarity) - 2-4kHz
  eq_presence_frequency_hz: number;
  eq_presence_gain_db: number;
  eq_presence_q: number;

  // High-frequency shelf EQ (Air) - 8kHz+
  eq_high_enabled: boolean;
  eq_high_frequency_hz: number;
  eq_high_gain_db: number;
  eq_high_q: number;

  // Reverb
  reverb_enabled: boolean;
  reverb_type: string;
  reverb_room_size: number;
  reverb_damping: number;
  reverb_wet_level: number;
  reverb_width: number;
}

export interface PresetMetadata {
  name: string;
  description: string;
  category: 'vocal' | 'speech' | 'creative';
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  use_cases: string[];
  tags: string[];
}

export interface MixingPreset {
  id: string;
  name: string;
  description: string;
  category: string;
  difficulty: string;
  use_cases: string[];
  params: MixingParams;
}

export interface MixingRequest {
  session_id: string;
  params: MixingParams;
}

export interface FullProcessRequest {
  session_id: string;
  pitch_params: Omit<PitchCorrectionParams, 'session_id'>;
  mixing_params: MixingParams;
  processing_order: 'pitch_first' | 'mix_first';
}

export interface ProcessingMode {
  pitchCorrection: boolean;
  mixing: boolean;
  processingOrder: 'pitch_first' | 'mix_first';
}

export class AudioAPI {
  /**
   * Upload audio file for processing
   */
  static async uploadAudio(
    file: File,
    analysisParams: AudioAnalysisParams = { confidence_threshold: 0.85, analyze_vibrato: true }
  ): Promise<AudioUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('analysis_request', JSON.stringify(analysisParams));

    const response = await fetch(`${API_BASE_URL}/api/audio/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Upload failed: ${errorData.detail || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Start pitch analysis on uploaded audio
   */
  static async startAnalysis(sessionId: string): Promise<ProcessingStatus> {
    const response = await fetch(`${API_BASE_URL}/api/audio/analyze/${sessionId}`, {
      method: 'POST',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Analysis failed: ${errorData.detail || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Apply pitch correction to analyzed audio
   */
  static async correctPitch(
    sessionId: string,
    params: Omit<PitchCorrectionParams, 'session_id'>
  ): Promise<ProcessingStatus> {
    const response = await fetch(`${API_BASE_URL}/api/audio/correct/${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
        ...params,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Correction failed: ${errorData.detail || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get processing status
   */
  static async getStatus(sessionId: string): Promise<ProcessingStatus> {
    const response = await fetch(`${API_BASE_URL}/api/audio/status/${sessionId}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Status check failed: ${errorData.detail || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Download processed audio file in specified format
   */
  static async downloadAudio(
    sessionId: string,
    fileType: 'original' | 'corrected' | 'mixed' | 'processed' = 'corrected',
    format: string = 'wav'
  ): Promise<Blob> {
    const url = `${API_BASE_URL}/api/audio/download/${sessionId}/${fileType}?format=${encodeURIComponent(format)}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Download failed: ${response.statusText}`);
    }

    return response.blob();
  }

  /**
   * Get available export formats
   */
  static async getAvailableFormats(): Promise<AudioExportFormat[]> {
    const response = await fetch(`${API_BASE_URL}/api/audio/formats`);

    if (!response.ok) {
      throw new Error(`Failed to get formats: ${response.statusText}`);
    }

    const data = await response.json();
    return data.formats;
  }

  /**
   * Get available musical scales
   */
  static async getAvailableScales(): Promise<AvailableScales> {
    const response = await fetch(`${API_BASE_URL}/api/audio/scales`);

    if (!response.ok) {
      throw new Error(`Failed to get scales: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Clean up session data
   */
  static async deleteSession(sessionId: string): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/api/audio/session/${sessionId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error(`Cleanup failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Health check
   */
  static async healthCheck(): Promise<{ status: string; service: string; version: string }> {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Apply mixing effects to uploaded audio
   */
  static async mixAudio(
    sessionId: string,
    params: MixingParams
  ): Promise<ProcessingStatus> {
    const response = await fetch(`${API_BASE_URL}/api/audio/mix/${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
        params: params,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Mixing failed: ${errorData.detail || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Apply full processing (pitch correction + mixing) to analyzed audio
   */
  static async processAudio(
    sessionId: string,
    pitchParams: Omit<PitchCorrectionParams, 'session_id'>,
    mixingParams: MixingParams,
    processingOrder: 'pitch_first' | 'mix_first' = 'pitch_first'
  ): Promise<ProcessingStatus> {
    const response = await fetch(`${API_BASE_URL}/api/audio/process/${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
        pitch_params: pitchParams,
        mixing_params: mixingParams,
        processing_order: processingOrder,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Processing failed: ${errorData.detail || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get all available mixing presets
   */
  static async getMixingPresets(): Promise<MixingPreset[]> {
    const response = await fetch(`${API_BASE_URL}/api/audio/mixing/presets`);

    if (!response.ok) {
      throw new Error(`Failed to get mixing presets: ${response.statusText}`);
    }

    const data = await response.json();
    return data.presets;
  }

  /**
   * Get specific mixing preset by name
   */
  static async getMixingPreset(presetName: string): Promise<MixingPreset> {
    const response = await fetch(`${API_BASE_URL}/api/audio/mixing/preset/${encodeURIComponent(presetName)}`);

    if (!response.ok) {
      throw new Error(`Failed to get mixing preset "${presetName}": ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get list of active sessions (for debugging)
   */
  static async getActiveSessions(): Promise<{ active_sessions: number; sessions: Record<string, any> }> {
    const response = await fetch(`${API_BASE_URL}/api/audio/sessions`);

    if (!response.ok) {
      throw new Error(`Failed to get sessions: ${response.statusText}`);
    }

    return response.json();
  }
}

/**
 * WebSocket client for real-time processing updates
 */
export class AudioProcessingWebSocket {
  private ws: WebSocket | null = null;
  private onStatusCallback?: (status: ProcessingStatus) => void;
  private onErrorCallback?: (error: string) => void;

  connect(sessionId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/api/audio/ws/${sessionId}`;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connected for session:', sessionId);
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const status: ProcessingStatus = JSON.parse(event.data);
          console.log('WebSocket status update:', status);

          if (this.onStatusCallback) {
            this.onStatusCallback(status);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
          if (this.onErrorCallback) {
            this.onErrorCallback('Failed to parse status update');
          }
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (this.onErrorCallback) {
          this.onErrorCallback('WebSocket connection error');
        }
        reject(error);
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
      };
    });
  }

  onStatusUpdate(callback: (status: ProcessingStatus) => void): void {
    this.onStatusCallback = callback;
  }

  onError(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.onStatusCallback = undefined;
    this.onErrorCallback = undefined;
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

/**
 * Utility functions
 */
export const audioUtils = {
  /**
   * Convert audio file to ArrayBuffer for analysis
   */
  fileToArrayBuffer(file: File): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as ArrayBuffer);
      reader.onerror = reject;
      reader.readAsArrayBuffer(file);
    });
  },

  /**
   * Create audio context for client-side analysis
   */
  async createAudioContext(audioData: ArrayBuffer): Promise<{
    audioContext: AudioContext;
    audioBuffer: AudioBuffer;
  }> {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(audioData);

    return { audioContext, audioBuffer };
  },

  /**
   * Extract waveform data for visualization
   */
  extractWaveformData(audioBuffer: AudioBuffer, targetSamples: number = 2000): Float32Array {
    const channelData = audioBuffer.getChannelData(0); // Use first channel
    const blockSize = Math.floor(channelData.length / targetSamples);
    const waveformData = new Float32Array(targetSamples);

    for (let i = 0; i < targetSamples; i++) {
      let sum = 0;
      const start = i * blockSize;
      const end = Math.min(start + blockSize, channelData.length);

      for (let j = start; j < end; j++) {
        sum += Math.abs(channelData[j]);
      }

      waveformData[i] = sum / (end - start);
    }

    return waveformData;
  },

  /**
   * Format file size for display
   */
  formatFileSize(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }
};