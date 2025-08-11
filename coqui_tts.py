"""
Coqui TTS module for text-to-speech generation.
"""

import logging
import os
import torch
from typing import Dict, Any, Optional
import numpy as np

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    logging.warning("Coqui TTS not available, falling back to subprocess calls")

logger = logging.getLogger(__name__)


class CoquiTTS:
    """Coqui TTS text-to-speech generator."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the TTS generator."""
        self.config = config
        self.device = config.get("gpu_device", "cuda:0")
        
        # TTS configuration
        tts_config = config.get("coqui_voice", {})
        self.voice = tts_config.get("voice", "alloy")
        self.sample_rate = tts_config.get("sample_rate", 24000)
        self.model_name = tts_config.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")
        
        if COQUI_AVAILABLE:
            self._load_tts()
        else:
            logger.warning("Using subprocess fallback for TTS generation")
    
    def _load_tts(self):
        """Load the TTS model."""
        try:
            logger.info(f"Loading TTS model: {self.model_name}")
            
            # Initialize TTS
            self.tts = TTS(model_name=self.model_name)
            
            # Move to device if available
            if torch.cuda.is_available() and "cuda" in self.device:
                self.tts.to(self.device)
            
            logger.info("TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise
    
    def generate(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        sample_rate: Optional[int] = None,
        speed: float = 1.0,
        emotion: Optional[str] = None
    ) -> str:
        """
        Generate speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Output audio path
            voice: Voice to use (overrides config)
            sample_rate: Sample rate (overrides config)
            speed: Speech speed multiplier
            emotion: Emotion to apply (if supported)
            
        Returns:
            Path to generated audio
        """
        if not COQUI_AVAILABLE:
            return self._generate_subprocess(
                text, output_path, voice, sample_rate, speed, emotion
            )
        
        # Use provided parameters or defaults
        voice = voice or self.voice
        sample_rate = sample_rate or self.sample_rate
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            logger.info(f"Generating TTS for text: {text[:50]}...")
            
            # Generate speech
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                voice=voice,
                speaker_wav=None,  # Use default voice
                language="en",
                speed=speed
            )
            
            logger.info(f"TTS audio generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
    
    def _generate_subprocess(
        self,
        text: str,
        output_path: str,
        voice: Optional[str],
        sample_rate: Optional[int],
        speed: float,
        emotion: Optional[str]
    ) -> str:
        """Fallback to subprocess call for TTS generation."""
        import subprocess
        import tempfile
        import json
        
        # Create temporary config file
        config_data = {
            "text": text,
            "output_path": output_path,
            "voice": voice or self.voice,
            "sample_rate": sample_rate or self.sample_rate,
            "speed": speed,
            "emotion": emotion,
            "model_name": self.model_name
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # Call external TTS script (placeholder)
            cmd = [
                "python", "scripts/generate_tts.py",
                "--config", config_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"TTS generation failed: {result.stderr}")
            
            return output_path
            
        finally:
            os.unlink(config_file)
    
    def generate_batch(
        self,
        texts: list,
        output_dir: str,
        **kwargs
    ) -> list:
        """
        Generate multiple audio files from a list of texts.
        
        Args:
            texts: List of text strings
            output_dir: Output directory for audio files
            **kwargs: Additional arguments for generate()
            
        Returns:
            List of generated audio paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        generated_paths = []
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"audio_{i:04d}.wav")
            try:
                path = self.generate(text, output_path, **kwargs)
                generated_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to generate audio {i}: {e}")
                continue
        
        logger.info(f"Generated {len(generated_paths)} audio files")
        return generated_paths
    
    def get_available_voices(self) -> list:
        """Get list of available voices."""
        if not COQUI_AVAILABLE:
            return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        try:
            return self.tts.voices
        except:
            return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            return duration
        except ImportError:
            # Fallback using ffprobe
            import subprocess
            
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                logger.warning("Could not determine audio duration")
                return 0.0
    
    def create_audio_segment(
        self,
        text: str,
        start_time: float,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create an audio segment with timing information.
        
        Args:
            text: Text to synthesize
            start_time: Start time in seconds
            output_path: Output audio path
            **kwargs: Additional arguments for generate()
            
        Returns:
            Dictionary with audio info
        """
        # Generate audio
        audio_path = self.generate(text, output_path, **kwargs)
        
        # Get duration
        duration = self.get_audio_duration(audio_path)
        
        return {
            "text": text,
            "audio_path": audio_path,
            "start_time": start_time,
            "end_time": start_time + duration,
            "duration": duration
        }
    
    def create_subtitle_file(
        self,
        audio_segments: list,
        output_path: str,
        format: str = "srt"
    ):
        """
        Create subtitle file from audio segments.
        
        Args:
            audio_segments: List of audio segment dictionaries
            output_path: Output subtitle file path
            format: Subtitle format ("srt" or "vtt")
        """
        if format == "srt":
            self._create_srt_file(audio_segments, output_path)
        elif format == "vtt":
            self._create_vtt_file(audio_segments, output_path)
        else:
            raise ValueError(f"Unsupported subtitle format: {format}")
    
    def _create_srt_file(self, audio_segments: list, output_path: str):
        """Create SRT subtitle file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(audio_segments, 1):
                start_time = self._format_srt_time(segment["start_time"])
                end_time = self._format_srt_time(segment["end_time"])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
    
    def _create_vtt_file(self, audio_segments: list, output_path: str):
        """Create VTT subtitle file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for i, segment in enumerate(audio_segments, 1):
                start_time = self._format_vtt_time(segment["start_time"])
                end_time = self._format_vtt_time(segment["end_time"])
                
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _format_vtt_time(self, seconds: float) -> str:
        """Format time for VTT format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
