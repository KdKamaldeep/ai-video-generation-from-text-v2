"""
Wav2Lip lip-sync module for WhyWouldYou-v2 pipeline.
"""

import cv2
import logging
import numpy as np
import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class Wav2LipInfer:
    """Wav2Lip lip-sync generator."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Wav2Lip generator."""
        self.config = config
        self.device = config.get("gpu_device", "cuda:0")
        
        # Wav2Lip configuration
        self.model_path = config.get("wav2lip_model", "models/wav2lip.pth")
        self.face_detection_model = config.get("face_detection_model", "models/s3fd.pth")
        
        # Check if Wav2Lip is available
        self.wav2lip_available = self._check_wav2lip_availability()
        
        if self.wav2lip_available:
            logger.info("Wav2Lip available")
        else:
            logger.warning("Wav2Lip not available, will use fallback methods")
    
    def _check_wav2lip_availability(self) -> bool:
        """Check if Wav2Lip is available."""
        try:
            # Check if Wav2Lip repository exists
            wav2lip_dir = "external/Wav2Lip"
            if os.path.exists(wav2lip_dir):
                return True
            
            # Check if Wav2Lip is installed via pip
            import wav2lip
            return True
        except ImportError:
            return False
    
    def sync(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        face_detection: str = "s3fd",
        resize_factor: int = 1,
        crop: Optional[Tuple[int, int, int, int]] = None,
        smooth: bool = True,
        wav2lip_batch_size: int = 128
    ) -> str:
        """
        Perform lip-sync using Wav2Lip.
        
        Args:
            video_path: Input video path
            audio_path: Input audio path
            output_path: Output synced video path
            face_detection: Face detection method ("s3fd" or "dlib")
            resize_factor: Resize factor for processing
            crop: Crop region (x1, y1, x2, y2)
            smooth: Apply temporal smoothing
            wav2lip_batch_size: Batch size for Wav2Lip processing
            
        Returns:
            Path to synced video
        """
        if not self.wav2lip_available:
            return self._sync_fallback(video_path, audio_path, output_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            logger.info(f"Starting Wav2Lip sync: {video_path} + {audio_path}")
            
            # Prepare Wav2Lip command
            cmd = self._build_wav2lip_command(
                video_path, audio_path, output_path,
                face_detection, resize_factor, crop, smooth, wav2lip_batch_size
            )
            
            # Run Wav2Lip
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Wav2Lip failed: {result.stderr}")
                return self._sync_fallback(video_path, audio_path, output_path)
            
            logger.info(f"Wav2Lip sync completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Wav2Lip sync failed: {e}")
            return self._sync_fallback(video_path, audio_path, output_path)
    
    def _build_wav2lip_command(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        face_detection: str,
        resize_factor: int,
        crop: Optional[Tuple[int, int, int, int]],
        smooth: bool,
        wav2lip_batch_size: int
    ) -> list:
        """Build Wav2Lip command."""
        cmd = [
            "python", "external/Wav2Lip/inference.py",
            "--checkpoint_path", self.model_path,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--face_det_batch_size", str(wav2lip_batch_size),
            "--wav2lip_batch_size", str(wav2lip_batch_size),
            "--resize_factor", str(resize_factor)
        ]
        
        if face_detection == "dlib":
            cmd.append("--pads")
            cmd.extend(["0", "20", "0", "0"])
        
        if crop:
            cmd.extend(["--crop", str(crop[0]), str(crop[1]), str(crop[2]), str(crop[3])])
        
        if smooth:
            cmd.append("--smooth")
        
        return cmd
    
    def _sync_fallback(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> str:
        """
        Fallback method when Wav2Lip is not available.
        Simply combines video and audio without lip-sync.
        """
        logger.info("Using fallback method: combining video and audio")
        
        try:
            # Use FFmpeg to combine video and audio
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            logger.info(f"Fallback sync completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Fallback sync failed: {e}")
            # Return original video if everything fails
            return video_path
    
    def detect_faces(
        self,
        video_path: str,
        output_dir: str,
        face_detection: str = "s3fd"
    ) -> list:
        """
        Detect faces in video for debugging.
        
        Args:
            video_path: Input video path
            output_dir: Output directory for face detection results
            face_detection: Face detection method
            
        Returns:
            List of detected face regions
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.wav2lip_available:
            logger.warning("Wav2Lip not available, cannot detect faces")
            return []
        
        try:
            # Use Wav2Lip's face detection
            cmd = [
                "python", "external/Wav2Lip/detect_faces.py",
                "--input", video_path,
                "--output", output_dir,
                "--method", face_detection
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Face detection failed: {result.stderr}")
                return []
            
            # Parse face detection results
            faces = self._parse_face_detection_results(output_dir)
            
            logger.info(f"Detected {len(faces)} faces")
            return faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _parse_face_detection_results(self, output_dir: str) -> list:
        """Parse face detection results from output directory."""
        faces = []
        
        # Look for face detection output files
        for file in os.listdir(output_dir):
            if file.endswith(".json"):
                import json
                with open(os.path.join(output_dir, file), 'r') as f:
                    data = json.load(f)
                    if "faces" in data:
                        faces.extend(data["faces"])
        
        return faces
    
    def create_sync_preview(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        preview_duration: float = 5.0
    ) -> str:
        """
        Create a short preview of lip-sync for testing.
        
        Args:
            video_path: Input video path
            audio_path: Input audio path
            output_path: Output preview path
            preview_duration: Duration of preview in seconds
            
        Returns:
            Path to preview video
        """
        try:
            # Create temporary short versions
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video_path = temp_video.name
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Extract short video segment
            cmd_video = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-t", str(preview_duration),
                "-c", "copy",
                temp_video_path
            ]
            
            subprocess.run(cmd_video, capture_output=True)
            
            # Extract short audio segment
            cmd_audio = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-t", str(preview_duration),
                "-c", "copy",
                temp_audio_path
            ]
            
            subprocess.run(cmd_audio, capture_output=True)
            
            # Perform lip-sync on short segments
            result = self.sync(temp_video_path, temp_audio_path, output_path)
            
            # Clean up temporary files
            os.unlink(temp_video_path)
            os.unlink(temp_audio_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Preview creation failed: {e}")
            return video_path
    
    def get_sync_quality_metrics(
        self,
        original_video: str,
        synced_video: str
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for lip-sync.
        
        Args:
            original_video: Original video path
            synced_video: Synced video path
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        try:
            # Get video durations
            orig_duration = self._get_video_duration(original_video)
            sync_duration = self._get_video_duration(synced_video)
            
            metrics["duration_match"] = abs(orig_duration - sync_duration)
            metrics["duration_ratio"] = sync_duration / orig_duration if orig_duration > 0 else 0
            
            # Get video resolutions
            orig_res = self._get_video_resolution(original_video)
            sync_res = self._get_video_resolution(synced_video)
            
            if orig_res and sync_res:
                metrics["resolution_match"] = orig_res == sync_res
                metrics["resolution_ratio"] = (sync_res[0] * sync_res[1]) / (orig_res[0] * orig_res[1])
            
            # Get file sizes
            orig_size = os.path.getsize(original_video)
            sync_size = os.path.getsize(synced_video)
            
            metrics["size_ratio"] = sync_size / orig_size if orig_size > 0 else 0
            
        except Exception as e:
            logger.error(f"Failed to calculate quality metrics: {e}")
        
        return metrics
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using FFprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        return 0.0
    
    def _get_video_resolution(self, video_path: str) -> Optional[Tuple[int, int]]:
        """Get video resolution using FFprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                width, height = map(int, result.stdout.strip().split(','))
                return (width, height)
        except:
            pass
        
        return None
    
    def batch_sync(
        self,
        video_audio_pairs: list,
        output_dir: str,
        **kwargs
    ) -> list:
        """
        Perform batch lip-sync on multiple video-audio pairs.
        
        Args:
            video_audio_pairs: List of (video_path, audio_path) tuples
            output_dir: Output directory
            **kwargs: Additional arguments for sync()
            
        Returns:
            List of output video paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, (video_path, audio_path) in enumerate(video_audio_pairs):
            output_path = os.path.join(output_dir, f"synced_{i:04d}.mp4")
            
            try:
                result = self.sync(video_path, audio_path, output_path, **kwargs)
                results.append(result)
                logger.info(f"Batch sync {i+1}/{len(video_audio_pairs)} completed")
            except Exception as e:
                logger.error(f"Batch sync {i+1} failed: {e}")
                results.append(video_path)  # Use original video as fallback
        
        return results
