"""
Post-processing module for video enhancement and effects.
Includes RIFE interpolation, motion blur, squash & stretch effects, and more.
"""

import cv2
import logging
import numpy as np
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from moviepy.editor import VideoFileClip, CompositeVideoClip, vfx
import tempfile

logger = logging.getLogger(__name__)


class PostProcessor:
    """Video post-processor with various enhancement effects."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the post-processor."""
        self.config = config
        self.device = config.get("gpu_device", "cuda:0")
        
        # Post-processing configuration
        self.rife_enabled = config.get("post_rife", False)
        self.rife_fps = config.get("rife_fps", 48)
        self.smear_enabled = config.get("post_smear", False)
        self.squash_enabled = config.get("post_squash", False)
        
        # Check for RIFE availability
        self.rife_available = self._check_rife_availability()
        
        if self.rife_available:
            logger.info("RIFE interpolation available")
        else:
            logger.warning("RIFE not available, using OpenCV interpolation")
    
    def _check_rife_availability(self) -> bool:
        """Check if RIFE is available."""
        try:
            # Check if RIFE repository exists
            rife_dir = "external/RIFE"
            if os.path.exists(rife_dir):
                return True
            
            # Check if RIFE is installed via pip
            import RIFE
            return True
        except ImportError:
            return False
    
    def process(
        self,
        input_path: str,
        output_path: str,
        rife: bool = False,
        rife_fps: int = 48,
        smear: bool = False,
        squash: bool = False,
        upscale: bool = False,
        color_grade: bool = False,
        motion_blur: bool = False
    ) -> str:
        """
        Apply post-processing effects to video.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            rife: Enable RIFE frame interpolation
            rife_fps: Target FPS for RIFE
            smear: Add motion blur smears
            squash: Add squash & stretch effects
            upscale: Upscale video using Real-ESRGAN
            color_grade: Apply color grading
            motion_blur: Add motion blur
            
        Returns:
            Path to processed video
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        current_path = input_path
        
        try:
            logger.info(f"Starting post-processing: {input_path}")
            
            # Apply effects in sequence
            if rife and self.rife_available:
                logger.info("Applying RIFE interpolation...")
                current_path = self._apply_rife_interpolation(current_path, rife_fps)
            
            if smear:
                logger.info("Applying motion blur smears...")
                current_path = self._apply_motion_smears(current_path)
            
            if squash:
                logger.info("Applying squash & stretch effects...")
                current_path = self._apply_squash_stretch(current_path)
            
            if motion_blur:
                logger.info("Applying motion blur...")
                current_path = self._apply_motion_blur(current_path)
            
            if color_grade:
                logger.info("Applying color grading...")
                current_path = self._apply_color_grading(current_path)
            
            if upscale:
                logger.info("Upscaling video...")
                current_path = self._upscale_video(current_path)
            
            # Final copy to output path
            if current_path != output_path:
                import shutil
                shutil.copy2(current_path, output_path)
                if current_path != input_path:
                    os.unlink(current_path)  # Clean up temp file
            
            logger.info(f"Post-processing completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            # Return original video if processing fails
            if current_path != input_path and current_path != output_path:
                import shutil
                shutil.copy2(input_path, output_path)
            return output_path
    
    def _apply_rife_interpolation(self, input_path: str, target_fps: int) -> str:
        """Apply RIFE frame interpolation."""
        if not self.rife_available:
            return self._apply_opencv_interpolation(input_path, target_fps)
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_output = temp_file.name
        
        try:
            # Get input video FPS
            input_fps = self._get_video_fps(input_path)
            
            # Calculate interpolation factor
            factor = target_fps / input_fps
            
            # Run RIFE
            cmd = [
                "python", "external/RIFE/inference_video.py",
                "--video", input_path,
                "--output", temp_output,
                "--fps", str(target_fps),
                "--model", "v4.6"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"RIFE failed: {result.stderr}")
                return self._apply_opencv_interpolation(input_path, target_fps)
            
            return temp_output
            
        except Exception as e:
            logger.warning(f"RIFE interpolation failed: {e}")
            return self._apply_opencv_interpolation(input_path, target_fps)
    
    def _apply_opencv_interpolation(self, input_path: str, target_fps: int) -> str:
        """Apply OpenCV-based frame interpolation."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_output = temp_file.name
        
        try:
            # Use FFmpeg for frame interpolation
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-filter:v", f"fps=fps={target_fps}:round=up",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                temp_output
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg interpolation failed: {result.stderr}")
            
            return temp_output
            
        except Exception as e:
            logger.error(f"OpenCV interpolation failed: {e}")
            return input_path
    
    def _apply_motion_smears(self, input_path: str) -> str:
        """Apply motion blur smears for cartoon effect."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_output = temp_file.name
        
        try:
            # Use MoviePy for motion smears
            clip = VideoFileClip(input_path)
            
            # Apply motion blur effect
            def add_motion_smear(frame):
                # Create motion smear effect
                blurred = cv2.GaussianBlur(frame, (15, 15), 0)
                # Blend with original frame
                alpha = 0.3
                return cv2.addWeighted(frame, 1 - alpha, blurred, alpha, 0)
            
            # Apply effect
            processed_clip = clip.fl_image(add_motion_smear)
            processed_clip.write_videofile(temp_output, codec='libx264')
            
            clip.close()
            processed_clip.close()
            
            return temp_output
            
        except Exception as e:
            logger.error(f"Motion smears failed: {e}")
            return input_path
    
    def _apply_squash_stretch(self, input_path: str) -> str:
        """Apply squash & stretch effects for cartoon animation."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_output = temp_file.name
        
        try:
            # Use MoviePy for squash & stretch
            clip = VideoFileClip(input_path)
            
            # Apply squash & stretch effect
            def apply_squash_stretch(get_frame, t):
                frame = get_frame(t)
                
                # Calculate squash factor based on time
                # This is a simplified effect - in practice you'd analyze motion
                squash_factor = 1.0 + 0.1 * np.sin(t * 2 * np.pi / clip.duration)
                
                # Apply scaling
                height, width = frame.shape[:2]
                new_height = int(height * squash_factor)
                new_width = int(width / squash_factor)
                
                # Resize frame
                resized = cv2.resize(frame, (new_width, new_height))
                
                # Pad to original size
                result = np.zeros((height, width, 3), dtype=np.uint8)
                y_offset = (height - new_height) // 2
                x_offset = (width - new_width) // 2
                
                result[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
                
                return result
            
            # Apply effect
            processed_clip = clip.fl(apply_squash_stretch)
            processed_clip.write_videofile(temp_output, codec='libx264')
            
            clip.close()
            processed_clip.close()
            
            return temp_output
            
        except Exception as e:
            logger.error(f"Squash & stretch failed: {e}")
            return input_path
    
    def _apply_motion_blur(self, input_path: str) -> str:
        """Apply motion blur effect."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_output = temp_file.name
        
        try:
            # Use FFmpeg for motion blur
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-filter:v", "tmix=frames=3:weights=0.5,0.3,0.2",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                temp_output
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Motion blur failed: {result.stderr}")
            
            return temp_output
            
        except Exception as e:
            logger.error(f"Motion blur failed: {e}")
            return input_path
    
    def _apply_color_grading(self, input_path: str) -> str:
        """Apply color grading effects."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_output = temp_file.name
        
        try:
            # Use FFmpeg for color grading
            # Apply cartoon-style color grading
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-filter:v", "eq=saturation=1.2:contrast=1.1:brightness=0.05",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                temp_output
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Color grading failed: {result.stderr}")
            
            return temp_output
            
        except Exception as e:
            logger.error(f"Color grading failed: {e}")
            return input_path
    
    def _upscale_video(self, input_path: str) -> str:
        """Upscale video using Real-ESRGAN."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_output = temp_file.name
        
        try:
            # Check if Real-ESRGAN is available
            try:
                import basicsr
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                esrgan_available = True
            except ImportError:
                esrgan_available = False
            
            if esrgan_available:
                # Use Real-ESRGAN
                cmd = [
                    "python", "external/Real-ESRGAN/inference_realesrgan_video.py",
                    "-i", input_path,
                    "-o", temp_output,
                    "-n", "RealESRGAN_x4plus"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    return temp_output
            
            # Fallback to FFmpeg upscaling
            logger.warning("Real-ESRGAN not available, using FFmpeg upscaling")
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vf", "scale=iw*2:ih*2:flags=lanczos",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                temp_output
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Upscaling failed: {result.stderr}")
            
            return temp_output
            
        except Exception as e:
            logger.error(f"Upscaling failed: {e}")
            return input_path
    
    def _get_video_fps(self, video_path: str) -> float:
        """Get video FPS using FFprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                fps_str = result.stdout.strip()
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    return num / den
                else:
                    return float(fps_str)
        except:
            pass
        
        return 30.0  # Default FPS
    
    def create_preview(
        self,
        input_path: str,
        output_path: str,
        duration: float = 5.0
    ) -> str:
        """
        Create a preview of post-processing effects.
        
        Args:
            input_path: Input video path
            output_path: Output preview path
            duration: Preview duration in seconds
            
        Returns:
            Path to preview video
        """
        try:
            # Extract short segment
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_input = temp_file.name
            
            # Extract segment
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-t", str(duration),
                "-c", "copy",
                temp_input
            ]
            
            subprocess.run(cmd, capture_output=True)
            
            # Apply post-processing to segment
            result = self.process(
                temp_input, output_path,
                rife=self.rife_enabled,
                rife_fps=self.rife_fps,
                smear=self.smear_enabled,
                squash=self.squash_enabled
            )
            
            # Clean up
            os.unlink(temp_input)
            
            return result
            
        except Exception as e:
            logger.error(f"Preview creation failed: {e}")
            return input_path
    
    def batch_process(
        self,
        input_paths: list,
        output_dir: str,
        **kwargs
    ) -> list:
        """
        Apply post-processing to multiple videos.
        
        Args:
            input_paths: List of input video paths
            output_dir: Output directory
            **kwargs: Additional arguments for process()
            
        Returns:
            List of output video paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, input_path in enumerate(input_paths):
            output_path = os.path.join(output_dir, f"processed_{i:04d}.mp4")
            
            try:
                result = self.process(input_path, output_path, **kwargs)
                results.append(result)
                logger.info(f"Batch processing {i+1}/{len(input_paths)} completed")
            except Exception as e:
                logger.error(f"Batch processing {i+1} failed: {e}")
                results.append(input_path)  # Use original video as fallback
        
        return results
