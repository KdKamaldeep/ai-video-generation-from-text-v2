"""
Video assembly module for final video composition.
Handles combining clips, adding audio, captions, and effects.
"""

import cv2
import logging
import numpy as np
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
# from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_videoclips
# Using FFmpeg as primary method, OpenCV for fallback operations
import tempfile

logger = logging.getLogger(__name__)


class VideoAssembler:
    """Video assembler for final video composition."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the video assembler."""
        self.config = config
        
        # Assembly configuration
        self.output_resolution = config.get("output_resolution", [1080, 1920])
        self.default_fps = config.get("fps", 24)
        self.crossfade_duration = config.get("crossfade_duration", 0.5)
        
        # Check for FFmpeg availability
        self.ffmpeg_available = self._check_ffmpeg_availability()
        
        if not self.ffmpeg_available:
            logger.warning("FFmpeg not available, some features may not work")
    
    def _check_ffmpeg_availability(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def assemble(
        self,
        clips: List[str],
        output_path: str,
        background_audio: Optional[str] = None,
        crossfade_duration: float = 0.5,
        add_captions: bool = False,
        caption_file: Optional[str] = None,
        intro_clip: Optional[str] = None,
        outro_clip: Optional[str] = None,
        watermark: Optional[str] = None
    ) -> str:
        """
        Assemble final video from clips.
        
        Args:
            clips: List of video clip paths
            output_path: Output video path
            background_audio: Background audio path
            crossfade_duration: Crossfade duration in seconds
            add_captions: Whether to add captions
            caption_file: Path to caption file (SRT/VTT)
            intro_clip: Intro video clip path
            outro_clip: Outro video clip path
            watermark: Watermark image path
            
        Returns:
            Path to assembled video
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            logger.info(f"Starting video assembly with {len(clips)} clips")
            
            # Validate input clips
            valid_clips = self._validate_clips(clips)
            if not valid_clips:
                raise ValueError("No valid video clips provided")
            
            # Prepare clips list
            final_clips = []
            
            # Add intro if provided
            if intro_clip and os.path.exists(intro_clip):
                final_clips.append(intro_clip)
            
            # Add main clips
            final_clips.extend(valid_clips)
            
            # Add outro if provided
            if outro_clip and os.path.exists(outro_clip):
                final_clips.append(outro_clip)
            
            # Assemble video using FFmpeg
            if self.ffmpeg_available:
                result_path = self._assemble_with_ffmpeg(
                    final_clips, output_path, background_audio,
                    crossfade_duration, add_captions, caption_file, watermark
                )
            else:
                raise RuntimeError("FFmpeg is required for video assembly. Please install FFmpeg.")
            
            logger.info(f"Video assembly completed: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Video assembly failed: {e}")
            raise
    
    def _validate_clips(self, clips: List[str]) -> List[str]:
        """Validate and filter video clips."""
        valid_clips = []
        
        for clip_path in clips:
            if os.path.exists(clip_path):
                # Check if it's a valid video file
                try:
                    cap = cv2.VideoCapture(clip_path)
                    if cap.isOpened():
                        valid_clips.append(clip_path)
                        cap.release()
                    else:
                        logger.warning(f"Invalid video file: {clip_path}")
                except Exception as e:
                    logger.warning(f"Error validating clip {clip_path}: {e}")
            else:
                logger.warning(f"Clip not found: {clip_path}")
        
        return valid_clips
    
    def _assemble_with_ffmpeg(
        self,
        clips: List[str],
        output_path: str,
        background_audio: Optional[str],
        crossfade_duration: float,
        add_captions: bool,
        caption_file: Optional[str],
        watermark: Optional[str]
    ) -> str:
        """Assemble video using FFmpeg."""
        
        # Create filter complex for concatenation with crossfades
        filter_complex = self._build_ffmpeg_filter_complex(
            clips, crossfade_duration, add_captions, caption_file, watermark
        )
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y"
        ]
        
        # Add input files
        for clip in clips:
            cmd.extend(["-i", clip])
        
        # Add background audio if provided
        if background_audio and os.path.exists(background_audio):
            cmd.extend(["-i", background_audio])
        
        # Add subtitle file if provided
        if add_captions and caption_file and os.path.exists(caption_file):
            cmd.extend(["-i", caption_file])
        
        # Add watermark if provided
        if watermark and os.path.exists(watermark):
            cmd.extend(["-i", watermark])
        
        # Add filter complex
        if filter_complex:
            cmd.extend(["-filter_complex", filter_complex])
        
        # Add output options
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ])
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg assembly failed: {result.stderr}")
        
        return output_path
    
    def _build_ffmpeg_filter_complex(
        self,
        clips: List[str],
        crossfade_duration: float,
        add_captions: bool,
        caption_file: Optional[str],
        watermark: Optional[str]
    ) -> str:
        """Build FFmpeg filter complex string."""
        filters = []
        
        # Scale all clips to output resolution
        scaled_inputs = []
        for i, clip in enumerate(clips):
            scaled_inputs.append(f"[{i}:v]scale={self.output_resolution[0]}:{self.output_resolution[1]}:force_original_aspect_ratio=decrease,pad={self.output_resolution[0]}:{self.output_resolution[1]}:(ow-iw)/2:(oh-ih)/2[v{i}]")
        
        filters.extend(scaled_inputs)
        
        # Add watermark if provided
        if watermark and os.path.exists(watermark):
            watermark_idx = len(clips)
            filters.append(f"[{watermark_idx}:v]scale=200:-1[watermark]")
            
            # Overlay watermark on each clip
            for i in range(len(clips)):
                filters.append(f"[v{i}][watermark]overlay=W-w-10:10[v{i}_w]")
        
        # Concatenate clips with crossfades
        if len(clips) > 1 and crossfade_duration > 0:
            concat_filters = []
            for i in range(len(clips) - 1):
                if i == 0:
                    concat_filters.append(f"[v{i}_w]" if watermark else f"[v{i}]")
                else:
                    # Add crossfade
                    prev_idx = i - 1
                    curr_idx = i
                    filters.append(f"[v{prev_idx}_w][v{curr_idx}_w]xfade=transition=fade:duration={crossfade_duration}:offset={i}[xfade{i}]" if watermark else f"[v{prev_idx}][v{curr_idx}]xfade=transition=fade:duration={crossfade_duration}:offset={i}[xfade{i}]")
                    concat_filters.append(f"[xfade{i}]")
            
            # Add last clip
            last_idx = len(clips) - 1
            concat_filters.append(f"[v{last_idx}_w]" if watermark else f"[v{last_idx}]")
            
            # Final concatenation
            filters.append(f"{''.join(concat_filters)}concat=n={len(concat_filters)}:v=1:a=0[outv]")
        else:
            # Simple concatenation without crossfades
            concat_inputs = []
            for i in range(len(clips)):
                concat_inputs.append(f"[v{i}_w]" if watermark else f"[v{i}]")
            
            filters.append(f"{''.join(concat_inputs)}concat=n={len(clips)}:v=1:a=0[outv]")
        
        # Add captions if provided
        if add_captions and caption_file and os.path.exists(caption_file):
            subtitle_idx = len(clips) + (1 if background_audio else 0)
            filters.append(f"[outv][{subtitle_idx}:s]overlay[outv_sub]")
            filters.append("[outv_sub]copy[finalv]")
        else:
            filters.append("[outv]copy[finalv]")
        
        return ";".join(filters)
    
    # MoviePy methods removed - using FFmpeg as primary method
    # FFmpeg provides better performance and reliability for video assembly
    
    def create_thumbnail(
        self,
        video_path: str,
        output_path: str,
        time_position: float = 1.0
    ) -> str:
        """
        Create a thumbnail from video.
        
        Args:
            video_path: Input video path
            output_path: Output thumbnail path
            time_position: Time position in seconds
            
        Returns:
            Path to thumbnail
        """
        try:
            if self.ffmpeg_available:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-ss", str(time_position),
                    "-vframes", "1",
                    "-q:v", "2",
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Thumbnail creation failed: {result.stderr}")
                
                return output_path
            else:
                # Fallback using OpenCV
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_MSEC, time_position * 1000)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    cv2.imwrite(output_path, frame)
                    return output_path
                else:
                    raise RuntimeError("Failed to read frame from video")
                    
        except Exception as e:
            logger.error(f"Thumbnail creation failed: {e}")
            raise
    
    def add_intro_outro(
        self,
        video_path: str,
        output_path: str,
        intro_path: Optional[str] = None,
        outro_path: Optional[str] = None,
        intro_duration: float = 3.0,
        outro_duration: float = 3.0
    ) -> str:
        """
        Add intro and outro to video using FFmpeg.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            intro_path: Intro video path
            outro_path: Outro video path
            intro_duration: Intro duration in seconds
            outro_duration: Outro duration in seconds
            
        Returns:
            Path to video with intro/outro
        """
        try:
            if not self.ffmpeg_available:
                raise RuntimeError("FFmpeg is required for intro/outro addition")
            
            # Prepare input files
            input_files = []
            filter_parts = []
            
            # Add intro if provided
            if intro_path and os.path.exists(intro_path):
                input_files.extend(["-i", intro_path])
                filter_parts.append(f"[0:v]trim=duration={intro_duration}[intro]")
            
            # Add main video
            input_files.extend(["-i", video_path])
            filter_parts.append("[1:v]")
            
            # Add outro if provided
            if outro_path and os.path.exists(outro_path):
                input_files.extend(["-i", outro_path])
                filter_parts.append(f"[2:v]trim=duration={outro_duration}[outro]")
            
            # Build filter complex
            if len(filter_parts) == 1:
                # Only main video
                filter_complex = "[1:v]"
            elif len(filter_parts) == 2 and intro_path:
                # Intro + main video
                filter_complex = f"[intro][1:v]concat=n=2:v=1[outv]"
            elif len(filter_parts) == 2 and outro_path:
                # Main video + outro
                filter_complex = f"[1:v][outro]concat=n=2:v=1[outv]"
            else:
                # Intro + main + outro
                filter_complex = f"[intro][1:v][outro]concat=n=3:v=1[outv]"
            
            # Build FFmpeg command
            cmd = ["ffmpeg", "-y"] + input_files + [
                "-filter_complex", filter_complex,
                "-map", "[outv]",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Intro/outro addition failed: {result.stderr}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Intro/outro addition failed: {e}")
            raise
    
    def create_vertical_video(
        self,
        video_path: str,
        output_path: str,
        target_resolution: Tuple[int, int] = (1080, 1920)
    ) -> str:
        """
        Convert video to vertical format (for Shorts/TikTok).
        
        Args:
            video_path: Input video path
            output_path: Output video path
            target_resolution: Target resolution (width, height)
            
        Returns:
            Path to vertical video
        """
        try:
            if self.ffmpeg_available:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-vf", f"scale={target_resolution[0]}:{target_resolution[1]}:force_original_aspect_ratio=decrease,pad={target_resolution[0]}:{target_resolution[1]}:(ow-iw)/2:(oh-ih)/2:color=black",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Vertical conversion failed: {result.stderr}")
                
                return output_path
            else:
                # Fallback using OpenCV
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, target_resolution)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize and pad frame
                    h, w = frame.shape[:2]
                    target_w, target_h = target_resolution
                    
                    # Calculate scaling
                    scale = min(target_w / w, target_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # Resize
                    resized = cv2.resize(frame, (new_w, new_h))
                    
                    # Create padded frame
                    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    y_offset = (target_h - new_h) // 2
                    x_offset = (target_w - new_w) // 2
                    
                    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
                    
                    out.write(padded)
                
                cap.release()
                out.release()
                
                return output_path
                
        except Exception as e:
            logger.error(f"Vertical conversion failed: {e}")
            raise
