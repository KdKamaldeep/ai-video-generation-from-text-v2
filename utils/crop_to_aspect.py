"""
Aspect ratio and cropping utilities for video processing.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def crop_to_aspect_ratio(
    image: np.ndarray, 
    target_aspect: float, 
    center_crop: bool = True
) -> np.ndarray:
    """
    Crop image to target aspect ratio.
    
    Args:
        image: Input image as numpy array (H, W, C)
        target_aspect: Target aspect ratio (width / height)
        center_crop: If True, crop from center; if False, crop from top-left
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    current_aspect = w / h
    
    if abs(current_aspect - target_aspect) < 0.01:
        return image  # Already correct aspect ratio
    
    if current_aspect > target_aspect:
        # Image is too wide, crop width
        new_w = int(h * target_aspect)
        if center_crop:
            start_x = (w - new_w) // 2
        else:
            start_x = 0
        return image[:, start_x:start_x + new_w]
    else:
        # Image is too tall, crop height
        new_h = int(w / target_aspect)
        if center_crop:
            start_y = (h - new_h) // 2
        else:
            start_y = 0
        return image[start_y:start_y + new_h, :]


def resize_to_resolution(
    image: np.ndarray, 
    target_resolution: Tuple[int, int],
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize image to target resolution.
    
    Args:
        image: Input image as numpy array
        target_resolution: Target (width, height)
        maintain_aspect: If True, maintain aspect ratio and pad if needed
        
    Returns:
        Resized image
    """
    target_w, target_h = target_resolution
    h, w = image.shape[:2]
    
    if maintain_aspect:
        # Calculate scaling factor to fit within target resolution
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    else:
        # Direct resize without maintaining aspect ratio
        return cv2.resize(image, target_resolution, interpolation=cv2.INTER_LANCZOS4)


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """
    Get video information.
    
    Returns:
        Tuple of (width, height, frame_count, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    
    return width, height, frame_count, fps


def create_video_writer(
    output_path: str,
    resolution: Tuple[int, int],
    fps: float,
    codec: str = "mp4v"
) -> cv2.VideoWriter:
    """
    Create a video writer with specified parameters.
    
    Args:
        output_path: Output video path
        resolution: Video resolution (width, height)
        fps: Frames per second
        codec: Video codec (mp4v, avc1, etc.)
        
    Returns:
        VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    if not writer.isOpened():
        raise ValueError(f"Could not create video writer for: {output_path}")
    
    return writer


def aspect_ratio_from_resolution(resolution: Tuple[int, int]) -> float:
    """Calculate aspect ratio from resolution."""
    width, height = resolution
    return width / height


def resolution_from_aspect_ratio(
    aspect_ratio: float, 
    height: int
) -> Tuple[int, int]:
    """Calculate resolution from aspect ratio and height."""
    width = int(height * aspect_ratio)
    return (width, height)
