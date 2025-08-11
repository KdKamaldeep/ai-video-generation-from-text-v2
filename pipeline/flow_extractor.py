"""
Optical flow extraction module for motion conditioning.
Supports RAFT and OpenCV Farneback for extracting optical flow from videos.
"""

import cv2
import logging
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class FlowExtractor:
    """Extract optical flow from videos using RAFT or OpenCV."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the flow extractor."""
        self.config = config
        self.device = config.get("gpu_device", "cuda:0")
        
        # Try to import RAFT
        self.raft_available = self._check_raft_availability()
        
        if self.raft_available:
            logger.info("RAFT optical flow available")
        else:
            logger.info("Using OpenCV Farneback for optical flow")
    
    def _check_raft_availability(self) -> bool:
        """Check if RAFT is available."""
        try:
            # This is a placeholder - in practice you'd check for RAFT installation
            # import raft
            return False  # For now, default to OpenCV
        except ImportError:
            return False
    
    def extract_flow(
        self,
        video_path: str,
        output_dir: str,
        method: str = "opencv",
        save_visualization: bool = True,
        save_flow_maps: bool = True
    ) -> str:
        """
        Extract optical flow from video.
        
        Args:
            video_path: Input video path
            output_dir: Output directory for flow data
            method: Extraction method ("raft" or "opencv")
            save_visualization: Save flow visualization images
            save_flow_maps: Save flow map files
            
        Returns:
            Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if method == "raft" and self.raft_available:
            return self._extract_raft(video_path, output_dir, save_visualization, save_flow_maps)
        else:
            return self._extract_opencv(video_path, output_dir, save_visualization, save_flow_maps)
    
    def _extract_opencv(
        self,
        video_path: str,
        output_dir: str,
        save_visualization: bool,
        save_flow_maps: bool
    ) -> str:
        """Extract optical flow using OpenCV Farneback."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Create subdirectories
        if save_visualization:
            vis_dir = os.path.join(output_dir, "flow_visualization")
            os.makedirs(vis_dir, exist_ok=True)
        
        if save_flow_maps:
            flow_dir = os.path.join(output_dir, "flow_maps")
            os.makedirs(flow_dir, exist_ok=True)
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        frame_count = 0
        flow_data = []
        
        logger.info("Extracting optical flow with OpenCV Farneback...")
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5,  # Pyramid scale
                levels=3,       # Number of pyramid levels
                winsize=15,     # Window size
                iterations=3,   # Iterations
                poly_n=5,       # Polynomial degree
                poly_sigma=1.2, # Gaussian sigma
                flags=0
            )
            
            # Save flow map
            if save_flow_maps:
                flow_file = os.path.join(flow_dir, f"flow_{frame_count:06d}.npy")
                np.save(flow_file, flow)
            
            # Save visualization
            if save_visualization:
                vis_frame = self._create_flow_visualization(flow)
                vis_file = os.path.join(vis_dir, f"flow_{frame_count:06d}.png")
                cv2.imwrite(vis_file, vis_frame)
            
            # Store flow data
            flow_data.append({
                "frame": frame_count,
                "flow_shape": flow.shape,
                "flow_mean": float(np.mean(flow)),
                "flow_std": float(np.std(flow))
            })
            
            # Update previous frame
            prev_gray = curr_gray
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        
        # Save flow summary
        if save_flow_maps:
            summary_file = os.path.join(output_dir, "flow_summary.json")
            import json
            with open(summary_file, 'w') as f:
                json.dump({
                    "total_frames": frame_count,
                    "flow_frames": len(flow_data),
                    "flow_data": flow_data
                }, f, indent=2)
        
        logger.info(f"Flow extraction complete. Processed {frame_count} frames")
        return output_dir
    
    def _extract_raft(
        self,
        video_path: str,
        output_dir: str,
        save_visualization: bool,
        save_flow_maps: bool
    ) -> str:
        """Extract optical flow using RAFT (placeholder)."""
        # This is a placeholder for RAFT integration
        # In practice, you'd load the RAFT model and process frames
        
        logger.warning("RAFT extraction not fully implemented, falling back to OpenCV")
        return self._extract_opencv(video_path, output_dir, save_visualization, save_flow_maps)
    
    def _create_flow_visualization(self, flow: np.ndarray) -> np.ndarray:
        """Create a visualization of optical flow."""
        # Convert flow to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Normalize magnitude for visualization
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Create HSV image
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
        hsv[..., 1] = 255  # Saturation
        hsv[..., 2] = magnitude.astype(np.uint8)  # Value
        
        # Convert to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return bgr
    
    def load_flow_sequence(
        self,
        flow_dir: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Load optical flow sequence from directory.
        
        Args:
            flow_dir: Directory containing flow map files
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all frames)
            
        Returns:
            List of flow arrays
        """
        flow_sequence = []
        
        # Load flow files
        flow_files = sorted([
            f for f in os.listdir(flow_dir) 
            if f.endswith(".npy") and f.startswith("flow_")
        ])
        
        for flow_file in flow_files:
            frame_num = int(flow_file.split("_")[1].split(".")[0])
            if start_frame <= frame_num and (end_frame is None or frame_num <= end_frame):
                flow_path = os.path.join(flow_dir, flow_file)
                flow = np.load(flow_path)
                flow_sequence.append(flow)
        
        return flow_sequence
    
    def create_flow_video(
        self,
        flow_sequence: List[np.ndarray],
        output_path: str,
        fps: float = 30.0
    ):
        """
        Create a video from flow sequence.
        
        Args:
            flow_sequence: List of flow arrays
            output_path: Output video path
            fps: Frames per second
        """
        if not flow_sequence:
            raise ValueError("Empty flow sequence")
        
        # Get frame size from first flow
        height, width = flow_sequence[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for flow in flow_sequence:
            # Create visualization
            vis_frame = self._create_flow_visualization(flow)
            out.write(vis_frame)
        
        out.release()
        logger.info(f"Flow video saved: {output_path}")
    
    def compute_flow_statistics(
        self,
        flow_sequence: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute statistics from flow sequence.
        
        Args:
            flow_sequence: List of flow arrays
            
        Returns:
            Dictionary of flow statistics
        """
        if not flow_sequence:
            return {}
        
        # Stack all flows
        flows = np.stack(flow_sequence)
        
        # Compute magnitude
        magnitude = np.sqrt(flows[..., 0]**2 + flows[..., 1]**2)
        
        stats = {
            "mean_magnitude": float(np.mean(magnitude)),
            "std_magnitude": float(np.std(magnitude)),
            "max_magnitude": float(np.max(magnitude)),
            "min_magnitude": float(np.min(magnitude)),
            "mean_flow_x": float(np.mean(flows[..., 0])),
            "mean_flow_y": float(np.mean(flows[..., 1])),
            "std_flow_x": float(np.std(flows[..., 0])),
            "std_flow_y": float(np.std(flows[..., 1]))
        }
        
        return stats
