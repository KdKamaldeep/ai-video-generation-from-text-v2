"""
Pose extraction module for motion conditioning.
Supports MediaPipe and OpenPose for extracting pose keypoints from videos.
"""

import cv2
import json
import logging
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import mediapipe as mp

logger = logging.getLogger(__name__)


class PoseExtractor:
    """Extract pose keypoints from videos using MediaPipe or OpenPose."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the pose extractor."""
        self.config = config
        self.device = config.get("gpu_device", "cuda:0")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # MediaPipe pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_pose(
        self,
        video_path: str,
        output_dir: str,
        method: str = "mediapipe",
        save_visualization: bool = True,
        save_keypoints: bool = True
    ) -> str:
        """
        Extract pose keypoints from video.
        
        Args:
            video_path: Input video path
            output_dir: Output directory for pose data
            method: Extraction method ("mediapipe" or "openpose")
            save_visualization: Save pose visualization images
            save_keypoints: Save keypoint JSON files
            
        Returns:
            Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if method == "mediapipe":
            return self._extract_mediapipe(video_path, output_dir, save_visualization, save_keypoints)
        elif method == "openpose":
            return self._extract_openpose(video_path, output_dir, save_visualization, save_keypoints)
        else:
            raise ValueError(f"Unsupported pose extraction method: {method}")
    
    def _extract_mediapipe(
        self,
        video_path: str,
        output_dir: str,
        save_visualization: bool,
        save_keypoints: bool
    ) -> str:
        """Extract pose using MediaPipe."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = 0
        keypoints_data = []
        
        # Create subdirectories
        if save_visualization:
            vis_dir = os.path.join(output_dir, "visualization")
            os.makedirs(vis_dir, exist_ok=True)
        
        if save_keypoints:
            keypoints_dir = os.path.join(output_dir, "keypoints")
            os.makedirs(keypoints_dir, exist_ok=True)
        
        logger.info("Extracting pose keypoints with MediaPipe...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract keypoints
                keypoints = self._extract_mediapipe_keypoints(results.pose_landmarks, frame.shape)
                keypoints_data.append({
                    "frame": frame_count,
                    "keypoints": keypoints,
                    "confidence": self._get_pose_confidence(results.pose_landmarks)
                })
                
                # Save keypoints JSON
                if save_keypoints:
                    keypoints_file = os.path.join(keypoints_dir, f"frame_{frame_count:06d}.json")
                    with open(keypoints_file, 'w') as f:
                        json.dump(keypoints_data[-1], f, indent=2)
                
                # Save visualization
                if save_visualization:
                    vis_frame = frame.copy()
                    self.mp_drawing.draw_landmarks(
                        vis_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    vis_file = os.path.join(vis_dir, f"frame_{frame_count:06d}.png")
                    cv2.imwrite(vis_file, vis_frame)
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        
        # Save summary keypoints file
        if save_keypoints:
            summary_file = os.path.join(output_dir, "keypoints_summary.json")
            with open(summary_file, 'w') as f:
                json.dump({
                    "total_frames": frame_count,
                    "frames_with_pose": len(keypoints_data),
                    "keypoints": keypoints_data
                }, f, indent=2)
        
        logger.info(f"Pose extraction complete. Processed {frame_count} frames, found pose in {len(keypoints_data)} frames")
        return output_dir
    
    def _extract_mediapipe_keypoints(
        self,
        landmarks,
        frame_shape: Tuple[int, int, int]
    ) -> Dict[str, List[float]]:
        """Extract keypoints from MediaPipe landmarks."""
        height, width = frame_shape[:2]
        
        keypoints = {}
        for i, landmark in enumerate(landmarks.landmark):
            # Normalize coordinates to image dimensions
            x = landmark.x * width
            y = landmark.y * height
            z = landmark.z * width  # MediaPipe uses width for depth
            
            keypoints[f"landmark_{i}"] = [x, y, z, landmark.visibility]
        
        return keypoints
    
    def _get_pose_confidence(self, landmarks) -> float:
        """Calculate overall pose confidence from landmarks."""
        visibilities = [landmark.visibility for landmark in landmarks.landmark]
        return np.mean(visibilities)
    
    def _extract_openpose(
        self,
        video_path: str,
        output_dir: str,
        save_visualization: bool,
        save_keypoints: bool
    ) -> str:
        """Extract pose using OpenPose (subprocess call)."""
        import subprocess
        import tempfile
        
        # Check if OpenPose is available
        try:
            result = subprocess.run(["OpenPoseDemo", "--help"], capture_output=True)
            if result.returncode != 0:
                raise FileNotFoundError("OpenPose not found")
        except FileNotFoundError:
            raise RuntimeError("OpenPose not installed or not in PATH")
        
        # Create temporary directory for OpenPose output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run OpenPose
            cmd = [
                "OpenPoseDemo",
                "--video", video_path,
                "--write_json", temp_dir,
                "--display", "0",
                "--render_pose", "0"
            ]
            
            if save_visualization:
                cmd.extend(["--write_images", temp_dir])
            
            logger.info("Running OpenPose...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"OpenPose failed: {result.stderr}")
            
            # Copy results to output directory
            self._copy_openpose_results(temp_dir, output_dir, save_visualization, save_keypoints)
        
        return output_dir
    
    def _copy_openpose_results(
        self,
        temp_dir: str,
        output_dir: str,
        save_visualization: bool,
        save_keypoints: bool
    ):
        """Copy OpenPose results from temporary directory."""
        import shutil
        
        # Copy JSON keypoints
        if save_keypoints:
            keypoints_dir = os.path.join(output_dir, "keypoints")
            os.makedirs(keypoints_dir, exist_ok=True)
            
            for file in os.listdir(temp_dir):
                if file.endswith("_keypoints.json"):
                    src = os.path.join(temp_dir, file)
                    dst = os.path.join(keypoints_dir, file)
                    shutil.copy2(src, dst)
        
        # Copy visualization images
        if save_visualization:
            vis_dir = os.path.join(output_dir, "visualization")
            os.makedirs(vis_dir, exist_ok=True)
            
            for file in os.listdir(temp_dir):
                if file.endswith(".png") and not file.endswith("_keypoints.png"):
                    src = os.path.join(temp_dir, file)
                    dst = os.path.join(vis_dir, file)
                    shutil.copy2(src, dst)
    
    def get_pose_sequence(
        self,
        keypoints_dir: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load pose keypoints sequence from directory.
        
        Args:
            keypoints_dir: Directory containing keypoint JSON files
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all frames)
            
        Returns:
            List of pose keypoint dictionaries
        """
        keypoints = []
        
        # Load summary file if available
        summary_file = os.path.join(keypoints_dir, "keypoints_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                data = json.load(f)
                keypoints = data["keypoints"]
        else:
            # Load individual frame files
            frame_files = sorted([
                f for f in os.listdir(keypoints_dir) 
                if f.endswith(".json") and f.startswith("frame_")
            ])
            
            for frame_file in frame_files:
                frame_num = int(frame_file.split("_")[1].split(".")[0])
                if start_frame <= frame_num and (end_frame is None or frame_num <= end_frame):
                    with open(os.path.join(keypoints_dir, frame_file), 'r') as f:
                        keypoints.append(json.load(f))
        
        return keypoints
    
    def visualize_pose_sequence(
        self,
        keypoints_sequence: List[Dict[str, Any]],
        output_path: str,
        frame_size: Tuple[int, int] = (512, 512)
    ):
        """
        Create a visualization video of pose sequence.
        
        Args:
            keypoints_sequence: List of pose keypoint dictionaries
            output_path: Output video path
            frame_size: Video frame size
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, frame_size)
        
        for pose_data in keypoints_sequence:
            # Create blank frame
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            
            # Draw pose keypoints (simplified visualization)
            if "keypoints" in pose_data:
                self._draw_keypoints_on_frame(frame, pose_data["keypoints"])
            
            out.write(frame)
        
        out.release()
        logger.info(f"Pose visualization saved: {output_path}")
    
    def _draw_keypoints_on_frame(self, frame: np.ndarray, keypoints: Dict[str, List[float]]):
        """Draw keypoints on a frame."""
        # This is a simplified visualization
        # In practice, you'd want to draw proper pose connections
        
        for key, coords in keypoints.items():
            if len(coords) >= 2:
                x, y = int(coords[0]), int(coords[1])
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
