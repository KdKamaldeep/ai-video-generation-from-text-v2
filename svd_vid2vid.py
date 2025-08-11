"""
Stable Video Diffusion (SVD) vid2vid module for WhyWouldYou-v2 pipeline.
"""

import cv2
import logging
import numpy as np
import os
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import json

try:
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import export_to_video
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("diffusers not available, falling back to subprocess calls")

logger = logging.getLogger(__name__)


class SVDVid2Vid:
    """Stable Video Diffusion vid2vid generator."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the SVD generator."""
        self.config = config
        self.device = config.get("gpu_device", "cuda:0")
        self.model_id = config.get("svd_model", "stabilityai/stable-video-diffusion-img2vid-xt")
        
        if DIFFUSERS_AVAILABLE:
            self._load_pipeline()
        else:
            logger.warning("Using subprocess fallback for SVD generation")
    
    def _load_pipeline(self):
        """Load the SVD pipeline."""
        try:
            logger.info(f"Loading SVD model: {self.model_id}")
            
            # Load SVD pipeline
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                variant="fp16" if torch.cuda.is_available() else None
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                self.pipeline.enable_xformers_memory_efficient_attention()
            
            logger.info("SVD pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SVD pipeline: {e}")
            raise
    
    def generate(
        self,
        input_image: str,
        motion_dir: str,
        output_path: str,
        frames: int = 48,
        fps: int = 24,
        strength: float = 0.56,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        motion_consistency: float = 0.8,
        num_inference_steps: int = 25
    ) -> str:
        """
        Generate video using SVD vid2vid.
        
        Args:
            input_image: Base image path
            motion_dir: Directory with motion conditioning data
            output_path: Output video path
            frames: Number of frames to generate
            fps: Output FPS
            strength: Motion strength
            guidance_scale: Guidance scale
            seed: Random seed
            motion_consistency: Motion consistency parameter
            num_inference_steps: Number of denoising steps
            
        Returns:
            Path to generated video
        """
        if not DIFFUSERS_AVAILABLE:
            return self._generate_subprocess(
                input_image, motion_dir, output_path, frames, fps,
                strength, guidance_scale, seed, motion_consistency, num_inference_steps
            )
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            logger.info(f"Generating SVD video from {input_image}")
            
            # Load input image
            image = Image.open(input_image).convert("RGB")
            
            # Load motion conditioning if available
            motion_conditioning = self._load_motion_conditioning(motion_dir)
            
            # Generate video
            result = self.pipeline(
                image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_frames=frames,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None,
                motion_consistency=motion_consistency,
                # Add motion conditioning if available
                **motion_conditioning
            )
            
            # Save video
            export_to_video(result.frames[0], output_path, fps=fps)
            
            logger.info(f"SVD video generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"SVD video generation failed: {e}")
            raise
    
    def _load_motion_conditioning(self, motion_dir: str) -> Dict[str, Any]:
        """Load motion conditioning data from directory."""
        conditioning = {}
        
        # Check for pose keypoints
        keypoints_dir = os.path.join(motion_dir, "keypoints")
        if os.path.exists(keypoints_dir):
            try:
                pose_sequence = self._load_pose_sequence(keypoints_dir)
                if pose_sequence:
                    conditioning["pose_conditioning"] = pose_sequence
                    logger.info(f"Loaded pose conditioning: {len(pose_sequence)} frames")
            except Exception as e:
                logger.warning(f"Failed to load pose conditioning: {e}")
        
        # Check for optical flow
        flow_dir = os.path.join(motion_dir, "flow_maps")
        if os.path.exists(flow_dir):
            try:
                flow_sequence = self._load_flow_sequence(flow_dir)
                if flow_sequence:
                    conditioning["flow_conditioning"] = flow_sequence
                    logger.info(f"Loaded flow conditioning: {len(flow_sequence)} frames")
            except Exception as e:
                logger.warning(f"Failed to load flow conditioning: {e}")
        
        return conditioning
    
    def _load_pose_sequence(self, keypoints_dir: str) -> List[np.ndarray]:
        """Load pose keypoints sequence."""
        from pose_extractor import PoseExtractor
        
        extractor = PoseExtractor(self.config)
        pose_data = extractor.get_pose_sequence(keypoints_dir)
        
        # Convert to numpy arrays for SVD
        pose_sequence = []
        for pose_frame in pose_data:
            if "keypoints" in pose_frame:
                # Convert keypoints to numpy array
                keypoints = pose_frame["keypoints"]
                pose_array = self._keypoints_to_array(keypoints)
                pose_sequence.append(pose_array)
        
        return pose_sequence
    
    def _load_flow_sequence(self, flow_dir: str) -> List[np.ndarray]:
        """Load optical flow sequence."""
        from flow_extractor import FlowExtractor
        
        extractor = FlowExtractor(self.config)
        return extractor.load_flow_sequence(flow_dir)
    
    def _keypoints_to_array(self, keypoints: Dict[str, List[float]]) -> np.ndarray:
        """Convert keypoints dictionary to numpy array."""
        # Extract coordinates from keypoints
        coords = []
        for key in sorted(keypoints.keys()):
            if key.startswith("landmark_"):
                coords.extend(keypoints[key][:2])  # Only x, y coordinates
        
        return np.array(coords, dtype=np.float32)
    
    def _generate_subprocess(
        self,
        input_image: str,
        motion_dir: str,
        output_path: str,
        frames: int,
        fps: int,
        strength: float,
        guidance_scale: float,
        seed: Optional[int],
        motion_consistency: float,
        num_inference_steps: int
    ) -> str:
        """Fallback to subprocess call for SVD generation."""
        import subprocess
        import tempfile
        
        # Create temporary config file
        config_data = {
            "input_image": input_image,
            "motion_dir": motion_dir,
            "output_path": output_path,
            "frames": frames,
            "fps": fps,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "seed": seed or 42,
            "motion_consistency": motion_consistency,
            "num_inference_steps": num_inference_steps,
            "model_id": self.model_id
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # Call external SVD script (placeholder)
            cmd = [
                "python", "scripts/generate_svd.py",
                "--config", config_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"SVD generation failed: {result.stderr}")
            
            return output_path
            
        finally:
            os.unlink(config_file)
    
    def generate_with_controlnet(
        self,
        input_image: str,
        controlnet_image: str,
        output_path: str,
        **kwargs
    ) -> str:
        """
        Generate video with ControlNet conditioning.
        
        Args:
            input_image: Base image path
            controlnet_image: ControlNet conditioning image path
            output_path: Output video path
            **kwargs: Additional arguments for generate()
            
        Returns:
            Path to generated video
        """
        # This is a placeholder for ControlNet integration
        # In practice, you'd modify the SVD pipeline to use ControlNet
        
        logger.warning("ControlNet integration not fully implemented")
        return self.generate(input_image, "", output_path, **kwargs)
    
    def create_debug_frames(
        self,
        input_image: str,
        motion_dir: str,
        output_dir: str,
        num_frames: int = 5
    ):
        """
        Create debug frames for troubleshooting.
        
        Args:
            input_image: Base image path
            motion_dir: Motion conditioning directory
            output_dir: Output directory for debug frames
            num_frames: Number of debug frames to generate
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load input image
        image = Image.open(input_image).convert("RGB")
        image_np = np.array(image)
        
        # Load motion conditioning
        motion_conditioning = self._load_motion_conditioning(motion_dir)
        
        # Create debug frames
        for i in range(num_frames):
            debug_frame = image_np.copy()
            
            # Add motion conditioning visualization
            if "pose_conditioning" in motion_conditioning and i < len(motion_conditioning["pose_conditioning"]):
                pose = motion_conditioning["pose_conditioning"][i]
                debug_frame = self._draw_pose_on_frame(debug_frame, pose)
            
            if "flow_conditioning" in motion_conditioning and i < len(motion_conditioning["flow_conditioning"]):
                flow = motion_conditioning["flow_conditioning"][i]
                debug_frame = self._draw_flow_on_frame(debug_frame, flow)
            
            # Save debug frame
            debug_path = os.path.join(output_dir, f"debug_{i:04d}.png")
            cv2.imwrite(debug_path, cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Debug frames saved to: {output_dir}")
    
    def _draw_pose_on_frame(self, frame: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Draw pose keypoints on frame."""
        # This is a simplified pose visualization
        # In practice, you'd draw proper pose connections
        
        for i in range(0, len(pose), 2):
            if i + 1 < len(pose):
                x, y = int(pose[i]), int(pose[i + 1])
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        return frame
    
    def _draw_flow_on_frame(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Draw optical flow on frame."""
        # Create flow visualization
        flow_vis = self._create_flow_visualization(flow)
        
        # Overlay on frame
        alpha = 0.3
        frame = cv2.addWeighted(frame, 1 - alpha, flow_vis, alpha, 0)
        
        return frame
    
    def _create_flow_visualization(self, flow: np.ndarray) -> np.ndarray:
        """Create flow visualization (same as in FlowExtractor)."""
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = magnitude.astype(np.uint8)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
