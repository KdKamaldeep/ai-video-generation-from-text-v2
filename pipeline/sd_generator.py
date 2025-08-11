"""
Stable Diffusion image generator for WhyWouldYou-v2 pipeline.
"""

import logging
import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import logging as diffusers_logging
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("diffusers not available, falling back to subprocess calls")

logger = logging.getLogger(__name__)


class SDGenerator:
    """Stable Diffusion image generator with LoRA and IP-Adapter support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the SD generator."""
        self.config = config
        self.device = config.get("gpu_device", "cuda:0")
        self.model_id = config.get("sd_model", "runwayml/stable-diffusion-v1-5")
        self.lora_path = config.get("lo_ra")
        self.ip_adapter_path = config.get("ip_adapter_path")
        
        if DIFFUSERS_AVAILABLE:
            self._load_pipeline()
        else:
            logger.warning("Using subprocess fallback for SD generation")
    
    def _load_pipeline(self):
        """Load the Stable Diffusion pipeline."""
        try:
            logger.info(f"Loading SD model: {self.model_id}")
            
            # Load base pipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Use DPM++ 2M scheduler for better quality
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Load LoRA if specified
            if self.lora_path and os.path.exists(self.lora_path):
                logger.info(f"Loading LoRA: {self.lora_path}")
                self.pipeline.load_lora_weights(self.lora_path)
            
            # Load IP-Adapter if specified
            if self.ip_adapter_path and os.path.exists(self.ip_adapter_path):
                logger.info(f"Loading IP-Adapter: {self.ip_adapter_path}")
                # Note: IP-Adapter loading would go here
                # This is a placeholder for IP-Adapter integration
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                self.pipeline.enable_xformers_memory_efficient_attention()
            
            logger.info("SD pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SD pipeline: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        output_path: str,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        negative_prompt: str = ""
    ) -> str:
        """
        Generate an image using Stable Diffusion.
        
        Args:
            prompt: Text prompt for generation
            output_path: Output image path
            seed: Random seed for reproducibility
            width: Image width
            height: Image height
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of denoising steps
            negative_prompt: Negative prompt
            
        Returns:
            Path to generated image
        """
        if not DIFFUSERS_AVAILABLE:
            return self._generate_subprocess(
                prompt, output_path, seed, width, height, 
                guidance_scale, num_inference_steps, negative_prompt
            )
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            logger.info(f"Generating image with prompt: {prompt[:50]}...")
            
            # Generate image
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
            )
            
            # Save image
            image = result.images[0]
            image.save(output_path)
            
            logger.info(f"Image generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    def _generate_subprocess(
        self,
        prompt: str,
        output_path: str,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        negative_prompt: str = ""
    ) -> str:
        """Fallback to subprocess call for SD generation."""
        import subprocess
        import json
        import tempfile
        
        # Create temporary config file
        config_data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed or 42,
            "output_path": output_path,
            "model_id": self.model_id
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # Call external SD script (placeholder)
            cmd = [
                "python", "scripts/generate_sd.py",
                "--config", config_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"SD generation failed: {result.stderr}")
            
            return output_path
            
        finally:
            os.unlink(config_file)
    
    def generate_batch(
        self,
        prompts: list,
        output_dir: str,
        **kwargs
    ) -> list:
        """
        Generate multiple images from a list of prompts.
        
        Args:
            prompts: List of text prompts
            output_dir: Output directory for images
            **kwargs: Additional arguments for generate()
            
        Returns:
            List of generated image paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        generated_paths = []
        for i, prompt in enumerate(prompts):
            output_path = os.path.join(output_dir, f"image_{i:04d}.png")
            try:
                path = self.generate(prompt, output_path, **kwargs)
                generated_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to generate image {i}: {e}")
                continue
        
        logger.info(f"Generated {len(generated_paths)} images")
        return generated_paths
