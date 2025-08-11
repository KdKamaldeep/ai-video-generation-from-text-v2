"""
Configuration utilities for WhyWouldYou-v2 pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and required fields."""
    required_fields = [
        "sd_model",
        "svd_model", 
        "output_resolution",
        "fps",
        "frames",
        "strength",
        "guidance_scale",
        "seed"
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in config:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Missing required configuration fields: {missing_fields}")
    
    # Validate specific field types and ranges
    if not isinstance(config["output_resolution"], list) or len(config["output_resolution"]) != 2:
        raise ValueError("output_resolution must be a list of [width, height]")
    
    if config["fps"] <= 0:
        raise ValueError("fps must be positive")
    
    if config["frames"] <= 0:
        raise ValueError("frames must be positive")
    
    if not (0.0 <= config["strength"] <= 1.0):
        raise ValueError("strength must be between 0.0 and 1.0")
    
    if config["guidance_scale"] <= 0:
        raise ValueError("guidance_scale must be positive")
    
    logger.info("Configuration validation passed")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        "sd_model": "runwayml/stable-diffusion-v1-5",
        "svd_model": "stabilityai/stable-video-diffusion-img2vid-xt",
        "controlnet_pose_model": None,
        "lo_ra": None,
        "ip_adapter_path": None,
        "output_resolution": [1080, 1920],  # Default for Shorts
        "fps": 24,
        "frames": 48,
        "strength": 0.56,
        "guidance_scale": 7.5,
        "seed": 42,
        "coqui_voice": {
            "voice": "alloy",
            "sample_rate": 24000
        },
        "wav2lip_enabled": True,
        "post_rife": False,
        "rife_fps": 48,
        "post_smear": False,
        "post_squash": False,
        "temp_dirs": {
            "images": "images",
            "motion": "motion", 
            "clips": "clips",
            "audio": "audio",
            "output": "output"
        },
        "gpu_device": "cuda:0"
    }


def create_config_template(output_path: str = "config.example.json") -> None:
    """Create a configuration template file."""
    config = get_default_config()
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration template created: {output_path}")
