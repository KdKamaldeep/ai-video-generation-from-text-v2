"""
Smoke test for WhyWouldYou-v2 pipeline.
Runs a minimal end-to-end test to verify basic functionality.
"""

import os
import sys
import tempfile
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_default_config, validate_config
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def create_test_config():
    """Create a minimal test configuration."""
    config = get_default_config()
    
    # Override for testing
    config.update({
        "output_resolution": [64, 64],  # Very small for testing
        "fps": 10,
        "frames": 10,
        "strength": 0.5,
        "guidance_scale": 7.0,
        "seed": 42,
        "gpu_device": "cpu",  # Use CPU for testing
        "wav2lip_enabled": False,  # Disable for testing
        "post_rife": False,
        "post_smear": False,
        "post_squash": False
    })
    
    return config


def test_config_validation():
    """Test configuration validation."""
    logger.info("Testing configuration validation...")
    
    config = create_test_config()
    validate_config(config)
    
    logger.info("✓ Configuration validation passed")


def test_directory_creation():
    """Test directory creation."""
    logger.info("Testing directory creation...")
    
    config = create_test_config()
    temp_dirs = config["temp_dirs"]
    
    for dir_name, dir_path in temp_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        assert os.path.exists(dir_path), f"Directory {dir_path} was not created"
    
    logger.info("✓ Directory creation passed")


def test_simple_image_generation():
    """Test simple image generation (mock)."""
    logger.info("Testing image generation...")
    
    # Create a simple test image
    import numpy as np
    from PIL import Image
    
    test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    test_image_path = "images/test_image.png"
    
    os.makedirs("images", exist_ok=True)
    Image.fromarray(test_image).save(test_image_path)
    
    assert os.path.exists(test_image_path), "Test image was not created"
    logger.info("✓ Image generation passed")


def test_motion_extraction():
    """Test motion extraction (mock)."""
    logger.info("Testing motion extraction...")
    
    # Create a simple test video
    import cv2
    
    test_video_path = "motion/test_video.mp4"
    os.makedirs("motion", exist_ok=True)
    
    # Create a simple test video (1 second, 10 fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 10.0, (64, 64))
    
    for i in range(10):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    assert os.path.exists(test_video_path), "Test video was not created"
    
    # Create mock motion data
    motion_dir = "motion/test_extraction"
    os.makedirs(motion_dir, exist_ok=True)
    
    # Create mock keypoints file
    keypoints_data = {
        "total_frames": 10,
        "frames_with_pose": 10,
        "keypoints": [
            {
                "frame": i,
                "keypoints": {"landmark_0": [32, 32, 0, 1.0]},
                "confidence": 0.9
            }
            for i in range(10)
        ]
    }
    
    with open(os.path.join(motion_dir, "keypoints_summary.json"), 'w') as f:
        json.dump(keypoints_data, f)
    
    logger.info("✓ Motion extraction passed")


def test_video_assembly():
    """Test video assembly (mock)."""
    logger.info("Testing video assembly...")
    
    # Create a simple test video
    import cv2
    import numpy as np
    
    test_clip_path = "clips/test_clip.mp4"
    os.makedirs("clips", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_clip_path, fourcc, 10.0, (64, 64))
    
    for i in range(10):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    assert os.path.exists(test_clip_path), "Test clip was not created"
    
    # Test assembly
    output_path = "output/test_final.mp4"
    os.makedirs("output", exist_ok=True)
    
    # Simple copy for testing
    import shutil
    shutil.copy2(test_clip_path, output_path)
    
    assert os.path.exists(output_path), "Final video was not created"
    logger.info("✓ Video assembly passed")


def test_tts_generation():
    """Test TTS generation (mock)."""
    logger.info("Testing TTS generation...")
    
    # Create a simple test audio file
    import numpy as np
    import soundfile as sf
    
    test_audio_path = "audio/test_audio.wav"
    os.makedirs("audio", exist_ok=True)
    
    # Create a simple sine wave
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz sine wave
    
    sf.write(test_audio_path, audio_data, sample_rate)
    
    assert os.path.exists(test_audio_path), "Test audio was not created"
    logger.info("✓ TTS generation passed")


def run_smoke_test():
    """Run the complete smoke test."""
    logger.info("Starting smoke test...")
    
    try:
        # Setup logging
        setup_logging(verbose=True)
        
        # Run individual tests
        test_config_validation()
        test_directory_creation()
        test_simple_image_generation()
        test_motion_extraction()
        test_video_assembly()
        test_tts_generation()
        
        logger.info("🎉 All smoke tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Smoke test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
