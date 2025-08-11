#!/usr/bin/env python3
"""
Create Example Dataset Script

This script creates a simple example dataset for testing the WhyWouldYou-v2 pipeline.
It generates a basic reference video and example images for demonstration.
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging import setup_logging

def create_stick_figure_video(output_path: str, duration: float = 3.0, fps: int = 24):
    """Create a simple stick figure animation video."""
    width, height = 640, 480
    total_frames = int(duration * fps)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(total_frames):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Calculate animation progress (0 to 1)
        progress = frame_idx / (total_frames - 1)
        
        # Simple stick figure animation (walking cycle)
        center_x = int(width * 0.5)
        center_y = int(height * 0.7)
        
        # Head
        head_radius = 30
        cv2.circle(frame, (center_x, center_y - 80), head_radius, (0, 0, 0), 2)
        
        # Body
        body_start = (center_x, center_y - 50)
        body_end = (center_x, center_y + 20)
        cv2.line(frame, body_start, body_end, (0, 0, 0), 2)
        
        # Arms (swinging)
        arm_swing = np.sin(progress * 4 * np.pi) * 20
        left_arm = (center_x - 40, center_y - 30)
        right_arm = (center_x + 40, center_y - 30)
        cv2.line(frame, body_start, left_arm, (0, 0, 0), 2)
        cv2.line(frame, body_start, right_arm, (0, 0, 0), 2)
        
        # Legs (walking motion)
        leg_swing = np.sin(progress * 4 * np.pi) * 15
        left_leg = (center_x - 20 + int(leg_swing), center_y + 60)
        right_leg = (center_x + 20 - int(leg_swing), center_y + 60)
        cv2.line(frame, body_end, left_leg, (0, 0, 0), 2)
        cv2.line(frame, body_end, right_leg, (0, 0, 0), 2)
        
        # Add some movement across the screen
        offset_x = int(progress * 200 - 100)
        frame = np.roll(frame, offset_x, axis=1)
        
        out.write(frame)
    
    out.release()
    print(f"Created stick figure video: {output_path}")

def create_example_image(output_path: str, prompt: str = "cartoon blue cat"):
    """Create a simple example image using basic shapes."""
    width, height = 512, 512
    
    # Create image with gradient background
    image = Image.new('RGB', (width, height), color='lightblue')
    draw = ImageDraw.Draw(image)
    
    # Add some simple shapes to represent a cartoon character
    # Body (ellipse)
    draw.ellipse([150, 200, 350, 400], fill='blue', outline='darkblue', width=3)
    
    # Head (circle)
    draw.ellipse([200, 100, 300, 200], fill='lightblue', outline='darkblue', width=3)
    
    # Eyes
    draw.ellipse([220, 130, 240, 150], fill='white', outline='black', width=2)
    draw.ellipse([260, 130, 280, 150], fill='white', outline='black', width=2)
    draw.ellipse([225, 135, 235, 145], fill='black')  # Pupils
    draw.ellipse([265, 135, 275, 145], fill='black')
    
    # Nose
    draw.polygon([(245, 160), (235, 170), (255, 170)], fill='pink')
    
    # Ears
    draw.ellipse([190, 80, 210, 120], fill='lightblue', outline='darkblue', width=2)
    draw.ellipse([290, 80, 310, 120], fill='lightblue', outline='darkblue', width=2)
    
    # Arms
    draw.ellipse([120, 250, 140, 320], fill='blue', outline='darkblue', width=2)
    draw.ellipse([360, 250, 380, 320], fill='blue', outline='darkblue', width=2)
    
    # Legs
    draw.ellipse([180, 380, 200, 450], fill='blue', outline='darkblue', width=2)
    draw.ellipse([300, 380, 320, 450], fill='blue', outline='darkblue', width=2)
    
    # Add text label
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((10, 10), f"Example: {prompt}", fill='black', font=font)
    
    image.save(output_path)
    print(f"Created example image: {output_path}")

def create_example_audio(output_path: str, text: str = "Hello, this is a test!"):
    """Create a simple audio file with silence (placeholder for TTS)."""
    # Create a simple sine wave tone
    sample_rate = 22050
    duration = 2.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Add some fade in/out
    fade_samples = int(0.1 * sample_rate)
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Convert to 16-bit PCM
    audio = (audio * 32767).astype(np.int16)
    
    # Save as WAV
    import wave
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    print(f"Created example audio: {output_path}")

def create_example_config(output_path: str):
    """Create an example configuration file for testing."""
    config = {
        "sd_model": "runwayml/stable-diffusion-v1-5",
        "svd_model": "stabilityai/stable-video-diffusion-img2vid-xt",
        "controlnet_pose_model": None,
        "lo_ra": None,
        "ip_adapter_path": None,
        "output_resolution": [480, 640],  # Smaller for testing
        "fps": 24,
        "frames": 24,  # Shorter for testing
        "strength": 0.56,
        "guidance_scale": 7.5,
        "seed": 42,
        "coqui_voice": {
            "voice": "alloy",
            "sample_rate": 24000,
            "model": "tts_models/multilingual/multi-dataset/xtts_v2"
        },
        "wav2lip_enabled": False,  # Disable for testing
        "wav2lip_model": "models/wav2lip.pth",
        "face_detection_model": "models/s3fd.pth",
        "post_rife": False,
        "rife_fps": 48,
        "post_smear": False,
        "post_squash": False,
        "crossfade_duration": 0.5,
        "temp_dirs": {
            "images": "images",
            "motion": "motion",
            "clips": "clips",
            "audio": "audio",
            "output": "output"
        },
        "gpu_device": "cpu"  # Use CPU for testing
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created example config: {output_path}")

def create_example_dataset():
    """Create a complete example dataset for testing."""
    # Setup logging
    setup_logging(verbose=True)
    
    # Create directories
    base_dir = project_root / "examples"
    base_dir.mkdir(exist_ok=True)
    
    (base_dir / "images").mkdir(exist_ok=True)
    (base_dir / "motion").mkdir(exist_ok=True)
    (base_dir / "audio").mkdir(exist_ok=True)
    (base_dir / "output").mkdir(exist_ok=True)
    
    print("Creating example dataset...")
    
    # Create example video
    video_path = base_dir / "motion" / "stick_figure_walking.mp4"
    create_stick_figure_video(str(video_path), duration=3.0, fps=24)
    
    # Create example images
    image_path = base_dir / "images" / "cartoon_cat.png"
    create_example_image(str(image_path), "cartoon blue cat")
    
    # Create example audio
    audio_path = base_dir / "audio" / "test_audio.wav"
    create_example_audio(str(audio_path), "Hello, this is a test!")
    
    # Create example config
    config_path = base_dir / "config_example.json"
    create_example_config(str(config_path))
    
    # Create README for the example
    readme_path = base_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("""# Example Dataset

This directory contains example files for testing the WhyWouldYou-v2 pipeline.

## Files

- `images/cartoon_cat.png` - Example base image for SVD
- `motion/stick_figure_walking.mp4` - Example reference motion video
- `audio/test_audio.wav` - Example audio file
- `config_example.json` - Example configuration file

## Usage

1. Test the pipeline with these files:
```bash
python main.py run-all --config examples/config_example.json \\
    --prompt "cartoon blue cat running" \\
    --motion_video examples/motion/stick_figure_walking.mp4 \\
    --tts_text "Hello, this is a test!"
```

2. Or run individual steps:
```bash
# Generate base image
python main.py image --prompt "cartoon blue cat" --config examples/config_example.json

# Extract motion
python main.py extract-motion --motion_video examples/motion/stick_figure_walking.mp4

# Generate TTS
python main.py tts --text "Hello, this is a test!" --config examples/config_example.json
```

## Notes

- These are simple placeholder files for testing
- The video is a basic stick figure animation
- The image is a simple cartoon character
- The audio is a basic tone (not real speech)
- Use CPU mode for testing (set in config)
""")
    
    print(f"\nExample dataset created in: {base_dir}")
    print("Files created:")
    print(f"  - {video_path}")
    print(f"  - {image_path}")
    print(f"  - {audio_path}")
    print(f"  - {config_path}")
    print(f"  - {readme_path}")
    print("\nYou can now test the pipeline with these example files!")

if __name__ == "__main__":
    create_example_dataset()
