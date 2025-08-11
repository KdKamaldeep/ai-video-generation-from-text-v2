# WhyWouldYou-v2: CLI Pipeline for SVD Video Generation

A complete, production-ready CLI pipeline for generating animated videos using Stable Video Diffusion (SVD). This pipeline takes a text scene description, reference motion video, and TTS text to create animated videos with lip-sync and post-processing effects.

## 🎯 Features

- **Stable Diffusion Image Generation**: Generate base images with LoRA and IP-Adapter support
- **Motion Conditioning**: Extract pose keypoints using MediaPipe or OpenPose
- **SVD Video Generation**: Apply motion to base images using Stable Video Diffusion
- **Text-to-Speech**: Generate audio using Coqui TTS
- **Lip-Sync**: Synchronize video with audio using Wav2Lip
- **Post-Processing**: RIFE interpolation, motion blur, squash & stretch effects
- **Video Assembly**: Combine clips with crossfades, captions, and background audio
- **Docker Support**: Containerized deployment with CUDA support

## 🚀 Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **CUDA**: CUDA 11.8 or 12.1
- **Python**: Python 3.10+
- **FFmpeg**: For video processing

> 💡 **Need help?** Check our [documentation](docs/README.md) for detailed guides!

## 🚀 RunPod Setup (Recommended)

The easiest way to get started is using RunPod with our automated setup script!

### Quick RunPod Setup

1. **Start a RunPod instance** with:
   - **GPU**: RTX 3080/4080/4090 or A100 (12GB+ VRAM recommended)
   - **RAM**: 32GB+
   - **Storage**: 100GB+
   - **OS**: Ubuntu 20.04/22.04

2. **Clone and setup** (copy-paste this into your RunPod terminal):
   ```bash
   cd /workspace
   git clone https://github.com/yourusername/WhyWouldYou-v2.git .
   chmod +x scripts/setup_runpod.sh
   sudo ./scripts/setup_runpod.sh
   ```

3. **Activate environment**:
   ```bash
   source /workspace/venv/bin/activate
   export WORKSPACE_DIR=/workspace
   export MODELS_DIR=/workspace/models
   export PYTHONPATH=/workspace:$PYTHONPATH
   ```

4. **Test the pipeline**:
   ```bash
   python scripts/create_example_dataset.py
   python main.py --help
   ```

### What the RunPod Setup Does

✅ **Automatically installs** all system dependencies (ffmpeg, opencv, etc.)  
✅ **Downloads all AI models** (Stable Diffusion, SVD, ControlNet, Wav2Lip, etc.)  
✅ **Sets up Python environment** with CUDA support  
✅ **Creates configuration files** ready to use  
✅ **Runs tests** to ensure everything works  
✅ **All models are FREE and open source!** 🎉  
✅ **No extra startup scripts** - direct environment activation

### Optional: Hugging Face Token

For faster model downloads, set your HF token:
```bash
export HF_TOKEN='your_token_here'
```
Get a free token at: https://huggingface.co/settings/tokens

### Recent Improvements

✅ **Fixed dependency conflicts** - Resolved scipy and torchaudio version issues  
✅ **Improved error handling** - Better fallback installation for problematic packages  
✅ **Enhanced compatibility** - Works with multiple Python versions (3.8+)  
✅ **Better troubleshooting** - Clear error messages and solution guidance  
✅ **No sudo required** - Works in restricted environments without elevated privileges

## 🐧 Linux Setup (One Command)

The fastest way to set up on any Linux system with our automated script!

### Quick Linux Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/WhyWouldYou-v2.git
   cd WhyWouldYou-v2
   ```

2. **Run the setup script** (one command):
   ```bash
   chmod +x scripts/setup_linux.sh
   ./scripts/setup_linux.sh
   ```

3. **Activate the environment**:
   ```bash
   source setup_env.sh
   ```

4. **Test the pipeline**:
   ```bash
   python main.py --help
   python tests/smoke_test.py
   ```

### What the Linux Setup Does

✅ **Detects your Linux distribution** (Ubuntu, Debian, CentOS, RHEL, Fedora, Arch)  
✅ **Installs all system dependencies** (Python, CUDA, FFmpeg, etc.)  
✅ **Creates Python virtual environment** with all dependencies  
✅ **Handles dependency conflicts** automatically with fallback installation  
✅ **Downloads all AI models** (Stable Diffusion, SVD, ControlNet, Wav2Lip, etc.)  
✅ **Sets up configuration files** ready to use  
✅ **Runs comprehensive tests** to ensure everything works  
✅ **Creates environment script** for easy activation  
✅ **All models are FREE and open source!** 🎉  

### System Requirements for Linux Setup

- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space (for models)
- **OS**: Any modern Linux distribution (Ubuntu 20.04+, CentOS 7+, Fedora 30+, Arch)

### Troubleshooting Setup Issues

If you encounter dependency conflicts or installation issues:

1. **Dependency conflicts**: The script automatically handles these with fallback installation
2. **Missing system packages**: The script will show you exactly what to install
3. **CUDA issues**: The script checks CUDA installation and provides guidance
4. **Permission issues**: Run as regular user (not root) - the script will guide you

### Optional: Hugging Face Token

For faster model downloads, set your HF token before running the setup:
```bash
export HF_TOKEN='your_token_here'
./scripts/setup_linux.sh
```
Get a free token at: https://huggingface.co/settings/tokens

### Local Installation

If you prefer to set up manually:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/WhyWouldYou-v2.git
   cd WhyWouldYou-v2
   ```

2. **Install system dependencies** (Ubuntu/Debian example):
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3.10 python3.10-venv python3-pip git ffmpeg
   ```

3. **Create virtual environment**:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools wheel
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up configuration**:
   ```bash
   cp config.example.json config.json
   # Edit config.json with your settings
   ```

6. **Download models**:
   ```bash
   chmod +x scripts/download_models.sh
   ./scripts/download_models.sh
   ```

7. **Set up environment**:
   ```bash
   source setup_env.sh
   ```

### Basic Usage

#### End-to-End Pipeline

```bash
# Run complete pipeline
python main.py run-all \
  --config config.json \
  --prompt "cartoon blue cat running, cel-shaded, Oggy-style" \
  --motion_video ref/run_cycle.mp4 \
  --tts_text "Watch out! Here I come!" \
  --out output/final_video.mp4
```

#### Step-by-Step Pipeline

```bash
# 1. Generate base image
python main.py image \
  --prompt "cartoon blue cat running, cel-shaded, Oggy-style" \
  --config config.json \
  --out images/base1.png

# 2. Extract motion from reference video
python main.py extract-motion \
  --motion_video ref/run_cycle.mp4 \
  --out motion/run1/ \
  --method mediapipe

# 3. Run SVD vid2vid
python main.py svd \
  --input_image images/base1.png \
  --motion_dir motion/run1/ \
  --out clips/scene1.mp4 \
  --config config.json

# 4. Generate TTS audio
python main.py tts \
  --text "Watch out!" \
  --out audio/line1.wav \
  --config config.json

# 5. Lip-sync (optional)
python main.py wav2lip \
  --video clips/scene1.mp4 \
  --audio audio/line1.wav \
  --out clips/scene1_synced.mp4

# 6. Post-process
python main.py post \
  --input clips/scene1_synced.mp4 \
  --out clips/scene1_final.mp4 \
  --rife \
  --smear

# 7. Assemble final video
python main.py assemble \
  --clips clips/scene1_final.mp4 \
  --audio bg/music.mp3 \
  --out final/final_video.mp4
```

## 📁 Project Structure

```
WhyWouldYou-v2/
├── main.py                 # Main CLI entrypoint
├── pipeline/               # Core pipeline modules
│   ├── __init__.py         # Pipeline package
│   ├── sd_generator.py     # Stable Diffusion image generation
│   ├── pose_extractor.py   # Pose keypoint extraction
│   ├── flow_extractor.py   # Optical flow extraction
│   ├── svd_vid2vid.py     # SVD video generation
│   ├── coqui_tts.py       # Text-to-speech generation
│   ├── wav2lip_infer.py   # Lip-sync processing
│   ├── postprocess.py     # Video post-processing
│   └── assemble.py        # Video assembly
├── utils/                 # Utility modules
│   ├── config.py          # Configuration management
│   ├── logging.py         # Logging setup
│   └── crop_to_aspect.py  # Aspect ratio utilities
├── docs/                  # Documentation
│   ├── README.md          # Documentation index
│   ├── ENVIRONMENT_SETUP.md # Environment setup guide
│   ├── RUNPOD_DEPLOYMENT.md # RunPod deployment guide
│   └── LICENSING.md       # Model licensing information
├── models/                # AI models directory
│   └── README.md          # Models documentation
├── scripts/               # Setup and utility scripts
│   ├── setup_runpod.sh    # RunPod setup script
│   ├── runpod_setup_one_liner.sh # Quick RunPod setup
│   ├── download_models.sh # Model download script
│   └── create_example_dataset.py # Example dataset creator
├── tests/                 # Test suite
│   └── smoke_test.py      # Smoke tests
├── config.example.json    # Example configuration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
└── README.md             # This file
```

## ⚙️ Configuration

The pipeline is configured via a JSON file. Key settings include:

### Core Settings
- `sd_model`: Stable Diffusion model (HuggingFace ID or local path)
- `svd_model`: SVD model for video generation
- `output_resolution`: Output video resolution `[width, height]`
- `fps`: Output frame rate
- `frames`: Number of frames to generate
- `strength`: Motion strength (0.0-1.0)
- `guidance_scale`: Guidance scale for generation

### TTS Settings
- `coqui_voice.voice`: Voice to use (alloy, echo, fable, etc.)
- `coqui_voice.sample_rate`: Audio sample rate

### Post-Processing
- `post_rife`: Enable RIFE frame interpolation
- `rife_fps`: Target FPS for interpolation
- `post_smear`: Add motion blur smears
- `post_squash`: Add squash & stretch effects

### Example Configuration
```json
{
  "sd_model": "runwayml/stable-diffusion-v1-5",
  "svd_model": "stabilityai/stable-video-diffusion-img2vid-xt",
  "output_resolution": [1080, 1920],
  "fps": 24,
  "frames": 48,
  "strength": 0.56,
  "guidance_scale": 7.5,
  "seed": 42,
  "coqui_voice": {
    "voice": "alloy",
    "sample_rate": 24000
  },
  "wav2lip_enabled": true,
  "post_rife": false,
  "rife_fps": 48,
  "gpu_device": "cuda:0"
}
```

## 🤖 Models

### Required Models

The pipeline uses several AI models. You can download them manually or let the pipeline download them automatically:

#### Stable Diffusion Models
- **Base Model**: `runwayml/stable-diffusion-v1-5` (auto-downloaded)
- **LoRA Models**: Place in `models/` directory
- **IP-Adapter**: Place in `models/` directory

#### SVD Models
- **SVD Model**: `stabilityai/stable-video-diffusion-img2vid-xt` (auto-downloaded)

#### TTS Models
- **Coqui TTS**: Auto-downloaded on first use

#### Wav2Lip Models (Optional)
- **Wav2Lip**: Download from [Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip)
- **Face Detection**: S3FD model for face detection

### Model Download Script

Create a script to download models:

```bash
#!/bin/bash
# scripts/download_models.sh

mkdir -p models

# Download Wav2Lip models
wget -O models/wav2lip.pth https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip.pth
wget -O models/s3fd.pth https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/s3fd.pth

# Download LoRA models (example)
# wget -O models/cartoon_lora.pt https://example.com/cartoon_lora.pt

echo "Models downloaded successfully!"
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t whywouldyou-v2 .
```

### Run with GPU Support

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  whywouldyou-v2 \
  python main.py run-all \
    --config config.json \
    --prompt "cartoon character" \
    --motion_video data/ref.mp4 \
    --tts_text "Hello world!"
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  whywouldyou-v2:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./output:/app/output
    command: ["python", "main.py", "--help"]
```

## 🧪 Testing

### Run Smoke Tests

```bash
python tests/smoke_test.py
```

### Run Individual Tests

```bash
# Test configuration
python -c "from utils.config import get_default_config, validate_config; validate_config(get_default_config())"

# Test directory creation
python -c "import os; [os.makedirs(d, exist_ok=True) for d in ['images', 'motion', 'clips', 'audio', 'output']]"
```

## 🔧 Troubleshooting

### Common Issues

#### GPU Memory Issues
- **Problem**: CUDA out of memory errors
- **Solution**: Reduce batch size, use smaller models, or enable gradient checkpointing

#### Model Download Issues
- **Problem**: Models fail to download
- **Solution**: Set `HF_TOKEN` environment variable for HuggingFace access

#### FFmpeg Issues
- **Problem**: Video processing fails
- **Solution**: Ensure FFmpeg is installed and in PATH

#### Wav2Lip Issues
- **Problem**: Lip-sync quality is poor
- **Solution**: Ensure good face detection, adjust face padding

### Performance Optimization

#### For Better Quality
- Increase `guidance_scale` (7.5-15.0)
- Use higher resolution models
- Enable RIFE interpolation
- Use LoRA models for specific styles

#### For Faster Processing
- Reduce `frames` count
- Use lower resolution
- Disable post-processing effects
- Use CPU for non-critical operations

### Debug Mode

Enable verbose logging:

```bash
python main.py --verbose run-all --config config.json ...
```

### Environment Issues

If you encounter import errors:

```bash
# Ensure environment is activated
source /workspace/venv/bin/activate

# Set environment variables
export WORKSPACE_DIR=/workspace
export MODELS_DIR=/workspace/models
export PYTHONPATH=/workspace:$PYTHONPATH

# Test imports
python -c "import sys; print('Python path:', sys.path)"
```

## 📊 Hardware Recommendations

### Minimum Requirements
- **GPU**: NVIDIA GTX 1080 (8GB VRAM)
- **RAM**: 16GB
- **Storage**: 50GB free space

### Recommended Setup
- **GPU**: NVIDIA RTX 3080/4080 (12GB+ VRAM)
- **RAM**: 32GB
- **Storage**: 100GB+ SSD

### Production Setup
- **GPU**: NVIDIA RTX 4090 or A100 (24GB+ VRAM)
- **RAM**: 64GB+
- **Storage**: 500GB+ NVMe SSD

## 📝 License and Usage

### 🎉 **All Models Are FREE!**

#### Model Licenses
- **Stable Diffusion**: ✅ CreativeML Open RAIL-M License (free for commercial use)
- **SVD**: ⚠️ Stability AI License (check terms for commercial use)
- **Wav2Lip**: ✅ MIT License (free for commercial use)
- **Coqui TTS**: ✅ MIT License (free for commercial use)
- **ControlNet**: ✅ Apache 2.0 License (free for commercial use)
- **IP-Adapter**: ✅ Apache 2.0 License (free for commercial use)
- **Real-ESRGAN**: ✅ BSD 3-Clause License (free for commercial use)
- **MediaPipe**: ✅ Apache 2.0 License (free for commercial use)

### Commercial Usage
✅ **Most models are FREE for commercial use!** No paid subscriptions required.

⚠️ **Note**: Only SVD and some LoRAs may have commercial restrictions - check individual licenses.

### Attribution
When using this pipeline, please attribute:
- Stable Diffusion by Stability AI
- SVD by Stability AI
- Wav2Lip by Rudrabha et al.
- Coqui TTS by Coqui AI

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 .
```

## 🚀 Quick Environment Setup

After running the setup script, activate your environment:

```bash
# Activate Python environment
source /workspace/venv/bin/activate

# Set environment variables
export WORKSPACE_DIR=/workspace
export MODELS_DIR=/workspace/models
export PYTHONPATH=/workspace:$PYTHONPATH

# Test the setup
python main.py --help
python scripts/create_example_dataset.py
```

## 📚 Examples

### Cartoon Character Animation
```bash
python main.py run-all \
  --config config.json \
  --prompt "cartoon blue cat running, cel-shaded, Oggy-style, high quality" \
  --motion_video ref/running_cycle.mp4 \
  --tts_text "Watch out! Here I come!" \
  --out output/cartoon_cat.mp4
```

### Anime Style Video
```bash
python main.py run-all \
  --config config.json \
  --prompt "anime girl dancing, Studio Ghibli style, detailed" \
  --motion_video ref/dance_moves.mp4 \
  --tts_text "Let's dance together!" \
  --out output/anime_dance.mp4
```

### Realistic Character
```bash
python main.py run-all \
  --config config.json \
  --prompt "realistic person walking, photorealistic, 4k" \
  --motion_video ref/walking_cycle.mp4 \
  --tts_text "Hello, how are you today?" \
  --out output/realistic_walk.mp4
```

## 🔗 Related Projects

- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models)
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [MediaPipe](https://github.com/google/mediapipe)

## 📚 Documentation

- **[Environment Setup](docs/ENVIRONMENT_SETUP.md)** - How to set up your environment (no .env file needed!)
- **[RunPod Deployment](docs/RUNPOD_DEPLOYMENT.md)** - Complete cloud GPU deployment guide
- **[Licensing Guide](docs/LICENSING.md)** - Model licenses (all FREE and open source!)
- **[Models Documentation](models/README.md)** - Required AI models and downloads

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/WhyWouldYou-v2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/WhyWouldYou-v2/discussions)
- **Wiki**: [Project Wiki](https://github.com/yourusername/WhyWouldYou-v2/wiki)

## 🙏 Acknowledgments

- Stability AI for Stable Diffusion and SVD
- Rudrabha et al. for Wav2Lip
- Coqui AI for TTS
- Google for MediaPipe
- The open-source AI community

---

**Note**: This project is for educational and research purposes. Please ensure compliance with all applicable licenses and regulations when using AI-generated content.
