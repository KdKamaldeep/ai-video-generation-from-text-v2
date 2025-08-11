# WhyWouldYou-v2 RunPod Deployment Guide

This guide will help you deploy the WhyWouldYou-v2 pipeline on RunPod for GPU-accelerated video generation.

## Prerequisites

- RunPod account with GPU credits
- Hugging Face account and token (for model downloads)
- Basic familiarity with Linux command line

## Quick Setup (Recommended)

### Option 1: One-Liner Setup

1. **Start a RunPod instance** with:
   - **Template**: `RunPod PyTorch` or `NVIDIA CUDA 12.1`
   - **GPU**: RTX 4090, RTX 3090, or A100 (recommended)
   - **RAM**: At least 32GB
   - **Storage**: At least 100GB

2. **Connect to your pod** via SSH or web terminal

3. **Run the one-liner setup script**:
   ```bash
   curl -sSL https://raw.githubusercontent.com/yourusername/WhyWouldYou-v2/main/scripts/runpod_setup_one_liner.sh | bash
   ```

4. **Set your Hugging Face token**:
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   ```

5. **Download models**:
   ```bash
   huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir models/sd --token $HF_TOKEN
   huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir models/svd --token $HF_TOKEN
   ```

6. **Activate the environment**:
   ```bash
   ./start.sh
   ```

### Option 2: Manual Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/WhyWouldYou-v2.git
   cd WhyWouldYou-v2
   ```

2. **Run the comprehensive setup script**:
   ```bash
   sudo bash scripts/setup_runpod.sh
   ```

3. **Activate the environment**:
   ```bash
   source /workspace/venv/bin/activate
   ./start.sh
   ```

## Hardware Recommendations

### Minimum Requirements
- **GPU**: RTX 3080 (10GB VRAM)
- **RAM**: 16GB
- **Storage**: 50GB
- **CPU**: 4 cores

### Recommended Configuration
- **GPU**: RTX 4090 (24GB VRAM) or A100 (40GB VRAM)
- **RAM**: 32GB or more
- **Storage**: 100GB or more
- **CPU**: 8+ cores

### High-End Configuration
- **GPU**: A100 80GB or H100
- **RAM**: 64GB+
- **Storage**: 200GB+
- **CPU**: 16+ cores

## Model Downloads

### Required Models

1. **Stable Diffusion**:
   ```bash
   huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir models/sd --token $HF_TOKEN
   ```

2. **Stable Video Diffusion**:
   ```bash
   huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir models/svd --token $HF_TOKEN
   ```

3. **ControlNet** (optional):
   ```bash
   huggingface-cli download lllyasviel/ControlNet-v1-1 --local-dir models/controlnet --token $HF_TOKEN
   ```

4. **IP-Adapter** (optional):
   ```bash
   huggingface-cli download h94/IP-Adapter --local-dir models/ip_adapter --token $HF_TOKEN
   ```

### External Models

The setup script automatically downloads:
- **Wav2Lip**: For lip synchronization
- **Real-ESRGAN**: For video upscaling
- **S3FD**: Face detection model

## Usage Examples

### Quick Test

1. **Create example dataset**:
   ```bash
   python scripts/create_example_dataset.py
   ```

2. **Run end-to-end test**:
   ```bash
   python main.py run-all \
       --config config.json \
       --prompt "cartoon blue cat running" \
       --motion_video examples/motion/stick_figure_walking.mp4 \
       --tts_text "Hello, this is a test!"
   ```

### Individual Steps

1. **Generate base image**:
   ```bash
   python main.py image \
       --prompt "cartoon blue cat" \
       --config config.json \
       --out images/base1.png
   ```

2. **Extract motion**:
   ```bash
   python main.py extract-motion \
       --motion_video your_reference_video.mp4 \
       --out motion/run1/ \
       --method mediapipe
   ```

3. **Run SVD**:
   ```bash
   python main.py svd \
       --input_image images/base1.png \
       --motion_dir motion/run1/ \
       --out clips/scene1.mp4 \
       --config config.json \
       --debug
   ```

4. **Generate TTS**:
   ```bash
   python main.py tts \
       --text "Hello world!" \
       --out audio/line1.wav \
       --config config.json
   ```

5. **Lip-sync**:
   ```bash
   python main.py wav2lip \
       --video clips/scene1.mp4 \
       --audio audio/line1.wav \
       --out clips/scene1_synced.mp4
   ```

## Configuration

### Main Configuration File

The setup creates `/workspace/config.json` with optimized settings for RunPod:

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
  "gpu_device": "cuda:0"
}
```

### Environment Variables

The setup creates `/workspace/start.sh` with all necessary environment variables:
- Model cache directories
- CUDA settings
- Memory optimization
- Python path configuration

### Performance Tuning

For different GPU configurations:

**RTX 3080 (10GB VRAM)**:
```json
{
  "frames": 24,
  "output_resolution": [720, 1280],
  "gpu_device": "cuda:0"
}
```

**RTX 4090 (24GB VRAM)**:
```json
{
  "frames": 48,
  "output_resolution": [1080, 1920],
  "gpu_device": "cuda:0"
}
```

**A100 (40GB+ VRAM)**:
```json
{
  "frames": 64,
  "output_resolution": [1440, 2560],
  "gpu_device": "cuda:0"
}
```

## Monitoring and Troubleshooting

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv
```

### Memory Monitoring

```bash
# System memory
free -h

# Process memory usage
ps aux --sort=-%mem | head -10
```

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `frames` in config
   - Lower `output_resolution`
   - Use CPU mode: set `gpu_device` to `cpu`

2. **Model Download Failures**:
   - Check internet connection
   - Verify HF_TOKEN is set correctly
   - Try manual download with `huggingface-cli`

3. **Slow Performance**:
   - Check GPU utilization with `nvidia-smi`
   - Monitor CPU usage with `htop`
   - Consider upgrading to a more powerful GPU

4. **Missing Dependencies**:
   - Re-run setup script
   - Install missing packages manually
   - Check Python environment activation

## File Structure

After setup, your workspace will look like:

```
/workspace/
├── main.py                 # Main CLI entrypoint
├── config.json            # Configuration file
├── start.sh               # Startup script with environment variables
├── venv/                  # Python virtual environment
├── pipeline/              # Core pipeline modules
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
├── docs/                  # Documentation
│   ├── README.md          # Documentation index
│   ├── ENVIRONMENT_SETUP.md # Environment setup guide
│   ├── RUNPOD_DEPLOYMENT.md # This file
│   └── LICENSING.md       # Model licensing information
├── models/                # AI models
│   ├── sd/               # Stable Diffusion
│   ├── svd/              # Stable Video Diffusion
│   ├── controlnet/       # ControlNet models
│   ├── lora/             # LoRA models
│   ├── ip_adapter/       # IP-Adapter models
│   ├── wav2lip.pth       # Wav2Lip model
│   ├── s3fd.pth          # Face detection
│   └── RealESRGAN_x4plus.pth
├── external/              # External repositories
│   ├── Wav2Lip/          # Wav2Lip repo
│   └── Real-ESRGAN/      # Real-ESRGAN repo
├── scripts/               # Setup and utility scripts
├── images/                # Generated images
├── motion/                # Motion data
├── clips/                 # Video clips
├── audio/                 # Audio files
├── output/                # Final outputs
├── examples/              # Example dataset
├── tests/                 # Test files
├── temp/                  # Temporary files
└── logs/                  # Log files
```

## Cost Optimization

### RunPod Pricing (approximate)

- **RTX 3080**: $0.40-0.60/hour
- **RTX 4090**: $0.80-1.20/hour
- **A100**: $2.00-3.00/hour

### Tips to Reduce Costs

1. **Use spot instances** when available
2. **Optimize batch processing** to maximize GPU utilization
3. **Use smaller models** for testing
4. **Shut down pods** when not in use
5. **Pre-download models** to avoid repeated downloads

## Support

### Getting Help

1. **Check logs**: Look in `/workspace/logs/` directory
2. **Run tests**: `python tests/smoke_test.py`
3. **Check model info**: `cat models/models_info.txt`
4. **Monitor resources**: Use `nvidia-smi` and `htop`
5. **Documentation**: See `/workspace/docs/` for detailed guides

### Useful Commands

```bash
# Check GPU status
nvidia-smi

# Monitor system resources
htop

# Check disk usage
df -h

# View recent logs
tail -f logs/pipeline.log

# Test pipeline
python tests/smoke_test.py

# Create example dataset
python scripts/create_example_dataset.py

# Start Jupyter notebook
./start.sh jupyter
```

### Emergency Recovery

If something goes wrong:

1. **Restart the pod** (preserves data)
2. **Re-run setup**: `sudo bash scripts/setup_runpod.sh`
3. **Check disk space**: `df -h`
4. **Verify models**: `ls -la models/`

## License and Legal

- **Stable Diffusion**: Check license for commercial use
- **SVD**: Check Stability AI license terms
- **Wav2Lip**: MIT License
- **Real-ESRGAN**: BSD 3-Clause License

Always verify model licenses before commercial use.

---

**Happy video generating! 🎬**
