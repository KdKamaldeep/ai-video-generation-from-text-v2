#!/bin/bash

# WhyWouldYou-v2 RunPod Setup Script
# This script sets up the complete pipeline environment on a RunPod instance

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="/workspace"
MODELS_DIR="$WORKSPACE_DIR/models"
EXTERNAL_DIR="$WORKSPACE_DIR/external"

echo -e "${GREEN}=== WhyWouldYou-v2 RunPod Setup Script ===${NC}"
echo -e "${BLUE}Setting up complete pipeline environment...${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU availability
check_gpu() {
    if command_exists nvidia-smi; then
        echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        return 0
    else
        echo -e "${YELLOW}⚠ No NVIDIA GPU detected, will use CPU mode${NC}"
        return 1
    fi
}

# Function to check system resources
check_system() {
    echo -e "${BLUE}Checking system resources...${NC}"
    
    # Check CPU
    CPU_CORES=$(nproc)
    echo -e "${GREEN}✓ CPU Cores: $CPU_CORES${NC}"
    
    # Check RAM
    RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    echo -e "${GREEN}✓ RAM: ${RAM_GB}GB${NC}"
    
    # Check disk space
    DISK_GB=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    echo -e "${GREEN}✓ Available disk space: ${DISK_GB}GB${NC}"
    
    # Check if we have enough disk space (need at least 50GB for models)
    #if [ "$DISK_GB" -lt 50 ]; then
    #    echo -e "${RED}✗ Insufficient disk space. Need at least 50GB, have ${DISK_GB}GB${NC}"
    #    exit 1
    #fi
}

# Function to update system packages
update_system() {
    echo -e "${BLUE}Updating system packages...${NC}"
    
    # Update package lists
    apt-get update
    
    # Upgrade existing packages
    apt-get upgrade -y
    
    # Install essential system dependencies
    apt-get install -y \
        git \
        wget \
        curl \
        unzip \
        build-essential \
        cmake \
        pkg-config \
        libssl-dev \
        libffi-dev \
        libsndfile1 \
        libportaudio2 \
        portaudio19-dev \
        python3-dev \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgtk-3-0 \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-dev \
        libhdf5-103 \
        python3-pyqt5 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer-plugins-bad1.0-0 \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        gstreamer1.0-tools \
        gstreamer1.0-x \
        gstreamer1.0-alsa \
        gstreamer1.0-gl \
        gstreamer1.0-gtk3 \
        gstreamer1.0-qt5 \
        gstreamer1.0-pulseaudio
    
    echo -e "${GREEN}✓ System packages updated${NC}"
}

# Function to setup Python environment
setup_python() {
    echo -e "${BLUE}Setting up Python environment...${NC}"
    
    # Create virtual environment
    python3 -m venv "$WORKSPACE_DIR/venv"
    source "$WORKSPACE_DIR/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA support if GPU is available
    if check_gpu; then
        echo -e "${BLUE}Installing PyTorch with CUDA support...${NC}"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo -e "${BLUE}Installing PyTorch CPU version...${NC}"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other dependencies
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    pip install -r "$PROJECT_ROOT/requirements.txt"
    
    # Install additional useful packages
    pip install \
        jupyter \
        ipywidgets \
        matplotlib \
        seaborn \
        pandas \
        scikit-learn \
        scipy \
        tqdm \
        psutil \
        GPUtil \
        nvidia-ml-py3
    
    echo -e "${GREEN}✓ Python environment setup complete${NC}"
}

# Function to setup project structure
setup_project() {
    echo -e "${BLUE}Setting up project structure...${NC}"
    
    # Create workspace directory
    mkdir -p "$WORKSPACE_DIR"
    cd "$WORKSPACE_DIR"
    
    # Copy project files
    cp -r "$PROJECT_ROOT"/* "$WORKSPACE_DIR/"
    
    # Create necessary directories
    mkdir -p "$MODELS_DIR"/{sd,svd,controlnet,lora,ip_adapter}
    mkdir -p "$EXTERNAL_DIR"
    mkdir -p "$WORKSPACE_DIR"/{images,motion,clips,audio,output,temp,logs}
    
    # Set permissions
    chmod -R 755 "$WORKSPACE_DIR"
    
    echo -e "${GREEN}✓ Project structure created${NC}"
}

# Function to download models
download_models() {
    echo -e "${BLUE}Downloading AI models...${NC}"
    
    # Check if HF_TOKEN is set
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${YELLOW}Note: HF_TOKEN environment variable not set.${NC}"
        echo -e "${BLUE}Most models are free and open source, but some require a free Hugging Face account.${NC}"
        echo -e "${BLUE}Set it with: export HF_TOKEN='your_token_here'${NC}"
        echo -e "${BLUE}Get a free token at: https://huggingface.co/settings/tokens${NC}"
    fi
    
    # Function to download with progress
    download_model() {
        local url="$1"
        local output="$2"
        local description="$3"
        
        echo -e "${BLUE}Downloading $description...${NC}"
        
        if command_exists wget; then
            wget --progress=bar:force:noscroll -O "$output" "$url"
        elif command_exists curl; then
            curl -L -o "$output" "$url"
        else
            echo -e "${RED}Error: Neither wget nor curl found.${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}✓ Downloaded $description${NC}"
    }
    
    # Function to download from Hugging Face
    download_hf_model() {
        local repo="$1"
        local local_dir="$2"
        local description="$3"
        
        echo -e "${BLUE}Downloading $description from Hugging Face...${NC}"
        
        if command_exists huggingface-cli; then
            if [ -n "$HF_TOKEN" ]; then
                huggingface-cli download "$repo" --local-dir "$local_dir" --token "$HF_TOKEN"
            else
                huggingface-cli download "$repo" --local-dir "$local_dir"
            fi
            echo -e "${GREEN}✓ Downloaded $description${NC}"
        else
            echo -e "${YELLOW}Warning: huggingface-cli not found. Installing...${NC}"
            pip install huggingface_hub
            if [ -n "$HF_TOKEN" ]; then
                huggingface-cli download "$repo" --local-dir "$local_dir" --token "$HF_TOKEN"
            else
                huggingface-cli download "$repo" --local-dir "$local_dir"
            fi
            echo -e "${GREEN}✓ Downloaded $description${NC}"
        fi
    }
    
    # Function to clone external repositories
    clone_repo() {
        local repo_url="$1"
        local local_dir="$2"
        local description="$3"
        
        if [ ! -d "$local_dir" ]; then
            echo -e "${BLUE}Cloning $description...${NC}"
            git clone "$repo_url" "$local_dir"
            echo -e "${GREEN}✓ Cloned $description${NC}"
        else
            echo -e "${YELLOW}✓ $description already exists${NC}"
        fi
    }
    
    # Download Stable Diffusion models
    echo -e "${BLUE}=== Stable Diffusion Models ===${NC}"
    download_hf_model "runwayml/stable-diffusion-v1-5" "$MODELS_DIR/sd" "Stable Diffusion v1.5"
    
    # Download SVD models
    echo -e "${BLUE}=== Stable Video Diffusion Models ===${NC}"
    download_hf_model "stabilityai/stable-video-diffusion-img2vid-xt" "$MODELS_DIR/svd" "SVD Img2Vid XT"
    
    # Download ControlNet models (optional)
    echo -e "${BLUE}=== ControlNet Models ===${NC}"
    download_hf_model "lllyasviel/ControlNet-v1-1" "$MODELS_DIR/controlnet" "ControlNet v1.1"
    
    # Download IP-Adapter models (optional)
    echo -e "${BLUE}=== IP-Adapter Models ===${NC}"
    download_hf_model "h94/IP-Adapter" "$MODELS_DIR/ip_adapter" "IP-Adapter"
    
    # Clone external repositories
    echo -e "${BLUE}=== External Repositories ===${NC}"
    clone_repo "https://github.com/Rudrabha/Wav2Lip.git" "$EXTERNAL_DIR/Wav2Lip" "Wav2Lip"
    clone_repo "https://github.com/xinntao/Real-ESRGAN.git" "$EXTERNAL_DIR/Real-ESRGAN" "Real-ESRGAN"
    
    # Download Wav2Lip models
    echo -e "${BLUE}=== Wav2Lip Models ===${NC}"
    WAV2LIP_MODEL_URL="https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip.pth"
    S3FD_MODEL_URL="https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/s3fd.pth"
    
    download_model "$WAV2LIP_MODEL_URL" "$MODELS_DIR/wav2lip.pth" "Wav2Lip model"
    download_model "$S3FD_MODEL_URL" "$MODELS_DIR/s3fd.pth" "S3FD face detection model"
    
    # Download Real-ESRGAN models
    echo -e "${BLUE}=== Real-ESRGAN Models ===${NC}"
    REALESRGAN_MODEL_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    download_model "$REALESRGAN_MODEL_URL" "$MODELS_DIR/RealESRGAN_x4plus.pth" "Real-ESRGAN x4+ model"
    
    echo -e "${GREEN}✓ All models downloaded successfully!${NC}"
}

# Function to create configuration files
setup_configs() {
    echo -e "${BLUE}Setting up configuration files...${NC}"
    
    # Create main config
    cat > "$WORKSPACE_DIR/config.json" << EOF
{
  "sd_model": "runwayml/stable-diffusion-v1-5",
  "svd_model": "stabilityai/stable-video-diffusion-img2vid-xt",
  "controlnet_pose_model": null,
  "lo_ra": null,
  "ip_adapter_path": null,
  "output_resolution": [1080, 1920],
  "fps": 24,
  "frames": 48,
  "strength": 0.56,
  "guidance_scale": 7.5,
  "seed": 42,
  "coqui_voice": {
    "voice": "alloy",
    "sample_rate": 24000,
    "model": "tts_models/multilingual/multi-dataset/xtts_v2"
  },
  "wav2lip_enabled": true,
  "wav2lip_model": "$MODELS_DIR/wav2lip.pth",
  "face_detection_model": "$MODELS_DIR/s3fd.pth",
  "post_rife": false,
  "rife_fps": 48,
  "post_smear": false,
  "post_squash": false,
  "crossfade_duration": 0.5,
  "temp_dirs": {
    "images": "images",
    "motion": "motion",
    "clips": "clips",
    "audio": "audio",
    "output": "output"
  },
  "gpu_device": "cuda:0"
}
EOF
    
    # Note: No .env file needed - all settings are in config.json and start.sh
    
    # Create startup script
    cat > "$WORKSPACE_DIR/start.sh" << 'EOF'
#!/bin/bash

# WhyWouldYou-v2 Startup Script for RunPod

echo "Starting WhyWouldYou-v2 environment..."

# Activate virtual environment
source /workspace/venv/bin/activate

# Set environment variables
export WORKSPACE_DIR=/workspace
export MODELS_DIR=/workspace/models
export EXTERNAL_DIR=/workspace/external
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export TORCH_HOME=/workspace/models
export PYTHONPATH=/workspace:$PYTHONPATH

# Change to workspace directory
cd /workspace

# Check GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "No GPU detected, using CPU mode"
fi

# Show available commands
echo ""
echo "Available commands:"
echo "  python main.py --help                    # Show all available commands"
echo "  python main.py run-all --help            # Show run-all options"
echo "  python scripts/create_example_dataset.py # Create test dataset"
echo "  python tests/smoke_test.py               # Run smoke test"
echo ""

# Start Jupyter if requested
if [ "$1" = "jupyter" ]; then
    echo "Starting Jupyter notebook..."
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
else
    echo "Environment ready! Use 'python main.py --help' to see available commands."
    echo "To start Jupyter: ./start.sh jupyter"
fi
EOF
    
    chmod +x "$WORKSPACE_DIR/start.sh"
    
    echo -e "${GREEN}✓ Configuration files created${NC}"
}

# Function to run tests
run_tests() {
    echo -e "${BLUE}Running tests...${NC}"
    
    cd "$WORKSPACE_DIR"
    source venv/bin/activate
    
    # Test CLI help
    echo -e "${BLUE}Testing CLI help...${NC}"
    python main.py --help > /dev/null
    python main.py image --help > /dev/null
    python main.py extract-motion --help > /dev/null
    python main.py svd --help > /dev/null
    python main.py tts --help > /dev/null
    python main.py wav2lip --help > /dev/null
    python main.py post --help > /dev/null
    python main.py assemble --help > /dev/null
    python main.py run-all --help > /dev/null
    
    # Test config loading
    echo -e "${BLUE}Testing config loading...${NC}"
    python -c "
from utils.config import load_config, validate_config, get_default_config
config = get_default_config()
validate_config(config)
print('Config validation passed')
"
    
    # Test logging setup
    echo -e "${BLUE}Testing logging setup...${NC}"
    python -c "
from utils.logging import setup_logging
import logging
setup_logging(verbose=True)
logger = logging.getLogger(__name__)
logger.info('Logging test successful')
"
    
    # Test image utilities
    echo -e "${BLUE}Testing image utilities...${NC}"
    python -c "
import numpy as np
from utils.crop_to_aspect import crop_to_aspect_ratio, resize_to_resolution
test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
cropped = crop_to_aspect_ratio(test_image, 1.0)
resized = resize_to_resolution(test_image, (50, 100))
print('Image utilities test passed')
"
    
    # Create example dataset
    echo -e "${BLUE}Creating example dataset...${NC}"
    python scripts/create_example_dataset.py
    
    # Run smoke test
    echo -e "${BLUE}Running smoke test...${NC}"
    python tests/smoke_test.py
    
    echo -e "${GREEN}✓ All tests passed!${NC}"
}

# Function to create model info file
create_model_info() {
    echo -e "${BLUE}Creating model information...${NC}"
    
    cat > "$MODELS_DIR/models_info.txt" << EOF
Models downloaded on: $(date)
Pipeline version: WhyWouldYou-v2
RunPod Setup: $(date)

Downloaded models:
- Stable Diffusion v1.5: $MODELS_DIR/sd/
- SVD Img2Vid XT: $MODELS_DIR/svd/
- ControlNet v1.1: $MODELS_DIR/controlnet/
- IP-Adapter: $MODELS_DIR/ip_adapter/
- Wav2Lip: $MODELS_DIR/wav2lip.pth
- S3FD: $MODELS_DIR/s3fd.pth
- Real-ESRGAN: $MODELS_DIR/RealESRGAN_x4plus.pth

External repositories:
- Wav2Lip: $EXTERNAL_DIR/Wav2Lip/
- Real-ESRGAN: $EXTERNAL_DIR/Real-ESRGAN/

System Information:
- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "None")
- CPU Cores: $(nproc)
- RAM: $(free -g | awk '/^Mem:/{print $2}')GB
- Disk: $(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')GB available

LICENSE INFORMATION:
- Stable Diffusion: CreativeML OpenRAIL-M (free for commercial use)
- SVD: Stability AI license (check terms for commercial use)
- ControlNet: Apache 2.0 (free for commercial use)
- IP-Adapter: Apache 2.0 (free for commercial use)
- Wav2Lip: MIT License (free for commercial use)
- Real-ESRGAN: BSD 3-Clause (free for commercial use)
- MediaPipe: Apache 2.0 (free for commercial use)

All models are FREE and OPEN SOURCE for personal use.
Most are also free for commercial use - check individual licenses above.
EOF
    
    echo -e "${GREEN}✓ Model information saved${NC}"
}

# Function to create usage instructions
create_usage_instructions() {
    echo -e "${BLUE}Creating usage instructions...${NC}"
    
    cat > "$WORKSPACE_DIR/USAGE.md" << EOF
# WhyWouldYou-v2 RunPod Usage Guide

## Quick Start

1. **Activate the environment:**
   \`\`\`bash
   source /workspace/venv/bin/activate
   \`\`\`

2. **Run the startup script:**
   \`\`\`bash
   ./start.sh
   \`\`\`

3. **Test the pipeline:**
   \`\`\`bash
   # Create example dataset
   python scripts/create_example_dataset.py
   
   # Run end-to-end test
   python main.py run-all \\
       --config config.json \\
       --prompt "cartoon blue cat running" \\
       --motion_video examples/motion/stick_figure_walking.mp4 \\
       --tts_text "Hello, this is a test!"
   \`\`\`

## Available Commands

### Generate base image
\`\`\`bash
python main.py image --prompt "cartoon blue cat" --config config.json --out images/base1.png
\`\`\`

### Extract motion from reference video
\`\`\`bash
python main.py extract-motion --motion_video ref.mp4 --out motion/run1/ --method mediapipe
\`\`\`

### Run SVD vid2vid
\`\`\`bash
python main.py svd --input_image images/base1.png --motion_dir motion/run1/ --out clips/scene1.mp4 --config config.json
\`\`\`

### Generate TTS audio
\`\`\`bash
python main.py tts --text "Hello world!" --out audio/line1.wav --config config.json
\`\`\`

### Run Wav2Lip lip-sync
\`\`\`bash
python main.py wav2lip --video clips/scene1.mp4 --audio audio/line1.wav --out clips/scene1_synced.mp4
\`\`\`

### Post-process video
\`\`\`bash
python main.py post --input clips/scene1_synced.mp4 --out clips/scene1_final.mp4 --rife --smear
\`\`\`

### Assemble final video
\`\`\`bash
python main.py assemble --clips clips/scene1_final.mp4 --audio bg/music.mp3 --out output/final.mp4
\`\`\`

## Configuration

- Main config: \`config.json\`
- Environment variables: \`.env\`
- Model directory: \`models/\`

## Troubleshooting

### GPU Issues
- Check GPU: \`nvidia-smi\`
- Monitor memory: \`watch -n 1 nvidia-smi\`

### Memory Issues
- Reduce batch size in config
- Use CPU mode: set \`gpu_device\` to \`cpu\`

### Model Issues
- Check model paths in \`config.json\`
- Verify models exist in \`models/\` directory

## File Structure

\`\`\`
/workspace/
├── main.py                 # Main CLI entrypoint
├── config.json            # Configuration file
├── start.sh               # Startup script
├── models/                # AI models
├── external/              # External repositories
├── images/                # Generated images
├── motion/                # Motion data
├── clips/                 # Video clips
├── audio/                 # Audio files
├── output/                # Final outputs
├── examples/              # Example dataset
└── tests/                 # Test files
\`\`\`

## Support

- Check logs in \`logs/\` directory
- Run smoke test: \`python tests/smoke_test.py\`
- View model info: \`cat models/models_info.txt\`
EOF
    
    echo -e "${GREEN}✓ Usage instructions created${NC}"
}

# Main setup function
main() {
    echo -e "${GREEN}Starting WhyWouldYou-v2 RunPod setup...${NC}"
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}This script must be run as root (use sudo)${NC}"
        exit 1
    fi
    
    # Check system
    check_system
    
    # Check GPU
    check_gpu
    
    # Update system
    update_system
    
    # Setup project structure
    setup_project
    
    # Setup Python environment
    setup_python
    
    # Download models
    download_models
    
    # Setup configurations
    setup_configs
    
    # Create model info
    create_model_info
    
    # Create usage instructions
    create_usage_instructions
    
    # Run tests
    run_tests
    
    # Set final permissions
    chown -R 1000:1000 "$WORKSPACE_DIR" 2>/dev/null || true
    
    echo -e "${GREEN}=== Setup Complete! ===${NC}"
    echo -e "${BLUE}Workspace: $WORKSPACE_DIR${NC}"
    echo -e "${BLUE}Models: $MODELS_DIR${NC}"
    echo -e "${BLUE}Configuration: $WORKSPACE_DIR/config.json${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "${BLUE}1. Activate environment: source $WORKSPACE_DIR/venv/bin/activate${NC}"
    echo -e "${BLUE}2. Run startup script: $WORKSPACE_DIR/start.sh${NC}"
    echo -e "${BLUE}3. Test pipeline: python main.py --help${NC}"
    echo -e "${BLUE}4. Create example: python scripts/create_example_dataset.py${NC}"
    echo ""
    echo -e "${GREEN}Setup completed successfully!${NC}"
}

# Run main function
main "$@"
