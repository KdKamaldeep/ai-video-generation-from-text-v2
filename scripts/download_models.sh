#!/bin/bash

# Model Download Script for WhyWouldYou-v2 Pipeline
# This script downloads all required models for the video generation pipeline

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
MODELS_DIR="$PROJECT_ROOT/models"
EXTERNAL_DIR="$PROJECT_ROOT/external"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN environment variable not set.${NC}"
    echo -e "${YELLOW}Some models may require authentication.${NC}"
    echo -e "${BLUE}Set it with: export HF_TOKEN='your_token_here'${NC}"
fi

# Create directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p "$MODELS_DIR"/{sd,svd,controlnet,lora,ip_adapter}
mkdir -p "$EXTERNAL_DIR"

# Function to download with progress
download_model() {
    local url="$1"
    local output="$2"
    local description="$3"
    
    echo -e "${BLUE}Downloading $description...${NC}"
    
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll -O "$output" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -o "$output" "$url"
    else
        echo -e "${RED}Error: Neither wget nor curl found. Please install one.${NC}"
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
    
    if command -v huggingface-cli &> /dev/null; then
        if [ -n "$HF_TOKEN" ]; then
            huggingface-cli download "$repo" --local-dir "$local_dir" --token "$HF_TOKEN"
        else
            huggingface-cli download "$repo" --local-dir "$local_dir"
        fi
        echo -e "${GREEN}✓ Downloaded $description${NC}"
    else
        echo -e "${YELLOW}Warning: huggingface-cli not found. Please install it:${NC}"
        echo -e "${BLUE}pip install huggingface_hub${NC}"
        echo -e "${YELLOW}Manual download required for: $repo${NC}"
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

echo -e "${GREEN}Starting model download...${NC}"

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

# Set permissions
echo -e "${BLUE}Setting permissions...${NC}"
chmod -R 755 "$MODELS_DIR"
chmod -R 755 "$EXTERNAL_DIR"

# Create model info file
echo -e "${BLUE}Creating model info...${NC}"
cat > "$MODELS_DIR/models_info.txt" << EOF
Models downloaded on: $(date)
Pipeline version: WhyWouldYou-v2

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

Note: Check individual model licenses for commercial use.
EOF

echo -e "${GREEN}✓ All models downloaded successfully!${NC}"
echo -e "${BLUE}Model information saved to: $MODELS_DIR/models_info.txt${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "${BLUE}1. Review model licenses for commercial use${NC}"
echo -e "${BLUE}2. Test the pipeline with: python main.py --help${NC}"
echo -e "${BLUE}3. Run smoke test: python tests/smoke_test.py${NC}"
