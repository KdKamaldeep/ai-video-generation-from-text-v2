#!/bin/bash

# WhyWouldYou-v2 RunPod One-Liner Setup
# Copy and paste this entire script into your RunPod terminal

echo "Setting up WhyWouldYou-v2 on RunPod..."

# Update system and install dependencies
apt-get update && apt-get install -y git wget curl unzip build-essential cmake pkg-config libssl-dev libffi-dev libsndfile1 libportaudio2 portaudio19-dev python3-dev python3-pip python3-venv ffmpeg libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0 libgl1-mesa-glx libgtk-3-0 libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5 libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 libgstreamer-plugins-bad1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

# Create workspace
mkdir -p /workspace && cd /workspace

# Clone the repository (replace with your actual repo URL)
git clone https://github.com/yourusername/WhyWouldYou-v2.git . || echo "Repository not found, creating structure manually"

# Create project structure
mkdir -p models/{sd,svd,controlnet,lora,ip_adapter} external images motion clips audio output temp logs

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
pip install diffusers transformers accelerate xformers opencv-python Pillow numpy moviepy ffmpeg-python librosa soundfile mediapipe TTS argparse json logging subprocess tempfile pathlib typing

# Install additional useful packages
pip install jupyter ipywidgets matplotlib seaborn pandas scikit-learn scipy tqdm psutil GPUtil nvidia-ml-py3

# Download models (if HF_TOKEN is set)
if [ -n "$HF_TOKEN" ]; then
    echo "Downloading models with HF_TOKEN..."
    pip install huggingface_hub
    huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir models/sd --token "$HF_TOKEN"
    huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir models/svd --token "$HF_TOKEN"
    huggingface-cli download lllyasviel/ControlNet-v1-1 --local-dir models/controlnet --token "$HF_TOKEN"
    huggingface-cli download h94/IP-Adapter --local-dir models/ip_adapter --token "$HF_TOKEN"
else
    echo "HF_TOKEN not set, skipping model downloads"
    echo "Note: All models are FREE and open source!"
    echo "Get a free HF token at: https://huggingface.co/settings/tokens"
fi

# Clone external repositories
git clone https://github.com/Rudrabha/Wav2Lip.git external/Wav2Lip
git clone https://github.com/xinntao/Real-ESRGAN.git external/Real-ESRGAN

# Download external models
wget -O models/wav2lip.pth https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip.pth
wget -O models/s3fd.pth https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/s3fd.pth
wget -O models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# Create config file
cat > config.json << 'EOF'
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
  "wav2lip_model": "/workspace/models/wav2lip.pth",
  "face_detection_model": "/workspace/models/s3fd.pth",
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

# Create startup script
cat > start.sh << 'EOF'
#!/bin/bash
echo "Starting WhyWouldYou-v2 environment..."
source /workspace/venv/bin/activate
export WORKSPACE_DIR=/workspace
export MODELS_DIR=/workspace/models
export EXTERNAL_DIR=/workspace/external
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export TORCH_HOME=/workspace/models
export PYTHONPATH=/workspace:$PYTHONPATH
cd /workspace

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "No GPU detected, using CPU mode"
fi

echo ""
echo "Available commands:"
echo "  python main.py --help                    # Show all available commands"
echo "  python main.py run-all --help            # Show run-all options"
echo "  python scripts/create_example_dataset.py # Create test dataset"
echo "  python tests/smoke_test.py               # Run smoke test"
echo ""

if [ "$1" = "jupyter" ]; then
    echo "Starting Jupyter notebook..."
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
else
    echo "Environment ready! Use 'python main.py --help' to see available commands."
    echo "To start Jupyter: ./start.sh jupyter"
fi
EOF

chmod +x start.sh

# Set permissions
chmod -R 755 /workspace

echo "Setup complete! Run './start.sh' to activate the environment."
echo ""
echo "All models are FREE and open source! 🎉"
echo "To download models, get a free HF token and run:"
echo "export HF_TOKEN='your_token_here'"
echo "huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir models/sd --token \$HF_TOKEN"
echo ""
echo "Get your free token at: https://huggingface.co/settings/tokens"
