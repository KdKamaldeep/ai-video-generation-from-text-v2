#!/bin/bash

# WhyWouldYou-v2 Linux Setup Script
# This script sets up the entire project on Linux with one command

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"
MODELS_DIR="$PROJECT_ROOT/models"
EXTERNAL_DIR="$PROJECT_ROOT/external"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_deps() {
    local distro=$(detect_distro)
    
    print_header "Installing System Dependencies"
    
    case $distro in
        "ubuntu"|"debian")
            print_status "Detected Ubuntu/Debian. Installing packages..."
            sudo apt-get update
            sudo apt-get install -y \
                python3.10 \
                python3.10-dev \
                python3.10-venv \
                python3-pip \
                git \
                build-essential \
                libsndfile1 \
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
                wget \
                curl \
                pkg-config \
                libssl-dev \
                libffi-dev \
                libbz2-dev \
                libreadline-dev \
                libsqlite3-dev \
                libncursesw5-dev \
                xz-utils \
                tk-dev \
                libxml2-dev \
                libxmlsec1-dev \
                liblzma-dev
            ;;
        "centos"|"rhel"|"fedora"|"rocky"|"almalinux")
            print_status "Detected CentOS/RHEL/Fedora. Installing packages..."
            if command_exists dnf; then
                sudo dnf update -y
                sudo dnf install -y \
                    python3.10 \
                    python3.10-devel \
                    python3.10-pip \
                    git \
                    gcc \
                    gcc-c++ \
                    make \
                    libsndfile \
                    ffmpeg \
                    ffmpeg-devel \
                    mesa-libGL \
                    mesa-libGL-devel \
                    gtk3 \
                    gtk3-devel \
                    wget \
                    curl \
                    pkg-config \
                    openssl-devel \
                    libffi-devel \
                    bzip2-devel \
                    readline-devel \
                    sqlite-devel \
                    ncurses-devel \
                    xz-devel \
                    tk-devel \
                    libxml2-devel \
                    libxmlsec1-devel \
                    xz-devel
            else
                sudo yum update -y
                sudo yum install -y \
                    python3.10 \
                    python3.10-devel \
                    python3.10-pip \
                    git \
                    gcc \
                    gcc-c++ \
                    make \
                    libsndfile \
                    ffmpeg \
                    ffmpeg-devel \
                    mesa-libGL \
                    mesa-libGL-devel \
                    gtk3 \
                    gtk3-devel \
                    wget \
                    curl \
                    pkg-config \
                    openssl-devel \
                    libffi-devel \
                    bzip2-devel \
                    readline-devel \
                    sqlite-devel \
                    ncurses-devel \
                    xz-devel \
                    tk-devel \
                    libxml2-devel \
                    libxmlsec1-devel \
                    xz-devel
            fi
            ;;
        "arch")
            print_status "Detected Arch Linux. Installing packages..."
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                python \
                python-pip \
                git \
                base-devel \
                libsndfile \
                ffmpeg \
                mesa \
                gtk3 \
                wget \
                curl \
                pkg-config \
                openssl \
                libffi \
                bzip2 \
                readline \
                sqlite \
                ncurses \
                xz \
                tk \
                libxml2 \
                libxmlsec1
            ;;
        *)
            print_warning "Unknown distribution: $distro"
            print_warning "Please install the following packages manually:"
            print_warning "- python3.10, python3.10-dev, python3.10-venv"
            print_warning "- git, build-essential, ffmpeg"
            print_warning "- libsndfile1, libsm6, libxext6, libxrender-dev"
            print_warning "- libgomp1, libglib2.0-0, libgl1-mesa-glx, libgtk-3-0"
            print_warning "- libavcodec-dev, libavformat-dev, libswscale-dev"
            print_warning "- wget, curl, pkg-config"
            ;;
    esac
    
    print_success "System dependencies installed"
}

# Function to check CUDA installation
check_cuda() {
    print_header "Checking CUDA Installation"
    
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        print_warning "NVIDIA GPU not detected or drivers not installed"
        print_warning "This project requires CUDA for optimal performance"
        print_warning "Please install NVIDIA drivers and CUDA toolkit"
    fi
    
    if command_exists nvcc; then
        print_success "CUDA toolkit found"
        nvcc --version | head -n 1
    else
        print_warning "CUDA toolkit not found"
        print_warning "Please install CUDA toolkit for GPU acceleration"
    fi
}

# Function to setup Python virtual environment
setup_python_env() {
    print_header "Setting up Python Virtual Environment"
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf "$VENV_DIR"
    fi
    
    print_status "Creating virtual environment..."
    python3.10 -m venv "$VENV_DIR"
    
    print_status "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    print_status "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    print_status "Installing Python dependencies..."
    
    # Try to install requirements with better error handling
    if ! pip install -r "$PROJECT_ROOT/requirements.txt"; then
        print_warning "First attempt failed. Trying with --no-deps for problematic packages..."
        
        # Install core packages first
        pip install torch>=2.0.0,<3.0.0 torchvision>=0.15.0,<1.0.0 torchaudio>=2.0.1,<3.0.0
        pip install diffusers>=0.24.0,<1.0.0 transformers>=4.35.0,<5.0.0 accelerate>=0.24.0,<1.0.0
        pip install opencv-python>=4.8.0 Pillow>=10.0.0 numpy>=1.24.0 scipy>=1.11.2
        pip install librosa>=0.10.0,<1.0.0 soundfile>=0.12.0 mediapipe>=0.10.0
        pip install TTS>=0.22.0,<1.0.0 scikit-learn>=1.3.0 matplotlib>=3.7.0
        pip install tqdm>=4.65.0 click>=8.1.0 pyyaml>=6.0 python-dotenv>=1.0.0
        
        # Try xformers separately (can be problematic)
        if ! pip install xformers>=0.0.22,<1.0.0; then
            print_warning "xformers installation failed. Continuing without it..."
        fi
    fi
    
    # Install additional useful packages
    print_status "Installing additional packages..."
    pip install huggingface_hub accelerate
    
    print_success "Python environment setup complete"
}

# Function to setup configuration
setup_config() {
    print_header "Setting up Configuration"
    
    if [ ! -f "$PROJECT_ROOT/config.json" ]; then
        print_status "Creating config.json from example..."
        cp "$PROJECT_ROOT/config.example.json" "$PROJECT_ROOT/config.json"
        print_success "Configuration file created"
    else
        print_warning "config.json already exists, skipping..."
    fi
    
    # Create necessary directories
    print_status "Creating project directories..."
    mkdir -p "$PROJECT_ROOT"/{images,motion,clips,audio,output,models,external}
    
    print_success "Configuration setup complete"
}

# Function to download models
download_models() {
    print_header "Downloading AI Models"
    
    # Check if HF_TOKEN is set
    if [ -z "$HF_TOKEN" ]; then
        print_warning "HF_TOKEN environment variable not set."
        print_warning "Some models may require authentication."
        print_warning "Set it with: export HF_TOKEN='your_token_here'"
        print_warning "Get a free token at: https://huggingface.co/settings/tokens"
    fi
    
    # Run the model download script
    if [ -f "$PROJECT_ROOT/scripts/download_models.sh" ]; then
        print_status "Running model download script..."
        chmod +x "$PROJECT_ROOT/scripts/download_models.sh"
        "$PROJECT_ROOT/scripts/download_models.sh"
    else
        print_error "Model download script not found!"
        exit 1
    fi
    
    print_success "Model download complete"
}

# Function to setup environment variables
setup_env_vars() {
    print_header "Setting up Environment Variables"
    
    # Create environment setup script
    cat > "$PROJECT_ROOT/setup_env.sh" << 'EOF'
#!/bin/bash
# Environment setup script for WhyWouldYou-v2

export WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODELS_DIR="$WORKSPACE_DIR/models"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"
export HF_HOME="$WORKSPACE_DIR/models"
export TRANSFORMERS_CACHE="$WORKSPACE_DIR/models"
export TORCH_HOME="$WORKSPACE_DIR/models"

# Activate virtual environment
if [ -f "$WORKSPACE_DIR/venv/bin/activate" ]; then
    source "$WORKSPACE_DIR/venv/bin/activate"
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found"
fi

echo "Environment variables set:"
echo "  WORKSPACE_DIR: $WORKSPACE_DIR"
echo "  MODELS_DIR: $MODELS_DIR"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  HF_HOME: $HF_HOME"
EOF
    
    chmod +x "$PROJECT_ROOT/setup_env.sh"
    print_success "Environment setup script created"
}

# Function to run tests
run_tests() {
    print_header "Running Tests"
    
    # Activate virtual environment for tests
    source "$VENV_DIR/bin/activate"
    
    # Set environment variables
    export WORKSPACE_DIR="$PROJECT_ROOT"
    export MODELS_DIR="$PROJECT_ROOT/models"
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export HF_HOME="$PROJECT_ROOT/models"
    export TRANSFORMERS_CACHE="$PROJECT_ROOT/models"
    export TORCH_HOME="$PROJECT_ROOT/models"
    
    print_status "Running smoke test..."
    if [ -f "$PROJECT_ROOT/tests/smoke_test.py" ]; then
        python "$PROJECT_ROOT/tests/smoke_test.py"
        print_success "Smoke test completed"
    else
        print_warning "Smoke test not found, skipping..."
    fi
    
    print_status "Testing main script..."
    python "$PROJECT_ROOT/main.py" --help
    print_success "Main script test completed"
}

# Function to create example dataset
create_example_dataset() {
    print_header "Creating Example Dataset"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Set environment variables
    export WORKSPACE_DIR="$PROJECT_ROOT"
    export MODELS_DIR="$PROJECT_ROOT/models"
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    if [ -f "$PROJECT_ROOT/scripts/create_example_dataset.py" ]; then
        print_status "Creating example dataset..."
        python "$PROJECT_ROOT/scripts/create_example_dataset.py"
        print_success "Example dataset created"
    else
        print_warning "Example dataset script not found, skipping..."
    fi
}

# Function to display final instructions
display_final_instructions() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}🎉 WhyWouldYou-v2 has been successfully set up!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "1. ${CYAN}Activate the environment:${NC}"
    echo -e "   ${YELLOW}source $PROJECT_ROOT/setup_env.sh${NC}"
    echo ""
    echo -e "2. ${CYAN}Test the pipeline:${NC}"
    echo -e "   ${YELLOW}python main.py --help${NC}"
    echo ""
    echo -e "3. ${CYAN}Run a quick test:${NC}"
    echo -e "   ${YELLOW}python tests/smoke_test.py${NC}"
    echo ""
    echo -e "4. ${CYAN}Create your first video:${NC}"
    echo -e "   ${YELLOW}python main.py --prompt \"A beautiful sunset\" --motion video.mp4 --tts \"Hello world\"${NC}"
    echo ""
    echo -e "${BLUE}Important notes:${NC}"
    echo -e "• ${YELLOW}GPU with 8GB+ VRAM recommended${NC}"
    echo -e "• ${YELLOW}Set HF_TOKEN for faster model downloads${NC}"
    echo -e "• ${YELLOW}Check config.json for customization${NC}"
    echo -e "• ${YELLOW}All models are free and open source!${NC}"
    echo ""
    echo -e "${PURPLE}Happy video generation! 🎬${NC}"
}

# Main setup function
main() {
    print_header "WhyWouldYou-v2 Linux Setup"
    echo -e "${CYAN}This script will set up the entire project on Linux${NC}"
    echo -e "${CYAN}Estimated time: 10-30 minutes (depending on internet speed)${NC}"
    echo ""
    
    # Check if running as root
    
    
    # Check if we're in the project directory
    if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Confirm with user
    echo -e "${YELLOW}This will install system packages and download large AI models.${NC}"
    echo -e "${YELLOW}Continue? (y/N):${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_status "Setup cancelled"
        exit 0
    fi
    
    # Run setup steps
    install_system_deps
    check_cuda
    setup_python_env
    setup_config
    download_models
    setup_env_vars
    run_tests
    create_example_dataset
    display_final_instructions
}

# Run main function
main "$@"
