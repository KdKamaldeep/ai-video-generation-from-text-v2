# Environment Setup Guide

## 🚫 **No .env File Needed!**

The WhyWouldYou-v2 pipeline is designed to work **without** a `.env` file. All configuration is handled through:

1. **`config.json`** - Pipeline settings and model paths
2. **`start.sh`** - Runtime environment variables (RunPod only)
3. **Command line arguments** - Override any config values

## 🔧 **Environment Variables (Optional)**

### For Model Downloads Only
```bash
# Set Hugging Face token for downloading models
export HF_TOKEN="your_token_here"

# Download models
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir models/sd --token $HF_TOKEN
```

### For RunPod Deployment
The `start.sh` script automatically sets these when you run it:
```bash
# Activate the environment
./start.sh

# This sets:
export WORKSPACE_DIR=/workspace
export MODELS_DIR=/workspace/models
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export TORCH_HOME=/workspace/models
export PYTHONPATH=/workspace:$PYTHONPATH
```

## 💡 **Why No .env File?**

1. **Security**: No sensitive data to store
2. **Simplicity**: Everything in `config.json`
3. **Portability**: Works on any system
4. **Flexibility**: Override via command line

## 🎯 **Configuration Priority**

1. **Command line arguments** (highest priority)
2. **config.json** (default values)
3. **Environment variables** (RunPod only)

## 📝 **Example Usage**

```bash
# Use config.json defaults
python main.py run-all --config config.json --prompt "cartoon cat"

# Override config values
python main.py run-all \
  --config config.json \
  --prompt "cartoon cat" \
  --frames 24 \
  --fps 30

# Set HF_TOKEN for model downloads
export HF_TOKEN="your_token"
python scripts/download_models.sh
```

## 🔍 **What Gets Set Where**

| Setting | Location | Purpose |
|---------|----------|---------|
| Model paths | `config.json` | Pipeline configuration |
| Generation params | `config.json` | SVD/SD settings |
| Output settings | `config.json` | Video/audio settings |
| Runtime paths | `start.sh` | RunPod environment |
| Model cache | `start.sh` | Hugging Face cache |
| CUDA settings | `start.sh` | GPU optimization |

## ✅ **Bottom Line**

**You don't need a `.env` file!** Just:
1. Copy `config.example.json` to `config.json`
2. Edit `config.json` with your settings
3. Run the pipeline with `python main.py`

The only environment variable you might need is `HF_TOKEN` for downloading models, and that's optional.
