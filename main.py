#!/usr/bin/env python3
"""
WhyWouldYou-v2 CLI Pipeline
A complete CLI pipeline for generating animated videos using Stable Video Diffusion.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import our modules
from sd_generator import SDGenerator
from pose_extractor import PoseExtractor
from flow_extractor import FlowExtractor
from svd_vid2vid import SVDVid2Vid
from coqui_tts import CoquiTTS
from wav2lip_infer import Wav2LipInfer
from postprocess import PostProcessor
from assemble import VideoAssembler
from utils.config import load_config, validate_config
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Set up the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        description="WhyWouldYou-v2: CLI pipeline for SVD video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate base image
  python main.py image --prompt "cartoon blue cat running" --config config.json --out images/base1.png
  
  # Extract motion from reference video
  python main.py extract-motion --motion_video ref/run_cycle.mp4 --out motion/run1/ --method mediapipe
  
  # Run SVD vid2vid
  python main.py svd --input_image images/base1.png --motion_dir motion/run1/ --out clips/scene1.mp4 --config config.json
  
  # Generate TTS audio
  python main.py tts --text "Watch out!" --out audio/line1.wav --config config.json
  
  # Run end-to-end pipeline
  python main.py run-all --config config.json --prompt "cartoon character" --motion_video ref.mp4 --tts_text "Hello world!"
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Image generation subcommand
    img_parser = subparsers.add_parser("image", help="Generate base image using Stable Diffusion")
    img_parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    img_parser.add_argument("--out", required=True, help="Output image path")
    img_parser.add_argument("--seed", type=int, help="Random seed")
    img_parser.add_argument("--width", type=int, default=512, help="Image width")
    img_parser.add_argument("--height", type=int, default=512, help="Image height")
    
    # Motion extraction subcommand
    motion_parser = subparsers.add_parser("extract-motion", help="Extract motion conditioning from reference video")
    motion_parser.add_argument("--motion_video", required=True, help="Reference motion video path")
    motion_parser.add_argument("--out", required=True, help="Output directory for motion frames")
    motion_parser.add_argument("--method", choices=["mediapipe", "openpose"], default="mediapipe", help="Pose extraction method")
    motion_parser.add_argument("--extract_flow", action="store_true", help="Also extract optical flow")
    
    # SVD vid2vid subcommand
    svd_parser = subparsers.add_parser("svd", help="Run SVD vid2vid to apply motion to base image")
    svd_parser.add_argument("--input_image", required=True, help="Base image path")
    svd_parser.add_argument("--motion_dir", required=True, help="Directory with motion conditioning frames")
    svd_parser.add_argument("--out", required=True, help="Output video path")
    svd_parser.add_argument("--frames", type=int, help="Number of frames to generate")
    svd_parser.add_argument("--fps", type=int, help="Output FPS")
    svd_parser.add_argument("--strength", type=float, help="Motion strength")
    svd_parser.add_argument("--guidance_scale", type=float, help="Guidance scale")
    svd_parser.add_argument("--seed", type=int, help="Random seed")
    
    # TTS subcommand
    tts_parser = subparsers.add_parser("tts", help="Generate TTS audio using Coqui")
    tts_parser.add_argument("--text", required=True, help="Text to synthesize")
    tts_parser.add_argument("--out", required=True, help="Output audio path")
    tts_parser.add_argument("--voice", help="Voice to use")
    tts_parser.add_argument("--sample_rate", type=int, help="Audio sample rate")
    
    # Wav2Lip subcommand
    wav2lip_parser = subparsers.add_parser("wav2lip", help="Run Wav2Lip for lip-sync")
    wav2lip_parser.add_argument("--video", required=True, help="Input video path")
    wav2lip_parser.add_argument("--audio", required=True, help="Input audio path")
    wav2lip_parser.add_argument("--out", required=True, help="Output synced video path")
    
    # Post-processing subcommand
    post_parser = subparsers.add_parser("post", help="Post-process video (RIFE, effects)")
    post_parser.add_argument("--input", required=True, help="Input video path")
    post_parser.add_argument("--out", required=True, help="Output video path")
    post_parser.add_argument("--rife", action="store_true", help="Enable RIFE interpolation")
    post_parser.add_argument("--rife_fps", type=int, help="Target FPS for RIFE")
    post_parser.add_argument("--smear", action="store_true", help="Add motion blur smears")
    post_parser.add_argument("--squash", action="store_true", help="Add squash & stretch effects")
    
    # Assembly subcommand
    assemble_parser = subparsers.add_parser("assemble", help="Assemble final video with FFmpeg")
    assemble_parser.add_argument("--clips", nargs="+", required=True, help="Input video clips")
    assemble_parser.add_argument("--audio", help="Background audio path")
    assemble_parser.add_argument("--out", required=True, help="Output video path")
    assemble_parser.add_argument("--crossfade", type=float, default=0.5, help="Crossfade duration in seconds")
    
    # Run-all subcommand
    runall_parser = subparsers.add_parser("run-all", help="Run complete end-to-end pipeline")
    runall_parser.add_argument("--prompt", required=True, help="Scene description prompt")
    runall_parser.add_argument("--motion_video", required=True, help="Reference motion video")
    runall_parser.add_argument("--tts_text", help="TTS text (optional)")
    runall_parser.add_argument("--out", default="output/final.mp4", help="Output video path")
    
    return parser


def run_image_generation(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run image generation subcommand."""
    logger.info("Generating base image...")
    
    generator = SDGenerator(config)
    generator.generate(
        prompt=args.prompt,
        output_path=args.out,
        seed=args.seed or config.get("seed", 42),
        width=args.width,
        height=args.height
    )
    logger.info(f"Image generated: {args.out}")


def run_motion_extraction(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run motion extraction subcommand."""
    logger.info("Extracting motion conditioning...")
    
    extractor = PoseExtractor(config)
    extractor.extract_pose(
        video_path=args.motion_video,
        output_dir=args.out,
        method=args.method
    )
    
    if args.extract_flow:
        logger.info("Extracting optical flow...")
        flow_extractor = FlowExtractor(config)
        flow_extractor.extract_flow(
            video_path=args.motion_video,
            output_dir=args.out
        )
    
    logger.info(f"Motion extracted to: {args.out}")


def run_svd_vid2vid(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run SVD vid2vid subcommand."""
    logger.info("Running SVD vid2vid...")
    
    svd = SVDVid2Vid(config)
    svd.generate(
        input_image=args.input_image,
        motion_dir=args.motion_dir,
        output_path=args.out,
        frames=args.frames or config.get("frames", 48),
        fps=args.fps or config.get("fps", 24),
        strength=args.strength or config.get("strength", 0.56),
        guidance_scale=args.guidance_scale or config.get("guidance_scale", 7.5),
        seed=args.seed or config.get("seed", 42)
    )
    logger.info(f"SVD video generated: {args.out}")


def run_tts(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run TTS subcommand."""
    logger.info("Generating TTS audio...")
    
    tts = CoquiTTS(config)
    tts.generate(
        text=args.text,
        output_path=args.out,
        voice=args.voice or config.get("coqui_voice", {}).get("voice", "alloy"),
        sample_rate=args.sample_rate or config.get("coqui_voice", {}).get("sample_rate", 24000)
    )
    logger.info(f"TTS audio generated: {args.out}")


def run_wav2lip(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run Wav2Lip subcommand."""
    logger.info("Running Wav2Lip lip-sync...")
    
    wav2lip = Wav2LipInfer(config)
    wav2lip.sync(
        video_path=args.video,
        audio_path=args.audio,
        output_path=args.out
    )
    logger.info(f"Lip-synced video generated: {args.out}")


def run_postprocess(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run post-processing subcommand."""
    logger.info("Post-processing video...")
    
    post = PostProcessor(config)
    post.process(
        input_path=args.input,
        output_path=args.out,
        rife=args.rife or config.get("post_rife", False),
        rife_fps=args.rife_fps or config.get("rife_fps", 48),
        smear=args.smear or config.get("post_smear", False),
        squash=args.squash or config.get("post_squash", False)
    )
    logger.info(f"Post-processed video: {args.out}")


def run_assemble(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run assembly subcommand."""
    logger.info("Assembling final video...")
    
    assembler = VideoAssembler(config)
    assembler.assemble(
        clips=args.clips,
        output_path=args.out,
        background_audio=args.audio,
        crossfade_duration=args.crossfade
    )
    logger.info(f"Final video assembled: {args.out}")


def run_complete_pipeline(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run complete end-to-end pipeline."""
    logger.info("Starting complete pipeline...")
    
    # Create output directories
    os.makedirs("images", exist_ok=True)
    os.makedirs("motion", exist_ok=True)
    os.makedirs("clips", exist_ok=True)
    os.makedirs("audio", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Step 1: Generate base image
    logger.info("Step 1: Generating base image...")
    generator = SDGenerator(config)
    base_image = "images/base_scene.png"
    generator.generate(
        prompt=args.prompt,
        output_path=base_image,
        seed=config.get("seed", 42)
    )
    
    # Step 2: Extract motion
    logger.info("Step 2: Extracting motion...")
    motion_dir = "motion/scene1"
    extractor = PoseExtractor(config)
    extractor.extract_pose(
        video_path=args.motion_video,
        output_dir=motion_dir,
        method="mediapipe"
    )
    
    # Step 3: Run SVD
    logger.info("Step 3: Running SVD vid2vid...")
    svd = SVDVid2Vid(config)
    scene_clip = "clips/scene1.mp4"
    svd.generate(
        input_image=base_image,
        motion_dir=motion_dir,
        output_path=scene_clip,
        frames=config.get("frames", 48),
        fps=config.get("fps", 24),
        strength=config.get("strength", 0.56),
        guidance_scale=config.get("guidance_scale", 7.5),
        seed=config.get("seed", 42)
    )
    
    # Step 4: TTS and lip-sync (if requested)
    if args.tts_text and config.get("wav2lip_enabled", True):
        logger.info("Step 4: Generating TTS and lip-sync...")
        tts = CoquiTTS(config)
        audio_file = "audio/dialogue.wav"
        tts.generate(
            text=args.tts_text,
            output_path=audio_file,
            voice=config.get("coqui_voice", {}).get("voice", "alloy"),
            sample_rate=config.get("coqui_voice", {}).get("sample_rate", 24000)
        )
        
        wav2lip = Wav2LipInfer(config)
        synced_clip = "clips/scene1_synced.mp4"
        wav2lip.sync(
            video_path=scene_clip,
            audio_path=audio_file,
            output_path=synced_clip
        )
        scene_clip = synced_clip
    
    # Step 5: Post-processing
    logger.info("Step 5: Post-processing...")
    post = PostProcessor(config)
    final_clip = "clips/scene1_final.mp4"
    post.process(
        input_path=scene_clip,
        output_path=final_clip,
        rife=config.get("post_rife", False),
        rife_fps=config.get("rife_fps", 48),
        smear=config.get("post_smear", False),
        squash=config.get("post_squash", False)
    )
    
    # Step 6: Final assembly
    logger.info("Step 6: Final assembly...")
    assembler = VideoAssembler(config)
    assembler.assemble(
        clips=[final_clip],
        output_path=args.out,
        crossfade_duration=0.5
    )
    
    logger.info(f"Pipeline complete! Output: {args.out}")


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        validate_config(config)
    
    try:
        # Route to appropriate subcommand
        if args.command == "image":
            run_image_generation(args, config)
        elif args.command == "extract-motion":
            run_motion_extraction(args, config)
        elif args.command == "svd":
            run_svd_vid2vid(args, config)
        elif args.command == "tts":
            run_tts(args, config)
        elif args.command == "wav2lip":
            run_wav2lip(args, config)
        elif args.command == "post":
            run_postprocess(args, config)
        elif args.command == "assemble":
            run_assemble(args, config)
        elif args.command == "run-all":
            run_complete_pipeline(args, config)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
