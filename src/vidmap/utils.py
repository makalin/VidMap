"""Utility functions and helper classes for VidMap."""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import zipfile

import aiofiles
import aiohttp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


logger = logging.getLogger(__name__)


class VideoUtils:
    """Utility functions for video processing."""
    
    @staticmethod
    def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive video information using ffprobe."""
        try:
            import ffmpeg
            
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(
                (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
                None
            )
            audio_stream = next(
                (stream for stream in probe["streams"] if stream["codec_type"] == "audio"),
                None
            )
            
            info = {
                "duration": float(probe["format"]["duration"]),
                "size_bytes": int(probe["format"]["size"]),
                "bitrate": int(probe["format"]["bit_rate"]),
                "format": probe["format"]["format_name"],
                "video": {
                    "codec": video_stream.get("codec_name", "unknown"),
                    "width": int(video_stream["width"]),
                    "height": int(video_stream["height"]),
                    "fps": VideoUtils._parse_fps(video_stream.get("r_frame_rate", "0/1")),
                    "pixel_format": video_stream.get("pix_fmt", "unknown"),
                    "bitrate": int(video_stream.get("bit_rate", 0)),
                    "rotation": int(video_stream.get("rotation", 0)),
                } if video_stream else {},
                "audio": {
                    "codec": audio_stream.get("codec_name", "unknown"),
                    "channels": int(audio_stream.get("channels", 0)),
                    "sample_rate": int(audio_stream.get("sample_rate", 0)),
                    "bitrate": int(audio_stream.get("bit_rate", 0)),
                } if audio_stream else {},
                "metadata": probe["format"].get("tags", {})
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {}
    
    @staticmethod
    def _parse_fps(fps_str: str) -> float:
        """Parse frame rate string to float."""
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den)
            else:
                return float(fps_str)
        except (ValueError, ZeroDivisionError):
            return 30.0
    
    @staticmethod
    def extract_frames(
        video_path: Union[str, Path],
        timestamps: List[float],
        output_dir: Union[str, Path],
        quality: int = 2
    ) -> List[str]:
        """Extract frames at specific timestamps."""
        try:
            import ffmpeg
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            frame_paths = []
            for i, timestamp in enumerate(timestamps):
                output_path = output_dir / f"frame_{i:04d}_{timestamp:.2f}.jpg"
                
                (
                    ffmpeg
                    .input(str(video_path), ss=timestamp)
                    .output(str(output_path), vframes=1, q=f"v={quality}")
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )
                
                frame_paths.append(str(output_path))
            
            return frame_paths
            
        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            return []
    
    @staticmethod
    def create_video_thumbnail(
        video_path: Union[str, Path],
        timestamp: float,
        output_path: Union[str, Path],
        size: Tuple[int, int] = (320, 180)
    ) -> bool:
        """Create a thumbnail from video at specific timestamp."""
        try:
            import ffmpeg
            
            (
                ffmpeg
                .input(str(video_path), ss=timestamp)
                .filter("scale", size[0], size[1])
                .output(str(output_path), vframes=1, q="v=2")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return False
    
    @staticmethod
    def extract_audio(
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        format: str = "wav",
        start_time: Optional[float] = None,
        duration: Optional[float] = None
    ) -> bool:
        """Extract audio from video file."""
        try:
            import ffmpeg
            
            input_stream = ffmpeg.input(str(video_path))
            
            if start_time is not None:
                input_stream = input_stream.filter("atrim", start=start_time)
            
            if duration is not None:
                input_stream = input_stream.filter("atrim", duration=duration)
            
            input_stream.output(str(output_path), acodec=format).overwrite_output().run(
                capture_stdout=True, capture_stderr=True, quiet=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return False
    
    @staticmethod
    def get_video_metadata(video_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract all available metadata from video file."""
        try:
            import ffmpeg
            
            probe = ffmpeg.probe(str(video_path))
            metadata = {}
            
            # Format metadata
            if "format" in probe:
                metadata.update(probe["format"])
            
            # Stream metadata
            for i, stream in enumerate(probe.get("streams", [])):
                stream_type = stream.get("codec_type", "unknown")
                metadata[f"stream_{i}_{stream_type}"] = stream
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return {}


class ImageUtils:
    """Utility functions for image processing."""
    
    @staticmethod
    def resize_image(
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        size: Tuple[int, int],
        maintain_aspect: bool = True
    ) -> bool:
        """Resize image to specified dimensions."""
        try:
            with Image.open(image_path) as img:
                if maintain_aspect:
                    img.thumbnail(size, Image.Resampling.LANCZOS)
                else:
                    img = img.resize(size, Image.Resampling.LANCZOS)
                
                img.save(output_path, quality=95)
                return True
                
        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            return False
    
    @staticmethod
    def create_collage(
        image_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        grid_size: Tuple[int, int],
        spacing: int = 10
    ) -> bool:
        """Create a collage from multiple images."""
        try:
            if len(image_paths) == 0:
                return False
            
            # Load first image to get dimensions
            with Image.open(image_paths[0]) as first_img:
                img_width, img_height = first_img.size
            
            # Calculate collage dimensions
            collage_width = grid_size[0] * img_width + (grid_size[0] - 1) * spacing
            collage_height = grid_size[1] * img_height + (grid_size[1] - 1) * spacing
            
            # Create blank collage
            collage = Image.new("RGB", (collage_width, collage_height), "white")
            
            # Place images
            for i, img_path in enumerate(image_paths[:grid_size[0] * grid_size[1]]):
                row = i // grid_size[0]
                col = i % grid_size[0]
                
                x = col * (img_width + spacing)
                y = row * (img_height + spacing)
                
                with Image.open(img_path) as img:
                    img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                    collage.paste(img, (x, y))
            
            collage.save(output_path, quality=95)
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collage: {e}")
            return False
    
    @staticmethod
    def extract_dominant_colors(
        image_path: Union[str, Path],
        num_colors: int = 5
    ) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Resize for faster processing
                img = img.resize((150, 150), Image.Resampling.LANCZOS)
                
                # Get colors
                colors = img.getcolors(maxcolors=img.size[0] * img.size[1])
                if colors is None:
                    return []
                
                # Sort by frequency and get top colors
                colors.sort(key=lambda x: x[0], reverse=True)
                return [color[1] for color in colors[:num_colors]]
                
        except Exception as e:
            logger.error(f"Failed to extract colors: {e}")
            return []
    
    @staticmethod
    def add_text_overlay(
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        text: str,
        position: Tuple[int, int] = (10, 10),
        font_size: int = 24,
        color: Tuple[int, int, int] = (255, 255, 255),
        background: Optional[Tuple[int, int, int]] = (0, 0, 0)
    ) -> bool:
        """Add text overlay to image."""
        try:
            with Image.open(image_path) as img:
                draw = ImageDraw.Draw(img)
                
                # Try to load font
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # Draw background if specified
                if background:
                    bbox = draw.textbbox(position, text, font=font)
                    draw.rectangle(bbox, fill=background)
                
                # Draw text
                draw.text(position, text, fill=color, font=font)
                
                img.save(output_path, quality=95)
                return True
                
        except Exception as e:
            logger.error(f"Failed to add text overlay: {e}")
            return False


class AudioUtils:
    """Utility functions for audio processing."""
    
    @staticmethod
    def analyze_audio_energy(
        audio_path: Union[str, Path],
        window_size: float = 1.0
    ) -> List[float]:
        """Analyze audio energy over time."""
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(str(audio_path))
            
            # Calculate energy
            hop_length = int(sr * window_size)
            energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            return energy.tolist()
            
        except ImportError:
            logger.warning("librosa not available, using fallback method")
            return AudioUtils._fallback_energy_analysis(audio_path, window_size)
        except Exception as e:
            logger.error(f"Failed to analyze audio energy: {e}")
            return []
    
    @staticmethod
    def _fallback_energy_analysis(
        audio_path: Union[str, Path],
        window_size: float
    ) -> List[float]:
        """Fallback energy analysis using scipy."""
        try:
            from scipy.io import wavfile
            from scipy.signal import windows
            
            # Read audio file
            sr, data = wavfile.read(str(audio_path))
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            # Calculate window size in samples
            window_samples = int(sr * window_size)
            
            # Calculate energy for each window
            energy = []
            for i in range(0, len(data), window_samples):
                window_data = data[i:i + window_samples]
                if len(window_data) > 0:
                    energy.append(np.sqrt(np.mean(window_data ** 2)))
            
            return energy
            
        except Exception as e:
            logger.error(f"Fallback energy analysis failed: {e}")
            return []
    
    @staticmethod
    def detect_silence(
        audio_path: Union[str, Path],
        threshold: float = 0.01,
        min_duration: float = 0.5
    ) -> List[Tuple[float, float]]:
        """Detect silence periods in audio."""
        try:
            energy = AudioUtils.analyze_audio_energy(audio_path)
            if not energy:
                return []
            
            # Find silence periods
            silence_periods = []
            start_time = None
            
            for i, e in enumerate(energy):
                if e < threshold and start_time is None:
                    start_time = i
                elif e >= threshold and start_time is not None:
                    duration = i - start_time
                    if duration >= min_duration:
                        silence_periods.append((start_time, duration))
                    start_time = None
            
            # Handle case where audio ends with silence
            if start_time is not None:
                duration = len(energy) - start_time
                if duration >= min_duration:
                    silence_periods.append((start_time, duration))
            
            return silence_periods
            
        except Exception as e:
            logger.error(f"Failed to detect silence: {e}")
            return []


class TextUtils:
    """Utility functions for text processing."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text
    
    @staticmethod
    def extract_keywords(
        text: str,
        max_keywords: int = 10,
        min_length: int = 3
    ) -> List[str]:
        """Extract keywords from text."""
        # Clean text
        text = TextUtils.clean_text(text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Split into words and filter
        words = text.split()
        keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
        
        # Count frequency
        from collections import Counter
        word_counts = Counter(keywords)
        
        # Return top keywords
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        except Exception:
            return 0.0
    
    @staticmethod
    def extract_timestamps(text: str) -> List[float]:
        """Extract timestamp references from text."""
        # Pattern for various timestamp formats
        patterns = [
            r'(\d{1,2}):(\d{2}):(\d{2})',  # HH:MM:SS
            r'(\d{1,2}):(\d{2})',          # MM:SS
            r'(\d+(?:\.\d+)?)s',            # 123.45s
            r'(\d+(?:\.\d+)?)m',            # 123.45m
        ]
        
        timestamps = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if pattern == r'(\d{1,2}):(\d{2}):(\d{2})':
                    hours, minutes, seconds = map(int, match.groups())
                    timestamps.append(hours * 3600 + minutes * 60 + seconds)
                elif pattern == r'(\d{1,2}):(\d{2})':
                    minutes, seconds = map(int, match.groups())
                    timestamps.append(minutes * 60 + seconds)
                elif pattern == r'(\d+(?:\.\d+)?)s':
                    timestamps.append(float(match.group(1)))
                elif pattern == r'(\d+(?:\.\d+)?)m':
                    timestamps.append(float(match.group(1)) * 60)
        
        return sorted(timestamps)


class FileUtils:
    """Utility functions for file operations."""
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
        """Calculate file hash."""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """Get file size in bytes."""
        return Path(file_path).stat().st_size
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    @staticmethod
    def create_backup(file_path: Union[str, Path], suffix: str = ".backup") -> str:
        """Create a backup of a file."""
        file_path = Path(file_path)
        backup_path = file_path.with_suffix(file_path.suffix + suffix)
        
        shutil.copy2(file_path, backup_path)
        return str(backup_path)
    
    @staticmethod
    def safe_delete(file_path: Union[str, Path]) -> bool:
        """Safely delete a file with error handling."""
        try:
            Path(file_path).unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def copy_directory(
        src: Union[str, Path],
        dst: Union[str, Path],
        ignore_patterns: Optional[List[str]] = None
    ) -> bool:
        """Copy directory with optional ignore patterns."""
        try:
            src = Path(src)
            dst = Path(dst)
            
            if ignore_patterns is None:
                ignore_patterns = ["__pycache__", "*.pyc", ".git", ".DS_Store"]
            
            def ignore_func(dir_path, filenames):
                return [name for name in filenames if any(
                    pattern in name or name.endswith(tuple(pattern for pattern in ignore_patterns if pattern.startswith("*")))
                )]
            
            shutil.copytree(src, dst, ignore=ignore_func)
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy directory: {e}")
            return False


class NetworkUtils:
    """Utility functions for network operations."""
    
    @staticmethod
    async def download_file(
        url: str,
        output_path: Union[str, Path],
        chunk_size: int = 8192,
        timeout: int = 30
    ) -> bool:
        """Download file from URL."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            f.write(chunk)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False
    
    @staticmethod
    async def check_url_accessible(url: str, timeout: int = 10) -> bool:
        """Check if URL is accessible."""
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.head(url) as response:
                    return response.status < 400
        except Exception:
            return False
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if string is a valid URL."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False


class TimeUtils:
    """Utility functions for time operations."""
    
    @staticmethod
    def format_duration(seconds: float, include_milliseconds: bool = False) -> str:
        """Format duration in human-readable format."""
        if seconds < 0:
            return "0:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if include_milliseconds:
            millisecs = int((seconds % 1) * 1000)
            if hours > 0:
                return f"{hours}:{minutes:02d}:{secs:02d},{millisecs:03d}"
            else:
                return f"{minutes}:{secs:02d},{millisecs:03d}"
        else:
            if hours > 0:
                return f"{hours}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes}:{secs:02d}"
    
    @staticmethod
    def parse_duration(duration_str: str) -> float:
        """Parse duration string to seconds."""
        try:
            # Handle various formats
            if ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(float, parts)
                    return hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:  # MM:SS
                    minutes, seconds = map(float, parts)
                    return minutes * 60 + seconds
            elif duration_str.endswith('s'):
                return float(duration_str[:-1])
            elif duration_str.endswith('m'):
                return float(duration_str[:-1]) * 60
            elif duration_str.endswith('h'):
                return float(duration_str[:-1]) * 3600
            else:
                return float(duration_str)
        except Exception:
            return 0.0
    
    @staticmethod
    def get_time_ranges(
        total_duration: float,
        segment_duration: float,
        overlap: float = 0.0
    ) -> List[Tuple[float, float]]:
        """Generate time ranges for segmentation."""
        ranges = []
        current_time = 0.0
        
        while current_time < total_duration:
            end_time = min(current_time + segment_duration, total_duration)
            ranges.append((current_time, end_time))
            
            # Move to next segment with overlap
            current_time = end_time - overlap
            if current_time >= total_duration:
                break
        
        return ranges


class ValidationUtils:
    """Utility functions for data validation."""
    
    @staticmethod
    def validate_video_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate video file and return validation results."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"valid": False, "error": "File does not exist"}
        
        # Check file size
        file_size = file_path.stat().st_size
        max_size = 50 * 1024 * 1024 * 1024  # 50GB limit
        
        if file_size > max_size:
            return {"valid": False, "error": f"File too large: {FileUtils.format_file_size(file_size)}"}
        
        # Check file extension
        valid_extensions = {
            '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v',
            '.3gp', '.ogv', '.ts', '.mts', '.m2ts', '.vob', '.asf', '.rm',
            '.rmvb', '.divx', '.xvid', '.h264', '.h265', '.hevc', '.vp8', '.vp9', '.av1'
        }
        
        if file_path.suffix.lower() not in valid_extensions:
            return {"valid": False, "error": f"Unsupported file format: {file_path.suffix}"}
        
        # Try to get video info
        try:
            video_info = VideoUtils.get_video_info(file_path)
            if not video_info:
                return {"valid": False, "error": "Could not read video file"}
            
            return {
                "valid": True,
                "file_size": file_size,
                "duration": video_info.get("duration", 0),
                "resolution": f"{video_info.get('video', {}).get('width', 0)}x{video_info.get('video', {}).get('height', 0)}",
                "format": video_info.get("format", "unknown")
            }
            
        except Exception as e:
            return {"valid": False, "error": f"Validation failed: {str(e)}"}
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Check required fields
        required_fields = ["project"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate project name
        if "project" in config:
            project_name = config["project"]
            if not isinstance(project_name, str) or len(project_name.strip()) == 0:
                errors.append("Project name must be a non-empty string")
            elif not re.match(r'^[a-zA-Z0-9_-]+$', project_name):
                errors.append("Project name contains invalid characters")
        
        # Validate numeric fields
        numeric_fields = {
            "ui.port": (1, 65535),
            "tiles.size": (64, 1024),
            "tiles.levels": (1, 10),
            "embeddings.dimension": (64, 2048),
            "embeddings.batch_size": (1, 128)
        }
        
        for field_path, (min_val, max_val) in numeric_fields.items():
            value = ValidationUtils._get_nested_value(config, field_path)
            if value is not None:
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    errors.append(f"{field_path} must be between {min_val} and {max_val}")
        
        return errors
    
    @staticmethod
    def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


class PerformanceUtils:
    """Utility functions for performance monitoring."""
    
    @staticmethod
    def measure_time(func):
        """Decorator to measure function execution time."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
            return result
        return wrapper
    
    @staticmethod
    async def measure_async_time(func):
        """Decorator to measure async function execution time."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            
            logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information."""
        try:
            import psutil
            
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
                "disk_usage": psutil.disk_usage('/').percent
            }
        except ImportError:
            return {"error": "psutil not available"}


# Convenience functions
def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """Get file hash (convenience function)."""
    return FileUtils.get_file_hash(file_path, algorithm)


def format_duration(seconds: float, include_milliseconds: bool = False) -> str:
    """Format duration (convenience function)."""
    return TimeUtils.format_duration(seconds, include_milliseconds)


def validate_video_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate video file (convenience function)."""
    return ValidationUtils.validate_video_file(file_path)


def clean_text(text: str) -> str:
    """Clean text (convenience function)."""
    return TextUtils.clean_text(text)


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords (convenience function)."""
    return TextUtils.extract_keywords(text, max_keywords)
