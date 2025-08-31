"""Video ingestion and metadata extraction."""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import json

import ffmpeg
from PIL import Image

from .config import Config
from .models import Video


logger = logging.getLogger(__name__)


class VideoIngester:
    """Handles video file ingestion and metadata extraction."""
    
    def __init__(self, config: Config):
        """Initialize the video ingester."""
        self.config = config
    
    async def ingest_video(
        self,
        video_path: Path,
        title: Optional[str] = None,
        **kwargs
    ) -> Video:
        """Ingest a video file and extract metadata."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract basic metadata
        metadata = await self._extract_metadata(video_path)
        
        # Generate title if not provided
        if not title:
            title = video_path.stem
        
        # Create video object
        video = Video(
            title=title,
            source_path=str(video_path.absolute()),
            duration=metadata["duration"],
            fps=metadata["fps"],
            width=metadata["width"],
            height=metadata["height"],
            size_bytes=video_path.stat().st_size,
            metadata=metadata
        )
        
        # Generate thumbnail
        await self._generate_thumbnail(video_path, video)
        
        logger.info(f"Ingested video: {title} ({metadata['duration']:.2f}s)")
        return video
    
    async def _extract_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract video metadata using ffmpeg."""
        try:
            # Use ffprobe to get metadata
            probe = ffmpeg.probe(str(video_path))
            
            # Get video stream info
            video_stream = next(
                (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found in file")
            
            # Extract basic metadata
            metadata = {
                "duration": float(probe["format"]["duration"]),
                "fps": self._parse_fps(video_stream.get("r_frame_rate", "0/1")),
                "width": int(video_stream["width"]),
                "height": int(video_stream["height"]),
                "codec": video_stream.get("codec_name", "unknown"),
                "bitrate": int(probe["format"].get("bit_rate", 0)),
                "format": probe["format"]["format_name"],
                "streams": len(probe["streams"]),
                "audio_streams": len([s for s in probe["streams"] if s["codec_type"] == "audio"]),
                "video_streams": len([s for s in probe["streams"] if s["codec_type"] == "video"]),
            }
            
            # Extract additional metadata if available
            if "tags" in probe["format"]:
                metadata["tags"] = probe["format"]["tags"]
            
            # Extract audio stream info
            audio_stream = next(
                (stream for stream in probe["streams"] if stream["codec_type"] == "audio"),
                None
            )
            
            if audio_stream:
                metadata["audio_codec"] = audio_stream.get("codec_name", "unknown")
                metadata["audio_channels"] = int(audio_stream.get("channels", 1))
                metadata["audio_sample_rate"] = int(audio_stream.get("sample_rate", 0))
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {video_path}: {e}")
            # Fallback to basic file info
            return {
                "duration": 0.0,
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "codec": "unknown",
                "bitrate": 0,
                "format": "unknown",
                "streams": 0,
                "audio_streams": 0,
                "video_streams": 0,
                "error": str(e)
            }
    
    def _parse_fps(self, fps_str: str) -> float:
        """Parse frame rate string to float."""
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den)
            else:
                return float(fps_str)
        except (ValueError, ZeroDivisionError):
            return 30.0  # Default fallback
    
    async def _generate_thumbnail(self, video_path: Path, video: Video) -> None:
        """Generate a thumbnail for the video."""
        try:
            storage_path = Path(self.config.storage.blob_path)
            thumbnails_dir = storage_path / "thumbnails"
            thumbnails_dir.mkdir(exist_ok=True)
            
            # Generate thumbnail at 10% of video duration
            timestamp = video.duration * 0.1
            
            thumbnail_path = thumbnails_dir / f"{video.id}.jpg"
            
            # Use ffmpeg to extract frame
            ffmpeg.input(str(video_path), ss=timestamp).filter(
                "scale", video.width, -1
            ).output(
                str(thumbnail_path), vframes=1, q="v=2"
            ).overwrite_output().run(
                capture_stdout=True, capture_stderr=True, quiet=True
            )
            
            # Update video with thumbnail path
            video.metadata["thumbnail_path"] = str(thumbnail_path)
            
            logger.debug(f"Generated thumbnail: {thumbnail_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail: {e}")
    
    async def validate_video(self, video_path: Path) -> Dict[str, Any]:
        """Validate video file and return validation results."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            return {"valid": False, "error": "File does not exist"}
        
        # Check file size
        file_size = video_path.stat().st_size
        max_size = 10 * 1024 * 1024 * 1024  # 10GB limit
        
        if file_size > max_size:
            return {"valid": False, "error": f"File too large: {file_size / (1024**3):.1f}GB"}
        
        # Check file extension
        valid_extensions = {
            ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
            ".3gp", ".ogv", ".ts", ".mts", ".m2ts", ".vob", ".asf", ".rm",
            ".rmvb", ".divx", ".xvid", ".h264", ".h265", ".hevc", ".vp8", ".vp9", ".av1"
        }
        
        if video_path.suffix.lower() not in valid_extensions:
            return {"valid": False, "error": f"Unsupported file format: {video_path.suffix}"}
        
        # Try to extract basic metadata
        try:
            metadata = await self._extract_metadata(video_path)
            if metadata.get("error"):
                return {"valid": False, "error": f"Metadata extraction failed: {metadata['error']}"}
            
            return {
                "valid": True,
                "metadata": metadata,
                "file_size": file_size
            }
            
        except Exception as e:
            return {"valid": False, "error": f"Validation failed: {str(e)}"}
    
    async def get_supported_formats(self) -> Dict[str, Any]:
        """Get list of supported video formats and codecs."""
        try:
            # Use ffmpeg to get supported formats
            result = subprocess.run(
                ["ffmpeg", "-formats"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            formats = {}
            for line in result.stdout.split("\n"):
                if line.strip() and not line.startswith("--"):
                    parts = line.split()
                    if len(parts) >= 2:
                        format_code = parts[1]
                        format_name = " ".join(parts[2:]) if len(parts) > 2 else format_code
                        formats[format_code] = format_name
            
            return {
                "supported_formats": formats,
                "total_formats": len(formats)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get supported formats: {e}")
            return {
                "supported_formats": {},
                "total_formats": 0,
                "error": str(e)
            }
