"""VidMap - Map-like indexing for long videos with AI-powered scene analysis."""

__version__ = "1.0.0"
__author__ = "VidMap Team"
__email__ = "makalin@github.com"

from .core import VidMap
from .models import Video, Scene, Segment, Speaker, Topic
from .config import Config

__all__ = [
    "VidMap",
    "Video",
    "Scene", 
    "Segment",
    "Speaker",
    "Topic",
    "Config",
    "__version__",
]
