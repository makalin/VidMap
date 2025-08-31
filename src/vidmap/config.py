"""Configuration management for VidMap."""

import os
from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator


class ASRConfig(BaseModel):
    """Automatic Speech Recognition configuration."""
    engine: str = Field(default="whisper", description="ASR engine to use")
    model: str = Field(default="small", description="Model size/type")
    language: Optional[str] = Field(default=None, description="Language code")
    device: str = Field(default="cpu", description="Device to use (cpu/cuda)")


class DiarizationConfig(BaseModel):
    """Speaker diarization configuration."""
    enabled: bool = Field(default=True, description="Enable speaker diarization")
    model: str = Field(default="pyannote/speaker-diarization", description="Model to use")
    min_speakers: int = Field(default=1, description="Minimum number of speakers")
    max_speakers: int = Field(default=10, description="Maximum number of speakers")


class OCRConfig(BaseModel):
    """Optical Character Recognition configuration."""
    enabled: bool = Field(default=True, description="Enable OCR")
    engine: str = Field(default="tesseract", description="OCR engine to use")
    languages: List[str] = Field(default=["eng"], description="Languages to detect")
    confidence_threshold: float = Field(default=0.5, description="Minimum confidence")


class VisionConfig(BaseModel):
    """Computer vision configuration."""
    enabled: bool = Field(default=True, description="Enable vision analysis")
    scene_detection: bool = Field(default=True, description="Enable scene detection")
    object_detection: bool = Field(default=False, description="Enable object detection")
    face_detection: bool = Field(default=True, description="Enable face detection")


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""
    model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    dimension: int = Field(default=384, description="Embedding dimension")
    batch_size: int = Field(default=32, description="Batch size for processing")


class TilesConfig(BaseModel):
    """Tile generation configuration."""
    size: int = Field(default=256, description="Tile size in pixels")
    overlap: int = Field(default=0, description="Tile overlap in pixels")
    levels: int = Field(default=5, description="Number of zoom levels")


class UIConfig(BaseModel):
    """User interface configuration."""
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=5173, description="Port to bind to")
    heatmaps: List[str] = Field(
        default=["salience", "laughter", "silence", "slide_density"],
        description="Available heatmap layers"
    )


class StorageConfig(BaseModel):
    """Storage configuration."""
    backend: str = Field(default="sqlite", description="Database backend")
    url: Optional[str] = Field(default=None, description="Database connection URL")
    blob_storage: str = Field(default="local", description="Blob storage backend")
    blob_path: Optional[str] = Field(default=None, description="Local blob storage path")


class Config(BaseModel):
    """Main configuration for VidMap."""
    project: str = Field(..., description="Project name")
    storage: StorageConfig = Field(default_factory=StorageConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    tiles: TilesConfig = Field(default_factory=TilesConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    
    @validator("storage", pre=True)
    def set_default_storage_path(cls, v, values):
        """Set default storage path based on project name."""
        if isinstance(v, dict) and "blob_path" not in v:
            project_name = values.get("project", "default")
            v["blob_path"] = f".vidmap/{project_name}"
        return v
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def create_default(cls, project: str, **kwargs) -> "Config":
        """Create default configuration for a project."""
        config = cls(project=project, **kwargs)
        
        # Set default storage path
        if not config.storage.blob_path:
            config.storage.blob_path = f".vidmap/{project}"
        
        return config
