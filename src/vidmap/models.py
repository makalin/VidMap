"""Data models for VidMap."""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from uuid import uuid4, UUID


class Video(BaseModel):
    """Video metadata and information."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Video title")
    source_path: str = Field(..., description="Path to video file")
    duration: float = Field(..., description="Duration in seconds")
    fps: float = Field(..., description="Frames per second")
    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")
    size_bytes: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Speaker(BaseModel):
    """Speaker information from diarization."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = Field(None, description="Speaker name if known")
    gender: Optional[str] = Field(None, description="Speaker gender")
    confidence: float = Field(..., description="Speaker detection confidence")
    segments: List["Segment"] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Topic(BaseModel):
    """Topic or theme identified in the video."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Topic name")
    description: Optional[str] = Field(None, description="Topic description")
    confidence: float = Field(..., description="Topic detection confidence")
    keywords: List[str] = Field(default_factory=list)
    segments: List["Segment"] = Field(default_factory=list)
    start_time: float = Field(..., description="Topic start time")
    end_time: float = Field(..., description="Topic end time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Segment(BaseModel):
    """Base segment class for different types of video segments."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    video_id: str = Field(..., description="Parent video ID")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    duration: float = Field(..., description="Duration in seconds")
    confidence: float = Field(..., description="Detection confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("duration", pre=True, always=True)
    def calculate_duration(cls, v, values):
        """Calculate duration from start and end times."""
        if "start_time" in values and "end_time" in values:
            return values["end_time"] - values["start_time"]
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Scene(Segment):
    """Scene segment with visual analysis."""
    type: str = Field(default="scene", description="Segment type")
    shot_type: Optional[str] = Field(None, description="Type of shot (close-up, wide, etc.)")
    visual_tags: List[str] = Field(default_factory=list, description="Visual content tags")
    dominant_colors: List[str] = Field(default_factory=list, description="Dominant colors")
    brightness: Optional[float] = Field(None, description="Average brightness")
    motion_score: Optional[float] = Field(None, description="Motion intensity score")
    thumbnail_path: Optional[str] = Field(None, description="Path to scene thumbnail")
    keyframes: List[float] = Field(default_factory=list, description="Keyframe timestamps")


class AudioSegment(Segment):
    """Audio segment with transcription and analysis."""
    type: str = Field(default="audio", description="Segment type")
    text: str = Field(..., description="Transcribed text")
    language: str = Field(default="en", description="Detected language")
    speaker_id: Optional[str] = Field(None, description="Speaker ID if diarized")
    words: List[Dict[str, Any]] = Field(default_factory=list, description="Word-level timing")
    sentiment: Optional[str] = Field(None, description="Sentiment analysis")
    energy: Optional[float] = Field(None, description="Audio energy level")
    silence: bool = Field(default=False, description="Whether segment is silent")


class OCRSegment(Segment):
    """OCR segment with extracted text from video frames."""
    type: str = Field(default="ocr", description="Segment type")
    text: str = Field(..., description="Extracted text")
    language: str = Field(default="en", description="Detected language")
    confidence: float = Field(..., description="OCR confidence")
    bounding_box: List[float] = Field(..., description="Text bounding box [x, y, w, h]")
    font_size: Optional[float] = Field(None, description="Estimated font size")
    is_slide: bool = Field(default=False, description="Whether text is from a slide")


class Chapter(BaseModel):
    """Chapter or section of the video."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Chapter title")
    description: Optional[str] = Field(None, description="Chapter description")
    start_time: float = Field(..., description="Chapter start time")
    end_time: float = Field(..., description="Chapter end time")
    duration: float = Field(..., description="Chapter duration")
    segments: List[Union[Scene, AudioSegment, OCRSegment]] = Field(default_factory=list)
    
    @validator("duration", pre=True, always=True)
    def calculate_duration(cls, v, values):
        """Calculate duration from start and end times."""
        if "start_time" in values and "end_time" in values:
            return values["end_time"] - values["start_time"]
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Project(BaseModel):
    """VidMap project containing videos and analysis."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    videos: List[Video] = Field(default_factory=list)
    chapters: List[Chapter] = Field(default_factory=list)
    speakers: List[Speaker] = Field(default_factory=list)
    topics: List[Topic] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchResult(BaseModel):
    """Search result with relevance scoring."""
    segment: Union[Scene, AudioSegment, OCRSegment]
    relevance_score: float = Field(..., description="Search relevance score")
    matched_text: Optional[str] = Field(None, description="Matched text for text search")
    matched_region: Optional[List[float]] = Field(None, description="Matched region coordinates")
    highlight_times: List[float] = Field(default_factory=list, description="Highlight timestamps")


class MapTile(BaseModel):
    """Map tile for zoomable interface."""
    level: int = Field(..., description="Zoom level")
    x: int = Field(..., description="Tile X coordinate")
    y: int = Field(..., description="Tile Y coordinate")
    segments: List[Union[Scene, AudioSegment, OCRSegment]] = Field(default_factory=list)
    thumbnail_path: Optional[str] = Field(None, description="Path to tile thumbnail")
    metadata: Dict[str, Any] = Field(default_factory=dict)
