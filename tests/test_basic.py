"""Basic tests for VidMap."""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from vidmap.config import Config
from vidmap.models import Video, Project, Scene, AudioSegment, OCRSegment


class TestConfig:
    """Test configuration management."""
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        config = Config.create_default("test_project")
        assert config.project == "test_project"
        assert config.storage.blob_path == ".vidmap/test_project"
        assert config.asr.engine == "whisper"
        assert config.vision.enabled is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config.create_default("test_project")
        assert config.tiles.size == 256
        assert config.ui.port == 5173
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = Config.create_default("test_project")
        config_dict = config.dict()
        assert config_dict["project"] == "test_project"
        assert "storage" in config_dict
        assert "asr" in config_dict


class TestModels:
    """Test data models."""
    
    def test_video_creation(self):
        """Test video model creation."""
        video = Video(
            title="Test Video",
            source_path="/path/to/video.mp4",
            duration=120.5,
            fps=30.0,
            width=1920,
            height=1080,
            size_bytes=1024 * 1024 * 100
        )
        
        assert video.title == "Test Video"
        assert video.duration == 120.5
        assert video.width == 1920
        assert video.height == 1080
    
    def test_scene_creation(self):
        """Test scene model creation."""
        scene = Scene(
            video_id="test_video_id",
            start_time=10.0,
            end_time=25.0,
            confidence=0.8,
            shot_type="medium",
            visual_tags=["bright", "colorful"],
            dominant_colors=["blue", "green"]
        )
        
        assert scene.type == "scene"
        assert scene.duration == 15.0
        assert scene.shot_type == "medium"
        assert "bright" in scene.visual_tags
    
    def test_audio_segment_creation(self):
        """Test audio segment model creation."""
        audio_seg = AudioSegment(
            video_id="test_video_id",
            start_time=30.0,
            end_time=45.0,
            confidence=0.9,
            text="This is a test transcription.",
            language="en"
        )
        
        assert audio_seg.type == "audio"
        assert audio_seg.duration == 15.0
        assert audio_seg.text == "This is a test transcription."
        assert audio_seg.language == "en"
    
    def test_ocr_segment_creation(self):
        """Test OCR segment model creation."""
        ocr_seg = OCRSegment(
            video_id="test_video_id",
            start_time=60.0,
            end_time=75.0,
            confidence=0.7,
            text="Slide Title",
            language="en",
            bounding_box=[100, 50, 300, 100],
            is_slide=True
        )
        
        assert ocr_seg.type == "ocr"
        assert ocr_seg.duration == 15.0
        assert ocr_seg.text == "Slide Title"
        assert ocr_seg.is_slide is True
    
    def test_project_creation(self):
        """Test project model creation."""
        project = Project(
            name="Test Project",
            description="A test project for VidMap"
        )
        
        assert project.name == "Test Project"
        assert project.description == "A test project for VidMap"
        assert len(project.videos) == 0
        assert len(project.chapters) == 0


class TestVideoIngester:
    """Test video ingestion functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return Config.create_default("test_project")
    
    def test_video_validation(self, temp_dir):
        """Test video file validation."""
        # This would test the video validation logic
        # For now, just test that the directory exists
        assert temp_dir.exists()
        assert temp_dir.is_dir()


class TestVideoAnalyzer:
    """Test video analysis functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return Config.create_default("test_project")
    
    def test_analyzer_initialization(self, sample_config):
        """Test analyzer initialization."""
        # This would test the analyzer setup
        # For now, just test that config is valid
        assert sample_config.vision.enabled is True
        assert sample_config.asr.engine == "whisper"


class TestVideoIndexer:
    """Test video indexing functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return Config.create_default("test_project")
    
    def test_indexer_initialization(self, sample_config):
        """Test indexer initialization."""
        # This would test the indexer setup
        # For now, just test that config is valid
        assert sample_config.embeddings.model == "all-MiniLM-L6-v2"
        assert sample_config.tiles.size == 256


class TestVideoSearcher:
    """Test video search functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return Config.create_default("test_project")
    
    def test_searcher_initialization(self, sample_config):
        """Test searcher initialization."""
        # This would test the searcher setup
        # For now, just test that config is valid
        assert sample_config.project == "test_project"


class TestProjectStorage:
    """Test project storage functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create sample configuration for testing."""
        config = Config.create_default("test_project")
        config.storage.blob_path = str(temp_dir)
        return config
    
    def test_storage_initialization(self, sample_config):
        """Test storage initialization."""
        # This would test the storage setup
        # For now, just test that config is valid
        assert sample_config.storage.blob_path is not None


# Integration tests
class TestIntegration:
    """Integration tests for VidMap."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create sample configuration for testing."""
        config = Config.create_default("test_project")
        config.storage.blob_path = str(temp_dir)
        return config
    
    def test_config_workflow(self, sample_config):
        """Test basic configuration workflow."""
        # Test that we can create and modify configuration
        assert sample_config.project == "test_project"
        
        # Test that we can access nested configurations
        assert sample_config.asr.engine == "whisper"
        assert sample_config.vision.scene_detection is True
        assert sample_config.tiles.levels == 5
    
    def test_model_relationships(self):
        """Test model relationships and validation."""
        # Test that models can be created with proper relationships
        video = Video(
            title="Test Video",
            source_path="/path/to/video.mp4",
            duration=120.0,
            fps=30.0,
            width=1920,
            height=1080,
            size_bytes=1024 * 1024 * 100
        )
        
        scene = Scene(
            video_id=video.id,
            start_time=0.0,
            end_time=30.0,
            confidence=0.8
        )
        
        # Test that scene references video correctly
        assert scene.video_id == video.id
        assert scene.duration == 30.0


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
