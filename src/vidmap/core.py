"""Core VidMap class for video analysis and indexing."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from .config import Config
from .models import Video, Project, Scene, AudioSegment, OCRSegment, Chapter, Speaker, Topic
from .ingest import VideoIngester
from .analysis import VideoAnalyzer
from .indexing import VideoIndexer
from .storage import ProjectStorage
from .search import VideoSearcher


logger = logging.getLogger(__name__)


class VidMap:
    """Main VidMap class for video analysis and indexing."""
    
    def __init__(self, config: Config):
        """Initialize VidMap with configuration."""
        self.config = config
        self.storage = ProjectStorage(config)
        self.ingester = VideoIngester(config)
        self.analyzer = VideoAnalyzer(config)
        self.indexer = VideoIndexer(config)
        self.searcher = VideoSearcher(config)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Ensure storage directories exist
        self._setup_storage()
    
    def _setup_storage(self):
        """Create necessary storage directories."""
        storage_path = Path(self.config.storage.blob_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (storage_path / "thumbnails").mkdir(exist_ok=True)
        (storage_path / "tiles").mkdir(exist_ok=True)
        (storage_path / "audio").mkdir(exist_ok=True)
        (storage_path / "transcripts").mkdir(exist_ok=True)
    
    async def create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Create a new VidMap project."""
        project = Project(
            name=name,
            description=description,
            config=self.config.dict()
        )
        
        await self.storage.save_project(project)
        logger.info(f"Created project: {name}")
        return project
    
    async def ingest_video(
        self,
        video_path: Union[str, Path],
        project_name: str,
        title: Optional[str] = None,
        **kwargs
    ) -> Video:
        """Ingest a video into a project."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get or create project
        project = await self.storage.get_project(project_name)
        if not project:
            project = await self.create_project(project_name)
        
        # Ingest video
        video = await self.ingester.ingest_video(video_path, title=title, **kwargs)
        
        # Add to project
        project.videos.append(video)
        await self.storage.save_project(project)
        
        logger.info(f"Ingested video: {video.title} into project: {project_name}")
        return video
    
    async def analyze_video(
        self,
        project_name: str,
        video_id: str,
        enable_scenes: bool = True,
        enable_asr: bool = True,
        enable_diarization: bool = True,
        enable_ocr: bool = True,
        enable_vision: bool = True
    ) -> Dict[str, Any]:
        """Analyze a video with specified features."""
        project = await self.storage.get_project(project_name)
        if not project:
            raise ValueError(f"Project not found: {project_name}")
        
        video = next((v for v in project.videos if v.id == video_id), None)
        if not video:
            raise ValueError(f"Video not found: {video_id}")
        
        results = {}
        
        # Scene detection
        if enable_scenes and self.config.vision.scene_detection:
            logger.info("Detecting scenes...")
            scenes = await self.analyzer.detect_scenes(video)
            results["scenes"] = scenes
            project.chapters.extend(self._scenes_to_chapters(scenes))
        
        # Audio transcription
        if enable_asr:
            logger.info("Transcribing audio...")
            audio_segments = await self.analyzer.transcribe_audio(video)
            results["audio_segments"] = audio_segments
        
        # Speaker diarization
        if enable_diarization and self.config.diarization.enabled:
            logger.info("Performing speaker diarization...")
            speakers = await self.analyzer.diarize_speakers(video, audio_segments)
            results["speakers"] = speakers
            project.speakers.extend(speakers)
        
        # OCR processing
        if enable_ocr and self.config.ocr.enabled:
            logger.info("Extracting text with OCR...")
            ocr_segments = await self.analyzer.extract_text(video)
            results["ocr_segments"] = ocr_segments
        
        # Vision analysis
        if enable_vision and self.config.vision.enabled:
            logger.info("Analyzing visual content...")
            vision_tags = await self.analyzer.analyze_vision(video)
            results["vision_tags"] = vision_tags
        
        # Update project
        await self.storage.save_project(project)
        
        logger.info(f"Analysis completed for video: {video.title}")
        return results
    
    async def build_index(
        self,
        project_name: str,
        video_id: str,
        enable_embeddings: bool = True,
        tile_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Build search index and generate map tiles."""
        project = await self.storage.get_project(project_name)
        if not project:
            raise ValueError(f"Project not found: {project_name}")
        
        video = next((v for v in project.videos if v.id == video_id), None)
        if not video:
            raise ValueError(f"Video not found: {video_id}")
        
        results = {}
        
        # Generate embeddings
        if enable_embeddings:
            logger.info("Generating embeddings...")
            embeddings = await self.indexer.generate_embeddings(video, project)
            results["embeddings"] = embeddings
        
        # Generate map tiles
        tile_size = tile_size or self.config.tiles.size
        logger.info(f"Generating map tiles (size: {tile_size})...")
        tiles = await self.indexer.generate_tiles(video, project, tile_size)
        results["tiles"] = tiles
        
        # Build search index
        logger.info("Building search index...")
        search_index = await self.indexer.build_search_index(video, project)
        results["search_index"] = search_index
        
        # Update project
        await self.storage.save_project(project)
        
        logger.info(f"Index built for video: {video.title}")
        return results
    
    async def search(
        self,
        project_name: str,
        query: str,
        video_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search for content in a project."""
        project = await self.storage.get_project(project_name)
        if not project:
            raise ValueError(f"Project not found: {project_name}")
        
        results = await self.searcher.search(
            project, query, video_id=video_id, filters=filters, limit=limit
        )
        
        return results
    
    async def get_map_data(
        self,
        project_name: str,
        video_id: str,
        level: int = 0,
        x: int = 0,
        y: int = 0
    ) -> Dict[str, Any]:
        """Get map tile data for the zoomable interface."""
        project = await self.storage.get_project(project_name)
        if not project:
            raise ValueError(f"Project not found: {project_name}")
        
        video = next((v for v in project.videos if v.id == video_id), None)
        if not video:
            raise ValueError(f"Video not found: {video_id}")
        
        tile_data = await self.indexer.get_tile_data(video, project, level, x, y)
        return tile_data
    
    async def export_project(
        self,
        project_name: str,
        format: str = "json",
        output_path: Optional[Union[str, Path]] = None
    ) -> Union[str, Path]:
        """Export project data in various formats."""
        project = await self.storage.get_project(project_name)
        if not project:
            raise ValueError(f"Project not found: {project_name}")
        
        if format == "json":
            return await self._export_json(project, output_path)
        elif format == "csv":
            return await self._export_csv(project, output_path)
        elif format == "edl":
            return await self._export_edl(project, output_path)
        elif format == "srt":
            return await self._export_srt(project, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _scenes_to_chapters(self, scenes: List[Scene]) -> List[Chapter]:
        """Convert detected scenes to chapters."""
        chapters = []
        for i, scene in enumerate(scenes):
            chapter = Chapter(
                title=f"Scene {i+1}",
                start_time=scene.start_time,
                end_time=scene.end_time,
                segments=[scene]
            )
            chapters.append(chapter)
        return chapters
    
    async def _export_json(self, project: Project, output_path: Optional[Union[str, Path]]) -> Union[str, Path]:
        """Export project as JSON."""
        import json
        
        if not output_path:
            output_path = Path(f"{project.name}.json")
        
        output_path = Path(output_path)
        
        with open(output_path, "w") as f:
            json.dump(project.dict(), f, indent=2, default=str)
        
        return output_path
    
    async def _export_csv(self, project: Project, output_path: Optional[Union[str, Path]]) -> Union[str, Path]:
        """Export project as CSV."""
        import csv
        
        if not output_path:
            output_path = Path(f"{project.name}.csv")
        
        output_path = Path(output_path)
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Type", "Start Time", "End Time", "Duration", "Content"])
            
            for video in project.videos:
                for chapter in project.chapters:
                    if chapter.segments:
                        writer.writerow([
                            "Chapter",
                            chapter.start_time,
                            chapter.end_time,
                            chapter.duration,
                            chapter.title
                        ])
        
        return output_path
    
    async def _export_edl(self, project: Project, output_path: Optional[Union[str, Path]]) -> Union[str, Path]:
        """Export project as EDL (Edit Decision List)."""
        if not output_path:
            output_path = Path(f"{project.name}.edl")
        
        output_path = Path(output_path)
        
        with open(output_path, "w") as f:
            f.write("TITLE: VidMap Export\n")
            f.write("FCM: NON-DROP FRAME\n\n")
            
            for i, chapter in enumerate(project.chapters, 1):
                f.write(f"{i:03d}  V     C        {chapter.start_time:08.2f} {chapter.end_time:08.2f} {chapter.start_time:08.2f} {chapter.end_time:08.2f}\n")
                f.write(f"* FROM CLIP NAME: {chapter.title}\n\n")
        
        return output_path
    
    async def _export_srt(self, project: Project, output_path: Optional[Union[str, Path]]) -> Union[str, Path]:
        """Export project as SRT subtitles."""
        if not output_path:
            output_path = Path(f"{project.name}.srt")
        
        output_path = Path(output_path)
        
        with open(output_path, "w") as f:
            for i, chapter in enumerate(project.chapters, 1):
                start_time = self._seconds_to_srt_time(chapter.start_time)
                end_time = self._seconds_to_srt_time(chapter.end_time)
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{chapter.title}\n\n")
        
        return output_path
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    async def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        await self.storage.close()
