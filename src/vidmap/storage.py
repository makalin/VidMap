"""Project storage and data persistence."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import sqlite3
from datetime import datetime

from .config import Config
from .models import Project, Video, Scene, AudioSegment, OCRSegment, Chapter, Speaker, Topic


logger = logging.getLogger(__name__)


class ProjectStorage:
    """Handles project data persistence and retrieval."""
    
    def __init__(self, config: Config):
        """Initialize the storage system."""
        self.config = config
        self.storage_path = Path(config.storage.blob_path)
        self.db_path = self.storage_path / "project.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create database connection
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                config TEXT NOT NULL
            )
        """)
        
        # Videos table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                title TEXT NOT NULL,
                source_path TEXT NOT NULL,
                duration REAL NOT NULL,
                fps REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """)
        
        # Scenes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scenes (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                confidence REAL NOT NULL,
                shot_type TEXT,
                visual_tags TEXT NOT NULL,
                dominant_colors TEXT NOT NULL,
                brightness REAL,
                motion_score REAL,
                thumbnail_path TEXT,
                keyframes TEXT NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        """)
        
        # Audio segments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_segments (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                confidence REAL NOT NULL,
                text TEXT NOT NULL,
                language TEXT NOT NULL,
                speaker_id TEXT,
                words TEXT NOT NULL,
                sentiment TEXT,
                energy REAL,
                silence BOOLEAN NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        """)
        
        # OCR segments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_segments (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                confidence REAL NOT NULL,
                text TEXT NOT NULL,
                language TEXT NOT NULL,
                bounding_box TEXT NOT NULL,
                font_size REAL,
                is_slide BOOLEAN NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        """)
        
        # Speakers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS speakers (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT,
                gender TEXT,
                confidence REAL NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """)
        
        # Topics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                confidence REAL NOT NULL,
                keywords TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """)
        
        # Chapters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chapters (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_project_id ON videos (project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scenes_video_id ON scenes (video_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audio_segments_video_id ON audio_segments (video_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ocr_segments_video_id ON ocr_segments (video_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_speakers_project_id ON speakers (project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_project_id ON topics (project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chapters_project_id ON chapters (project_id)")
        
        self.conn.commit()
    
    async def save_project(self, project: Project) -> None:
        """Save a project to storage."""
        try:
            cursor = self.conn.cursor()
            
            # Save project
            cursor.execute("""
                INSERT OR REPLACE INTO projects (id, name, description, created_at, updated_at, config)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                project.id,
                project.name,
                project.description,
                project.created_at.isoformat(),
                project.updated_at.isoformat(),
                json.dumps(project.config)
            ))
            
            # Save videos
            for video in project.videos:
                await self._save_video(cursor, video, project.id)
            
            # Save speakers
            for speaker in project.speakers:
                await self._save_speaker(cursor, speaker, project.id)
            
            # Save topics
            for topic in project.topics:
                await self._save_topic(cursor, topic, project.id)
            
            # Save chapters
            for chapter in project.chapters:
                await self._save_chapter(cursor, chapter, project.id)
            
            self.conn.commit()
            logger.info(f"Saved project: {project.name}")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to save project {project.name}: {e}")
            raise
    
    async def _save_video(self, cursor, video: Video, project_id: str) -> None:
        """Save a video to storage."""
        cursor.execute("""
            INSERT OR REPLACE INTO videos 
            (id, project_id, title, source_path, duration, fps, width, height, size_bytes, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video.id,
            project_id,
            video.title,
            video.source_path,
            video.duration,
            video.fps,
            video.width,
            video.height,
            video.size_bytes,
            video.created_at.isoformat(),
            json.dumps(video.metadata)
        ))
        
        # Save related segments
        for chapter in video.metadata.get("chapters", []):
            for segment in chapter.segments:
                if hasattr(segment, 'type'):
                    if segment.type == "scene":
                        await self._save_scene(cursor, segment, video.id)
                    elif segment.type == "audio":
                        await self._save_audio_segment(cursor, segment, video.id)
                    elif segment.type == "ocr":
                        await self._save_ocr_segment(cursor, segment, video.id)
    
    async def _save_scene(self, cursor, scene: Scene, video_id: str) -> None:
        """Save a scene to storage."""
        cursor.execute("""
            INSERT OR REPLACE INTO scenes 
            (id, video_id, start_time, end_time, duration, confidence, shot_type, visual_tags, 
             dominant_colors, brightness, motion_score, thumbnail_path, keyframes, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scene.id,
            video_id,
            scene.start_time,
            scene.end_time,
            scene.duration,
            scene.confidence,
            scene.shot_type,
            json.dumps(scene.visual_tags),
            json.dumps(scene.dominant_colors),
            scene.brightness,
            scene.motion_score,
            scene.thumbnail_path,
            json.dumps(scene.keyframes),
            json.dumps(scene.metadata)
        ))
    
    async def _save_audio_segment(self, cursor, segment: AudioSegment, video_id: str) -> None:
        """Save an audio segment to storage."""
        cursor.execute("""
            INSERT OR REPLACE INTO audio_segments 
            (id, video_id, start_time, end_time, duration, confidence, text, language, 
             speaker_id, words, sentiment, energy, silence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            segment.id,
            video_id,
            segment.start_time,
            segment.end_time,
            segment.duration,
            segment.confidence,
            segment.text,
            segment.language,
            segment.speaker_id,
            json.dumps(segment.words),
            segment.sentiment,
            segment.energy,
            segment.silence,
            json.dumps(segment.metadata)
        ))
    
    async def _save_ocr_segment(self, cursor, segment: OCRSegment, video_id: str) -> None:
        """Save an OCR segment to storage."""
        cursor.execute("""
            INSERT OR REPLACE INTO ocr_segments 
            (id, video_id, start_time, end_time, duration, confidence, text, language, 
             bounding_box, font_size, is_slide, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            segment.id,
            video_id,
            segment.start_time,
            segment.end_time,
            segment.duration,
            segment.confidence,
            segment.text,
            segment.language,
            json.dumps(segment.bounding_box),
            segment.font_size,
            segment.is_slide,
            json.dumps(segment.metadata)
        ))
    
    async def _save_speaker(self, cursor, speaker: Speaker, project_id: str) -> None:
        """Save a speaker to storage."""
        cursor.execute("""
            INSERT OR REPLACE INTO speakers (id, project_id, name, gender, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            speaker.id,
            project_id,
            speaker.name,
            speaker.gender,
            speaker.confidence,
            json.dumps(speaker.metadata)
        ))
    
    async def _save_topic(self, cursor, topic: Topic, project_id: str) -> None:
        """Save a topic to storage."""
        cursor.execute("""
            INSERT OR REPLACE INTO topics 
            (id, project_id, name, description, confidence, keywords, start_time, end_time, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            topic.id,
            project_id,
            topic.name,
            topic.description,
            topic.confidence,
            json.dumps(topic.keywords),
            topic.start_time,
            topic.end_time,
            json.dumps(topic.metadata)
        ))
    
    async def _save_chapter(self, cursor, chapter: Chapter, project_id: str) -> None:
        """Save a chapter to storage."""
        cursor.execute("""
            INSERT OR REPLACE INTO chapters 
            (id, project_id, title, description, start_time, end_time, duration, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chapter.id,
            project_id,
            chapter.title,
            chapter.description,
            chapter.start_time,
            chapter.end_time,
            chapter.duration,
            json.dumps(chapter.metadata)
        ))
    
    async def get_project(self, name: str) -> Optional[Project]:
        """Retrieve a project by name."""
        try:
            cursor = self.conn.cursor()
            
            # Get project
            cursor.execute("SELECT * FROM projects WHERE name = ?", (name,))
            project_row = cursor.fetchone()
            
            if not project_row:
                return None
            
            # Create project object
            project = Project(
                id=project_row["id"],
                name=project_row["name"],
                description=project_row["description"],
                created_at=datetime.fromisoformat(project_row["created_at"]),
                updated_at=datetime.fromisoformat(project_row["updated_at"]),
                config=json.loads(project_row["config"])
            )
            
            # Load videos
            project.videos = await self._load_videos(cursor, project.id)
            
            # Load speakers
            project.speakers = await self._load_speakers(cursor, project.id)
            
            # Load topics
            project.topics = await self._load_topics(cursor, project.id)
            
            # Load chapters
            project.chapters = await self._load_chapters(cursor, project.id)
            
            return project
            
        except Exception as e:
            logger.error(f"Failed to load project {name}: {e}")
            return None
    
    async def _load_videos(self, cursor, project_id: str) -> List[Video]:
        """Load videos for a project."""
        cursor.execute("SELECT * FROM videos WHERE project_id = ?", (project_id,))
        video_rows = cursor.fetchall()
        
        videos = []
        for row in video_rows:
            video = Video(
                id=row["id"],
                title=row["title"],
                source_path=row["source_path"],
                duration=row["duration"],
                fps=row["fps"],
                width=row["width"],
                height=row["height"],
                size_bytes=row["size_bytes"],
                created_at=datetime.fromisoformat(row["created_at"]),
                metadata=json.loads(row["metadata"])
            )
            videos.append(video)
        
        return videos
    
    async def _load_speakers(self, cursor, project_id: str) -> List[Speaker]:
        """Load speakers for a project."""
        cursor.execute("SELECT * FROM speakers WHERE project_id = ?", (project_id,))
        speaker_rows = cursor.fetchall()
        
        speakers = []
        for row in speaker_rows:
            speaker = Speaker(
                id=row["id"],
                name=row["name"],
                gender=row["gender"],
                confidence=row["confidence"],
                metadata=json.loads(row["metadata"])
            )
            speakers.append(speaker)
        
        return speakers
    
    async def _load_topics(self, cursor, project_id: str) -> List[Topic]:
        """Load topics for a project."""
        cursor.execute("SELECT * FROM topics WHERE project_id = ?", (project_id,))
        topic_rows = cursor.fetchall()
        
        topics = []
        for row in topic_rows:
            topic = Topic(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                confidence=row["confidence"],
                keywords=json.loads(row["keywords"]),
                start_time=row["start_time"],
                end_time=row["end_time"],
                metadata=json.loads(row["metadata"])
            )
            topics.append(topic)
        
        return topics
    
    async def _load_chapters(self, cursor, project_id: str) -> List[Chapter]:
        """Load chapters for a project."""
        cursor.execute("SELECT * FROM chapters WHERE project_id = ?", (project_id,))
        chapter_rows = cursor.fetchall()
        
        chapters = []
        for row in chapter_rows:
            chapter = Chapter(
                id=row["id"],
                title=row["title"],
                description=row["description"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                duration=row["duration"],
                metadata=json.loads(row["metadata"])
            )
            chapters.append(chapter)
        
        return chapters
    
    async def list_projects(self) -> List[str]:
        """List all project names."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM projects ORDER BY created_at DESC")
            return [row["name"] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []
    
    async def delete_project(self, name: str) -> bool:
        """Delete a project and all its data."""
        try:
            cursor = self.conn.cursor()
            
            # Get project ID
            cursor.execute("SELECT id FROM projects WHERE name = ?", (name,))
            project_row = cursor.fetchone()
            
            if not project_row:
                return False
            
            project_id = project_row["id"]
            
            # Delete all related data
            cursor.execute("DELETE FROM scenes WHERE video_id IN (SELECT id FROM videos WHERE project_id = ?)", (project_id,))
            cursor.execute("DELETE FROM audio_segments WHERE video_id IN (SELECT id FROM videos WHERE project_id = ?)", (project_id,))
            cursor.execute("DELETE FROM ocr_segments WHERE video_id IN (SELECT id FROM videos WHERE project_id = ?)", (project_id,))
            cursor.execute("DELETE FROM videos WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM speakers WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM topics WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM chapters WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            
            self.conn.commit()
            logger.info(f"Deleted project: {name}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to delete project {name}: {e}")
            return False
    
    async def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
