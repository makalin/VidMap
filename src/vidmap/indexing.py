"""Video indexing and tile generation for the zoomable map interface."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import Config
from .models import Video, Project, Scene, AudioSegment, OCRSegment, MapTile


logger = logging.getLogger(__name__)


class VideoIndexer:
    """Handles video indexing, embedding generation, and map tile creation."""
    
    def __init__(self, config: Config):
        """Initialize the video indexer."""
        self.config = config
        self._setup_models()
    
    def _setup_models(self):
        """Setup AI models for indexing."""
        try:
            # Setup sentence transformer for embeddings
            if self.config.embeddings.model:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.config.embeddings.model)
            else:
                self.embedding_model = None
                
            # Setup FAISS for vector search
            try:
                import faiss
                self.faiss_index = None
                self.faiss_available = True
            except ImportError:
                self.faiss_available = False
                logger.warning("FAISS not available, vector search will be disabled")
                
        except ImportError as e:
            logger.warning(f"Some indexing features may not be available: {e}")
            self.embedding_model = None
            self.faiss_available = False
    
    async def generate_embeddings(
        self,
        video: Video,
        project: Project
    ) -> Dict[str, Any]:
        """Generate embeddings for video content."""
        if not self.embedding_model:
            logger.warning("Embedding model not available, skipping embedding generation")
            return {}
        
        try:
            embeddings = {
                "scenes": [],
                "audio_segments": [],
                "ocr_segments": [],
                "topics": []
            }
            
            # Generate embeddings for scenes
            for chapter in project.chapters:
                for segment in chapter.segments:
                    if hasattr(segment, 'type'):
                        if segment.type == "scene":
                            embedding = await self._generate_scene_embedding(segment)
                            embeddings["scenes"].append({
                                "segment_id": segment.id,
                                "embedding": embedding
                            })
                        elif segment.type == "audio":
                            embedding = await self._generate_text_embedding(segment.text)
                            embeddings["audio_segments"].append({
                                "segment_id": segment.id,
                                "embedding": embedding
                            })
                        elif segment.type == "ocr":
                            embedding = await self._generate_text_embedding(segment.text)
                            embeddings["ocr_segments"].append({
                                "segment_id": segment.id,
                                "embedding": embedding
                            })
            
            # Generate embeddings for topics
            for topic in project.topics:
                topic_text = f"{topic.name} {topic.description or ''} {' '.join(topic.keywords)}"
                embedding = await self._generate_text_embedding(topic_text)
                embeddings["topics"].append({
                    "topic_id": topic.id,
                    "embedding": embedding
                })
            
            logger.info(f"Generated {len(embeddings['scenes'])} scene embeddings")
            logger.info(f"Generated {len(embeddings['audio_segments'])} audio embeddings")
            logger.info(f"Generated {len(embeddings['ocr_segments'])} OCR embeddings")
            logger.info(f"Generated {len(embeddings['topics'])} topic embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return {}
    
    async def _generate_scene_embedding(self, scene: Scene) -> List[float]:
        """Generate embedding for a scene based on visual and metadata features."""
        try:
            # Create a text representation of the scene
            scene_text = f"scene {scene.shot_type or 'unknown'} {' '.join(scene.visual_tags)} {' '.join(scene.dominant_colors)}"
            
            if scene.brightness:
                scene_text += f" brightness:{scene.brightness:.2f}"
            if scene.motion_score:
                scene_text += f" motion:{scene.motion_score:.2f}"
            
            return await self._generate_text_embedding(scene_text)
            
        except Exception as e:
            logger.debug(f"Scene embedding generation failed: {e}")
            return [0.0] * self.config.embeddings.dimension
    
    async def _generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text content."""
        try:
            if not text.strip():
                return [0.0] * self.config.embeddings.dimension
            
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            
            # Convert to list and ensure correct dimension
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
            # Pad or truncate to expected dimension
            if len(embedding_list) < self.config.embeddings.dimension:
                embedding_list.extend([0.0] * (self.config.embeddings.dimension - len(embedding_list)))
            elif len(embedding_list) > self.config.embeddings.dimension:
                embedding_list = embedding_list[:self.config.embeddings.dimension]
            
            return embedding_list
            
        except Exception as e:
            logger.debug(f"Text embedding generation failed: {e}")
            return [0.0] * self.config.embeddings.dimension
    
    async def generate_tiles(
        self,
        video: Video,
        project: Project,
        tile_size: int
    ) -> List[MapTile]:
        """Generate map tiles for the zoomable interface."""
        try:
            tiles = []
            duration = video.duration
            
            # Generate tiles at different zoom levels
            for level in range(self.config.tiles.levels):
                # Calculate tile parameters for this level
                tiles_per_level = 2 ** level
                tile_duration = duration / tiles_per_level
                
                for tile_idx in range(tiles_per_level):
                    start_time = tile_idx * tile_duration
                    end_time = (tile_idx + 1) * tile_duration
                    
                    # Find segments in this tile
                    tile_segments = self._find_segments_in_timerange(
                        project, start_time, end_time
                    )
                    
                    # Generate tile thumbnail
                    thumbnail_path = await self._generate_tile_thumbnail(
                        video, start_time, end_time, level, tile_idx, tile_size
                    )
                    
                    # Create tile
                    tile = MapTile(
                        level=level,
                        x=tile_idx,
                        y=0,  # Single row for now
                        segments=tile_segments,
                        thumbnail_path=thumbnail_path,
                        metadata={
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": tile_duration,
                            "segment_count": len(tile_segments)
                        }
                    )
                    
                    tiles.append(tile)
            
            logger.info(f"Generated {len(tiles)} map tiles")
            return tiles
            
        except Exception as e:
            logger.error(f"Tile generation failed: {e}")
            return []
    
    def _find_segments_in_timerange(
        self,
        project: Project,
        start_time: float,
        end_time: float
    ) -> List:
        """Find all segments that fall within a time range."""
        segments = []
        
        for chapter in project.chapters:
            for segment in chapter.segments:
                # Check if segment overlaps with time range
                if (segment.start_time < end_time and segment.end_time > start_time):
                    segments.append(segment)
        
        return segments
    
    async def _generate_tile_thumbnail(
        self,
        video: Video,
        start_time: float,
        end_time: float,
        level: int,
        tile_idx: int,
        tile_size: int
    ) -> Optional[str]:
        """Generate a thumbnail for a map tile."""
        try:
            # Create tile directory
            tiles_dir = Path(self.config.storage.blob_path) / "tiles" / str(level)
            tiles_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate thumbnail path
            thumbnail_path = tiles_dir / f"tile_{tile_idx}.png"
            
            # Create a simple tile visualization
            await self._create_tile_visualization(
                video, start_time, end_time, tile_size, thumbnail_path
            )
            
            return str(thumbnail_path)
            
        except Exception as e:
            logger.debug(f"Tile thumbnail generation failed: {e}")
            return None
    
    async def _create_tile_visualization(
        self,
        video: Video,
        start_time: float,
        end_time: float,
        tile_size: int,
        output_path: Path
    ):
        """Create a visual representation of a tile."""
        try:
            # Create a blank image
            img = Image.new('RGB', (tile_size, tile_size), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Draw tile information
            duration = end_time - start_time
            draw.text((10, 10), f"Time: {start_time:.1f}s - {end_time:.1f}s", fill='black', font=font)
            draw.text((10, 30), f"Duration: {duration:.1f}s", fill='black', font=font)
            
            # Draw a simple timeline representation
            timeline_y = tile_size - 40
            draw.line([(20, timeline_y), (tile_size - 20, timeline_y)], fill='blue', width=2)
            
            # Add some visual elements based on content type
            # This is a placeholder - in a real implementation you'd analyze the actual content
            draw.rectangle([(20, 60), (tile_size - 20, 80)], outline='green', width=2)
            draw.text((25, 65), "Content Preview", fill='green', font=font)
            
            # Save the image
            img.save(output_path)
            
        except Exception as e:
            logger.debug(f"Tile visualization creation failed: {e}")
            # Create a simple fallback image
            img = Image.new('RGB', (tile_size, tile_size), color='gray')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Tile", fill='white')
            img.save(output_path)
    
    async def build_search_index(
        self,
        video: Video,
        project: Project
    ) -> Dict[str, Any]:
        """Build a search index for the video content."""
        try:
            search_index = {
                "text_index": {},
                "vector_index": None,
                "metadata_index": {}
            }
            
            # Build text index for full-text search
            text_index = {}
            
            for chapter in project.chapters:
                for segment in chapter.segments:
                    if hasattr(segment, 'type'):
                        if segment.type == "audio" and segment.text:
                            # Index audio transcriptions
                            words = segment.text.lower().split()
                            for word in words:
                                if word not in text_index:
                                    text_index[word] = []
                                text_index[word].append({
                                    "segment_id": segment.id,
                                    "start_time": segment.start_time,
                                    "end_time": segment.end_time,
                                    "type": "audio"
                                })
                        
                        elif segment.type == "ocr" and segment.text:
                            # Index OCR text
                            words = segment.text.lower().split()
                            for word in words:
                                if word not in text_index:
                                    text_index[word] = []
                                text_index[word].append({
                                    "segment_id": segment.id,
                                    "start_time": segment.start_time,
                                    "end_time": segment.end_time,
                                    "type": "ocr"
                                })
            
            search_index["text_index"] = text_index
            
            # Build vector index if FAISS is available
            if self.faiss_available and self.embedding_model:
                vector_index = await self._build_vector_index(project)
                search_index["vector_index"] = vector_index
            
            # Build metadata index
            metadata_index = {
                "speakers": {},
                "topics": {},
                "visual_tags": {},
                "shot_types": {}
            }
            
            # Index speakers
            for speaker in project.speakers:
                if speaker.name:
                    metadata_index["speakers"][speaker.name.lower()] = speaker.id
            
            # Index topics
            for topic in project.topics:
                metadata_index["topics"][topic.name.lower()] = topic.id
                for keyword in topic.keywords:
                    if keyword.lower() not in metadata_index["topics"]:
                        metadata_index["topics"][keyword.lower()] = []
                    metadata_index["topics"][keyword.lower()].append(topic.id)
            
            # Index visual tags and shot types
            for chapter in project.chapters:
                for segment in chapter.segments:
                    if hasattr(segment, 'type') and segment.type == "scene":
                        # Index shot types
                        if segment.shot_type:
                            if segment.shot_type not in metadata_index["shot_types"]:
                                metadata_index["shot_types"][segment.shot_type] = []
                            metadata_index["shot_types"][segment.shot_type].append(segment.id)
                        
                        # Index visual tags
                        for tag in segment.visual_tags:
                            if tag not in metadata_index["visual_tags"]:
                                metadata_index["visual_tags"][tag] = []
                            metadata_index["visual_tags"][tag].append(segment.id)
            
            search_index["metadata_index"] = metadata_index
            
            logger.info("Search index built successfully")
            return search_index
            
        except Exception as e:
            logger.error(f"Search index building failed: {e}")
            return {}
    
    async def _build_vector_index(self, project: Project) -> Any:
        """Build a FAISS vector index for similarity search."""
        try:
            import faiss
            
            # Collect all embeddings
            embeddings = []
            segment_ids = []
            
            for chapter in project.chapters:
                for segment in chapter.segments:
                    if hasattr(segment, 'type'):
                        if segment.type == "audio" and segment.text:
                            embedding = await self._generate_text_embedding(segment.text)
                            embeddings.append(embedding)
                            segment_ids.append(segment.id)
                        elif segment.type == "ocr" and segment.text:
                            embedding = await self._generate_text_embedding(segment.text)
                            embeddings.append(embedding)
                            segment_ids.append(segment.id)
            
            if not embeddings:
                return None
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add vectors to index
            index.add(embeddings_array)
            
            # Store segment IDs for retrieval
            index.segment_ids = segment_ids
            
            logger.info(f"Built FAISS index with {len(embeddings)} vectors")
            return index
            
        except Exception as e:
            logger.error(f"Vector index building failed: {e}")
            return None
    
    async def get_tile_data(
        self,
        video: Video,
        project: Project,
        level: int,
        x: int,
        y: int
    ) -> Dict[str, Any]:
        """Get data for a specific map tile."""
        try:
            # Calculate time range for this tile
            tiles_per_level = 2 ** level
            tile_duration = video.duration / tiles_per_level
            start_time = x * tile_duration
            end_time = (x + 1) * tile_duration
            
            # Find segments in this tile
            segments = self._find_segments_in_timerange(project, start_time, end_time)
            
            # Get tile metadata
            tile_data = {
                "level": level,
                "x": x,
                "y": y,
                "start_time": start_time,
                "end_time": end_time,
                "duration": tile_duration,
                "segments": [],
                "summary": self._generate_tile_summary(segments)
            }
            
            # Add segment information
            for segment in segments:
                segment_info = {
                    "id": segment.id,
                    "type": segment.type,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "confidence": segment.confidence
                }
                
                if hasattr(segment, 'text') and segment.text:
                    segment_info["text"] = segment.text[:100] + "..." if len(segment.text) > 100 else segment.text
                
                if hasattr(segment, 'visual_tags') and segment.visual_tags:
                    segment_info["visual_tags"] = segment.visual_tags
                
                if hasattr(segment, 'shot_type') and segment.shot_type:
                    segment_info["shot_type"] = segment.shot_type
                
                tile_data["segments"].append(segment_info)
            
            return tile_data
            
        except Exception as e:
            logger.error(f"Failed to get tile data: {e}")
            return {
                "level": level,
                "x": x,
                "y": y,
                "error": str(e)
            }
    
    def _generate_tile_summary(self, segments: List) -> Dict[str, Any]:
        """Generate a summary of tile content."""
        summary = {
            "total_segments": len(segments),
            "types": {},
            "duration": 0.0,
            "keywords": []
        }
        
        for segment in segments:
            # Count types
            seg_type = getattr(segment, 'type', 'unknown')
            summary["types"][seg_type] = summary["types"].get(seg_type, 0) + 1
            
            # Calculate total duration
            summary["duration"] += segment.duration
            
            # Extract keywords from text
            if hasattr(segment, 'text') and segment.text:
                words = segment.text.lower().split()
                summary["keywords"].extend(words[:5])  # Top 5 words
        
        # Get most common keywords
        from collections import Counter
        if summary["keywords"]:
            word_counts = Counter(summary["keywords"])
            summary["keywords"] = [word for word, count in word_counts.most_common(10)]
        
        return summary
