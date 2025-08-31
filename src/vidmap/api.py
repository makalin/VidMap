"""FastAPI web server for VidMap."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Path as APIPath, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from .config import Config
from .core import VidMap
from .models import Project, Video, Scene, AudioSegment, OCRSegment, Chapter, Speaker, Topic


logger = logging.getLogger(__name__)


class VidMapAPI:
    """FastAPI web server for VidMap."""
    
    def __init__(self, config: Config):
        """Initialize the API server."""
        self.config = config
        self.app = FastAPI(
            title="VidMap API",
            description="Map-like indexing for long videos with AI-powered scene analysis",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Initialize VidMap instance
        self.vidmap = None
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize VidMap on startup."""
            self.vidmap = VidMap(self.config)
            logger.info("VidMap API server started")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown."""
            if self.vidmap:
                await self.vidmap.close()
            logger.info("VidMap API server stopped")
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        # Projects
        @self.app.get("/api/projects", response_model=List[Dict[str, Any]])
        async def list_projects():
            """List all projects."""
            try:
                projects = await self.vidmap.storage.list_projects()
                return [{"name": name} for name in projects]
            except Exception as e:
                logger.error(f"Failed to list projects: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/projects/{project_name}", response_model=Dict[str, Any])
        async def get_project(project_name: str = APIPath(..., description="Project name")):
            """Get project details."""
            try:
                project = await self.vidmap.storage.get_project(project_name)
                if not project:
                    raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")
                
                return {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "created_at": project.created_at.isoformat(),
                    "updated_at": project.updated_at.isoformat(),
                    "videos": [
                        {
                            "id": video.id,
                            "title": video.title,
                            "duration": video.duration,
                            "width": video.width,
                            "height": video.height,
                            "fps": video.fps
                        }
                        for video in project.videos
                    ],
                    "chapters": [
                        {
                            "id": chapter.id,
                            "title": chapter.title,
                            "start_time": chapter.start_time,
                            "end_time": chapter.end_time,
                            "duration": chapter.duration
                        }
                        for chapter in project.chapters
                    ],
                    "speakers": [
                        {
                            "id": speaker.id,
                            "name": speaker.name,
                            "confidence": speaker.confidence
                        }
                        for speaker in project.speakers
                    ],
                    "topics": [
                        {
                            "id": topic.id,
                            "name": topic.name,
                            "description": topic.description,
                            "start_time": topic.start_time,
                            "end_time": topic.end_time
                        }
                        for topic in project.topics
                    ]
                }
            except Exception as e:
                logger.error(f"Failed to get project {project_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/projects")
        async def create_project(
            name: str = Query(..., description="Project name"),
            description: Optional[str] = Query(None, description="Project description")
        ):
            """Create a new project."""
            try:
                project = await self.vidmap.create_project(name, description)
                return {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "created_at": project.created_at.isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to create project {name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Videos
        @self.app.post("/api/projects/{project_name}/videos")
        async def ingest_video(
            project_name: str = APIPath(..., description="Project name"),
            video_path: str = Query(..., description="Path to video file"),
            title: Optional[str] = Query(None, description="Video title")
        ):
            """Ingest a video into a project."""
            try:
                video = await self.vidmap.ingest_video(
                    video_path=video_path,
                    project_name=project_name,
                    title=title
                )
                
                return {
                    "id": video.id,
                    "title": video.title,
                    "duration": video.duration,
                    "width": video.width,
                    "height": video.height,
                    "fps": video.fps
                }
            except Exception as e:
                logger.error(f"Failed to ingest video: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Analysis
        @self.app.post("/api/projects/{project_name}/analyze")
        async def analyze_video(
            project_name: str = APIPath(..., description="Project name"),
            video_id: str = Query(..., description="Video ID to analyze"),
            enable_scenes: bool = Query(True, description="Enable scene detection"),
            enable_asr: bool = Query(True, description="Enable audio transcription"),
            enable_diarization: bool = Query(True, description="Enable speaker diarization"),
            enable_ocr: bool = Query(True, description="Enable OCR text extraction"),
            enable_vision: bool = Query(True, description="Enable visual analysis")
        ):
            """Analyze a video with specified features."""
            try:
                results = await self.vidmap.analyze_video(
                    project_name=project_name,
                    video_id=video_id,
                    enable_scenes=enable_scenes,
                    enable_asr=enable_asr,
                    enable_diarization=enable_diarization,
                    enable_ocr=enable_ocr,
                    enable_vision=enable_vision
                )
                
                return {
                    "status": "success",
                    "results": results
                }
            except Exception as e:
                logger.error(f"Failed to analyze video: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Indexing
        @self.app.post("/api/projects/{project_name}/build")
        async def build_index(
            project_name: str = APIPath(..., description="Project name"),
            video_id: str = Query(..., description="Video ID to index"),
            enable_embeddings: bool = Query(True, description="Generate embeddings"),
            tile_size: int = Query(256, description="Tile size in pixels")
        ):
            """Build search index and generate map tiles."""
            try:
                results = await self.vidmap.build_index(
                    project_name=project_name,
                    video_id=video_id,
                    enable_embeddings=enable_embeddings,
                    tile_size=tile_size
                )
                
                return {
                    "status": "success",
                    "results": results
                }
            except Exception as e:
                logger.error(f"Failed to build index: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Search
        @self.app.get("/api/projects/{project_name}/search")
        async def search_content(
            project_name: str = APIPath(..., description="Project name"),
            q: str = Query(..., description="Search query"),
            video_id: Optional[str] = Query(None, description="Filter by video ID"),
            limit: int = Query(50, description="Maximum number of results")
        ):
            """Search for content in a project."""
            try:
                results = await self.vidmap.search(
                    project_name=project_name,
                    query=q,
                    video_id=video_id,
                    limit=limit
                )
                
                # Convert results to serializable format
                serializable_results = []
                for result in results:
                    segment = result["segment"]
                    serializable_results.append({
                        "segment_id": segment.id,
                        "type": getattr(segment, 'type', 'unknown'),
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "relevance_score": result["relevance_score"],
                        "matched_text": result.get("matched_text", ""),
                        "search_type": result.get("search_type", "unknown")
                    })
                
                return {
                    "query": q,
                    "total_results": len(serializable_results),
                    "results": serializable_results
                }
            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Map tiles
        @self.app.get("/api/projects/{project_name}/map")
        async def get_map_data(
            project_name: str = APIPath(..., description="Project name"),
            video_id: str = Query(..., description="Video ID"),
            level: int = Query(0, description="Zoom level"),
            x: int = Query(0, description="Tile X coordinate"),
            y: int = Query(0, description="Tile Y coordinate")
        ):
            """Get map tile data for the zoomable interface."""
            try:
                tile_data = await self.vidmap.get_map_data(
                    project_name=project_name,
                    video_id=video_id,
                    level=level,
                    x=x,
                    y=y
                )
                
                return tile_data
            except Exception as e:
                logger.error(f"Failed to get map data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Export
        @self.app.get("/api/projects/{project_name}/export")
        async def export_project(
            project_name: str = APIPath(..., description="Project name"),
            format: str = Query("json", description="Export format"),
            output_path: Optional[str] = Query(None, description="Output file path")
        ):
            """Export project data in various formats."""
            try:
                output_path = await self.vidmap.export_project(
                    project_name=project_name,
                    format=format,
                    output_path=output_path
                )
                
                # Return file if it's a local path
                if Path(output_path).exists():
                    return FileResponse(
                        output_path,
                        media_type="application/octet-stream",
                        filename=f"{project_name}.{format}"
                    )
                else:
                    return {"status": "success", "output_path": str(output_path)}
                    
            except Exception as e:
                logger.error(f"Export failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Static files (for web UI)
        try:
            static_dir = Path(self.config.storage.blob_path) / "static"
            if static_dir.exists():
                self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        except Exception as e:
            logger.warning(f"Could not mount static files: {e}")
        
        # Serve tile images
        @self.app.get("/tiles/{level}/{tile_id}")
        async def get_tile_image(
            level: str = APIPath(..., description="Zoom level"),
            tile_id: str = APIPath(..., description="Tile ID")
        ):
            """Serve map tile images."""
            try:
                tile_path = Path(self.config.storage.blob_path) / "tiles" / level / f"tile_{tile_id}.png"
                if tile_path.exists():
                    return FileResponse(tile_path, media_type="image/png")
                else:
                    raise HTTPException(status_code=404, detail="Tile not found")
            except Exception as e:
                logger.error(f"Failed to serve tile: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Serve thumbnails
        @self.app.get("/thumbnails/{video_id}")
        async def get_video_thumbnail(
            video_id: str = APIPath(..., description="Video ID")
        ):
            """Serve video thumbnails."""
            try:
                thumb_path = Path(self.config.storage.blob_path) / "thumbnails" / f"{video_id}.jpg"
                if thumb_path.exists():
                    return FileResponse(thumb_path, media_type="image/jpeg")
                else:
                    raise HTTPException(status_code=404, detail="Thumbnail not found")
            except Exception as e:
                logger.error(f"Failed to serve thumbnail: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = None, port: int = None):
        """Run the FastAPI server."""
        host = host or self.config.ui.host
        port = port or self.config.ui.port
        
        logger.info(f"Starting VidMap API server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


def create_api_server(config: Config) -> VidMapAPI:
    """Create a VidMap API server instance."""
    return VidMapAPI(config)


def run_api_server(config_path: Union[str, Path] = None, host: str = None, port: int = None):
    """Run the VidMap API server."""
    if config_path:
        config = Config.from_file(config_path)
    else:
        config = Config.create_default("default")
    
    api_server = create_api_server(config)
    api_server.run(host=host, port=port)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = None
    
    run_api_server(config_path)
