"""Command-line interface for VidMap."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config
from .core import VidMap


console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """VidMap - Map-like indexing for long videos with AI-powered scene analysis."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--project", "-p", required=True, help="Project name")
@click.option("--title", "-t", help="Video title (defaults to filename)")
@click.option("--description", "-d", help="Project description")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
def ingest(video_path: str, project: str, title: Optional[str], description: Optional[str], config: Optional[str]):
    """Ingest a video into a project."""
    asyncio.run(_ingest_video(video_path, project, title, description, config))


async def _ingest_video(video_path: str, project: str, title: Optional[str], description: Optional[str], config: Optional[str]):
    """Ingest video implementation."""
    try:
        # Load configuration
        if config:
            vidmap_config = Config.from_file(config)
        else:
            vidmap_config = Config.create_default(project, description=description)
        
        # Initialize VidMap
        vidmap = VidMap(vidmap_config)
        
        with console.status("[bold green]Ingesting video...", spinner="dots"):
            # Ingest video
            video = await vidmap.ingest_video(
                video_path=video_path,
                project_name=project,
                title=title
            )
        
        # Display results
        table = Table(title=f"Video Ingested: {video.title}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("ID", video.id)
        table.add_row("Title", video.title)
        table.add_row("Duration", f"{video.duration:.2f}s")
        table.add_row("Resolution", f"{video.width}x{video.height}")
        table.add_row("FPS", f"{video.fps:.2f}")
        table.add_row("Size", f"{video.size_bytes / (1024**3):.2f} GB")
        
        console.print(table)
        console.print(f"\n[green]✓[/green] Video '{video.title}' successfully ingested into project '{project}'")
        
        await vidmap.close()
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to ingest video: {e}")
        sys.exit(1)


@main.command()
@click.option("--project", "-p", required=True, help="Project name")
@click.option("--scenes", is_flag=True, help="Enable scene detection")
@click.option("--asr", is_flag=True, help="Enable audio transcription")
@click.option("--diar", is_flag=True, help="Enable speaker diarization")
@click.option("--ocr", is_flag=True, help="Enable OCR text extraction")
@click.option("--vision", is_flag=True, help="Enable visual analysis")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
def index(project: str, scenes: bool, asr: bool, diar: bool, ocr: bool, vision: bool, config: Optional[str]):
    """Analyze and index video content."""
    asyncio.run(_index_video(project, scenes, asr, diar, ocr, vision, config))


async def _index_video(project: str, scenes: bool, asr: bool, diar: bool, ocr: bool, vision: bool, config: Optional[str]):
    """Index video implementation."""
    try:
        # Load configuration
        if config:
            vidmap_config = Config.from_file(config)
        else:
            vidmap_config = Config.create_default(project)
        
        # Initialize VidMap
        vidmap = VidMap(vidmap_config)
        
        # Get project
        project_obj = await vidmap.storage.get_project(project)
        if not project_obj or not project_obj.videos:
            console.print(f"[red]✗[/red] No videos found in project '{project}'")
            await vidmap.close()
            return
        
        # Analyze each video
        for video in project_obj.videos:
            console.print(f"\n[bold blue]Analyzing video: {video.title}[/bold blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing...", total=None)
                
                # Run analysis
                results = await vidmap.analyze_video(
                    project_name=project,
                    video_id=video.id,
                    enable_scenes=scenes,
                    enable_asr=asr,
                    enable_diarization=diar,
                    enable_ocr=ocr,
                    enable_vision=vision
                )
                
                progress.update(task, description="Analysis completed!")
            
            # Display results summary
            _display_analysis_results(results)
        
        await vidmap.close()
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to index video: {e}")
        sys.exit(1)


def _display_analysis_results(results: dict):
    """Display analysis results in a table."""
    if not results:
        console.print("[yellow]No analysis results to display[/yellow]")
        return
    
    table = Table(title="Analysis Results")
    table.add_column("Feature", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Details", style="green")
    
    for feature, data in results.items():
        if isinstance(data, list):
            count = len(data)
            details = f"Processed {count} items"
        else:
            count = "N/A"
            details = str(data)
        
        table.add_row(feature.replace("_", " ").title(), str(count), details)
    
    console.print(table)


@main.command()
@click.option("--project", "-p", required=True, help="Project name")
@click.option("--embeddings", is_flag=True, help="Generate embeddings")
@click.option("--tile-size", type=int, default=256, help="Tile size in pixels")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
def build(project: str, embeddings: bool, tile_size: int, config: Optional[str]):
    """Build search index and generate map tiles."""
    asyncio.run(_build_index(project, embeddings, tile_size, config))


async def _build_index(project: str, embeddings: bool, tile_size: int, config: Optional[str]):
    """Build index implementation."""
    try:
        # Load configuration
        if config:
            vidmap_config = Config.from_file(config)
        else:
            vidmap_config = Config.create_default(project)
        
        # Initialize VidMap
        vidmap = VidMap(vidmap_config)
        
        # Get project
        project_obj = await vidmap.storage.get_project(project)
        if not project_obj or not project_obj.videos:
            console.print(f"[red]✗[/red] No videos found in project '{project}'")
            await vidmap.close()
            return
        
        # Build index for each video
        for video in project_obj.videos:
            console.print(f"\n[bold blue]Building index for: {video.title}[/bold blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Building index...", total=None)
                
                # Build index
                results = await vidmap.build_index(
                    project_name=project,
                    video_id=video.id,
                    enable_embeddings=embeddings,
                    tile_size=tile_size
                )
                
                progress.update(task, description="Index built!")
            
            # Display results summary
            _display_build_results(results)
        
        await vidmap.close()
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to build index: {e}")
        sys.exit(1)


def _display_build_results(results: dict):
    """Display build results in a table."""
    if not results:
        console.print("[yellow]No build results to display[/yellow]")
        return
    
    table = Table(title="Build Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green")
    
    for component, data in results.items():
        if component == "embeddings":
            if data:
                status = "✓ Generated"
                details = f"{len(data.get('scenes', []))} scene embeddings"
            else:
                status = "✗ Skipped"
                details = "Embeddings disabled"
        elif component == "tiles":
            status = "✓ Generated"
            details = f"{len(data)} map tiles"
        elif component == "search_index":
            status = "✓ Built"
            details = "Search index ready"
        else:
            status = "✓ Completed"
            details = str(data)
        
        table.add_row(component.replace("_", " ").title(), status, details)
    
    console.print(table)


@main.command()
@click.option("--project", "-p", required=True, help="Project name")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=5173, help="Port to bind to")
@click.option("--open", is_flag=True, help="Open browser automatically")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
def serve(project: str, host: str, port: int, open: bool, config: Optional[str]):
    """Serve the VidMap web interface."""
    console.print(f"[bold blue]Starting VidMap server for project: {project}[/bold blue]")
    console.print(f"[green]Server will be available at: http://{host}:{port}[/green]")
    
    if open:
        console.print("[yellow]Note: Browser auto-open not implemented yet[/yellow]")
    
    # This would start the FastAPI server
    # For now, just show a placeholder
    console.print("[yellow]Web interface not yet implemented[/yellow]")
    console.print("Use the API endpoints directly or implement the web UI")


@main.command()
@click.option("--project", "-p", required=True, help="Project name")
@click.option("--format", "-f", type=click.Choice(["json", "csv", "edl", "srt", "vtt"]), default="json", help="Export format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
def export(project: str, format: str, output: Optional[str], config: Optional[str]):
    """Export project data in various formats."""
    asyncio.run(_export_project(project, format, output, config))


async def _export_project(project: str, format: str, output: Optional[str], config: Optional[str]):
    """Export project implementation."""
    try:
        # Load configuration
        if config:
            vidmap_config = Config.from_file(config)
        else:
            vidmap_config = Config.create_default(project)
        
        # Initialize VidMap
        vidmap = VidMap(vidmap_config)
        
        with console.status(f"[bold green]Exporting project '{project}' as {format.upper()}...", spinner="dots"):
            # Export project
            output_path = await vidmap.export_project(
                project_name=project,
                format=format,
                output_path=output
            )
        
        console.print(f"\n[green]✓[/green] Project '{project}' exported to: {output_path}")
        
        await vidmap.close()
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to export project: {e}")
        sys.exit(1)


@main.command()
@click.option("--project", "-p", required=True, help="Project name")
@click.option("--query", "-q", required=True, help="Search query")
@click.option("--limit", "-l", type=int, default=10, help="Maximum number of results")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
def search(project: str, query: str, limit: int, config: Optional[str]):
    """Search for content in a project."""
    asyncio.run(_search_project(project, query, limit, config))


async def _search_project(project: str, query: str, limit: int, config: Optional[str]):
    """Search project implementation."""
    try:
        # Load configuration
        if config:
            vidmap_config = Config.from_file(config)
        else:
            vidmap_config = Config.create_default(project)
        
        # Initialize VidMap
        vidmap = VidMap(vidmap_config)
        
        with console.status(f"[bold green]Searching for '{query}'...", spinner="dots"):
            # Search project
            results = await vidmap.search(
                project_name=project,
                query=query,
                limit=limit
            )
        
        if not results:
            console.print(f"[yellow]No results found for query: {query}[/yellow]")
            await vidmap.close()
            return
        
        # Display results
        table = Table(title=f"Search Results for '{query}'")
        table.add_column("Score", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Time", style="green")
        table.add_column("Content", style="white")
        
        for result in results:
            segment = result["segment"]
            score = f"{result['relevance_score']:.2f}"
            seg_type = getattr(segment, 'type', 'unknown')
            time = f"{segment.start_time:.1f}s - {segment.end_time:.1f}s"
            
            # Truncate content
            content = result.get("matched_text", "")
            if len(content) > 60:
                content = content[:57] + "..."
            
            table.add_row(score, seg_type, time, content)
        
        console.print(table)
        console.print(f"\n[green]✓[/green] Found {len(results)} results")
        
        await vidmap.close()
        
    except Exception as e:
        console.print(f"[red]✗[/red] Search failed: {e}")
        sys.exit(1)


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
def list_projects(config: Optional[str]):
    """List all available projects."""
    asyncio.run(_list_projects(config))


async def _list_projects(config: Optional[str]):
    """List projects implementation."""
    try:
        # Load configuration
        if config:
            vidmap_config = Config.from_file(config)
        else:
            vidmap_config = Config.create_default("default")
        
        # Initialize VidMap
        vidmap = VidMap(vidmap_config)
        
        # List projects
        projects = await vidmap.storage.list_projects()
        
        if not projects:
            console.print("[yellow]No projects found[/yellow]")
            await vidmap.close()
            return
        
        # Display projects
        table = Table(title="Available Projects")
        table.add_column("Project Name", style="cyan")
        table.add_column("Status", style="magenta")
        
        for project_name in projects:
            # Get project details
            project = await vidmap.storage.get_project(project_name)
            if project and project.videos:
                status = f"✓ {len(project.videos)} videos"
            else:
                status = "Empty"
            
            table.add_row(project_name, status)
        
        console.print(table)
        console.print(f"\n[green]✓[/green] Found {len(projects)} projects")
        
        await vidmap.close()
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list projects: {e}")
        sys.exit(1)


@main.command()
@click.option("--project", "-p", required=True, help="Project name")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
def info(project: str, config: Optional[str]):
    """Show detailed information about a project."""
    asyncio.run(_show_project_info(project, config))


async def _show_project_info(project: str, config: Optional[str]):
    """Show project info implementation."""
    try:
        # Load configuration
        if config:
            vidmap_config = Config.from_file(config)
        else:
            vidmap_config = Config.create_default(project)
        
        # Initialize VidMap
        vidmap = VidMap(vidmap_config)
        
        # Get project
        project_obj = await vidmap.storage.get_project(project)
        if not project_obj:
            console.print(f"[red]✗[/red] Project '{project}' not found")
            await vidmap.close()
            return
        
        # Display project information
        console.print(f"\n[bold blue]Project: {project_obj.name}[/bold blue]")
        if project_obj.description:
            console.print(f"[dim]{project_obj.description}[/dim]")
        
        # Project stats
        stats_table = Table(title="Project Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")
        
        stats_table.add_row("Videos", str(len(project_obj.videos)))
        stats_table.add_row("Chapters", str(len(project_obj.chapters)))
        stats_table.add_row("Speakers", str(len(project_obj.speakers)))
        stats_table.add_row("Topics", str(len(project_obj.topics)))
        stats_table.add_row("Created", project_obj.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        stats_table.add_row("Updated", project_obj.updated_at.strftime("%Y-%m-%d %H:%M:%S"))
        
        console.print(stats_table)
        
        # Video details
        if project_obj.videos:
            console.print(f"\n[bold green]Videos:[/bold green]")
            for video in project_obj.videos:
                console.print(f"  • {video.title} ({video.duration:.1f}s, {video.width}x{video.height})")
        
        await vidmap.close()
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to show project info: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
