# VidMap

Map-like indexing for long videos. VidMap analyzes media, auto-generates a zoomable "map" (scene graph + topic tiles), and lets users navigate by panning, zooming, and searching—like Google Maps, but for timelines.

---

## Why VidMap?

* **Skip linear scrubbing.** See the whole video at once as clustered, labeled regions.
* **Zoom for detail.** Zoom in for shots and quotes; zoom out for acts/chapters/themes.
* **Shareable insights.** Export maps, chapter lists, and JSON for downstream tools.

---

## Features

* **Zoomable Map UI**
  * Pan/zoom timeline canvas (LOD tiles).
  * Semantic clusters (topic, speaker, scene).
  * Thumbnails + keyframes; heatmap overlays (salience, applause, laughter).

* **AI Segmentation & Tagging**
  * Shot/scene detection (visual), topic segmentation (audio/text).
  * Speaker diarization, face/name linking (optional).
  * OCR for slides/whiteboards; on-frame keyword extraction.

* **Search & Jump**
  * Natural-language search ("where he explains gradient clipping").
  * Faceted filters (topic, speaker, location, slide text, visual objects).

* **Chapters & Exports**
  * Auto chapters → editable.
  * Export to JSON/CSV/EDL/SRT/WebVTT; static PNG/SVG of map.

* **Multi-file Atlas**
  * Index playlists/courses into a unified "country → province → city" map metaphor.

* **Toolbox**
  * **vidmap ingest**: add videos (local/URL), batch extract tracks & metadata.
  * **vidmap index**: run detectors (scenes, ASR, diarization, OCR).
  * **vidmap build**: tile pyramid + embeddings + graph.
  * **vidmap serve**: local web UI.
  * **vidmap export**: chapters, map image, captions, EDL.

> **v2 (planned):** drag-drop tiles, manual merge/split, change order, sticky notes, per-tile color tags, multi-user edit mode.

---

## Architecture

* **Core**: Python pipeline (ingest → analyze → tile → index)
* **Workers**: Async task queue (Celery/RQ) for GPU/CPU jobs
* **Embeddings**: Vector DB (FAISS/pgvector)
* **Search**: BM25 + vector hybrid
* **UI**: React + WebGL/Canvas; virtualized canvas for millions of nodes
* **Store**: SQLite/Postgres project db + blob storage for tiles/thumbnails
* **Schema**: Scene graph (nodes: scene/shot/segment/speaker/topic; edges: temporal, semantic, visual)

---

## Tech Stack

* **Video**: `ffmpeg`, `PySceneDetect`
* **Audio/ASR**: Whisper (or Vosk), RMS/envelope for energy
* **Diarization**: `pyannote.audio`
* **OCR**: Tesseract or PaddleOCR
* **Vision Tags**: `torchvision`/`opencv`
* **NLP**: `spacy` + transformer embeddings
* **Server**: FastAPI
* **UI**: React + Zustand + React-Query, WebGL/Canvas rendering

---

## Installation

### Prerequisites

* Python 3.10+
* FFmpeg (for video processing)
* CUDA-compatible GPU (optional, for acceleration)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/makalin/VidMap.git
cd VidMap

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Optional Dependencies

```bash
# GPU support
pip install torch[cu118] torchvision[cu118] faiss-gpu

# ASR with Whisper
pip install openai-whisper

# OCR support
pip install pytesseract paddleocr

# Speaker diarization
pip install pyannote.audio
```

---

## Quick Start

### 1. Create a Project

```bash
# Create a new project
vidmap create --project demo --description "My first VidMap project"
```

### 2. Ingest a Video

```bash
# Ingest a video file
vidmap ingest ./media/talk.mp4 --project demo --title "Introduction to Machine Learning"
```

### 3. Analyze & Index

```bash
# Run full analysis (scenes, ASR, diarization, OCR, vision)
vidmap index --project demo --scenes --asr --diar --ocr --vision

# Build search index and generate map tiles
vidmap build --project demo --embeddings --tile-size 256
```

### 4. Launch Web Interface

```bash
# Start the web server
vidmap serve --project demo --host 0.0.0.0 --port 5173 --open
```

Open `http://localhost:5173` → pan/zoom, search, click tiles to jump.

---

## CLI Reference

### Project Management

```bash
# Create a new project
vidmap create --project <name> [--description "..."] [--config <path>]

# List all projects
vidmap list-projects [--config <path>]

# Show project information
vidmap info --project <name> [--config <path>]
```

### Video Processing

```bash
# Ingest a video
vidmap ingest <path|url> --project <name> [--title "..."] [--config <path>]

# Analyze video content
vidmap index --project <name> [--scenes] [--asr] [--diar] [--ocr] [--vision] [--config <path>]

# Build search index
vidmap build --project <name> [--embeddings] [--tile-size <size>] [--config <path>]
```

### Web Interface

```bash
# Serve web UI
vidmap serve --project <name> [--host <host>] [--port <port>] [--open] [--config <path>]

# Start API server only
python -m vidmap.api [config_path]
```

### Search & Export

```bash
# Search for content
vidmap search --project <name> --query <query> [--limit <number>] [--config <path>]

# Export project data
vidmap export --project <name> --format {json,csv,edl,srt,vtt} [--output <path>] [--config <path>]
```

---

## Configuration

### Configuration File (`vidmap.yaml`)

```yaml
project: demo
description: "Example VidMap project"

# Storage configuration
storage:
  backend: sqlite
  blob_storage: local
  blob_path: .vidmap/demo

# ASR configuration
asr:
  engine: whisper
  model: small
  language: en
  device: cpu

# Speaker diarization
diarization:
  enabled: true
  model: pyannote/speaker-diarization
  min_speakers: 1
  max_speakers: 5

# OCR configuration
ocr:
  enabled: true
  engine: tesseract
  languages: [eng]
  confidence_threshold: 0.6

# Computer vision
vision:
  enabled: true
  scene_detection: true
  object_detection: false
  face_detection: true

# Embeddings
embeddings:
  model: all-MiniLM-L6-v2
  dimension: 384
  batch_size: 32

# Map tiles
tiles:
  size: 256
  overlap: 0
  levels: 5

# Web UI
ui:
  host: 0.0.0.0
  port: 5173
  heatmaps: [salience, laughter, silence, slide_density]
```

### Environment Variables

```bash
# Set environment variables for sensitive configuration
export VIDMAP_OPENAI_API_KEY="your-key-here"
export VIDMAP_HUGGINGFACE_TOKEN="your-token-here"
export VIDMAP_STORAGE_PATH="/path/to/storage"
```

---

## API Reference

### REST API Endpoints

The FastAPI server provides the following endpoints:

#### Projects
- `GET /api/projects` - List all projects
- `GET /api/projects/{name}` - Get project details
- `POST /api/projects` - Create new project

#### Videos
- `POST /api/projects/{name}/videos` - Ingest video
- `POST /api/projects/{name}/analyze` - Analyze video
- `POST /api/projects/{name}/build` - Build index

#### Search & Navigation
- `GET /api/projects/{name}/search` - Search content
- `GET /api/projects/{name}/map` - Get map tile data

#### Export
- `GET /api/projects/{name}/export` - Export project data

#### Static Files
- `GET /tiles/{level}/{tile_id}` - Serve map tiles
- `GET /thumbnails/{video_id}` - Serve video thumbnails

### API Documentation

When running the server, visit:
- Swagger UI: `http://localhost:5173/docs`
- ReDoc: `http://localhost:5173/redoc`

---

## Web Interface

### Features

- **Project Selection**: Choose from available projects
- **Video Navigation**: Browse videos within projects
- **Timeline Map**: Zoomable timeline with scene/segment visualization
- **Search Interface**: Full-text search across video content
- **Export Tools**: Download data in various formats

### Controls

- **Zoom**: Mouse wheel or zoom buttons
- **Pan**: Click and drag
- **Search**: Type query and press Enter
- **Navigation**: Click on timeline tiles to jump to content

---

## Development

### Project Structure

```
vidmap/
├── src/vidmap/           # Source code
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration management
│   ├── models.py         # Data models
│   ├── core.py           # Main VidMap class
│   ├── ingest.py         # Video ingestion
│   ├── analysis.py       # Video analysis
│   ├── indexing.py       # Indexing and tiles
│   ├── search.py         # Search functionality
│   ├── storage.py        # Data persistence
│   ├── api.py            # FastAPI server
│   ├── cli.py            # Command-line interface
│   └── static/           # Web UI assets
├── tests/                # Test suite
├── examples/             # Example configurations
├── requirements.txt      # Python dependencies
├── pyproject.toml       # Project configuration
└── README.md            # This file
```

### Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=vidmap

# Run specific test file
pytest tests/test_basic.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

---

## Workflow Examples

### Academic Lecture Analysis

```bash
# Create project for lecture series
vidmap create --project "CS101_Lectures" --description "Introduction to Computer Science"

# Ingest lecture videos
vidmap ingest ./lectures/lecture01.mp4 --project CS101_Lectures --title "Introduction"
vidmap ingest ./lectures/lecture02.mp4 --project CS101_Lectures --title "Variables and Types"

# Analyze content
vidmap index --project CS101_Lectures --scenes --asr --ocr --vision

# Build searchable index
vidmap build --project CS101_Lectures --embeddings

# Search for specific topics
vidmap search --project CS101_Lectures --query "variable declaration"
```

### Podcast Content Indexing

```bash
# Create podcast project
vidmap create --project "TechPodcast" --description "Weekly Tech Discussion"

# Ingest episodes
vidmap ingest ./podcasts/episode001.mp3 --project TechPodcast --title "AI Trends 2024"

# Focus on audio content
vidmap index --project TechPodcast --asr --diar

# Build index
vidmap build --project TechPodcast --embeddings

# Search for discussions
vidmap search --project TechPodcast --query "machine learning applications"
```

---

## Performance & Scaling

### Optimization Tips

- **GPU Acceleration**: Use CUDA for faster processing
- **Batch Processing**: Process multiple videos in sequence
- **Storage**: Use SSD for better I/O performance
- **Memory**: Increase RAM for large video files

### Scaling Considerations

- **Distributed Processing**: Use Celery for task distribution
- **Database**: Switch to PostgreSQL for multi-user scenarios
- **Caching**: Implement Redis for search result caching
- **CDN**: Use CDN for tile and thumbnail delivery

---

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Install FFmpeg
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **CUDA out of memory**
   ```bash
   # Reduce batch size in configuration
   embeddings:
     batch_size: 16  # Reduce from 32
   ```

3. **Slow processing**
   ```bash
   # Enable GPU acceleration
   asr:
     device: cuda
   
   # Use smaller models
   asr:
     model: tiny  # Instead of small
   ```

### Debug Mode

```bash
# Enable verbose logging
vidmap --verbose ingest ./video.mp4 --project demo

# Check system requirements
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/makalin/VidMap.git
cd VidMap

# Create feature branch
git checkout -b feature/amazing-feature

# Install development dependencies
pip install -e ".[dev]"

# Make changes and test
pytest

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Include tests for new features

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built on the shoulders of:
- `ffmpeg` - Video processing
- `PySceneDetect` - Scene detection
- `pyannote.audio` - Speaker diarization
- `Whisper` - Speech recognition
- `FAISS` - Vector similarity search
- `FastAPI` - Web framework
- And the open-source community

---

## Support

- **Documentation**: [GitHub README](https://github.com/makalin/VidMap#readme)
- **Issues**: [GitHub Issues](https://github.com/makalin/VidMap/issues)
- **Discussions**: [GitHub Discussions](https://github.com/makalin/VidMap/discussions)
- **Email**: [Create an issue](https://github.com/makalin/VidMap/issues) for support

---

## Roadmap

### v1.1 (Current)
- [x] Core video analysis pipeline
- [x] Scene detection and segmentation
- [x] Audio transcription (Whisper)
- [x] Basic OCR support
- [x] Search functionality
- [x] Web interface
- [x] Export capabilities

### v1.2 (Next)
- [ ] Advanced speaker diarization
- [ ] Object detection and tracking
- [ ] Emotion analysis
- [ ] Multi-language support
- [ ] Batch processing improvements

### v2.0 (Future)
- [ ] Interactive timeline editing
- [ ] Collaborative features
- [ ] Advanced visualization options
- [ ] Mobile app
- [ ] Cloud deployment options
