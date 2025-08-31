# VidMap

Map-like indexing for long videos. VidMap analyzes media, auto-generates a zoomable “map” (scene graph + topic tiles), and lets users navigate by panning, zooming, and searching—like Google Maps, but for timelines.

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

  * Natural-language search (“where he explains gradient clipping”).
  * Faceted filters (topic, speaker, location, slide text, visual objects).
* **Chapters & Exports**

  * Auto chapters → editable.
  * Export to JSON/CSV/EDL/SRT/WebVTT; static PNG/SVG of map.
* **Multi-file Atlas**

  * Index playlists/courses into a unified “country → province → city” map metaphor.
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

## Install

```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate
pip install -U vidmap

# with extras (gpu + asr + ocr)
pip install -U "vidmap[gpu,whisper,ocr]"
```

---

## Quick Start

```bash
# 1) Ingest a video
vidmap ingest ./media/talk.mp4 --project demo

# 2) Analyze & index (scenes, asr, diar, ocr, tags)
vidmap index --project demo --scenes --asr --diar --ocr --vision

# 3) Build zoom tiles + embeddings
vidmap build --project demo

# 4) Launch UI
vidmap serve --project demo --open
```

Open `http://localhost:5173` → pan/zoom, search, click tiles to jump.

---

## UI Controls

* **Zoom**: mousewheel / trackpad pinch
* **Pan**: click-drag
* **Jump**: click a tile
* **Search**: `f` then type; hit `Enter` to navigate results
* **Layers**: toggle heatmaps (salience, laughter, silence, slide density)
* **Bookmarks**: `b` to bookmark, export later

> **v2 edit mode:** drag-drop tiles, reorder chapters, merge/split, inline rename, sticky notes.

---

## Data Model (simplified)

```json
{
  "video": { "id": "vid_01", "duration": 3921.4, "fps": 29.97, "source": "talk.mp4" },
  "nodes": [
    { "id": "scn_12", "type": "scene", "t0": 841.2, "t1": 903.8,
      "thumb": "tiles/2/12.png", "tags": ["optimizer","demo"], "speaker": ["spk_a"] }
  ],
  "edges": [
    { "from": "scn_11", "to": "scn_12", "type": "temporal" },
    { "from": "scn_12", "to": "topic_gradclip", "type": "semantic" }
  ],
  "chapters": [
    { "title": "Optimizer Tuning", "t0": 812.0, "t1": 1140.0 }
  ]
}
```

---

## REST API (FastAPI)

```http
GET  /api/projects/:id/map?lod=3              # tile data
GET  /api/projects/:id/search?q=grad+clipping  # hybrid search
GET  /api/projects/:id/chapters                # chapters
POST /api/projects/:id/chapters                # (v2) edit
```

---

## CLI Reference

```bash
vidmap ingest <path|url> --project <name> [--title "..."] [--lang tr]
vidmap index  --project <name> [--scenes] [--asr] [--diar] [--ocr] [--vision]
vidmap build  --project <name> [--embeddings all-MiniLM] [--tile-size 256]
vidmap serve  --project <name> [--host 0.0.0.0] [--port 5173] [--open]
vidmap export --project <name> --format {json,csv,edl,srt,vtt,png,svg}
```

---

## Configuration (`vidmap.yaml`)

```yaml
project: demo
storage: .vidmap/demo
asr:
  engine: whisper
  model: small
diarization:
  enabled: true
ocr:
  enabled: true
embeddings:
  model: all-MiniLM-L6-v2
tiles:
  size: 256
ui:
  heatmaps: [salience, laughter, silence, slide_density]
```

---

## Workflow Tips

* For talking-head content: prioritize **ASR + diarization**; enable slide OCR if slides exist.
* For documentaries/music videos: enable **scene + vision tags**; use salience heatmap for hooks.
* For long courses/playlists: create a **Collection** project and build an atlas map.

---

## Roadmap

* **v1**

  * Zoomable map UI, hybrid search, exports
  * Multi-video atlas, heatmap layers
* **v1.x**

  * Face/Name linker, cross-video entity timelines
  * Slide-to-segment alignment; quote extract
* **v2 (Editing)**

  * **Drag-drop** tiles & **change order**
  * Manual **merge/split** segments; in-place chapter titles
  * Sticky notes, color labels, bulk edits
  * Multi-user sessions with history/undo

---

## Contributing

Issues/PRs welcome. Please include:

* Repro clip (<1 min) and `vidmap.yaml`
* Pipeline step & logs
* Before/after screenshots for UI changes

---

## License

Apache-2.0

---

## Acknowledgements

Built on the shoulders of `ffmpeg`, `PySceneDetect`, `pyannote`, Whisper, FAISS, and the open-source community.
