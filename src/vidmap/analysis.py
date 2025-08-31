"""Video analysis including scene detection, transcription, and OCR."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from .config import Config
from .models import Video, Scene, AudioSegment, OCRSegment, Speaker


logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Handles video analysis tasks like scene detection, transcription, and OCR."""
    
    def __init__(self, config: Config):
        """Initialize the video analyzer."""
        self.config = config
        self._setup_models()
    
    def _setup_models(self):
        """Setup AI models for analysis."""
        try:
            if self.config.vision.enabled:
                import cv2
                self.cv2 = cv2
            else:
                self.cv2 = None
                
            if self.config.asr.engine == "whisper":
                try:
                    import whisper
                    self.whisper_model = whisper.load_model(self.config.asr.model)
                except ImportError:
                    logger.warning("Whisper not available, ASR will be disabled")
                    self.whisper_model = None
            else:
                self.whisper_model = None
                
        except ImportError as e:
            logger.warning(f"Some analysis features may not be available: {e}")
    
    async def detect_scenes(self, video: Video) -> List[Scene]:
        """Detect scene changes in the video."""
        if not self.config.vision.scene_detection:
            return []
        
        try:
            scenes = []
            video_path = Path(video.source_path)
            
            # Use OpenCV for scene detection
            if self.cv2:
                scenes = await self._detect_scenes_opencv(video_path, video)
            else:
                # Fallback to simple time-based segmentation
                scenes = await self._detect_scenes_simple(video_path, video)
            
            logger.info(f"Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return []
    
    async def _detect_scenes_opencv(self, video_path: Path, video: Video) -> List[Scene]:
        """Detect scenes using OpenCV."""
        scenes = []
        cap = self.cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        try:
            fps = cap.get(self.cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
            
            # Parameters for scene detection
            threshold = 30.0  # Brightness difference threshold
            min_scene_duration = 2.0  # Minimum scene duration in seconds
            min_frames = int(min_scene_duration * fps)
            
            prev_frame = None
            scene_start = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for comparison
                gray = self.cv2.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = self.cv2.absdiff(gray, prev_frame)
                    mean_diff = np.mean(diff)
                    
                    # Check if this is a scene change
                    if mean_diff > threshold and frame_count - scene_start >= min_frames:
                        # Create scene
                        scene = Scene(
                            video_id=video.id,
                            start_time=scene_start / fps,
                            end_time=frame_count / fps,
                            confidence=min(mean_diff / 100.0, 1.0),
                            shot_type=self._classify_shot(frame),
                            visual_tags=self._extract_visual_tags(frame),
                            dominant_colors=self._extract_dominant_colors(frame),
                            brightness=np.mean(gray),
                            motion_score=mean_diff / 100.0
                        )
                        scenes.append(scene)
                        scene_start = frame_count
                
                prev_frame = gray
                frame_count += 1
                
                # Progress logging
                if frame_count % 1000 == 0:
                    logger.debug(f"Processed {frame_count}/{total_frames} frames")
            
            # Add final scene
            if frame_count - scene_start >= min_frames:
                scene = Scene(
                    video_id=video.id,
                    start_time=scene_start / fps,
                    end_time=frame_count / fps,
                    confidence=0.8,
                    shot_type=self._classify_shot(frame),
                    visual_tags=self._extract_visual_tags(frame),
                    dominant_colors=self._extract_dominant_colors(frame),
                    brightness=np.mean(gray),
                    motion_score=0.0
                )
                scenes.append(scene)
        
        finally:
            cap.release()
        
        return scenes
    
    async def _detect_scenes_simple(self, video_path: Path, video: Video) -> List[Scene]:
        """Simple time-based scene detection as fallback."""
        scenes = []
        duration = video.duration
        
        # Create scenes every 30 seconds
        scene_duration = 30.0
        current_time = 0.0
        
        while current_time < duration:
            end_time = min(current_time + scene_duration, duration)
            
            scene = Scene(
                video_id=video.id,
                start_time=current_time,
                end_time=end_time,
                confidence=0.5,
                shot_type="unknown",
                visual_tags=[],
                dominant_colors=[],
                brightness=0.5,
                motion_score=0.0
            )
            scenes.append(scene)
            
            current_time = end_time
        
        return scenes
    
    def _classify_shot(self, frame) -> str:
        """Classify the type of shot in a frame."""
        try:
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            
            # Simple shot classification based on aspect ratio and content
            if aspect_ratio > 2.0:
                return "wide"
            elif aspect_ratio < 1.0:
                return "tall"
            else:
                # Check if it's a close-up by analyzing face detection
                if self.config.vision.face_detection:
                    faces = self._detect_faces(frame)
                    if len(faces) > 0:
                        return "close-up"
                return "medium"
        except Exception:
            return "unknown"
    
    def _detect_faces(self, frame) -> List:
        """Detect faces in a frame."""
        try:
            face_cascade = self.cv2.CascadeClassifier(
                self.cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = self.cv2.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            return faces
        except Exception:
            return []
    
    def _extract_visual_tags(self, frame) -> List[str]:
        """Extract visual content tags from a frame."""
        tags = []
        
        try:
            # Simple color-based tagging
            hsv = self.cv2.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
            
            # Check for dark scenes
            if np.mean(hsv[:, :, 2]) < 50:
                tags.append("dark")
            elif np.mean(hsv[:, :, 2]) > 200:
                tags.append("bright")
            
            # Check for colorful scenes
            colorfulness = np.std(hsv[:, :, 1])
            if colorfulness > 50:
                tags.append("colorful")
            else:
                tags.append("monochrome")
                
        except Exception:
            pass
        
        return tags
    
    def _extract_dominant_colors(self, frame) -> List[str]:
        """Extract dominant colors from a frame."""
        colors = []
        
        try:
            # Simple color extraction
            hsv = self.cv2.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
            
            # Define color ranges
            color_ranges = {
                "red": ([0, 50, 50], [10, 255, 255]),
                "green": ([40, 50, 50], [80, 255, 255]),
                "blue": ([100, 50, 50], [130, 255, 255]),
                "yellow": ([20, 50, 50], [40, 255, 255]),
                "purple": ([130, 50, 50], [170, 255, 255])
            }
            
            for color_name, (lower, upper) in color_ranges.items():
                mask = self.cv2.inRange(hsv, np.array(lower), np.array(upper))
                if np.sum(mask) > frame.shape[0] * frame.shape[1] * 0.1:  # 10% threshold
                    colors.append(color_name)
                    
        except Exception:
            pass
        
        return colors
    
    async def transcribe_audio(self, video: Video) -> List[AudioSegment]:
        """Transcribe audio from the video."""
        if not self.whisper_model:
            logger.warning("Whisper model not available, skipping transcription")
            return []
        
        try:
            logger.info("Starting audio transcription...")
            
            # Use Whisper for transcription
            result = self.whisper_model.transcribe(
                video.source_path,
                language=self.config.asr.language,
                word_timestamps=True
            )
            
            # Convert Whisper results to AudioSegment objects
            segments = []
            for seg in result["segments"]:
                audio_segment = AudioSegment(
                    video_id=video.id,
                    start_time=seg["start"],
                    end_time=seg["end"],
                    text=seg["text"].strip(),
                    language=result["language"],
                    confidence=seg.get("avg_logprob", 0.5),
                    words=seg.get("words", [])
                )
                segments.append(audio_segment)
            
            logger.info(f"Transcribed {len(segments)} audio segments")
            return segments
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return []
    
    async def diarize_speakers(
        self,
        video: Video,
        audio_segments: List[AudioSegment]
    ) -> List[Speaker]:
        """Perform speaker diarization on audio segments."""
        if not self.config.diarization.enabled:
            return []
        
        try:
            # Simple speaker diarization based on audio characteristics
            # In a real implementation, you would use pyannote.audio or similar
            
            speakers = []
            current_speaker_id = 0
            
            for segment in audio_segments:
                # Simple heuristic: new speaker if silence gap > 2 seconds
                if speakers and segment.start_time - speakers[-1].segments[-1].end_time > 2.0:
                    current_speaker_id += 1
                
                # Create or find existing speaker
                if current_speaker_id >= len(speakers):
                    speaker = Speaker(
                        id=f"spk_{current_speaker_id}",
                        confidence=0.8,
                        segments=[]
                    )
                    speakers.append(speaker)
                
                # Add segment to speaker
                speakers[current_speaker_id].segments.append(segment)
                segment.speaker_id = speakers[current_speaker_id].id
            
            logger.info(f"Identified {len(speakers)} speakers")
            return speakers
            
        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            return []
    
    async def extract_text(self, video: Video) -> List[OCRSegment]:
        """Extract text from video frames using OCR."""
        if not self.config.ocr.enabled:
            return []
        
        try:
            ocr_segments = []
            
            # Sample frames for OCR (every 5 seconds)
            sample_interval = 5.0
            current_time = 0.0
            
            while current_time < video.duration:
                # Extract frame at current time
                frame = await self._extract_frame(video, current_time)
                if frame is not None:
                    # Perform OCR on frame
                    text = await self._perform_ocr(frame)
                    
                    if text and len(text.strip()) > 3:  # Minimum text length
                        ocr_segment = OCRSegment(
                            video_id=video.id,
                            start_time=current_time,
                            end_time=min(current_time + sample_interval, video.duration),
                            text=text.strip(),
                            language=self.config.ocr.languages[0],
                            confidence=0.7,  # Default confidence
                            bounding_box=[0, 0, frame.shape[1], frame.shape[0]],
                            is_slide=self._is_likely_slide(frame, text)
                        )
                        ocr_segments.append(ocr_segment)
                
                current_time += sample_interval
            
            logger.info(f"Extracted text from {len(ocr_segments)} frames")
            return ocr_segments
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []
    
    async def _extract_frame(self, video: Video, timestamp: float):
        """Extract a frame from the video at a specific timestamp."""
        try:
            import cv2
            
            cap = cv2.VideoCapture(video.source_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return frame
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Frame extraction failed at {timestamp}s: {e}")
            return None
    
    async def _perform_ocr(self, frame) -> str:
        """Perform OCR on a frame."""
        try:
            if self.config.ocr.engine == "tesseract":
                import pytesseract
                text = pytesseract.image_to_string(frame)
                return text
            elif self.config.ocr.engine == "paddleocr":
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(use_angle_cls=True, lang='en')
                result = ocr.ocr(frame)
                
                if result and result[0]:
                    text = " ".join([line[1][0] for line in result[0]])
                    return text
                else:
                    return ""
            else:
                return ""
                
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return ""
    
    def _is_likely_slide(self, frame, text: str) -> bool:
        """Determine if a frame is likely a slide based on visual characteristics."""
        try:
            # Simple heuristic: slides often have high contrast and structured text
            gray = self.cv2.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            
            # Check contrast
            contrast = np.std(gray)
            
            # Check if text is structured (multiple lines)
            lines = text.count('\n')
            
            # Slides typically have high contrast and multiple text lines
            return contrast > 50 and lines > 1
            
        except Exception:
            return False
    
    async def analyze_vision(self, video: Video) -> Dict[str, Any]:
        """Analyze visual content of the video."""
        if not self.config.vision.enabled:
            return {}
        
        try:
            vision_analysis = {
                "objects": [],
                "faces": [],
                "text_regions": [],
                "motion_patterns": []
            }
            
            # Sample frames for analysis
            sample_interval = 10.0  # Every 10 seconds
            current_time = 0.0
            
            while current_time < video.duration:
                frame = await self._extract_frame(video, current_time)
                if frame is not None:
                    # Object detection
                    if self.config.vision.object_detection:
                        objects = await self._detect_objects(frame)
                        vision_analysis["objects"].extend(objects)
                    
                    # Face detection
                    if self.config.vision.face_detection:
                        faces = self._detect_faces(frame)
                        if faces:
                            vision_analysis["faces"].append({
                                "timestamp": current_time,
                                "count": len(faces)
                            })
                
                current_time += sample_interval
            
            logger.info("Vision analysis completed")
            return vision_analysis
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {}
    
    async def _detect_objects(self, frame) -> List[Dict[str, Any]]:
        """Detect objects in a frame."""
        # Placeholder for object detection
        # In a real implementation, you would use YOLO, Detectron2, or similar
        return []
