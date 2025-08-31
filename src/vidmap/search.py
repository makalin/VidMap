"""Video search functionality with hybrid text and vector search."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict

from .config import Config
from .models import Project, SearchResult


logger = logging.getLogger(__name__)


class VideoSearcher:
    """Handles video content search with hybrid text and vector search."""
    
    def __init__(self, config: Config):
        """Initialize the video searcher."""
        self.config = config
    
    async def search(
        self,
        project: Project,
        query: str,
        video_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search for content in a project."""
        try:
            query = query.strip().lower()
            if not query:
                return []
            
            # Parse query for different search types
            search_results = []
            
            # Text search
            text_results = await self._text_search(project, query, video_id, filters)
            search_results.extend(text_results)
            
            # Vector search (if available)
            vector_results = await self._vector_search(project, query, video_id, filters)
            search_results.extend(vector_results)
            
            # Metadata search
            metadata_results = await self._metadata_search(project, query, video_id, filters)
            search_results.extend(metadata_results)
            
            # Deduplicate and rank results
            ranked_results = self._rank_and_deduplicate(search_results, query)
            
            # Apply limit
            return ranked_results[:limit]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _text_search(
        self,
        project: Project,
        query: str,
        video_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform full-text search on video content."""
        results = []
        query_words = query.split()
        
        for chapter in project.chapters:
            for segment in chapter.segments:
                # Skip if video filter is applied
                if video_id and segment.video_id != video_id:
                    continue
                
                # Apply other filters
                if filters and not self._apply_filters(segment, filters):
                    continue
                
                # Search in audio segments
                if hasattr(segment, 'text') and segment.text:
                    relevance_score = self._calculate_text_relevance(segment.text, query_words)
                    
                    if relevance_score > 0:
                        result = {
                            "segment": segment,
                            "relevance_score": relevance_score,
                            "matched_text": segment.text,
                            "search_type": "text",
                            "highlight_times": [segment.start_time]
                        }
                        results.append(result)
        
        return results
    
    async def _vector_search(
        self,
        project: Project,
        query: str,
        video_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        # This would use the FAISS index built during indexing
        # For now, return empty results as a placeholder
        return []
    
    async def _metadata_search(
        self,
        project: Project,
        query: str,
        video_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search through metadata like speakers, topics, visual tags."""
        results = []
        query_lower = query.lower()
        
        # Search speakers
        for speaker in project.speakers:
            if video_id and not self._speaker_in_video(speaker, video_id, project):
                continue
            
            if speaker.name and query_lower in speaker.name.lower():
                # Find segments for this speaker
                speaker_segments = self._find_speaker_segments(speaker, project)
                
                for segment in speaker_segments:
                    if filters and not self._apply_filters(segment, filters):
                        continue
                    
                    result = {
                        "segment": segment,
                        "relevance_score": 0.8,
                        "matched_text": f"Speaker: {speaker.name}",
                        "search_type": "metadata",
                        "highlight_times": [segment.start_time]
                    }
                    results.append(result)
        
        # Search topics
        for topic in project.topics:
            if video_id and not self._topic_in_video(topic, video_id, project):
                continue
            
            if (query_lower in topic.name.lower() or 
                (topic.description and query_lower in topic.description.lower()) or
                any(query_lower in keyword.lower() for keyword in topic.keywords)):
                
                # Find segments for this topic
                topic_segments = self._find_topic_segments(topic, project)
                
                for segment in topic_segments:
                    if filters and not self._apply_filters(segment, filters):
                        continue
                    
                    result = {
                        "segment": segment,
                        "relevance_score": 0.7,
                        "matched_text": f"Topic: {topic.name}",
                        "search_type": "metadata",
                        "highlight_times": [segment.start_time]
                    }
                    results.append(result)
        
        # Search visual tags and shot types
        for chapter in project.chapters:
            for segment in chapter.segments:
                if video_id and segment.video_id != video_id:
                    continue
                
                if filters and not self._apply_filters(segment, filters):
                    continue
                
                # Search visual tags
                if hasattr(segment, 'visual_tags') and segment.visual_tags:
                    for tag in segment.visual_tags:
                        if query_lower in tag.lower():
                            result = {
                                "segment": segment,
                                "relevance_score": 0.6,
                                "matched_text": f"Visual tag: {tag}",
                                "search_type": "metadata",
                                "highlight_times": [segment.start_time]
                            }
                            results.append(result)
                
                # Search shot types
                if hasattr(segment, 'shot_type') and segment.shot_type:
                    if query_lower in segment.shot_type.lower():
                        result = {
                            "segment": segment,
                            "relevance_score": 0.6,
                            "matched_text": f"Shot type: {segment.shot_type}",
                            "search_type": "metadata",
                            "highlight_times": [segment.start_time]
                        }
                        results.append(result)
        
        return results
    
    def _calculate_text_relevance(self, text: str, query_words: List[str]) -> float:
        """Calculate text relevance score based on word matches."""
        if not text or not query_words:
            return 0.0
        
        text_lower = text.lower()
        total_words = len(text.split())
        matched_words = 0
        exact_phrase_bonus = 0.0
        
        # Check for exact phrase match
        if len(query_words) > 1:
            query_phrase = " ".join(query_words)
            if query_phrase in text_lower:
                exact_phrase_bonus = 0.3
        
        # Count word matches
        for word in query_words:
            if word in text_lower:
                matched_words += 1
        
        # Calculate base relevance
        if total_words == 0:
            return 0.0
        
        word_relevance = matched_words / len(query_words)
        density_relevance = matched_words / total_words
        
        # Combine scores with weights
        relevance = (word_relevance * 0.6 + density_relevance * 0.3 + exact_phrase_bonus)
        
        return min(relevance, 1.0)
    
    def _apply_filters(self, segment, filters: Dict[str, Any]) -> bool:
        """Apply search filters to a segment."""
        for filter_key, filter_value in filters.items():
            if filter_key == "type" and hasattr(segment, 'type'):
                if segment.type != filter_value:
                    return False
            elif filter_key == "speaker_id" and hasattr(segment, 'speaker_id'):
                if segment.speaker_id != filter_value:
                    return False
            elif filter_key == "time_range":
                start_time, end_time = filter_value
                if segment.start_time > end_time or segment.end_time < start_time:
                    return False
            elif filter_key == "confidence" and hasattr(segment, 'confidence'):
                if segment.confidence < filter_value:
                    return False
        
        return True
    
    def _speaker_in_video(self, speaker, video_id: str, project: Project) -> bool:
        """Check if a speaker appears in a specific video."""
        for chapter in project.chapters:
            for segment in chapter.segments:
                if (hasattr(segment, 'speaker_id') and 
                    segment.speaker_id == speaker.id and 
                    segment.video_id == video_id):
                    return True
        return False
    
    def _topic_in_video(self, topic, video_id: str, project: Project) -> bool:
        """Check if a topic appears in a specific video."""
        for chapter in project.chapters:
            for segment in chapter.segments:
                if (segment.video_id == video_id and
                    segment.start_time >= topic.start_time and
                    segment.end_time <= topic.end_time):
                    return True
        return False
    
    def _find_speaker_segments(self, speaker, project: Project) -> List:
        """Find all segments for a specific speaker."""
        segments = []
        for chapter in project.chapters:
            for segment in chapter.segments:
                if (hasattr(segment, 'speaker_id') and 
                    segment.speaker_id == speaker.id):
                    segments.append(segment)
        return segments
    
    def _find_topic_segments(self, topic, project: Project) -> List:
        """Find all segments for a specific topic."""
        segments = []
        for chapter in project.chapters:
            for segment in chapter.segments:
                if (segment.start_time >= topic.start_time and
                    segment.end_time <= topic.end_time):
                    segments.append(segment)
        return segments
    
    def _rank_and_deduplicate(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Rank and deduplicate search results."""
        if not results:
            return []
        
        # Group by segment ID to deduplicate
        unique_results = {}
        for result in results:
            segment_id = result["segment"].id
            
            if segment_id not in unique_results:
                unique_results[segment_id] = result
            else:
                # Combine scores for the same segment
                existing = unique_results[segment_id]
                existing["relevance_score"] = max(
                    existing["relevance_score"],
                    result["relevance_score"]
                )
                existing["matched_text"] += f" | {result['matched_text']}"
                existing["search_type"] = "hybrid"
        
        # Convert back to list and sort by relevance
        ranked_results = list(unique_results.values())
        ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return ranked_results
    
    async def search_by_timestamp(
        self,
        project: Project,
        timestamp: float,
        video_id: Optional[str] = None,
        tolerance: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Search for content at a specific timestamp."""
        try:
            results = []
            
            for chapter in project.chapters:
                for segment in chapter.segments:
                    # Skip if video filter is applied
                    if video_id and segment.video_id != video_id:
                        continue
                    
                    # Check if timestamp falls within segment
                    if (segment.start_time - tolerance <= timestamp <= segment.end_time + tolerance):
                        result = {
                            "segment": segment,
                            "relevance_score": 1.0 - abs(timestamp - segment.start_time) / tolerance,
                            "matched_text": f"Timestamp: {timestamp:.2f}s",
                            "search_type": "timestamp",
                            "highlight_times": [segment.start_time]
                        }
                        results.append(result)
            
            # Sort by relevance (closest to timestamp)
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Timestamp search failed: {e}")
            return []
    
    async def search_by_duration(
        self,
        project: Project,
        min_duration: float,
        max_duration: Optional[float] = None,
        video_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for segments by duration."""
        try:
            results = []
            
            if max_duration is None:
                max_duration = float('inf')
            
            for chapter in project.chapters:
                for segment in chapter.segments:
                    # Skip if video filter is applied
                    if video_id and segment.video_id != video_id:
                        continue
                    
                    # Check duration
                    if min_duration <= segment.duration <= max_duration:
                        result = {
                            "segment": segment,
                            "relevance_score": 0.8,
                            "matched_text": f"Duration: {segment.duration:.2f}s",
                            "search_type": "duration",
                            "highlight_times": [segment.start_time]
                        }
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Duration search failed: {e}")
            return []
    
    async def get_search_suggestions(
        self,
        project: Project,
        partial_query: str,
        limit: int = 10
    ) -> List[str]:
        """Get search suggestions based on partial query."""
        try:
            suggestions = set()
            partial_lower = partial_query.lower()
            
            # Get suggestions from topics
            for topic in project.topics:
                if partial_lower in topic.name.lower():
                    suggestions.add(topic.name)
                for keyword in topic.keywords:
                    if partial_lower in keyword.lower():
                        suggestions.add(keyword)
            
            # Get suggestions from speaker names
            for speaker in project.speakers:
                if speaker.name and partial_lower in speaker.name.lower():
                    suggestions.add(speaker.name)
            
            # Get suggestions from visual tags
            for chapter in project.chapters:
                for segment in chapter.segments:
                    if hasattr(segment, 'visual_tags'):
                        for tag in segment.visual_tags:
                            if partial_lower in tag.lower():
                                suggestions.add(tag)
                    
                    if hasattr(segment, 'shot_type') and segment.shot_type:
                        if partial_lower in segment.shot_type.lower():
                            suggestions.add(segment.shot_type)
            
            # Convert to list and sort
            suggestions_list = list(suggestions)
            suggestions_list.sort()
            
            return suggestions_list[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []
