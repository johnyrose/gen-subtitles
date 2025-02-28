"""
Video Subtitles Generator using OpenAI Whisper API
------------------------------------------------
This module contains the core functionality for generating subtitles from video files.
"""

import os
import tempfile
import json
import time
from typing import List, Dict, Any

from openai import OpenAI
from moviepy import VideoFileClip

# Constants
DEFAULT_VIDEO_PATH = "input_video.mp4"
DEFAULT_OUTPUT_PATH = "subtitles.srt"
DEFAULT_LANGUAGE = "en"  # English by default
DEFAULT_SUBTITLE_FORMAT = "srt"
MAX_CHARS_PER_SEGMENT = 40  # Maximum characters per subtitle segment
MAX_FILE_SIZE_MB = 25  # OpenAI's limit for Whisper API

# Supported languages by Whisper
SUPPORTED_LANGUAGES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "mi": "Maori",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh"
}

# Supported subtitle formats
SUBTITLE_FORMATS = ["srt", "vtt", "json", "txt"]


def extract_audio_from_video(video_path: str, output_audio_path: str) -> None:
    """Extract audio from a video file and save it to a specified path"""
    try:
        video = VideoFileClip(video_path)
        print("Extracting audio...")
        video.audio.write_audiofile(output_audio_path, ffmpeg_params=["-ac", "1"])
    except Exception as e:
        raise Exception(f"Error extracting audio from video: {e}")


def split_audio_if_needed(audio_path: str, max_size_mb: int = MAX_FILE_SIZE_MB) -> List[str]:
    """Split audio file into smaller chunks if it exceeds the maximum size"""
    # Check file size
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    
    if file_size_mb <= max_size_mb:
        return [audio_path]
    
    # If file is too large, split it using pydub
    from pydub import AudioSegment
    from pydub.silence import detect_silence
    
    # Create temporary directory for chunks
    temp_dir = tempfile.mkdtemp()
    
    # Load audio file
    print("Audio file is too large. Loading for splitting...")
    audio = AudioSegment.from_file(audio_path)
    
    # Calculate how many chunks we need
    num_chunks = int(file_size_mb / max_size_mb) + 1
    total_duration_ms = len(audio)
    chunk_duration_ms = total_duration_ms // num_chunks
    
    print("Detecting silence for optimal split points...")
    # Try to find silence intervals to use as split points
    silence_intervals = detect_silence(
        audio, 
        min_silence_len=500,
        silence_thresh=-40
    )
    
    # Find ideal split points at silence intervals
    split_points = [0]  # Start with beginning of audio
    
    for i in range(1, num_chunks):
        target_time = i * chunk_duration_ms
        
        # Find silence closest to target time
        closest_silence = None
        min_distance = float('inf')
        
        for start, end in silence_intervals:
            middle = (start + end) // 2
            distance = abs(middle - target_time)
            
            if distance < min_distance:
                min_distance = distance
                closest_silence = middle
        
        # If no silence found near the target time, just use the target time
        split_points.append(closest_silence if closest_silence is not None 
                          and min_distance < chunk_duration_ms * 0.2 
                          else target_time)
    
    split_points.append(total_duration_ms)  # End of audio
    
    # Create chunks
    chunk_paths = []
    for i in range(len(split_points) - 1):
        print(f"Creating chunk {i+1}/{num_chunks}...")
        chunk = audio[split_points[i]:split_points[i+1]]
        
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunk_paths.append(chunk_path)
    
    return chunk_paths


def transcribe_audio_with_whisper(audio_path: str, language: str, client: OpenAI) -> Dict:
    """Transcribe audio file using OpenAI Whisper API"""
    with open(audio_path, "rb") as audio_file:
        try:
            print("Sending to Whisper API...")
            start_time = time.time()
            
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"]
            )
            
            elapsed_time = time.time() - start_time
            print(f"Transcription completed in {elapsed_time:.2f} seconds")
            
            # Convert to dict if needed
            if hasattr(transcription, "model_dump"):
                transcription = transcription.model_dump()
            elif not isinstance(transcription, dict):
                transcription = json.loads(json.dumps(transcription, default=lambda o: o.__dict__))
                
            return transcription
        except Exception as e:
            raise Exception(f"Error transcribing audio with Whisper API: {e}")


def merge_transcriptions(transcriptions: List[Dict]) -> Dict:
    """Merge multiple transcription results into one"""
    if len(transcriptions) == 1:
        return transcriptions[0]
    
    # Make a copy of the first transcription
    merged = transcriptions[0].copy()
    
    # Adjust timestamps for segments after the first chunk
    current_end_time = merged["segments"][-1]["end"]
    
    for trans in transcriptions[1:]:
        for segment in trans["segments"]:
            adjusted_segment = segment.copy()
            adjusted_segment["start"] += current_end_time
            adjusted_segment["end"] += current_end_time
            
            # Adjust word timestamps if available
            if "words" in adjusted_segment:
                for word in adjusted_segment["words"]:
                    word["start"] += current_end_time
                    word["end"] += current_end_time
            
            merged["segments"].append(adjusted_segment)
        
        # Update current end time for next chunk
        current_end_time = merged["segments"][-1]["end"]
    
    # Update text field to include all transcriptions
    merged["text"] = " ".join(t["text"] for t in transcriptions)
    
    return merged


def format_timestamp(seconds: float, format_type: str = "srt") -> str:
    """Format time in seconds to the specified subtitle format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_value = seconds % 60
    milliseconds = int((seconds_value - int(seconds_value)) * 1000)
    
    if format_type == "vtt":
        return f"{hours:02d}:{minutes:02d}:{int(seconds_value):02d}.{milliseconds:03d}"
    else:  # srt format
        return f"{hours:02d}:{minutes:02d}:{int(seconds_value):02d},{milliseconds:03d}"


def split_long_segments(segments: List[Dict[str, Any]], max_chars: int = MAX_CHARS_PER_SEGMENT) -> List[Dict[str, Any]]:
    """Split subtitle segments that have more than the maximum number of characters
    
    This creates multiple sequential subtitle segments with appropriate timings.
    Prioritizes splitting at sentence boundaries, then punctuation, then word boundaries.
    """
    if not segments:
        return []
        
    result = []
    
    for segment in segments:
        text = segment['text'].strip()
        
        # If the segment is already short enough, keep it as is
        if len(text) <= max_chars:
            result.append(segment)
            continue
            
        # Get timing information - Whisper API provides start and end times
        start_time = segment['start']
        end_time = segment['end']
        total_duration = end_time - start_time
        
        # Split the text into chunks of appropriate length
        chunks = []
        remaining_text = text
        
        while remaining_text:
            if len(remaining_text) <= max_chars:
                # Add the remaining text as the final chunk
                chunks.append(remaining_text)
                break
                
            # Find an appropriate split point
            split_point = find_split_point(remaining_text, max_chars)
            
            # Add the chunk and continue with remaining text
            chunks.append(remaining_text[:split_point].strip())
            remaining_text = remaining_text[split_point:].strip()
        
        # Calculate duration for each chunk (proportional to number of characters)
        total_chars = sum(len(chunk) for chunk in chunks)
        chunk_durations = []
        
        for chunk in chunks:
            duration_fraction = len(chunk) / total_chars if total_chars > 0 else 1.0 / len(chunks)
            chunk_durations.append(max(0.5, duration_fraction * total_duration))  # At least 0.5s per segment
        
        # Create new segments with proper timing
        current_start = start_time
        for i, (chunk, duration) in enumerate(zip(chunks, chunk_durations)):
            chunk_end = min(end_time, current_start + duration)  # Ensure we don't exceed the original end time
            
            # Create new segment using Whisper API format
            new_segment = {
                'id': segment['id'] if i == 0 else f"{segment['id']}.{i}",
                'start': current_start,
                'end': chunk_end,
                'text': chunk
            }
            
            # Copy any other fields from original segment
            for key, value in segment.items():
                if key not in ('id', 'start', 'end', 'text'):
                    new_segment[key] = value
            
            result.append(new_segment)
            current_start = chunk_end
    
    # Re-index all segments
    for i, seg in enumerate(result):
        seg['id'] = i
        
    return result


def find_split_point(text: str, max_length: int) -> int:
    """Find the best point to split text, prioritizing sentence and word boundaries"""
    if len(text) <= max_length:
        return len(text)
    
    # Try to find sentence boundaries first (.!?)
    for i in range(max_length, 0, -1):
        if i < len(text) and text[i-1] in ".!?":
            return i
    
    # Try other punctuation next (,:;)
    for i in range(max_length, 0, -1):
        if i < len(text) and text[i-1] in ",:;":
            return i
    
    # Fall back to word boundaries (spaces)
    space_pos = text[:max_length].rfind(' ')
    if space_pos > max_length // 2:  # Only use if reasonably placed
        return space_pos + 1  # Include the space in the first chunk
    
    # If all else fails, just split at max_length
    return max_length


def parse_timestamp(timestamp: str) -> float:
    """Parse a subtitle timestamp string into seconds"""
    # Handle both SRT (00:00:00,000) and VTT (00:00:00.000) formats
    timestamp = timestamp.replace(',', '.').strip()
    
    # Split into components
    parts = timestamp.split(':')
    if len(parts) != 3:
        return 0.0
        
    hours = int(parts[0])
    minutes = int(parts[1])
    
    # Last part may have milliseconds
    seconds_parts = parts[2].split('.')
    seconds = float(seconds_parts[0])
    
    if len(seconds_parts) > 1:
        milliseconds = float(f"0.{seconds_parts[1]}")
        seconds += milliseconds
    
    return hours * 3600 + minutes * 60 + seconds


def create_subtitle_content(transcription: Dict, format_type: str) -> str:
    """Create subtitle content in the specified format"""
    if format_type == "json":
        return json.dumps(transcription, indent=2)
    
    if format_type == "txt":
        return transcription["text"]
    
    subtitle_content = ""
    
    # Add header for WebVTT format
    if format_type == "vtt":
        subtitle_content = "WEBVTT\n\n"
    
    for i, segment in enumerate(transcription["segments"]):
        if format_type in ["srt", "vtt"]:
            # Add index for SRT format (not needed for VTT)
            if format_type == "srt":
                subtitle_content += f"{i+1}\n"
            
            # Add timestamps
            start_time = format_timestamp(segment["start"], format_type)
            end_time = format_timestamp(segment["end"], format_type)
            
            subtitle_content += f"{start_time} --> {end_time}\n"
            
            # Add text
            subtitle_content += f"{segment['text'].strip()}\n\n"
    
    return subtitle_content


def generate_subtitles(
    video_path: str, 
    output_path: str, 
    language: str,
    subtitle_format: str,
    client: OpenAI,
    limit_chars: bool = True
) -> None:
    """Generate subtitles for a video file"""
    # Create a temporary file for the extracted audio
    temp_audio_path = tempfile.mktemp(suffix=".mp3")
    audio_chunks = []
    
    try:
        print(f"Step 1/4: Extracting audio from video: {video_path}")
        extract_audio_from_video(video_path, temp_audio_path)
        
        print(f"Step 2/4: Processing audio")
        audio_chunks = split_audio_if_needed(temp_audio_path)
        
        print(f"Step 3/4: Transcribing audio using Whisper API (language: {language})")
        transcriptions = []
        for i, chunk_path in enumerate(audio_chunks):
            print(f"  Processing chunk {i+1}/{len(audio_chunks)}")
            transcription = transcribe_audio_with_whisper(chunk_path, language, client)
            transcriptions.append(transcription)
        
        print(f"Step 4/4: Creating {subtitle_format.upper()} subtitles")
        
        if len(audio_chunks) > 1:
            print("  Merging transcriptions...")
            merged_transcription = merge_transcriptions(transcriptions)
        else:
            merged_transcription = transcriptions[0]
        
        # Apply character limiting if enabled
        if limit_chars:
            print(f"  Limiting subtitle segments to max {MAX_CHARS_PER_SEGMENT} characters...")
            segments = split_long_segments(merged_transcription["segments"])
            merged_transcription["segments"] = segments
        
        subtitle_content = create_subtitle_content(merged_transcription, subtitle_format)
        
        print(f"Writing subtitles to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(subtitle_content)
        
        print(f"✓ Subtitle generation completed successfully!")
        print(f"Output saved to: {output_path}")
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        # Clean up audio chunks if they were created in a temp directory
        if len(audio_chunks) > 1 and audio_chunks[0] != temp_audio_path:
            temp_dir = os.path.dirname(audio_chunks[0])
            for chunk_path in audio_chunks:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)