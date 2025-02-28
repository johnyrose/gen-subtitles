"""
Subtitle Translator using OpenAI GPT
-----------------------------------
This module contains the core functionality for translating subtitle files.
"""

import os
import re
import json
import time
from typing import List, Dict, Any

from openai import OpenAI
from pydantic import BaseModel, Field


# Constants
DEFAULT_INPUT_PATH = "subtitles.srt"
DEFAULT_OUTPUT_PATH = "translated.srt"
DEFAULT_TARGET_LANGUAGE = "es"  # Spanish by default
DEFAULT_MODEL = "gpt-4o-mini"

# System Prompt - Instructions for the model
TRANSLATION_PROMPT = """
You are a subtitle translation expert with deep knowledge of languages, cultures, and entertainment media.

Your task is to translate subtitles while carefully preserving:
1. The original meaning and intent
2. Cultural references, jokes, wordplay, and idioms (adapt them appropriately for the target language)
3. The tone and style of speech (formal, casual, slang, etc.)
4. Character voice and personality
5. Emotional content and subtext

Guidelines:
- Translate naturally, not literally. Prioritize how a native speaker would express the same idea.
- Preserve humor, sarcasm, and cultural references when possible.
- If a direct translation would lose meaning, adapt to an equivalent in the target language.
- Keep translations concise and suitable for reading as subtitles.
- Maintain any formatting like italics, bold, etc.

IMPORTANT: Translate each subtitle segment independently while keeping context from surrounding segments.
"""

# Supported subtitle formats and their file extensions
SUBTITLE_FORMATS = {
    "srt": ".srt",
    "vtt": ".vtt",
    "txt": ".txt"
}

# Languages supported by OpenAI models
SUPPORTED_LANGUAGES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "zh": "Chinese"
}

# Pydantic models for structured outputs
class TranslatedSegment(BaseModel):
    """Model for a translated subtitle segment"""
    translation: str = Field(description="The translated text of the subtitle segment")


# Helper functions for subtitle parsing
def detect_subtitle_format(file_path: str) -> str:
    """Detect subtitle format based on file extension"""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    for format_name, format_ext in SUBTITLE_FORMATS.items():
        if ext == format_ext:
            return format_name
    
    # Default to SRT if unknown
    return "srt"


def parse_srt_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse an SRT subtitle file into a list of subtitle segments"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split file into subtitle blocks
    subtitle_blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []
    
    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        try:
            index = int(lines[0])
            timing = lines[1]
            text = '\n'.join(lines[2:])
            
            subtitles.append({
                'index': index,
                'timing': timing,
                'text': text
            })
        except ValueError:
            continue
    
    return subtitles


def parse_vtt_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a VTT subtitle file into a list of subtitle segments"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove WebVTT header
    if content.startswith('WEBVTT'):
        content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
    
    # Split file into subtitle blocks
    subtitle_blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []
    index = 1
    
    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        
        # Check if the first line is a cue identifier
        timing_line = 0
        if '-->' not in lines[0]:
            timing_line = 1
            
        if timing_line >= len(lines):
            continue
            
        if '-->' in lines[timing_line]:
            timing = lines[timing_line]
            text = '\n'.join(lines[timing_line + 1:])
            
            subtitles.append({
                'index': index,
                'timing': timing,
                'text': text
            })
            index += 1
    
    return subtitles


def parse_subtitle_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a subtitle file based on its format"""
    format_type = detect_subtitle_format(file_path)
    
    if format_type == "srt":
        return parse_srt_file(file_path)
    elif format_type == "vtt":
        return parse_vtt_file(file_path)
    elif format_type == "txt":
        # For plain text files, just read lines and create basic subtitles
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        subtitles = []
        for i, line in enumerate(lines):
            if line.strip():
                subtitles.append({
                    'index': i + 1,
                    'timing': f"00:00:{i*5:02d},000 --> 00:00:{(i+1)*5-1:02d},999",
                    'text': line.strip()
                })
        
        return subtitles
    else:
        raise ValueError(f"Unsupported subtitle format: {format_type}")


def translate_subtitle_segment(
    segment: Dict[str, Any],
    target_language: str,
    keep_original: bool,
    client: OpenAI,
    model: str,
    context_text: str = ""
) -> Dict[str, Any]:
    """Translate a single subtitle segment using OpenAI API with structured output"""
    
    text_to_translate = segment['text']
    language_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
    try:
        prompt = f"Translate the following subtitle to {language_name}:\n\n{text_to_translate}"
        if context_text:
            prompt = (
                f"This is the context for the translation (DO NOT translate this part, it's just for reference):\n"
                f"```\n{context_text}\n```\n\n"
                f"Translate ONLY the following subtitle to {language_name}:\n\n{text_to_translate}"
            )
        
        messages = [
            {"role": "system", "content": TRANSLATION_PROMPT + "\nIMPORTANT: Respond with a JSON object that has a 'translation' field containing your translation."},
            {"role": "user", "content": prompt}
        ]
        
        # Using structured outputs for reliable formatting
        # Use a simpler approach - just JSON response format instead of structured schema
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3,  # Lower temperature for more consistent translations
        )
        
        # Parse the JSON response
        response_text = response.choices[0].message.content
        translation_data = json.loads(response_text)
        translated_text = translation_data.get("translation", "")
        
        if keep_original:
            # Combine original and translated text
            combined_text = f"{text_to_translate}\n{translated_text}"
            return {**segment, 'text': combined_text}
        else:
            # Replace with translated text only
            return {**segment, 'text': translated_text}
            
    except Exception as e:
        print(f"Error translating segment: {e}")
        # Return original text if translation fails
        return segment


def get_context_for_segment(subtitles: List[Dict[str, Any]], current_index: int, context_size: int = 600) -> str:
    """Get surrounding subtitle context for a segment, limited to context_size characters"""
    all_text = []
    current_text = subtitles[current_index]['text']
    total_chars = len(current_text)
    
    # Add text before current segment (going backwards)
    chars_before = context_size // 2
    i = current_index - 1
    before_segments = []
    
    while i >= 0 and total_chars < chars_before:
        segment_text = subtitles[i]['text']
        before_segments.insert(0, segment_text)
        total_chars += len(segment_text)
        i -= 1
    
    # Add text after current segment (going forwards)
    chars_after = context_size // 2
    i = current_index + 1
    after_segments = []
    
    while i < len(subtitles) and total_chars < chars_before + chars_after:
        segment_text = subtitles[i]['text']
        after_segments.append(segment_text)
        total_chars += len(segment_text)
        i += 1
    
    # Create context with clear markers
    context = []
    
    if before_segments:
        context.append("PREVIOUS SUBTITLES:")
        context.extend(before_segments)
    
    context.append("\nCURRENT SUBTITLE TO TRANSLATE:")
    context.append(current_text)
    
    if after_segments:
        context.append("\nFOLLOWING SUBTITLES:")
        context.extend(after_segments)
    
    return "\n".join(context)


def batch_translate_subtitles(
    subtitles: List[Dict[str, Any]],
    target_language: str,
    keep_original: bool,
    client: OpenAI,
    model: str,
    batch_size: int = 10,  # Process in batches to avoid rate limits
    with_context: bool = True
) -> List[Dict[str, Any]]:
    """Translate subtitles in batches"""
    translated_subtitles = []
    total_segments = len(subtitles)
    
    for i in range(0, total_segments, batch_size):
        batch = subtitles[i:i+batch_size]
        print(f"Translating batch {i//batch_size + 1}/{(total_segments + batch_size - 1)//batch_size} "
              f"({i+1}-{min(i+batch_size, total_segments)}/{total_segments})...")
        
        for j, segment in enumerate(batch):
            segment_index = i + j
            
            # Get context if enabled
            context_text = ""
            if with_context:
                context_text = get_context_for_segment(subtitles, segment_index)
            
            translated_segment = translate_subtitle_segment(
                segment, target_language, keep_original, client, model, context_text
            )
            translated_subtitles.append(translated_segment)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.2)
    
    return translated_subtitles


def write_srt_file(subtitles: List[Dict[str, Any]], output_path: str) -> None:
    """Write subtitles to an SRT file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for subtitle in subtitles:
            f.write(f"{subtitle['index']}\n")
            f.write(f"{subtitle['timing']}\n")
            f.write(f"{subtitle['text']}\n\n")


def write_vtt_file(subtitles: List[Dict[str, Any]], output_path: str) -> None:
    """Write subtitles to a VTT file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        
        for subtitle in subtitles:
            # VTT cue identifiers are optional
            f.write(f"{subtitle['timing']}\n")
            f.write(f"{subtitle['text']}\n\n")


def write_subtitle_file(subtitles: List[Dict[str, Any]], output_path: str) -> None:
    """Write subtitles to a file based on its format"""
    format_type = detect_subtitle_format(output_path)
    
    if format_type == "srt":
        write_srt_file(subtitles, output_path)
    elif format_type == "vtt":
        write_vtt_file(subtitles, output_path)
    elif format_type == "txt":
        # For plain text files, just write the text content
        with open(output_path, 'w', encoding='utf-8') as f:
            for subtitle in subtitles:
                f.write(f"{subtitle['text']}\n")
    else:
        raise ValueError(f"Unsupported output format: {format_type}")


def translate_subtitle_file(
    input_path: str,
    output_path: str,
    target_language: str,
    keep_original: bool,
    client: OpenAI,
    model: str,
    with_context: bool = True
) -> None:
    """Translate a subtitle file and save the translated version"""
    
    print(f"Step 1/3: Parsing subtitle file: {input_path}")
    subtitles = parse_subtitle_file(input_path)
    
    print(f"Step 2/3: Translating subtitles to {SUPPORTED_LANGUAGES.get(target_language, target_language)}")
    if with_context:
        print("  Using surrounding context for improved translation quality")
    
    translated_subtitles = batch_translate_subtitles(
        subtitles=subtitles,
        target_language=target_language,
        keep_original=keep_original,
        client=client,
        model=model,
        with_context=with_context
    )
    
    print(f"Step 3/3: Writing translated subtitles to: {output_path}")
    write_subtitle_file(translated_subtitles, output_path)
    
    print(f"✓ Subtitle translation completed successfully!")
    print(f"Output saved to: {output_path}")
