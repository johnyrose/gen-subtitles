"""
CLI Interface for Subtitle Generator and Translator
-------------------------------------------------
This module provides a command-line interface using Typer for:
1. Generating subtitles from video files
2. Generating subtitles for all videos in a directory
3. Translating subtitle files
4. Translating all subtitle files in a directory
"""

import os
import shutil
import typer
from typing import Optional, List
from enum import Enum
from openai import OpenAI

from generator import (
    generate_subtitles, 
    DEFAULT_VIDEO_PATH, 
    DEFAULT_OUTPUT_PATH as DEFAULT_GEN_OUTPUT_PATH,
    DEFAULT_LANGUAGE as DEFAULT_GEN_LANGUAGE,
    DEFAULT_SUBTITLE_FORMAT,
    SUPPORTED_LANGUAGES as GEN_SUPPORTED_LANGUAGES,
    SUBTITLE_FORMATS
)
from translator import (
    translate_subtitle_file,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_TRANS_OUTPUT_PATH,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_MODEL,
    SUPPORTED_LANGUAGES as TRANS_SUPPORTED_LANGUAGES,
    SUBTITLE_FORMATS as TRANS_SUBTITLE_FORMATS
)
from utils import (
    process_in_parallel,
    scan_directory_for_files,
    get_output_path,
    logger
)

app = typer.Typer(help="Subtitle generator and translator CLI")


class SubtitleFormat(str, Enum):
    """Supported subtitle formats as an Enum for typer"""
    SRT = "srt"
    VTT = "vtt"
    JSON = "json"
    TXT = "txt"


@app.command()
def generate(
    video: str = typer.Argument(..., help="Path to the input video file"),
    language: str = typer.Option(..., "--language", "-l", help="Language of the audio (e.g., 'en' for English, 'fr' for French, 'nl' for Dutch)"),
    output: str = typer.Option(DEFAULT_GEN_OUTPUT_PATH, "--output", "-o", help="Path to save the output subtitle file"),
    format: SubtitleFormat = typer.Option(DEFAULT_SUBTITLE_FORMAT, "--format", "-f", help="Format of the output subtitle file"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="OpenAI API key (if not provided, will use OPENAI_API_KEY environment variable)"),
    limit_chars: bool = typer.Option(True, "--limit-chars/--no-limit-chars", help="Limit subtitle segments to a maximum of characters")
):
    """
    Generate subtitles for a video file using OpenAI Whisper API
    
    Examples:
        python main.py generate video.mp4 -l en -o subtitles.srt -f srt 
        python main.py generate movie.mp4 -l fr -o french_subs.vtt -f vtt
        python main.py generate documentary.mp4 -l nl -o dutch_subs.srt
    """
    
    # Set OpenAI API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        typer.echo("Error: OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        typer.echo("Please provide an API key with --api-key or set the OPENAI_API_KEY environment variable")
        raise typer.Exit(code=1)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Check if video file exists
    if not os.path.exists(video):
        typer.echo(f"Error: Video file not found: {video}")
        raise typer.Exit(code=1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate subtitles
    try:
        generate_subtitles(
            video, 
            output, 
            language, 
            format.value,
            client,
            limit_chars
        )
    except Exception as e:
        typer.echo(f"Error generating subtitles: {e}")
        raise typer.Exit(code=1)


@app.command("generate-dir")
def generate_directory(
    directory: str = typer.Argument(..., help="Directory containing video files to process"),
    language: str = typer.Option(..., "--language", "-l", help="Language of the audio (e.g., 'en' for English, 'fr' for French, 'nl' for Dutch)"),
    extension: str = typer.Option("mp4", "--extension", "-e", help="File extension to look for (default: mp4)"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-od", help="Directory to save subtitle files (default: same directory as videos)"),
    format: SubtitleFormat = typer.Option(DEFAULT_SUBTITLE_FORMAT, "--format", "-f", help="Format of the output subtitle file"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="OpenAI API key (if not provided, will use OPENAI_API_KEY environment variable)"),
    limit_chars: bool = typer.Option(True, "--limit-chars/--no-limit-chars", help="Limit subtitle segments to a maximum of characters"),
    max_workers: int = typer.Option(5, "--max-workers", "-w", help="Maximum number of videos to process in parallel"),
    max_retries: int = typer.Option(3, "--max-retries", "-r", help="Maximum number of retries for failed videos")
):
    """
    Generate subtitles for all video files in a directory using OpenAI Whisper API
    
    Examples:
        python main.py generate-dir /videos -l en -f srt
        python main.py generate-dir /movies -l fr -e mkv -od /subtitles -w 8
        python main.py generate-dir /documentaries -l nl -f vtt -od /nl_subs
    """
    
    # Set OpenAI API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        typer.echo("Error: OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        typer.echo("Please provide an API key with --api-key or set the OPENAI_API_KEY environment variable")
        raise typer.Exit(code=1)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Check if directory exists
    if not os.path.exists(directory):
        typer.echo(f"Error: Directory not found: {directory}")
        raise typer.Exit(code=1)
    
    # Scan directory for video files
    try:
        video_files = scan_directory_for_files(directory, extension)
    except Exception as e:
        typer.echo(f"Error scanning directory: {e}")
        raise typer.Exit(code=1)
    
    if not video_files:
        typer.echo(f"No files with extension .{extension} found in {directory}")
        raise typer.Exit(code=1)
    
    typer.echo(f"Found {len(video_files)} video files to process.")
    
    # Define the function to process a single video file
    def process_video(video_path, client=None, language=None, format=None, limit_chars=None):
        output_path = get_output_path(video_path, output_dir, format)
        logger.info(f"Processing {video_path} -> {output_path}")
        
        try:
            generate_subtitles(
                video_path,
                output_path,
                language,
                format,
                client,
                limit_chars
            )
            return output_path
        except Exception as e:
            # Re-raise the exception for the retry mechanism
            logger.error(f"Error processing {video_path}: {e}")
            raise
    
    # Process all videos in parallel
    typer.echo(f"Starting parallel processing with {max_workers} workers...")
    
    # Prepare arguments for the processing function
    func_args = {
        "client": client,
        "language": language,
        "format": format.value,
        "limit_chars": limit_chars
    }
    
    # Process videos in parallel
    successful_results, failures = process_in_parallel(
        items=video_files,
        process_func=process_video,
        max_workers=max_workers,
        max_retries=max_retries,
        func_args=func_args
    )
    
    # Print summary
    typer.echo("\nProcessing Summary:")
    typer.echo(f"Successfully processed: {len(successful_results)} videos")
    typer.echo(f"Failed to process: {len(failures)} videos")
    
    if failures:
        typer.echo("\nFailed videos:")
        for video, error in failures:
            typer.echo(f"  - {video}: {error}")
    
    if successful_results:
        typer.echo("\nSuccessfully generated subtitle files:")
        for output_path in successful_results:
            typer.echo(f"  - {output_path}")


@app.command()
def translate(
    input: str = typer.Argument(..., help="Path to the input subtitle file"),
    language: str = typer.Option(..., "--language", "-l", help="Target language code (e.g., 'en' for English, 'fr' for French, 'nl' for Dutch)"),
    output: str = typer.Option(DEFAULT_TRANS_OUTPUT_PATH, "--output", "-o", help="Path to save the translated subtitle file"),
    keep_original: bool = typer.Option(False, "--keep-original", "-k", help="Keep original subtitles alongside translations"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="OpenAI model to use for translation"),
    context: bool = typer.Option(True, "--context/--no-context", "-c", help="Include surrounding subtitle context for better translations"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-api", help="OpenAI API key (if not provided, will use OPENAI_API_KEY environment variable)")
):
    """
    Translate subtitle files using OpenAI GPT
    
    Examples:
        python main.py translate subtitles.srt -l fr -o french_subs.srt
        python main.py translate english.srt -l nl -o dutch_subs.srt -k
        python main.py translate original.srt -l es -m gpt-4-turbo
    """
    
    # Set OpenAI API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        typer.echo("Error: OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        typer.echo("Please provide an API key with --api-key or set the OPENAI_API_KEY environment variable")
        raise typer.Exit(code=1)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Check if input file exists
    if not os.path.exists(input):
        typer.echo(f"Error: Input subtitle file not found: {input}")
        raise typer.Exit(code=1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Translate subtitles
    try:
        translate_subtitle_file(
            input,
            output,
            language,
            keep_original,
            client,
            model,
            context
        )
    except Exception as e:
        typer.echo(f"Error translating subtitles: {e}")
        raise typer.Exit(code=1)


@app.command("translate-dir")
def translate_directory(
    directory: str = typer.Argument(..., help="Directory containing subtitle files to translate"),
    language: str = typer.Option(..., "--language", "-l", help="Target language code (e.g., 'en' for English, 'fr' for French, 'nl' for Dutch)"),
    extension: str = typer.Option("srt", "--extension", "-e", help="File extension to look for (default: srt)"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-od", help="Directory to save translated subtitle files (default: same directory as originals)"),
    keep_original: bool = typer.Option(False, "--keep-original", "-k", help="Keep original subtitles alongside translations"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="OpenAI model to use for translation"),
    context: bool = typer.Option(True, "--context/--no-context", "-c", help="Include surrounding subtitle context for better translations"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-api", help="OpenAI API key (if not provided, will use OPENAI_API_KEY environment variable)"),
    max_workers: int = typer.Option(5, "--max-workers", "-w", help="Maximum number of files to process in parallel"),
    max_retries: int = typer.Option(3, "--max-retries", "-r", help="Maximum number of retries for failed files")
):
    """
    Translate all subtitle files in a directory using OpenAI GPT
    
    Examples:
        python main.py translate-dir /subtitles -l fr
        python main.py translate-dir /movies/subs -l nl -e vtt -od /translated_subs -w 8
        python main.py translate-dir /docs -l es -m gpt-4-turbo -k
    """
    
    # Set OpenAI API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        typer.echo("Error: OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        typer.echo("Please provide an API key with --api-key or set the OPENAI_API_KEY environment variable")
        raise typer.Exit(code=1)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Check if directory exists
    if not os.path.exists(directory):
        typer.echo(f"Error: Directory not found: {directory}")
        raise typer.Exit(code=1)
    
    # Scan directory for subtitle files
    try:
        subtitle_files = scan_directory_for_files(directory, extension)
    except Exception as e:
        typer.echo(f"Error scanning directory: {e}")
        raise typer.Exit(code=1)
    
    if not subtitle_files:
        typer.echo(f"No files with extension .{extension} found in {directory}")
        raise typer.Exit(code=1)
    
    typer.echo(f"Found {len(subtitle_files)} subtitle files to translate.")
    
    # Create backup directory for originals
    backup_dir = os.path.join(directory, "original_subtitles")
    os.makedirs(backup_dir, exist_ok=True)
    typer.echo(f"Created backup directory at: {backup_dir}")
    
    # Define the function to process a single subtitle file
    def process_subtitle(subtitle_path, client=None, language=None, model=None, context=None, 
                         keep_original=None, backup_dir=None, output_dir=None):
        # Get the output path
        if output_dir:
            # Create output directory if needed
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(subtitle_path)
            output_path = os.path.join(output_dir, base_name)
        else:
            output_path = subtitle_path
        
        # Create a backup copy of the original
        backup_path = os.path.join(backup_dir, os.path.basename(subtitle_path))
        shutil.copy2(subtitle_path, backup_path)
        logger.info(f"Backed up {subtitle_path} to {backup_path}")
        
        # Translate the subtitle file
        logger.info(f"Translating {subtitle_path} -> {output_path}")
        translate_subtitle_file(
            subtitle_path,
            output_path,
            language,
            keep_original,
            client,
            model,
            context
        )
        
        return output_path
    
    # Process all subtitle files in parallel
    typer.echo(f"Starting parallel translation with {max_workers} workers...")
    
    # Prepare arguments for the processing function
    func_args = {
        "client": client,
        "language": language,
        "model": model,
        "context": context,
        "keep_original": keep_original,
        "backup_dir": backup_dir,
        "output_dir": output_dir
    }
    
    # Process files in parallel
    successful_results, failures = process_in_parallel(
        items=subtitle_files,
        process_func=process_subtitle,
        max_workers=max_workers,
        max_retries=max_retries,
        func_args=func_args
    )
    
    # Print summary
    typer.echo("\nTranslation Summary:")
    typer.echo(f"Successfully translated: {len(successful_results)} files")
    typer.echo(f"Failed to translate: {len(failures)} files")
    
    if failures:
        typer.echo("\nFailed files:")
        for subtitle, error in failures:
            typer.echo(f"  - {subtitle}: {error}")
    
    if successful_results:
        typer.echo("\nSuccessfully translated subtitle files:")
        for output_path in successful_results:
            typer.echo(f"  - {output_path}")