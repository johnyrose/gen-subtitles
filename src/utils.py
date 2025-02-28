"""
Utility functions for subtitle processing
----------------------------------------
This module provides utilities for parallel processing and other common operations.
"""

import os
import time
import concurrent.futures
import logging
from typing import List, Callable, TypeVar, Dict, Any, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')
R = TypeVar('R')


def process_in_parallel(
    items: List[T],
    process_func: Callable[[T, Dict[str, Any]], R],
    max_workers: int = 5,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    func_args: Dict[str, Any] = None
) -> Tuple[List[R], List[Tuple[T, Exception]]]:
    """
    Process a list of items in parallel with automatic retries.
    
    Args:
        items: List of items to process
        process_func: Function to call for each item
        max_workers: Maximum number of parallel workers
        max_retries: Maximum number of retries per item
        retry_delay: Delay in seconds between retries
        func_args: Additional arguments to pass to process_func
    
    Returns:
        Tuple containing:
        - List of successful results
        - List of failed items with their exceptions
    """
    if func_args is None:
        func_args = {}
    
    results = []
    failures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a map of future to (item, attempt_number)
        future_to_item = {}
        
        # Submit all items for processing
        for item in items:
            future = executor.submit(
                _process_with_retry,
                item=item,
                process_func=process_func,
                max_retries=max_retries,
                retry_delay=retry_delay,
                func_args=func_args
            )
            future_to_item[future] = item
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result, success = future.result()
                if success:
                    results.append(result)
                    logger.info(f"Successfully processed: {item}")
                else:
                    failures.append((item, result))
                    logger.error(f"Failed to process after {max_retries} attempts: {item}. Error: {result}")
            except Exception as exc:
                failures.append((item, exc))
                logger.error(f"Exception processing item: {item}. Error: {exc}")

    return results, failures


def _process_with_retry(
    item: T, 
    process_func: Callable, 
    max_retries: int, 
    retry_delay: float,
    func_args: Dict[str, Any]
) -> Tuple[Any, bool]:
    """
    Process an item with automatic retries.
    
    Args:
        item: The item to process
        process_func: The function to call for processing
        max_retries: Maximum number of retries
        retry_delay: Delay in seconds between retries
        func_args: Additional arguments to pass to process_func
    
    Returns:
        Tuple containing:
        - Result or exception
        - Success flag (True if successful, False if failed)
    """
    for attempt in range(max_retries):
        try:
            # Call the processing function with the item and additional arguments
            result = process_func(item, **func_args)
            return result, True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {item}: {str(e)}")
            if attempt < max_retries - 1:
                # Wait before retrying
                time.sleep(retry_delay)
            else:
                # Return the last exception if all retries failed
                return e, False


def scan_directory_for_files(directory: str, extension: str) -> List[str]:
    """
    Scan a directory for files with a specific extension.
    
    Args:
        directory: Directory to scan
        extension: File extension to look for (without the dot)
    
    Returns:
        List of full paths to matching files
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")
    
    if not os.path.isdir(directory):
        raise ValueError(f"Not a directory: {directory}")
    
    # Normalize extension format (ensure it doesn't have a leading dot)
    if extension.startswith('.'):
        extension = extension[1:]
    
    matching_files = []
    
    # Walk through directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(f".{extension.lower()}"):
                full_path = os.path.join(root, file)
                matching_files.append(full_path)
    
    return matching_files


def get_output_path(video_path: str, output_dir: str, subtitle_format: str) -> str:
    """
    Generate the output path for a subtitle file based on the video path.
    
    Args:
        video_path: Path to the video file
        output_dir: Output directory (or None to use the same directory as the video)
        subtitle_format: Subtitle format extension (srt, vtt, etc.)
    
    Returns:
        Full path to the output subtitle file
    """
    # Get the video filename without extension
    video_basename = os.path.basename(video_path)
    video_name = os.path.splitext(video_basename)[0]
    
    # Create the subtitle filename
    subtitle_filename = f"{video_name}.{subtitle_format}"
    
    # Determine the output directory
    if output_dir:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, subtitle_filename)
    else:
        # Use the same directory as the video
        return os.path.join(os.path.dirname(video_path), subtitle_filename)