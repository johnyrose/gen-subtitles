# Gen-Subtitles

A command-line tool for generating and translating subtitles using OpenAI's APIs. Generate accurate subtitles for your videos with Whisper API and translate them to any language with GPT models, all from your terminal.

## Features

- **Generate Subtitles** from video files using OpenAI's Whisper API
- **Translate Subtitles** to any language using OpenAI's GPT models
- **Batch Processing** of entire directories of videos or subtitle files
- **Parallel Processing** for faster generation and translation
- **Automatic Retries** for handling API transient errors
- **Backup Original Subtitles** during translation
- **Multiple Format Support** (SRT, VTT, JSON, TXT)
- **Context-Aware Translation** preserves dialogue flow and meaning

## Installation

### Prerequisites

- Python 3.8 or newer
- An OpenAI API key with access to Whisper and GPT models

### Install from PyPI

```bash
pip install gen-subtitles
```

### Install from Source

```bash
git clone https://github.com/yourusername/gen-subtitles.git
cd gen-subtitles
pip install -e .
```

## Quick Start

### OpenAI API Key

To use the tool, you need the `OPENAI_API_KEY` environment variable set. You can set it as an environment variable instead of passing it with each command:

On Linux or MacOS:

```bash
export OPENAI_API_KEY=your-api-key-here
```

On Windows:

```bash
set OPENAI_API_KEY=your-api-key-here
```

### Generate Subtitles for a Video

```bash
gen-subtitles generate video.mp4 -l en
```

### Translate Subtitles to Another Language

```bash
gen-subtitles translate subtitles.srt -l fr -o french_subtitles.srt
```

### Process All Videos in a Directory

```bash
gen-subtitles generate-dir /path/to/videos -l en
```

### Translate All Subtitle Files in a Directory

```bash
gen-subtitles translate-dir /path/to/subtitles -l es
```

## Command Reference

### Generate Command

Generate subtitles for a single video file.

```
gen-subtitles generate VIDEO -l LANGUAGE [OPTIONS]
```

**Arguments:**
- `VIDEO`: Path to the input video file

**Required Options:**
- `-l, --language TEXT`: Language of the audio (e.g., 'en' for English, 'fr' for French, 'nl' for Dutch)

**Options:**
- `-o, --output TEXT`: Path to save the output subtitle file (default: filename.srt)
- `-f, --format [srt|vtt|json|txt]`: Format of the output subtitle file (default: srt)
- `-k, --api-key TEXT`: OpenAI API key (if not set in environment variable)
- `--limit-chars / --no-limit-chars`: Limit subtitle segments to a maximum of characters (default: enabled)

**Example:**
```bash
gen-subtitles generate documentary.mp4 -l en -o documentary_subs.srt -f vtt
```

### Generate-Dir Command

Generate subtitles for all video files in a directory.

```
gen-subtitles generate-dir DIRECTORY -l LANGUAGE [OPTIONS]
```

**Arguments:**
- `DIRECTORY`: Directory containing video files to process

**Required Options:**
- `-l, --language TEXT`: Language of the audio (e.g., 'en' for English, 'fr' for French, 'nl' for Dutch)

**Options:**
- `-e, --extension TEXT`: File extension to look for (default: mp4)
- `-od, --output-dir TEXT`: Directory to save subtitle files (default: same as videos)
- `-f, --format [srt|vtt|json|txt]`: Format of the output subtitle file (default: srt)
- `-k, --api-key TEXT`: OpenAI API key (if not set in environment variable)
- `-w, --max-workers INTEGER`: Maximum number of videos to process in parallel (default: 5)
- `-r, --max-retries INTEGER`: Maximum number of retries for failed videos (default: 3)
- `--limit-chars / --no-limit-chars`: Limit subtitle segments to a maximum of characters (default: enabled)

**Example:**
```bash
gen-subtitles generate-dir /path/to/lectures -l fr -e mkv -od /path/to/subtitles -w 8
```

### Translate Command

Translate a subtitle file to another language.

```
gen-subtitles translate INPUT -l LANGUAGE [OPTIONS]
```

**Arguments:**
- `INPUT`: Path to the input subtitle file

**Required Options:**
- `-l, --language TEXT`: Target language code (e.g., 'en' for English, 'fr' for French, 'nl' for Dutch)

**Options:**
- `-o, --output TEXT`: Path to save the translated subtitle file (default: translated.srt)
- `-k, --keep-original`: Keep original subtitles alongside translations (default: disabled)
- `-m, --model TEXT`: OpenAI model to use for translation (default: gpt-4o-mini)
- `-c, --context / --no-context`: Include surrounding subtitle context for better translations (default: enabled)
- `-api, --api-key TEXT`: OpenAI API key (if not set in environment variable)

**Example:**
```bash
gen-subtitles translate english_subtitles.srt -l de -o german_subtitles.srt -k -m gpt-4
```

### Translate-Dir Command

Translate all subtitle files in a directory.

```
gen-subtitles translate-dir DIRECTORY -l LANGUAGE [OPTIONS]
```

**Arguments:**
- `DIRECTORY`: Directory containing subtitle files to translate

**Required Options:**
- `-l, --language TEXT`: Target language code (e.g., 'en' for English, 'fr' for French, 'nl' for Dutch)

**Options:**
- `-e, --extension TEXT`: File extension to look for (default: srt)
- `-od, --output-dir TEXT`: Directory to save translated subtitle files (default: same as originals)
- `-k, --keep-original`: Keep original subtitles alongside translations (default: disabled)
- `-m, --model TEXT`: OpenAI model to use for translation (default: gpt-4o-mini)
- `-c, --context / --no-context`: Include surrounding subtitle context for better translations (default: enabled)
- `-api, --api-key TEXT`: OpenAI API key (if not set in environment variable)
- `-w, --max-workers INTEGER`: Maximum number of files to process in parallel (default: 5)
- `-r, --max-retries INTEGER`: Maximum number of retries for failed files (default: 3)

**Example:**
```bash
gen-subtitles translate-dir /path/to/english_subs -l ja -e vtt -od /path/to/japanese_subs -k -w 3
```

## Advanced Features

### Parallel Processing

Both directory commands (`generate-dir` and `translate-dir`) use parallel processing to speed up operations. You can control the degree of parallelism with the `-w, --max-workers` option.

### Automatic Retries

The tool automatically retries failed API operations to handle transient errors. Configure retries with the `-r, --max-retries` option.

### Subtitle Format Support

The tool supports multiple subtitle formats:
- **SRT**: Standard subtitle format supported by most players
- **VTT**: Web Video Text Tracks format, used for HTML5 video
- **JSON**: Machine-readable format with detailed timestamp data
- **TXT**: Plain text transcription without timing information

### Context-Aware Translation

When translating subtitles, the tool can consider surrounding subtitles for better context-aware translations. This produces more coherent and natural translations, especially for dialogues and multi-part sentences.

### Backup of Original Subtitle Files

When using the `translate-dir` command, the tool automatically creates an `original_subtitles` directory and backs up all original files before processing.

## Environment Variables

You can set the OpenAI API key as an environment variable instead of passing it with each command:

```bash
export OPENAI_API_KEY=your-api-key-here
```
