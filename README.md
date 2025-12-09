# ClinicalTrials.gov ETL Pipeline

An Extract, Transform, Load (ETL) pipeline for fetching clinical trial data from ClinicalTrials.gov API, transforming it with AI-powered classification using Google Gemini, and loading the results into CSV format.

## Overview

This ETL pipeline follows a three-phase process:

1. **EXTRACT**: Fetches clinical trial data from the ClinicalTrials.gov API v2
2. **TRANSFORM**: Converts raw API responses into structured data and optionally applies AI-powered classification
3. **LOAD**: Saves the transformed data to a CSV file

## Features

- **Extract Phase**:
  - Fetches clinical trials from ClinicalTrials.gov API
  - Handles pagination automatically
  - Configurable API filters via YAML configuration

- **Transform Phase**:
  - Converts raw API JSON to structured format
  - Optional AI-powered classification using Google Gemini
  - Configurable transformation rules

- **Load Phase**:
  - Exports results to CSV
  - Configurable output filename
  - Includes all extracted fields plus AI-determined values

## Requirements

- Python 3.12+
- Virtual environment (recommended)

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Gemini API key (required for AI features):
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Configuration

The pipeline uses a YAML configuration file (`config.yaml`). Copy the example file to get started:

```bash
cp config.example.yaml config.yaml
```

### Configuration Sections

#### `ctgov` - API Configuration
- `api_url`: ClinicalTrials.gov API endpoint
- `page_size`: Number of studies per page (default: 1000)
- `filter_advanced`: Array of filter conditions (joined with `AND`)

#### `gemini` - AI Configuration
- `api_key_env`: Environment variable name for API key (default: `GEMINI_API_KEY`)
- `model`: Gemini model to use (default: `gemini-2.5-flash`)
- `api_delay`: Delay between API calls in seconds (default: 0.5)
- `system_instruction`: System-level instructions for AI classification (multiline YAML)
- `row_prompt_template`: Template for per-row AI prompts (multiline YAML with placeholders)

#### `ai_processing` - AI Processing Options
- `enabled`: Enable/disable AI processing (default: `true`)
- `column_name`: Name of the AI-determined column (default: `ai_determined_value`)
- `max_rows`: Limit number of rows processed by AI (set to `null` for all)
- `debug_only_tuning_trials`: Process only trials listed in `tuning_trials` (default: `false`)

#### `output` - Output Configuration
- `csv_filename`: Output CSV filename (default: `clinical_trials_filtered.csv`)

#### `tuning_trials` - Trial IDs for Testing
- List of NCT IDs to use for testing/debugging AI prompts

## Usage

Run the ETL pipeline:

```bash
python3 etl.py
```

Or specify a custom config file:

```bash
python3 etl.py --config my_config.yaml
```

### ETL Process Flow

1. **Extract**: The pipeline fetches all studies matching your filter criteria from the API
2. **Transform**: Each study is converted to structured format, and optionally processed with AI
3. **Load**: All transformed studies are saved to the configured CSV file

## Output

The pipeline generates a CSV file with the following columns:

- `nct_id`: Clinical trial identifier
- `brief_title`: Brief title of the study
- `official_title`: Official title of the study
- `overall_status`: Current status of the trial
- `study_type`: Type of study (e.g., INTERVENTIONAL)
- `start_date`: Study start date
- `start_year`: Year extracted from start date
- `gender`: Participant gender criteria
- `brief_summary`: Brief summary of the study
- `detailed_description`: Detailed description
- `criteria`: Eligibility criteria
- `ai_determined_value`: AI classification result (if AI processing is enabled)

The AI column name can be customized via the `ai_processing.column_name` configuration option.

## Architecture

### Extract Functions
- `extract_clinical_trials()`: Fetches data from ClinicalTrials.gov API

### Transform Functions
- `transform_study_data()`: Converts raw study data to structured format
- `transform_studies_with_ai()`: Applies AI-powered classification to studies
- `process_study_with_ai()`: Processes a single study with AI
- `initialize_gemini()`: Initializes Gemini API client
- `get_gemini_response()`: Gets response from Gemini API

### Load Functions
- `load_to_csv()`: Saves transformed data to CSV file

## Notes

- The pipeline uses Gemini system instructions for efficiency (context set once per model initialization)
- Rate limiting is built-in to respect API limits
- The pipeline handles pagination automatically
- YAML configuration allows for easy editing of prompts and filters with proper multiline support
- The `filter_advanced` configuration uses an array format for better readability and maintainability

## Example Configuration

See `config.example.yaml` for a complete example configuration file with detailed comments.
