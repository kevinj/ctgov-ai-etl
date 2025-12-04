# Clinical Trials API Script

A Python script to fetch, filter, and analyze clinical trial data from ClinicalTrials.gov API with AI-powered classification using Google Gemini.

## Features

- Fetches clinical trials from ClinicalTrials.gov API
- Filters trials based on:
  - Study type (Interventional)
  - Date range (2013-2024)
  - Location (Canada)
  - Age (18-64 years)
  - Sex (Female or All)
  - Phase (Early Phase 1 through Phase 4)
  - Keywords (pregnancy-related)
- AI-powered classification using Google Gemini API
- Exports results to CSV

## Requirements

- Python 3.12+
- Virtual environment (recommended)

## Setup

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

## Usage

Run the script:
```bash
./clinical_trials_api.py
```

Or with Python:
```bash
python3 clinical_trials_api.py
```

The script will:
1. Fetch clinical trials matching the filter criteria
2. Process studies with AI classification (if enabled)
3. Save results to `clinical_trials_filtered.csv`

## Configuration

Edit the following variables in `clinical_trials_api.py`:

- `GEMINI_MODEL`: Gemini model to use (default: `gemini-2.5-flash`)
- `GENERAL_CONTEXT`: System instructions for AI classification
- `ROW_PROMPT_TEMPLATE`: Template for per-row AI prompts
- `MAX_AI_ROWS`: Limit number of rows processed by AI (set to `None` for all)
- `ENABLE_AI_PROCESSING`: Enable/disable AI features
- `API_FILTER_ADVANCED`: Advanced API filter string

## Output

The script generates a CSV file (`clinical_trials_filtered.csv`) with the following columns:
- NCT_ID
- Brief Title
- Study Type
- Overall Status
- Start Date
- Completion Date
- Minimum Age
- Maximum Age
- Sex
- Phase
- Enrollment
- Brief Summary
- Detailed Description
- Eligibility Criteria
- Locations
- Interventions
- AI-determined value (if AI processing is enabled)

## Notes

- The script uses system instructions for Gemini, which is more efficient than sending context with each request
- Rate limiting is built-in to respect API limits
- The script handles pagination automatically

