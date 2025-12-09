#!/usr/bin/env python3
"""
ClinicalTrials.gov ETL Script with Gemini AI Integration
Fetches clinical trials from ClinicalTrials.gov API, processes them with Gemini AI,
and outputs a CSV with specified columns.

Configuration is loaded from a JSON or YAML file (default: config.yaml).
"""

import requests
import json
import csv
import sys
import os
import time
import argparse
from typing import Dict, Any, Optional, List
from datetime import datetime
from urllib.parse import urlencode

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Warning: google-generativeai not available. AI features will be disabled.")


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file (auto-detected by extension).
    
    Args:
        config_path: Path to configuration file (.json or .yaml/.yml)
        
    Returns:
        Dict containing configuration
    """
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print(f"   Please create a config file based on config.example.yaml or config.example.json")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith(('.yaml', '.yml')):
                if not YAML_AVAILABLE:
                    print("❌ YAML support not available. Install with: pip install pyyaml")
                    sys.exit(1)
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        print(f"✅ Loaded configuration from {config_path}")
        return config
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)


# Global variable to store configuration (loaded in main)
CONFIG: Dict[str, Any] = {}

# Global variable to store the actual model being used
_ACTUAL_GEMINI_MODEL = None

# ============================================================================
# EXTRACT - Functions that fetch data from external sources
# ============================================================================

def extract_clinical_trials() -> Optional[Dict[Any, Any]]:
    """
    Extract all clinical trial data from ClinicalTrials.gov API using /studies endpoint.
    
    Returns:
        Optional[Dict[Any, Any]]: The JSON response data or None if failed
    """
    ctgov_config = CONFIG.get('ctgov', {})
    # Construct the API URL for the /studies endpoint with pagination
    api_url = ctgov_config.get('api_url', 'https://clinicaltrials.gov/api/v2/studies')
    
    # Handle filter_advanced as either array or string (for backward compatibility)
    filter_advanced = ctgov_config.get('filter_advanced', '')
    if isinstance(filter_advanced, list):
        filter_advanced = ' AND '.join(filter_advanced)
    
    params = {
        'pageSize': ctgov_config.get('page_size', 1000),
        'filter.advanced': filter_advanced
    }
    
    all_studies = []
    page_count = 0
    
    try:
        print("Extracting clinical trials from API...")
        print(f"API URL: {api_url}")
        print("-" * 50)
        
        while True:
            page_count += 1
            print(f"Fetching page {page_count}...")
            print(f"URL: {api_url}?{urlencode(params)}")
            
            # Send GET request to the API
            response = requests.get(api_url, params=params, timeout=60)
            
            # Check if the request was successful
            if response.status_code != 200:
                print(f"❌ Failed to retrieve data: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return None
            
            data = response.json()
            studies = data.get('studies', [])
            all_studies.extend(studies)
            
            print(f"  Retrieved {len(studies)} studies (total: {len(all_studies)})")
            
            # Debug: show response structure and total count if available
            if page_count == 1:
                print(f"  Response keys: {list(data.keys())}")
                # Check if API provides total count
                if 'totalCount' in data:
                    print(f"  Total studies matching filters: {data.get('totalCount')}")
                if 'nextPageToken' in data:
                    print(f"  Next page token: {data.get('nextPageToken')}")
                else:
                    print("  No nextPageToken in response (this is the only/last page)")
            
            # Check for next page
            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                print("  No more pages available")
                break
                
            params['pageToken'] = next_page_token
            print(f"  Next page token: {next_page_token}")
            
            # Safety limit to prevent infinite loops
            if page_count > 100:
                print("⚠️  Reached maximum page limit (100 pages)")
                break
        
        print(f"✅ Successfully extracted {len(all_studies)} total studies")
        return {'studies': all_studies}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error occurred: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON response: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


# ============================================================================
# TRANSFORM - Functions that transform and process data
# ============================================================================

def transform_study_data(study: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Transform raw study data into structured format for CSV output.
    
    Args:
        study (Dict[Any, Any]): Raw study data from API
        
    Returns:
        Dict[str, Any]: Transformed study data
    """
    protocol_section = study.get('protocolSection', {})
    
    # Basic identification
    identification = protocol_section.get('identificationModule', {})
    nct_id = identification.get('nctId', 'N/A')
    brief_title = identification.get('briefTitle', 'N/A')
    official_title = identification.get('officialTitle', 'N/A')
    
    # Status
    status = protocol_section.get('statusModule', {})
    overall_status = status.get('overallStatus', 'N/A')
    start_date_struct = status.get('startDateStruct', {})
    start_date = start_date_struct.get('date', 'N/A')
    start_year = start_date.split('-')[0] if start_date != 'N/A' and '-' in start_date else 'N/A'
    
    # Design and enrollment
    design = protocol_section.get('designModule', {})
    study_type = design.get('studyType', 'N/A')
    
    # Eligibility
    eligibility = protocol_section.get('eligibilityModule', {})
    gender = eligibility.get('sex', 'N/A')
    criteria_text = eligibility.get('eligibilityCriteria', 'N/A')
    
    # Description
    description = protocol_section.get('descriptionModule', {})
    brief_summary = description.get('briefSummary', 'N/A')
    detailed_description = description.get('detailedDescription', 'N/A')
    
    return {
        'nct_id': nct_id,
        'brief_title': brief_title,
        'official_title': official_title,
        'overall_status': overall_status,
        'study_type': study_type,
        'start_date': start_date,
        'gender': gender,
        'brief_summary': brief_summary,
        'detailed_description': detailed_description,
        'criteria': criteria_text,
        'start_year': start_year
    }


# ============================================================================
# GEMINI AI FUNCTIONS (for transformation)
# ============================================================================

def initialize_gemini() -> Optional[genai.GenerativeModel]:
    """
    Initialize Gemini API client with system instruction.
    
    Returns:
        Optional[genai.GenerativeModel]: Initialized model with system instruction, or None if failed
    """
    global _ACTUAL_GEMINI_MODEL
    
    if not GEMINI_AVAILABLE:
        print("❌ Gemini library not available")
        return None
    
    gemini_config = CONFIG.get('gemini', {})
    api_key_env = gemini_config.get('api_key_env', 'GEMINI_API_KEY')
    gemini_api_key = os.getenv(api_key_env, '')
    
    if not gemini_api_key:
        print(f"❌ {api_key_env} environment variable not set")
        print(f"   Set it with: export {api_key_env}='your-api-key'")
        return None
    
    try:
        genai.configure(api_key=gemini_api_key)
        # Create model with system instruction (context set once)
        model_name = gemini_config.get('model')
        system_instruction = gemini_config.get('system_instruction', '')
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction
        )
        print(f"✅ Gemini API initialized (model: {model_name})")
        print(f"   System instruction set (context loaded once)")
        _ACTUAL_GEMINI_MODEL = model_name
        return model
    except Exception as e:
        print(f"❌ Failed to initialize Gemini API: {e}")
        return None


def get_gemini_response(model: genai.GenerativeModel, row_prompt: str) -> Optional[str]:
    """
    Get response from Gemini API for a single row.
    
    Args:
        model (genai.GenerativeModel): Initialized model with system instruction
        row_prompt (str): Row-specific prompt (context already set via system instruction)
        
    Returns:
        Optional[str]: AI response or None if failed
    """
    if not GEMINI_AVAILABLE:
        return None
    
    try:
        # Model already has system instruction (context), just send the row prompt
        response = model.generate_content(row_prompt)
        
        # Extract text from response
        if response and response.text:
            return response.text.strip()
        else:
            return None
            
    except Exception as e:
        print(f"⚠️  Gemini API error: {e}")
        return None


def process_study_with_ai(model: genai.GenerativeModel, study_data: Dict[str, Any]) -> Optional[str]:
    """
    Process a single study with AI to determine a column value.
    
    Args:
        model (genai.GenerativeModel): Initialized model with system instruction
        study_data (Dict[str, Any]): Study data dictionary
        
    Returns:
        Optional[str]: AI-determined value or None if failed
    """
    # Format the row prompt with study data
    gemini_config = CONFIG.get('gemini', {})
    row_prompt_template = gemini_config.get('row_prompt_template', '')
    try:
        row_prompt = row_prompt_template.format(**study_data)
    except KeyError as e:
        print(f"⚠️  Missing field in prompt template: {e}")
        return None
    
    # Get AI response (context already set via system instruction)
    result = get_gemini_response(model, row_prompt)
    
    # Add small delay to respect rate limits
    api_delay = gemini_config.get('api_delay', 0.5)
    if api_delay > 0:
        time.sleep(api_delay)
    
    return result


def transform_studies_with_ai(studies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform studies by adding AI-determined column values.
    
    Args:
        studies (List[Dict[str, Any]]): List of study data dictionaries
        
    Returns:
        List[Dict[str, Any]]: Studies with AI-determined values added
    """
    # Initialize model with system instruction (context set once)
    model = initialize_gemini()
    if not model:
        print("❌ Failed to initialize Gemini API. AI processing is required.")
        sys.exit(1)
    
    # Get AI processing configuration
    ai_config = CONFIG.get('ai_processing', {})
    
    # Determine which studies to process
    max_ai_rows = ai_config.get('max_rows')
    if max_ai_rows is None:
        studies_to_process = studies
        remaining_studies = []
        limit_msg = "all"
    else:
        studies_to_process = studies[:max_ai_rows]
        remaining_studies = studies[max_ai_rows:]
        limit_msg = f"first {max_ai_rows}"
    
    # Filter to only tuning trials if debug mode is enabled
    if ai_config.get('debug_only_tuning_trials', False):
        tuning_trials = CONFIG.get('tuning_trials', [])
        tuning_trials_set = set(tuning_trials)
        original_count = len(studies_to_process)
        # Separate studies into tuning trials and others
        filtered_tuning = [s for s in studies_to_process if s.get('nct_id') in tuning_trials_set]
        filtered_out = [s for s in studies_to_process if s.get('nct_id') not in tuning_trials_set]
        studies_to_process = filtered_tuning
        # Add filtered-out studies to remaining_studies
        remaining_studies.extend(filtered_out)
        if original_count != len(studies_to_process):
            print(f"🔍 Debug mode: Filtered to {len(studies_to_process)} tuning trials (from {original_count} studies)")
    
    ai_column_name = ai_config.get('column_name', 'ai_determined_value')
    gemini_config = CONFIG.get('gemini', {})
    model_name = gemini_config.get('model', 'gemini-2.5-flash')
    
    print(f"\n🤖 Transforming {len(studies_to_process)} studies with Gemini AI ({limit_msg} studies)...")
    print(f"   Model: {model_name}")
    print(f"   Column: {ai_column_name}")
    print("-" * 60)
    
    processed_studies = []
    success_count = 0
    error_count = 0
    total_studies = len(studies_to_process)
    
    for i, study in enumerate(studies_to_process):
        nct_id = study.get('nct_id', 'Unknown')
        print(f"  [{i+1}/{total_studies}] Transforming {nct_id} with Gemini AI...", end=' ', flush=True)
        
        ai_value = process_study_with_ai(model, study)
        
        if ai_value:
            study[ai_column_name] = ai_value
            success_count += 1
            print("✓")
        else:
            study[ai_column_name] = 'N/A'
            error_count += 1
            print("✗")
        
        processed_studies.append(study)
    
    # Add remaining studies without AI transformation (set AI column to 'N/A')
    for study in remaining_studies:
        study[ai_column_name] = 'N/A'
        processed_studies.append(study)
    
    print(f"\n✅ AI transformation complete:")
    print(f"   Processed with AI: {len(studies_to_process)} studies")
    print(f"   Successful: {success_count}/{len(studies_to_process)}")
    print(f"   Errors: {error_count}/{len(studies_to_process)}")
    if remaining_studies:
        print(f"   Remaining {len(remaining_studies)} studies set to 'N/A'")
    
    return processed_studies




# ============================================================================
# LOAD - Functions that save data to output destinations
# ============================================================================

def load_to_csv(studies: List[Dict[str, Any]], filename: Optional[str] = None) -> None:
    """
    Load transformed studies to CSV file.
    
    Args:
        studies (List[Dict[str, Any]]): List of study data
        filename (str): Output filename (if None, uses config default)
    """
    if not studies:
        print("No studies to save")
        return
    
    if filename is None:
        output_config = CONFIG.get('output', {})
        filename = output_config.get('csv_filename', 'clinical_trials_filtered.csv')
    
    # Base fieldnames
    fieldnames = [
        'nct_id', 'brief_title', 'official_title', 'overall_status',
        'minimum_age', 'maximum_age', 'study_type', 'start_date',
        'gender', 'brief_summary', 'detailed_description', 'criteria',
        'start_year'
    ]
    
    # Add AI column if it exists in any study
    ai_config = CONFIG.get('ai_processing', {})
    ai_column_name = ai_config.get('column_name', 'ai_determined_value')
    if studies and ai_column_name in studies[0]:
        fieldnames.append(ai_column_name)
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(studies)
        print(f"\n💾 Data saved to: {filename}")
    except Exception as e:
        print(f"❌ Failed to save CSV: {e}")


def main():
    """Main function to execute the clinical trials ETL script."""
    parser = argparse.ArgumentParser(description='ClinicalTrials.gov ETL with Gemini AI')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (.json or .yaml/.yml, default: config.yaml)'
    )
    args = parser.parse_args()
    
    # Load configuration
    global CONFIG
    CONFIG = load_config(args.config)
    
    print("ClinicalTrials.gov ETL - Data Fetcher with Gemini AI")
    print("="*60)
    
    # EXTRACT: Fetch all studies from API
    study_data = extract_clinical_trials()
    
    if not study_data:
        print("❌ Failed to extract data from API")
        sys.exit(1)
    
    # TRANSFORM: Transform raw studies to structured format
    raw_studies = study_data.get('studies', [])
    print(f"\n📊 Found {len(raw_studies)} total studies")
    
    transformed_studies = []
    for i, study in enumerate(raw_studies):
        if i % 1000 == 0:
            print(f"Transforming study {i+1}/{len(raw_studies)}...")
        transformed_studies.append(transform_study_data(study))

    
    if not transformed_studies:
        print("❌ No studies after transformation")
        sys.exit(1)
    
    # TRANSFORM: Apply AI transformation
    transformed_studies = transform_studies_with_ai(transformed_studies)
    
    # LOAD: Save transformed studies to CSV
    load_to_csv(transformed_studies)
    
    print(f"\n✅ Successfully processed {len(transformed_studies)} studies")


if __name__ == "__main__":
    main()