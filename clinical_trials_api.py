#!/usr/bin/env python3
"""
ClinicalTrials.gov API Script
Fetches all clinical trials, filters for age < 65 and Canadian locations,
and outputs a CSV with specified columns.
"""

import requests
import json
import csv
import sys
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from urllib.parse import urlencode

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Warning: google-generativeai not available. AI features will be disabled.")


# ============================================================================
# GEMINI AI CONFIGURATION
# ============================================================================
# Set your Gemini API key as an environment variable: export GEMINI_API_KEY="your-key-here"
# Or uncomment and set it directly below (not recommended for production)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Name of the column to add with AI-determined values
AI_COLUMN_NAME = 'ai_determined_value'  # Change this to your desired column name

# General context/instructions that apply to all rows
GENERAL_CONTEXT = """
You will be analyzing information about clinical trials, including the eligibility criteria.
I want you to classify these trials into one of the following categories:

ONLY_PREGNANCY: All study participants must be pregnant
INCLUDE_PREGNANCY: Includes pregnant participants
EXCLUDE_PREGNANCY: Explicitly excludes pregnant participants
POSTPARTUM: All study participants must be postpartum
FERTILITY: All study participants must be trying to get pregnant, but are not pregnant yet.
PREGNANT OR POSTPARTUM: All study participants must either be pregnant or postpartum.
NOT MENTIONED: The criteria does not mention whether pregnanct, postpartum, or fertility participants are included or excluded.

I am going to provide an example of what it often, but not always looks like, demarcated by '==='.
You will be provided with examples of criteria and their classifications, then you will classify new criteria.

Example 1: === Inclusion Criteria:... Women of childbearing potential need a negative pregnancy test; Exclusion Criteria: Pregnant at enrollment; women who are pregnant; currently pregnant; pregnant or; positive pregnancy test; positive serum pregnancy test; pregnancy; positive urine pregnancy test ... === EXCLUDE_PREGNANCY 'Pregnant';
Example 2: === Inclusion Criteria: pregnant at enrollment; pregnancy; positive pregnancy test; positive serum pregnancy test; active labor; negative pregnancy test; negative urine pregnancy test; serum pregnancy test must be negative ... Exclusion Criteria: ... === INCLUDE_PREGNANCY 'pregnant at enrollment';
Example 3: === Inclusion Criteria: (no mention of pregnancy) ... Exclusion Criteria: ... (no mention of pregnancy) === NOT_MENTIONED;
Example 4: === Inclusion Criteria: negative serum pregnancy test; pregnancy test is negative; serum pregnancy test must be negative ... Exclusion Criteria: Positive pregnancy test; trying to get pregnant... === EXCLUDE_PREGNANCY 'negative pregnancy test';
Example 5: === Inclusion Criteria: postpartum... Exclusion Criteria:... === POSTPARTUM 'postpartum';
Example 6: === Inclusion Criteria: postpartum or pregnant... Exclusion Criteria:... === PREGNANT OR POSTPARTUM 'pregnant or postpartum'
Example 7: === Inclusion Criteria: trying to get pregnant... Exclusion Criteria:... === FERTILITY 'trying to get pregnant';

You are to output the outcome of the classification (Only the category name, not the full text of the category),
and then quote the part of the criteria that gave you confidence as to the result.
"""

# Prompt template for each row (use {field_name} to reference study data fields)
# Available fields: nct_id, brief_title, official_title, overall_status, minimum_age, 
# maximum_age, study_type, start_date, gender, brief_summary, detailed_description,
# num_canadian_sites, criteria, is_prenatal, min_age_in_months, max_age_in_months, start_year
ROW_PROMPT_TEMPLATE = """
Analyze the following clinical trial study:

NCT ID: {nct_id}
Title: {brief_title}
Official Title: {official_title}
Summary: {brief_summary}
Detailed Description: {detailed_description}
Eligibility Criteria: {criteria}
"""

# Gemini model to use
# Options: 'gemini-2.5-flash-lite' (fast/cheap), 'gemini-2.5-flash' (balanced), 'gemini-2.5-pro' (most capable)
GEMINI_MODEL = 'gemini-2.5-flash'

# Rate limiting: delay between API calls (seconds)
API_DELAY = 0.5  # Adjust based on your API rate limits

# Maximum number of rows to process with AI (None = process all rows)
MAX_AI_ROWS = None  # Set to a number (e.g., 10) to limit processing, or None for all rows

# Enable/disable AI processing
ENABLE_AI_PROCESSING = True  # Set to False to skip AI processing

# Debug mode: only process trials in TUNING_TRIALS list
DEBUG_ONLY_USE_TUNING_TRIALS = True  # Set to True to only process tuning trials

# API filter configuration
API_FILTER_ADVANCED = (
    'AREA[StudyType]INTERVENTIONAL AND '
    'AREA[StartDate]RANGE[2013-05-01,2024-12-31] AND '
    'SEARCH[Location](AREA[LocationCountry]Canada) AND '
    '(AREA[Sex]FEMALE OR AREA[Sex]ALL) AND '
    '(AREA[MaximumAge]RANGE[18 Years,MAX] OR AREA[MaximumAge]MISSING) AND '
    '(AREA[MinimumAge]RANGE[MIN,64 Years] OR AREA[MinimumAge]MISSING) AND '
    '(AREA[Phase]EARLY_PHASE1 OR AREA[Phase]PHASE1 OR AREA[Phase]PHASE2 OR AREA[Phase]PHASE3 OR AREA[Phase]PHASE4) AND '
    '(SEARCH[Study]pregnant OR SEARCH[Study]pregnancy)'
)

# Global variable to store the actual model being used (may differ from GEMINI_MODEL if auto-selected)
_ACTUAL_GEMINI_MODEL = None

TUNING_TRIALS = [
    'NCT06695221', 'NCT06964347', 'NCT07140172', 'NCT06638125', 'NCT04328584', 'NCT07149064', 'NCT07010276', 'NCT07108283',
    'NCT06938074', 'NCT07127471', 'NCT06574620', 'NCT06721949', 'NCT06767358', 'NCT06943573', 'NCT07004023', 'NCT07030920',
    'NCT07036003', 'NCT07049276', 'NCT07100977', 'NCT07127523', 'NCT07174713', 'NCT07188246', 'NCT07192536', 'NCT07193134',
    'NCT07195994', 'NCT07174973', 'NCT06035809', 'NCT07076914', 'NCT06478290', 'NCT07105449', 'NCT07159789', 'NCT07163702',
    'NCT07191145', 'NCT07196410', 'NCT05713630', 'NCT06072326', 'NCT06704022', 'NCT06768944', 'NCT06773949', 'NCT06785272',
    'NCT06807021', 'NCT06886815', 'NCT06906172', 'NCT06913647', 'NCT07034794', 'NCT07140939', 'NCT07157384', 'NCT07178353',
    'NCT06979453', 'NCT07190469', 'NCT07013201', 'NCT06599762', 'NCT06928272', 'NCT07014137', 'NCT07169188', 'NCT07174323',
    'NCT06638151', 'NCT05128734', 'NCT06498024', 'NCT06614543', 'NCT06705764', 'NCT06807528', 'NCT06929767', 'NCT06997133',
    'NCT07064954', 'NCT07103252', 'NCT07109245', 'NCT07142486', 'NCT04823299', 'NCT06356883', 'NCT06555393', 'NCT06983821',
    'NCT06989775', 'NCT07063862', 'NCT07091643', 'NCT07103291', 'NCT07113665', 'NCT07140575', 'NCT07020494', 'NCT07038746',
    'NCT07122531', 'NCT07155551', 'NCT07129239', 'NCT06776250', 'NCT07081646', 'NCT06949436', 'NCT07112339', 'NCT07118943',
    'NCT07076199', 'NCT05337241', 'NCT06965998', 'NCT07043478', 'NCT07085975', 'NCT07133308', 'NCT06333899', 'NCT03952845',
    'NCT06878417', 'NCT06885996', 'NCT06928506', 'NCT05643131', 'NCT06888193', 'NCT04041713', 'NCT06794723', 'NCT07030426',
    'NCT07097012', 'NCT06452407', 'NCT07173361', 'NCT06942208', 'NCT06965049', 'NCT07066631', 'NCT07132385', 'NCT07084298',
    'NCT07194551', 'NCT06954701', 'NCT06979180', 'NCT06546254', 'NCT06878443', 'NCT06920537', 'NCT06340568', 'NCT07065461',
    'NCT06977126', 'NCT06730347', 'NCT07080463', 'NCT06494878', 'NCT05299398', 'NCT05648253', 'NCT05407987', 'NCT06641141',
    'NCT06814756', 'NCT06660212', 'NCT06710197', 'NCT06513962', 'NCT07083037', 'NCT06813417', 'NCT06555575', 'NCT06744517',
    'NCT06462391', 'NCT06572241', 'NCT06690775', 'NCT06912230', 'NCT06655675', 'NCT06205095', 'NCT06659692', 'NCT06446089',
    'NCT06222788', 'NCT06781957', 'NCT06551168', 'NCT06486441', 'NCT06242756', 'NCT06454864', 'NCT06276257', 'NCT06440967',
    'NCT06365853', 'NCT04478305', 'NCT06251635', 'NCT06345937', 'NCT06433453', 'NCT06014983', 'NCT06333353', 'NCT06325449',
    'NCT06350591', 'NCT06356948', 'NCT06038032', 'NCT06107868', 'NCT06072781', 'NCT05432856', 'NCT06316778', 'NCT06389071',
    'NCT06391814', 'NCT05991414', 'NCT07092189', 'NCT01687127', 'NCT06187675', 'NCT05912517', 'NCT05946304', 'NCT05417516',
    'NCT05793944', 'NCT05961371', 'NCT06117982', 'NCT05226624', 'NCT06268405', 'NCT05683119', 'NCT05738239', 'NCT05936424',
    'NCT05952258', 'NCT05879926', 'NCT05411549', 'NCT05961124', 'NCT05710991', 'NCT05967884', 'NCT05994599', 'NCT05495906',
    'NCT05114096', 'NCT05946447', 'NCT06519071', 'NCT05592938', 'NCT05634499', 'NCT05601752', 'NCT05823467', 'NCT05876260',
    'NCT03949465', 'NCT05765487', 'NCT06719115', 'NCT05537519', 'NCT05990166', 'NCT05632601', 'NCT05773378', 'NCT05625659',
    'NCT05585242', 'NCT05731960', 'NCT05797480', 'NCT05065021', 'NCT05628727', 'NCT05705440', 'NCT05098574', 'NCT05503290',
    'NCT05527184', 'NCT05445778', 'NCT05555121', 'NCT05515354', 'NCT05593445', 'NCT05753176', 'NCT05358639', 'NCT05758857',
    'NCT05711030', 'NCT05511415', 'NCT05597358', 'NCT05115188', 'NCT05691140', 'NCT05456685', 'NCT05512065', 'NCT07103161',
    'NCT05280067', 'NCT04918576', 'NCT04918589', 'NCT05596812', 'NCT05299502', 'NCT06532162', 'NCT05022823', 'NCT05257408',
    'NCT05251493', 'NCT05040581', 'NCT05372549', 'NCT05342402', 'NCT05182008', 'NCT04918186', 'NCT04890925', 'NCT05347667',
    'NCT04050189', 'NCT05201547', 'NCT05140941', 'NCT04815291', 'NCT04831580', 'NCT04950868', 'NCT05173987', 'NCT04836585',
    'NCT05116189', 'NCT04930107', 'NCT05362435', 'NCT05064254', 'NCT05138835', 'NCT05110456', 'NCT05097586', 'NCT04511052',
    'NCT05103982', 'NCT05107609', 'NCT05045144', 'NCT05085366', 'NCT05139121', 'NCT05037617', 'NCT04182360', 'NCT05115643',
    'NCT05217966', 'NCT04931342', 'NCT04787289', 'NCT04630184', 'NCT05041257', 'NCT05058924', 'NCT04980391', 'NCT04902729',
    'NCT04510584', 'NCT04444440', 'NCT03754322', 'NCT04902378', 'NCT05355103', 'NCT04469101', 'NCT04893421', 'NCT04008563',
    'NCT04891029', 'NCT04860843', 'NCT04844138', 'NCT04680585', 'NCT04253717', 'NCT04781725', 'NCT04580927', 'NCT05329571',
    'NCT04412681',
]

# ============================================================================
# GEMINI AI FUNCTIONS
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
    
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("   Set it with: export GEMINI_API_KEY='your-api-key'")
        return None
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Create model with system instruction (context set once)
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=GENERAL_CONTEXT
        )
        print(f"✅ Gemini API initialized (model: {GEMINI_MODEL})")
        print(f"   System instruction set (context loaded once)")
        _ACTUAL_GEMINI_MODEL = GEMINI_MODEL
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
    if not ENABLE_AI_PROCESSING:
        return None
    
    # Format the row prompt with study data
    try:
        row_prompt = ROW_PROMPT_TEMPLATE.format(**study_data)
    except KeyError as e:
        print(f"⚠️  Missing field in prompt template: {e}")
        return None
    
    # Get AI response (context already set via system instruction)
    result = get_gemini_response(model, row_prompt)
    
    # Add small delay to respect rate limits
    if API_DELAY > 0:
        time.sleep(API_DELAY)
    
    return result


def process_studies_with_ai(studies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all studies with AI to add AI-determined column values.
    
    Args:
        studies (List[Dict[str, Any]]): List of study data dictionaries
        
    Returns:
        List[Dict[str, Any]]: Studies with AI-determined values added
    """
    if not ENABLE_AI_PROCESSING:
        print("ℹ️  AI processing is disabled")
        return studies
    
    # Initialize model with system instruction (context set once)
    model = initialize_gemini()
    if not model:
        print("⚠️  Skipping AI processing due to initialization failure")
        return studies
    
    # Determine which studies to process
    if MAX_AI_ROWS is None:
        studies_to_process = studies
        remaining_studies = []
        limit_msg = "all"
    else:
        studies_to_process = studies[:MAX_AI_ROWS]
        remaining_studies = studies[MAX_AI_ROWS:]
        limit_msg = f"first {MAX_AI_ROWS}"
    
    # Filter to only tuning trials if debug mode is enabled
    if DEBUG_ONLY_USE_TUNING_TRIALS:
        tuning_trials_set = set(TUNING_TRIALS)
        original_count = len(studies_to_process)
        # Separate studies into tuning trials and others
        filtered_tuning = [s for s in studies_to_process if s.get('nct_id') in tuning_trials_set]
        filtered_out = [s for s in studies_to_process if s.get('nct_id') not in tuning_trials_set]
        studies_to_process = filtered_tuning
        # Add filtered-out studies to remaining_studies
        remaining_studies.extend(filtered_out)
        if original_count != len(studies_to_process):
            print(f"🔍 Debug mode: Filtered to {len(studies_to_process)} tuning trials (from {original_count} studies)")
    
    print(f"\n🤖 Processing {len(studies_to_process)} studies with Gemini AI ({limit_msg} studies)...")
    print(f"   Model: {GEMINI_MODEL}")
    print(f"   Column: {AI_COLUMN_NAME}")
    print("-" * 60)
    
    processed_studies = []
    success_count = 0
    error_count = 0
    total_studies = len(studies_to_process)
    
    for i, study in enumerate(studies_to_process):
        nct_id = study.get('nct_id', 'Unknown')
        print(f"  [{i+1}/{total_studies}] Processing {nct_id}...", end=' ', flush=True)
        
        ai_value = process_study_with_ai(model, study)
        
        if ai_value:
            study[AI_COLUMN_NAME] = ai_value
            success_count += 1
            print("✓")
        else:
            study[AI_COLUMN_NAME] = 'N/A'
            error_count += 1
            print("✗")
        
        processed_studies.append(study)
    
    # Add remaining studies without AI processing (set AI column to 'N/A')
    for study in remaining_studies:
        study[AI_COLUMN_NAME] = 'N/A'
        processed_studies.append(study)
    
    print(f"\n✅ AI processing complete:")
    print(f"   Processed with AI: {len(studies_to_process)} studies")
    print(f"   Successful: {success_count}/{len(studies_to_process)}")
    print(f"   Errors: {error_count}/{len(studies_to_process)}")
    if remaining_studies:
        print(f"   Remaining {len(remaining_studies)} studies set to 'N/A'")
    
    return processed_studies


def fetch_all_clinical_trials() -> Optional[Dict[Any, Any]]:
    """
    Fetch all clinical trial data from ClinicalTrials.gov API using /studies endpoint.
    
    Returns:
        Optional[Dict[Any, Any]]: The JSON response data or None if failed
    """
    # Construct the API URL for the /studies endpoint with pagination
    api_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        'pageSize': 1000,  # Maximum page size
        'filter.advanced': API_FILTER_ADVANCED
    }
    
    all_studies = []
    page_count = 0
    
    try:
        print("Fetching all clinical trials from API...")
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
        
        print(f"✅ Successfully retrieved {len(all_studies)} total studies")
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


def extract_age_in_months(age_str: str) -> Optional[int]:
    """
    Extract age in months from age string.
    
    Args:
        age_str (str): Age string like "18 Years", "6 Months", etc.
        
    Returns:
        Optional[int]: Age in months or None if cannot parse
    """
    if not age_str or age_str == 'N/A':
        return None
    
    try:
        if 'Years' in age_str:
            years = float(age_str.replace('Years', '').strip())
            return int(years * 12)
        elif 'Months' in age_str:
            months = float(age_str.replace('Months', '').strip())
            return int(months)
        elif 'Days' in age_str:
            days = float(age_str.replace('Days', '').strip())
            return int(days / 30)  # Approximate months
        else:
            return None
    except (ValueError, AttributeError):
        return None


def is_geriatric_trial(min_age_str: str) -> bool:
    """
    Check if trial is geriatric (min age 65 or older).
    
    Args:
        min_age_str (str): Minimum age string
        
    Returns:
        bool: True if geriatric trial
    """
    min_age_months = extract_age_in_months(min_age_str)
    if min_age_months is None:
        return False
    return min_age_months >= (65 * 12)  # 65 years in months


def has_canadian_location(study: Dict[Any, Any]) -> bool:
    """
    Check if study has any Canadian locations.
    
    Args:
        study (Dict[Any, Any]): Study data
        
    Returns:
        bool: True if has Canadian location
    """
    protocol_section = study.get('protocolSection', {})
    contacts_locations = protocol_section.get('contactsLocationsModule', {})
    locations = contacts_locations.get('locations', [])
    
    for location in locations:
        country = location.get('country', '').upper()
        if 'CANADA' in country or 'CA' in country:
            return True
    return False


def count_canadian_sites(study: Dict[Any, Any]) -> int:
    """
    Count number of Canadian sites for a study.
    
    Args:
        study (Dict[Any, Any]): Study data
        
    Returns:
        int: Number of Canadian sites
    """
    protocol_section = study.get('protocolSection', {})
    contacts_locations = protocol_section.get('contactsLocationsModule', {})
    locations = contacts_locations.get('locations', [])
    
    canadian_count = 0
    for location in locations:
        country = location.get('country', '').upper()
        if 'CANADA' in country or 'CA' in country:
            canadian_count += 1
    return canadian_count


def is_prenatal_trial(study: Dict[Any, Any]) -> bool:
    """
    Check if trial is prenatal (pregnant women included).
    
    Args:
        study (Dict[Any, Any]): Study data
        
    Returns:
        bool: True if prenatal trial
    """
    protocol_section = study.get('protocolSection', {})
    eligibility = protocol_section.get('eligibilityModule', {})
    criteria_text = eligibility.get('eligibilityCriteria', '').upper()
    
    # Look for pregnancy-related terms
    pregnancy_terms = ['PREGNANT', 'PREGNANCY']
    return any(term in criteria_text for term in pregnancy_terms)


def extract_study_data(study: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Extract all required data from a study for CSV output.
    
    Args:
        study (Dict[Any, Any]): Study data
        
    Returns:
        Dict[str, Any]: Extracted study data
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
    min_age = eligibility.get('minimumAge', 'N/A')
    max_age = eligibility.get('maximumAge', 'N/A')
    gender = eligibility.get('sex', 'N/A')
    criteria_text = eligibility.get('eligibilityCriteria', 'N/A')
    
    # Age in months
    min_age_months = extract_age_in_months(min_age)
    max_age_months = extract_age_in_months(max_age)
    
    # Description
    description = protocol_section.get('descriptionModule', {})
    brief_summary = description.get('briefSummary', 'N/A')
    detailed_description = description.get('detailedDescription', 'N/A')
    
    # Canadian sites
    num_canadian_sites = count_canadian_sites(study)
    
    # Prenatal check
    is_prenatal = is_prenatal_trial(study)
    
    return {
        'nct_id': nct_id,
        'brief_title': brief_title,
        'official_title': official_title,
        'overall_status': overall_status,
        'minimum_age': min_age,
        'maximum_age': max_age,
        'study_type': study_type,
        'start_date': start_date,
        'gender': gender,
        'brief_summary': brief_summary,
        'detailed_description': detailed_description,
        'num_canadian_sites': num_canadian_sites,
        'criteria': criteria_text,
        'is_prenatal': is_prenatal,
        'min_age_in_months': min_age_months,
        'max_age_in_months': max_age_months,
        'start_year': start_year
    }


def is_study_after_2000(start_date: str) -> bool:
    """
    Check if study started after 2000/01/01.
    
    Args:
        start_date (str): Start date string
        
    Returns:
        bool: True if study started after 2000/01/01
    """
    if not start_date or start_date == 'N/A':
        return False
    
    try:
        # Parse date (format: YYYY-MM-DD or YYYY-MM)
        if '-' in start_date:
            year = int(start_date.split('-')[0])
            return year >= 2000
        return False
    except (ValueError, IndexError):
        return False


def filter_and_process_studies(study_data: Dict[Any, Any]) -> List[Dict[str, Any]]:
    """
    Filter studies and extract data for CSV.
    
    Args:
        study_data (Dict[Any, Any]): Raw API response
        
    Returns:
        List[Dict[str, Any]]: Filtered and processed studies
    """
    studies = study_data.get('studies', [])
    print(f"\n📊 Found {len(studies)} total studies")
    
    filtered_studies = []
    filter_stats = {
        'geriatric': 0,
        'passed': 0,
        'male_only': 0
    }
    
    for i, study in enumerate(studies):
        if i % 1000 == 0:
            print(f"Processing study {i+1}/{len(studies)}...")
        
        # Extract basic data first
        study_info = extract_study_data(study)
        
        # Apply client-side filters (server already filtered for country, study type, and date range)
        # Filter out male-only studies since API sex filter isn't working properly
        if study_info['gender'] == 'MALE':
            filter_stats['male_only'] += 1
            continue  # Skip male-only studies
        
        filter_stats['passed'] += 1
        filtered_studies.append(study_info)
    
    print(f"\n📊 Filter Statistics:")
    print(f"  Server-side filters applied: Canada, INTERVENTIONAL, 2000-2025")
    print(f"  Geriatric trials (age ≥65): {filter_stats['geriatric']}")
    print(f"  Male-only studies filtered: {filter_stats['male_only']}")
    print(f"  ✅ Passed all filters: {filter_stats['passed']}")
    print(f"\n✅ After filtering: {len(filtered_studies)} studies remain")
    return filtered_studies


def save_to_csv(studies: List[Dict[str, Any]], filename: str = "clinical_trials_filtered.csv") -> None:
    """
    Save filtered studies to CSV file.
    
    Args:
        studies (List[Dict[str, Any]]): List of study data
        filename (str): Output filename
    """
    if not studies:
        print("No studies to save")
        return
    
    # Base fieldnames
    fieldnames = [
        'nct_id', 'brief_title', 'official_title', 'overall_status',
        'minimum_age', 'maximum_age', 'study_type', 'start_date',
        'gender', 'brief_summary', 'detailed_description',
        'num_canadian_sites', 'criteria', 'is_prenatal',
        'min_age_in_months', 'max_age_in_months', 'start_year'
    ]
    
    # Add AI column if it exists in any study
    if studies and AI_COLUMN_NAME in studies[0]:
        fieldnames.append(AI_COLUMN_NAME)
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(studies)
        print(f"\n💾 Data saved to: {filename}")
    except Exception as e:
        print(f"❌ Failed to save CSV: {e}")


def main():
    """Main function to execute the clinical trials API script."""
    print("ClinicalTrials.gov API Data Fetcher - Filtered CSV Export")
    print("="*60)
    
    # Fetch all studies
    study_data = fetch_all_clinical_trials()
    
    if not study_data:
        print("❌ Failed to retrieve data from API")
        sys.exit(1)
    
    # Filter and process studies
    filtered_studies = filter_and_process_studies(study_data)
    
    if not filtered_studies:
        print("❌ No studies match the filtering criteria")
        sys.exit(1)
    
    # Process studies with AI (if enabled)
    if ENABLE_AI_PROCESSING:
        filtered_studies = process_studies_with_ai(filtered_studies)
    
    # Save to CSV
    save_to_csv(filtered_studies)
    
    print(f"\n✅ Successfully processed {len(filtered_studies)} studies")
    print("Filters applied:")
    print("  - Server-side: Canada, INTERVENTIONAL, 2000-2025")
    print("  - Client-side: Age < 65 years (non-geriatric)")


if __name__ == "__main__":
    main()