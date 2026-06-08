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
import urllib.parse
import re
import atexit
from functools import lru_cache
import yaml
from google import genai
from google.genai import types
import vertexai
from vertexai.generative_models import GenerativeModel

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
def fetch_pubmed_publications(nct_id: str) -> str:
    """
    Fetch publications from PubMed that reference the given NCT ID.
    Uses NCBI E-Utilities API.
    
    Args:
        nct_id: ClinicalTrials.gov NCT ID
        
    Returns:
        Formatted string of PubMed publications
    """
    if not nct_id or nct_id == 'N/A':
        return 'N/A'
    
    try:
        # Step 1: Search for PMIDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': f"{nct_id}[SI]",
            'retmode': 'json'
        }
        
        # Add small delay to respect NCBI rate limits
        time.sleep(0.34) 
        
        search_response = requests.get(search_url, params=search_params, timeout=15)
        if search_response.status_code != 200:
            return "N/A (PubMed search failed)"
        
        search_data = search_response.json()
        pmids = search_data.get('esearchresult', {}).get('idlist', [])
        
        if not pmids:
            return "N/A"
        
        # Step 2: Get summaries for these PMIDs
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'json'
        }
        
        time.sleep(0.34)
        
        summary_response = requests.get(summary_url, params=summary_params, timeout=15)
        if summary_response.status_code != 200:
            return f"N/A (PMIDs found: {', '.join(pmids)}, but summary failed)"
        
        summary_data = summary_response.json()
        results = summary_data.get('result', {})
        
        pub_list = []
        uids = results.get('uids', [])
        for pmid in uids:
            pub = results.get(pmid, {})
            title = pub.get('title', 'Unknown Title')
            source = pub.get('source', 'Unknown Journal')
            pubdate = pub.get('pubdate', 'Unknown Date')
            authors = pub.get('authors', [])
            author_names = [a.get('name', '') for a in authors[:3]]
            author_str = ", ".join(author_names)
            if len(authors) > 3:
                author_str += " et al."
            
            pub_list.append(f"{author_str} ({pubdate}) {title}. {source}. PMID: {pmid}")
        
        return "\n".join(pub_list) if pub_list else "N/A"
        
    except Exception as e:
        print(f"⚠️ Warning: PubMed fetch failed for {nct_id}: {e}")
        return "N/A (Error fetching PubMed data)"


# ============================================================================
# HEALTH CANADA API
# ============================================================================

# Dosage-form terms that appear in Health Canada drugproduct brand_name/descriptor
# (activeingredient uses INN names + dosage_unit codes like TAB/CAP/ML, not "cream"/"ointment")
HC_FORMULATION_WORDS = frozenset({
    'aerosol', 'caplet', 'caplets', 'capsule', 'capsules', 'cream', 'drops',
    'emulsion', 'film', 'foam', 'gel', 'granules', 'inhalation', 'inhaler',
    'injectable', 'injection', 'lotion', 'lozenge', 'lozenges', 'ointment',
    'paste', 'patch', 'powder', 'solution', 'spray', 'suppositories', 'suppository',
    'suspension', 'syrup', 'tablet', 'tablets',
})
# dosage_unit codes from activeingredient (e.g. TAB, CAP, LOZ) — strip if trailing token
HC_DOSAGE_UNIT_CODES = frozenset({
    'amp', 'cap', 'drop', 'film', 'loz', 'pad', 'spray', 'sup', 'syr', 'tab',
    'vial', 'vtab', 'waf',
})


def hc_ingredient_search_name(clean_name: str) -> str:
    """Strip dose/strength and dosage-form words for Health Canada ingredient lookup."""
    name = re.sub(
        r'\s+\d+(\.\d+)?\s*(%|mg|mcg|μg|ug|g|ml|l|iu|units?)(?:/\w+)?\b.*$',
        '',
        clean_name,
        flags=re.I,
    ).strip()
    words = name.split()
    while words:
        token = words[-1].lower().rstrip('.')
        if token in HC_FORMULATION_WORDS or token in HC_DOSAGE_UNIT_CODES:
            words.pop()
        else:
            break
    name = ' '.join(words).strip()
    return name or clean_name


def _hc_api_urls():
    hc_config = CONFIG.get('health_canada_api', {})
    base = hc_config.get('base_url', 'https://health-products.canada.ca')
    return (
        hc_config.get('active_ingredient_url', f'{base}/api/drug/activeingredient/'),
        hc_config.get('status_url', f'{base}/api/drug/status/'),
    )


_HC_CACHE_MISS = object()
_hc_disk_cache: Optional[Dict[str, Any]] = None
_hc_disk_cache_dirty = False


def _hc_cache_file() -> str:
    hc_config = CONFIG.get('health_canada_api', {})
    return hc_config.get('cache_file', 'health_canada_cache.yaml')


def _ensure_hc_disk_cache() -> Dict[str, Any]:
    global _hc_disk_cache
    if _hc_disk_cache is not None:
        return _hc_disk_cache

    path = _hc_cache_file()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
            _hc_disk_cache = loaded if isinstance(loaded, dict) else {}
            print(f"✅ Loaded Health Canada cache from {path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load Health Canada cache ({path}): {e}")
            _hc_disk_cache = {}
    else:
        _hc_disk_cache = {}

    _hc_disk_cache.setdefault('active_ingredients', {})
    _hc_disk_cache.setdefault('status', {})
    return _hc_disk_cache


def save_hc_disk_cache() -> None:
    """Persist in-memory Health Canada cache to disk if it has changed."""
    global _hc_disk_cache_dirty
    if not _hc_disk_cache_dirty or _hc_disk_cache is None:
        return

    path = _hc_cache_file()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                _hc_disk_cache,
                f,
                default_flow_style=False,
                sort_keys=True,
                allow_unicode=True,
            )
        _hc_disk_cache_dirty = False
        print(f"💾 Saved Health Canada cache to {path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save Health Canada cache ({path}): {e}")


atexit.register(save_hc_disk_cache)


def _hc_cache_get_active_ingredient(ingredient_name: str):
    cache = _ensure_hc_disk_cache()
    key = ingredient_name.lower()
    ingredients = cache['active_ingredients']
    if key not in ingredients:
        return _HC_CACHE_MISS
    return ingredients[key]


def _hc_cache_set_active_ingredient(ingredient_name: str, data: List[dict]) -> None:
    global _hc_disk_cache_dirty
    cache = _ensure_hc_disk_cache()
    cache['active_ingredients'][ingredient_name.lower()] = data
    _hc_disk_cache_dirty = True


def _hc_cache_get_status(drug_code: int):
    cache = _ensure_hc_disk_cache()
    key = str(drug_code)
    statuses = cache['status']
    if key not in statuses:
        return _HC_CACHE_MISS
    return statuses[key]


def _hc_cache_set_status(drug_code: int, data: List[dict]) -> None:
    global _hc_disk_cache_dirty
    cache = _ensure_hc_disk_cache()
    cache['status'][str(drug_code)] = data
    _hc_disk_cache_dirty = True


@lru_cache(maxsize=512)
def fetch_hc_active_ingredients(ingredient_name: str):
    cached = _hc_cache_get_active_ingredient(ingredient_name)
    if cached is not _HC_CACHE_MISS:
        return cached

    active_url, _ = _hc_api_urls()
    try:
        r = requests.get(
            active_url,
            params={'ingredientname': ingredient_name, 'lang': 'en', 'type': 'json'},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        result = r.json()
        if not isinstance(result, list):
            result = []
        _hc_cache_set_active_ingredient(ingredient_name, result)
        return result
    except Exception as e:
        print(f"⚠️ Warning: Health Canada active ingredient fetch failed for {ingredient_name}: {e}")
        return None


@lru_cache(maxsize=512)
def fetch_hc_status(drug_code: int):
    cached = _hc_cache_get_status(drug_code)
    if cached is not _HC_CACHE_MISS:
        return cached

    _, status_url = _hc_api_urls()
    try:
        r = requests.get(
            status_url,
            params={'id': drug_code, 'lang': 'en', 'type': 'json'},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list):
            result = data
        elif isinstance(data, dict):
            result = [data]
        else:
            result = []
        _hc_cache_set_status(drug_code, result)
        return result
    except Exception as e:
        print(f"⚠️ Warning: Health Canada status fetch failed for drug_code {drug_code}: {e}")
        return None


def _hc_strength_label(entry: dict) -> Optional[str]:
    strength = (entry.get('strength') or '').strip()
    unit = (entry.get('strength_unit') or '').strip()
    if strength and unit:
        return f"{strength} {unit}"
    return strength or None


def _hc_approval_year(statuses: List[dict]) -> Optional[str]:
    """Earliest original_market_date year from approved/marketed status records."""
    years = []
    for st in statuses:
        if st.get('status', '').lower() not in ('approved', 'marketed'):
            continue
        date_str = st.get('original_market_date') or ''
        if date_str and len(date_str) >= 4:
            years.append(date_str[:4])
    return min(years) if years else None


def fetch_health_canada_approval(interventions: List[str]) -> str:
    """Check each intervention against Health Canada database."""
    if not interventions:
        return "N/A"

    print(f"Health Canada lookup interventions: {interventions}")

    search_names: List[str] = []
    seen = set()
    for intv in interventions:
        if not intv or intv == 'N/A':
            continue
        clean_name = re.sub(r'^[^:]+:\s*', '', intv).strip().lower()
        if not clean_name:
            continue
        search_name = hc_ingredient_search_name(clean_name)
        print(f"  {intv} -> {search_name}")
        if search_name not in seen:
            seen.add(search_name)
            search_names.append(search_name)

    checks = []
    for search_name in search_names:
        print(f"  Searching Health Canada for: {search_name}")
        active = fetch_hc_active_ingredients(search_name)
        if active is None:
            checks.append(f"NOT APPROVED - {search_name} (API fetch failed)")
            continue
        if not active:
            checks.append(f"NOT APPROVED - {search_name} (not found in database)")
            continue

        drug_codes = {a['drug_code'] for a in active if a.get('drug_code')}
        approved = False
        approved_strengths: List[str] = []
        approval_year: Optional[str] = None
        reason = "not found in database"

        for drug_code in drug_codes:
            statuses = fetch_hc_status(drug_code)
            if not statuses:
                continue
            drug_approved = any(
                st.get('status', '').lower() in ('approved', 'marketed')
                for st in statuses
            )
            if drug_approved:
                approved = True
                year = _hc_approval_year(statuses)
                if year and (approval_year is None or year < approval_year):
                    approval_year = year
                for entry in active:
                    if entry.get('drug_code') != drug_code:
                        continue
                    label = _hc_strength_label(entry)
                    if label and label not in approved_strengths:
                        approved_strengths.append(label)
            else:
                for st in statuses:
                    status_text = st.get('status', '').lower()
                    if not status_text:
                        continue
                    reason = f"status is {status_text}"
                    if 'cancelled' in status_text:
                        cancel_date = st.get('history_date') or st.get('expiration_date') or ''
                        if cancel_date and len(cancel_date) >= 4:
                            reason += f" in {cancel_date[:4]}"

        if approved:
            details = []
            if approved_strengths:
                details.append(', '.join(approved_strengths))
            if approval_year:
                details.append(f"since {approval_year}")
            suffix = f" ({'; '.join(details)})" if details else ""
            checks.append(f"APPROVED - {search_name}{suffix}")
        else:
            checks.append(f"NOT APPROVED - {search_name} ({reason})")

    if not checks:
        return "N/A"
    return "\n".join(checks)


import xml.etree.ElementTree as ET

# ============================================================================
# FDA / DAILYMED API (For better vaccine coverage)
# ============================================================================

_FDA_CACHE_MISS = object()
_fda_disk_cache: Optional[Dict[str, Any]] = None
_fda_disk_cache_dirty = False


def _fda_cache_file() -> str:
    dm_config = CONFIG.get('dailymed_api', {})
    return dm_config.get('cache_file', 'fda_drug_cache.yaml')


def _ensure_fda_disk_cache() -> Dict[str, Any]:
    global _fda_disk_cache
    if _fda_disk_cache is not None:
        return _fda_disk_cache

    path = _fda_cache_file()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
            _fda_disk_cache = loaded if isinstance(loaded, dict) else {}
            print(f"✅ Loaded FDA drug cache from {path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load FDA drug cache ({path}): {e}")
            _fda_disk_cache = {}
    else:
        _fda_disk_cache = {}

    _fda_disk_cache.setdefault('labels', {})
    return _fda_disk_cache


def save_fda_disk_cache() -> None:
    """Persist in-memory FDA drug cache to disk if it has changed."""
    global _fda_disk_cache_dirty
    if not _fda_disk_cache_dirty or _fda_disk_cache is None:
        return

    path = _fda_cache_file()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                _fda_disk_cache,
                f,
                default_flow_style=False,
                sort_keys=True,
                allow_unicode=True,
            )
        _fda_disk_cache_dirty = False
        print(f"💾 Saved FDA drug cache to {path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save FDA drug cache ({path}): {e}")


atexit.register(save_fda_disk_cache)


def _fda_cache_get_label(drug_name: str):
    cache = _ensure_fda_disk_cache()
    key = drug_name.lower()
    labels = cache['labels']
    if key not in labels:
        return _FDA_CACHE_MISS
    return labels[key]


def _fda_cache_set_label(drug_name: str, data: Optional[Dict[str, Any]]) -> None:
    global _fda_disk_cache_dirty
    cache = _ensure_fda_disk_cache()
    key = drug_name.lower()
    if data is None:
        cache['labels'][key] = None
    else:
        stored = dict(data)
        nct_ids = stored.get('nct_ids') or []
        stored['nct_ids'] = sorted(nct_ids) if isinstance(nct_ids, set) else list(nct_ids)
        cache['labels'][key] = stored
    _fda_disk_cache_dirty = True


def _fda_label_from_cache(cached: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if cached is None:
        return None
    result = dict(cached)
    result['nct_ids'] = set(result.get('nct_ids') or [])
    return result


@lru_cache(maxsize=1024)
def fetch_fda_label(drug_name):
    cached = _fda_cache_get_label(drug_name)
    if cached is not _FDA_CACHE_MISS:
        return _fda_label_from_cache(cached)

    dm_config = CONFIG.get('dailymed_api', {})
    base_url = dm_config.get('spl_url', "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json")
    
    # 1. Search DailyMed with progressive word truncation to capture base drug names
    stop_words = {'vaccination', 'vaccine', 'with', 'injection', 'of', 'for', 'dose', 'regimen', 'therapy', 'treatment', 'placebo', 'saline', 'adjuvant', 'and'}
    words = drug_name.split()
    clean_words = [w for w in words if w not in stop_words]
    data = None
    
    # Generate all contiguous sequences of the remaining words (from longest to shortest)
    for length in range(len(clean_words), 0, -1):
        for start in range(len(clean_words) - length + 1):
            query = " ".join(clean_words[start:start+length])
            if len(query) < 4 and length < len(clean_words): continue # Prevent overly generic short queries
            
            search_url = f"{base_url}?drug_name={urllib.parse.quote(query)}"
            try:
                r = requests.get(search_url, timeout=5)
                if r.status_code == 200:
                    resp = r.json()
                    if resp.get('data'):
                        data = resp
                        break
            except Exception:
                pass
        if data:
            break
            
    if not data:
        _fda_cache_set_label(drug_name, None)
        return None
    # Use the first match
    setid = data['data'][0]['setid']
    pub_date = data['data'][0].get('published_date', '')
    app_num = pub_date.split()[-1] if pub_date else "Unknown Year"
    
    # 2. Fetch the actual SPL XML
    xml_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
    try:
        x_res = requests.get(xml_url, timeout=10)
        if x_res.status_code == 200:
            xml_data = x_res.text.replace('xmlns="urn:hl7-org:v3"', '') # Strip namespace for easier search
            root = ET.fromstring(xml_data)
            
            # Extract text based on LOINC codes for Pregnancy (42228-6 or 42228-7) and Lactation (77290-5 or 34080-2)
            def get_loinc_text(*codes):
                for code in codes:
                    elem = root.find(f".//code[@code='{code}']/..")
                    if elem is not None:
                        texts = [t for t in elem.itertext() if t.strip()]
                        joined = ' '.join(texts).replace('\n', ' ').strip()
                        return joined
                return None
            preg = get_loinc_text("42228-6", "42228-7")
            lact = get_loinc_text("77290-5", "34080-2")
            
            contra = get_loinc_text("34070-3")
            preg_status = "Unknown / Data Insufficient"
            lact_status = "Unknown / Data Insufficient"
            
            # Helper to check for strong restriction/contraindication phrases
            def get_safety_status(text):
                if not text: return "Unknown / Data Insufficient"
                t = text.lower()
                
                # Check for strict contraindications
                if any(p in t for p in ["contraindicat", "do not administer", "do not vaccinate", "avoid pregnancy", "discontinue"]):
                    return "Contraindicated"
                    
                # Check for recommendations against use
                if any(p in t for p in ["not recommended", "should be avoided", "not approved"]):
                    return "Not Recommended"
                    
                # Check for caution/weighing risks
                if any(p in t for p in ["only if clearly needed", "potential risk", "weigh the potential benefits", "should be considered along with"]):
                    return "Use with Caution"
                    
                # Check for relative safety/no harm
                if any(p in t for p in ["no evidence of harm", "do not suggest an increased risk", "no increased risk", "no adverse effects"]):
                    return "No Evidence of Risk"
                    
                return "Unknown / Data Insufficient"
            
            # Extract basic status from section text
            preg_status = get_safety_status(preg)
            lact_status = get_safety_status(lact)

            # Override with Contraindications section if explicit
            if contra:
                contra_lower = contra.lower()
                if "pregnan" in contra_lower: preg_status = "Contraindicated"
                if "lactat" in contra_lower or "breast" in contra_lower or "nurs" in contra_lower: lact_status = "Contraindicated"
                
            nct_ids = set([n.upper() for n in re.findall(r'NCT\d{8}', xml_data, re.IGNORECASE)])
                
            result = {
                'application_number': app_num,
                'pregnancy': preg,
                'lactation': lact,
                'preg_status': preg_status,
                'lact_status': lact_status,
                'nct_ids': nct_ids
            }
            _fda_cache_set_label(drug_name, result)
            return result
    except Exception:
        pass
    return None

def fetch_fda_info(interventions: List[str], current_nct_id: str):
    if not interventions:
        return "N/A", "N/A", "N/A", "N/A", "N/A"
        
    status_list = []
    safety_list = []
    preg_contra_list = []
    lact_contra_list = []
    cited_list = []
    
    for intv in interventions:
        if not intv or intv == 'N/A': continue
        clean_name = re.sub(r'^[^:]+:\s*', '', intv).strip().lower()
        if not clean_name: continue
        
        data = fetch_fda_label(clean_name)
        if data:
            status_list.append(f"APPROVED - {clean_name} ({data['application_number']})")
            preg_text = data['pregnancy'] or "No specific pregnancy section found."
            lact_text = data['lactation'] or "No specific lactation section found."
            safety_list.append(f"[{clean_name.upper()}]\nPregnancy: {preg_text}\nLactation: {lact_text}")
            preg_contra_list.append(f"[{clean_name.upper()}]: {data['preg_status']}")
            lact_contra_list.append(f"[{clean_name.upper()}]: {data['lact_status']}")
            
            cited = "True" if current_nct_id.upper() in data.get('nct_ids', set()) else "False"
            cited_list.append(f"[{clean_name.upper()}]: {cited}")
        else:
            status_list.append(f"NOT APPROVED - {clean_name} (Not found in DailyMed/FDA)")
            safety_list.append(f"[{clean_name.upper()}]\nFDA Safety: N/A")
            preg_contra_list.append(f"[{clean_name.upper()}]: N/A")
            lact_contra_list.append(f"[{clean_name.upper()}]: N/A")
            cited_list.append(f"[{clean_name.upper()}]: N/A")
            
    status_str = "\n".join(status_list) if status_list else "N/A"
    safety_str = "\n\n".join(safety_list) if safety_list else "N/A"
    preg_contra_str = "\n".join(preg_contra_list) if preg_contra_list else "N/A"
    lact_contra_str = "\n".join(lact_contra_list) if lact_contra_list else "N/A"
    cited_str = "\n".join(cited_list) if cited_list else "N/A"
    return status_str, safety_str, preg_contra_str, lact_contra_str, cited_str


# ============================================================================
# TRANSFORM - Functions that transform and process data
# ============================================================================

def parse_age_in_years(age_str: str) -> Optional[float]:
    if not age_str or age_str == 'N/A':
        return None
    age_str = age_str.lower()
    match = re.match(r'([\d.]+)\s*(year|month|week|day)', age_str)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit == 'year': return val
        elif unit == 'month': return val / 12.0
        elif unit == 'week': return val / 52.0
        elif unit == 'day': return val / 365.0
    return None

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
    
    # Conditions
    conditions_module = protocol_section.get('conditionsModule', {})
    conditions = conditions_module.get('conditions', [])
    conditions_str = ', '.join(conditions) if conditions else 'N/A'
    
    # Status
    status = protocol_section.get('statusModule', {})
    overall_status = status.get('overallStatus', 'N/A')
    start_date_struct = status.get('startDateStruct', {})
    start_date = start_date_struct.get('date', 'N/A')
    start_year = "N/A"
    if start_date != 'N/A':
        match = re.search(r'\b(\d{4})\b', start_date)
        if match:
            start_year = match.group(1)
            
    primary_completion_date = status.get('primaryCompletionDateStruct', {}).get('date', 'N/A')
    completion_date = status.get('completionDateStruct', {}).get('date', 'N/A')
    first_posted_date = status.get('studyFirstPostDateStruct', {}).get('date', 'N/A')
    last_update_date = status.get('lastUpdatePostDateStruct', {}).get('date', 'N/A')
    
    # Sponsors and Funders
    sponsor_module = protocol_section.get('sponsorCollaboratorsModule', {})
    lead_sponsor = sponsor_module.get('leadSponsor', {})
    lead_sponsor_name = lead_sponsor.get('name', 'N/A')
    lead_sponsor_class = lead_sponsor.get('class', 'N/A')
    
    collaborators = sponsor_module.get('collaborators', [])
    collab_names = [c.get('name', 'N/A') for c in collaborators]
    collaborators_str = ', '.join(collab_names) if collab_names else 'None'
    
    funder_info_str = f"Lead: {lead_sponsor_name} ({lead_sponsor_class})"
    if collab_names:
        funder_info_str += f"; Collaborators: {collaborators_str}"
    
    # Design and enrollment
    design = protocol_section.get('designModule', {})
    study_type = design.get('studyType', 'N/A')
    phases = design.get('phases', [])
    phase_str = ', '.join(phases) if phases else 'N/A'
    enrollment_info = design.get('enrollmentInfo', {})
    enrollment = enrollment_info.get('count', 'N/A')
    
    design_info = design.get('designInfo', {})
    allocation = design_info.get('allocation', 'N/A')
    intervention_model = design_info.get('interventionModel', 'N/A')
    primary_purpose = design_info.get('primaryPurpose', 'N/A')
    masking_info = design_info.get('maskingInfo', {})
    masking = masking_info.get('masking', 'N/A')
    
    # Eligibility
    eligibility = protocol_section.get('eligibilityModule', {})
    gender = eligibility.get('sex', 'N/A')
    min_age = eligibility.get('minimumAge', 'N/A')
    max_age = eligibility.get('maximumAge', 'N/A')
    healthy_volunteers = eligibility.get('healthyVolunteers', 'N/A')
    criteria_text = eligibility.get('eligibilityCriteria', 'N/A')
    
    applicable = "TRUE"
    if gender.upper() == "MALE":
        applicable = "FALSE"
    
    parsed_min = parse_age_in_years(min_age)
    parsed_max = parse_age_in_years(max_age)
    
    if parsed_max is not None and parsed_max < 18:
        applicable = "FALSE"
    if parsed_min is not None and parsed_min > 55:
        applicable = "FALSE"
    
    # Description
    description = protocol_section.get('descriptionModule', {})
    brief_summary = description.get('briefSummary', 'N/A')
    detailed_description = description.get('detailedDescription', 'N/A')
    
    # Locations / Countries
    contacts_locations = protocol_section.get('contactsLocationsModule', {})
    locations = contacts_locations.get('locations', [])
    unique_countries = sorted(list(set(loc.get('country', 'N/A') for loc in locations if 'country' in loc)))
    location_str = ', '.join(unique_countries) if unique_countries else 'N/A'
    
    # Detailed sites for AI counting
    sites_list = []
    for loc in locations:
        facility = loc.get('facility', 'Unknown Facility')
        country = loc.get('country', 'Unknown Country')
        sites_list.append(f"• {facility} ({country})")
    sites_detail_str = '\n'.join(sites_list) if sites_list else 'N/A'
    
    sites_detail_str = '\n'.join(sites_list) if sites_list else 'N/A'
    
    # New Columns
    site_count_type = "Multi-site" if len(locations) > 1 else ("Single-site" if len(locations) == 1 else "Unknown/None")
    has_canadian_site = "TRUE" if "Canada" in unique_countries else "FALSE"
    # Arms and Interventions
    arms_interventions = protocol_section.get('armsInterventionsModule', {})
    
    # Arm Groups
    arm_groups = arms_interventions.get('armGroups', [])
    arm_groups_list = []
    experimental_interventions = set()
    for arm in arm_groups:
        label = arm.get('label', 'N/A')
        a_type = arm.get('type', 'N/A')
        desc = arm.get('description', '')
        entry = f"{label} ({a_type})"
        if desc:
            entry += f": {desc}"
        arm_groups_list.append(entry)
        
        # Track which interventions are explicitly mapped to an EXPERIMENTAL arm
        if 'EXPERIMENTAL' in a_type.upper():
            for mapped_int in arm.get('interventionNames', []):
                clean_int = mapped_int.split(': ')[-1].strip().lower() if ': ' in mapped_int else mapped_int.strip().lower()
                experimental_interventions.add(clean_int)
    arm_groups_str = '\n'.join(arm_groups_list) if arm_groups_list else 'N/A'
    
    interventions = arms_interventions.get('interventions', [])
    intervention_types = []
    intervention_names = []
    intervention_descriptions = []
    
    for intv in interventions:
        i_type = intv.get('type', 'N/A')
        i_name = intv.get('name', 'N/A')
        i_desc = intv.get('description', 'N/A')
        
        intervention_types.append(i_type)
        
        # Filter out anything that isn't the primary experimental drug/biological to avoid API false positives
        is_placebo_or_saline = 'placebo' in i_name.lower() or 'saline' in i_name.lower()
        is_primary_type = i_type.upper() in ['BIOLOGICAL', 'DRUG']
        
        if not is_placebo_or_saline and is_primary_type:
            # If the trial provided explicit EXPERIMENTAL arm mappings, enforce it
            if experimental_interventions:
                if i_name.lower() in experimental_interventions:
                    intervention_names.append(i_name)
            else:
                intervention_names.append(i_name)
            
            
        intervention_descriptions.append(f"• {i_name} ({i_type}): {i_desc}")
    
    intervention_types_str = ', '.join(sorted(list(set(intervention_types)))) if intervention_types else 'N/A'
    intervention_names_str = ', '.join(intervention_names) if intervention_names else 'N/A'
    intervention_descriptions_str = '\n'.join(intervention_descriptions) if intervention_descriptions else 'N/A'
    
    health_canada_approval_str = fetch_health_canada_approval(intervention_names)
    fda_status_str, fda_safety_str, preg_contra_str, lact_contra_str, cited_str = fetch_fda_info(intervention_names, nct_id)
    
    # Calculate Consistency
    consistency_str = "N/A"
    if health_canada_approval_str != "N/A" and fda_status_str != "N/A" and intervention_names:
        is_inconsistent = False
        for line_hc in health_canada_approval_str.split('\n'):
            for line_fda in fda_status_str.split('\n'):
                if ' - ' not in line_hc or ' (' not in line_hc or ' - ' not in line_fda or ' (' not in line_fda:
                    continue
                name_hc = line_hc.split(' - ')[1].split(' (')[0].strip().lower()
                name_fda = hc_ingredient_search_name(
                    line_fda.split(' - ')[1].split(' (')[0].strip().lower()
                )
                if name_hc == name_fda:
                    app_hc = line_hc.startswith("APPROVED")
                    app_fda = line_fda.startswith("APPROVED")
                    if app_hc != app_fda:
                        is_inconsistent = True
        consistency_str = "Inconsistent" if is_inconsistent else "Consistent"
    
    # Outcomes
    outcomes_module = protocol_section.get('outcomesModule', {})
    
    primary_outcomes = []
    for out in outcomes_module.get('primaryOutcomes', []):
        measure = out.get('measure', 'N/A')
        timeframe = out.get('timeFrame', 'N/A')
        desc = out.get('description', '')
        entry = f"{measure} (Time frame: {timeframe})"
        if desc:
            entry += f" - {desc}"
        primary_outcomes.append(entry)
    
    primary_outcomes_str = '\n'.join(primary_outcomes) if primary_outcomes else 'N/A'
    
    secondary_outcomes = []
    for out in outcomes_module.get('secondaryOutcomes', []):
        measure = out.get('measure', 'N/A')
        timeframe = out.get('timeFrame', 'N/A')
        desc = out.get('description', '')
        entry = f"{measure} (Time frame: {timeframe})"
        if desc:
            entry += f" - {desc}"
        secondary_outcomes.append(entry)
    
    secondary_outcomes_str = '\n'.join(secondary_outcomes) if secondary_outcomes else 'N/A'
    
    # Results Link
    has_results = study.get('hasResults', False)
    results_link = f"https://clinicaltrials.gov/study/{nct_id}?tab=results" if has_results else "N/A"
    
    # References / Publications
    references_module = protocol_section.get('referencesModule', {})
    references = references_module.get('references', [])
    publications_list = []
    for ref in references:
        pmid = ref.get('pmid')
        ref_type = ref.get('type', 'N/A')
        citation = ref.get('citation', 'No citation provided')
        pub_entry = f"[{ref_type}] {citation}"
        if pmid:
            pub_entry += f" (PMID: {pmid})"
        publications_list.append(pub_entry)
    publications_str = '\n'.join(publications_list) if publications_list else 'N/A'
    # pubmed_publications_str = fetch_pubmed_publications(nct_id)
    pubmed_publications_str = "N/A (Skipped to speed up processing)"
    
    return {
        'nct_id': nct_id,
        'brief_title': brief_title,
        'official_title': official_title,
        'conditions': conditions_str,
        'overall_status': overall_status,
        'study_type': study_type,
        'start_date': start_date,
        'primary_completion_date': primary_completion_date,
        'completion_date': completion_date,
        'first_posted_date': first_posted_date,
        'last_update_date': last_update_date,
        'gender': gender,
        'minimum_age': min_age,
        'maximum_age': max_age,
        'Applicable': applicable,
        'healthy_volunteers': healthy_volunteers,
        'brief_summary': brief_summary,
        'detailed_description': detailed_description,
        'criteria': criteria_text,
        'start_year': start_year,
        'location': location_str,
        'countries_str': location_str,
        'sites_detail': sites_detail_str,
        'site_count_type': site_count_type,
        'has_canadian_site': has_canadian_site,
        'phase': phase_str,
        'enrollment': enrollment,
        'results_link': results_link,
        'publications': publications_str,
        'pubmed_publications': pubmed_publications_str,
        'Health Canada Approval': health_canada_approval_str,
        'FDA Approved Status': fda_status_str,
        'FDA Pregnancy & Lactation Safety': fda_safety_str,
        'FDA Pregnancy Status': preg_contra_str,
        'FDA Lactation Status': lact_contra_str,
        'Cited in FDA Label': cited_str,
        'Approval Consistency': consistency_str,
        'intervention_types': intervention_types_str,
        'intervention_names': intervention_names_str,
        'intervention_descriptions': intervention_descriptions_str,
        'primary_outcomes': primary_outcomes_str,
        'secondary_outcomes': secondary_outcomes_str,
        'allocation': allocation,
        'intervention_model': intervention_model,
        'primary_purpose': primary_purpose,
        'masking': masking,
        'arm_groups': arm_groups_str,
        'funder_info': funder_info_str,
        'lead_sponsor': lead_sponsor_name,
        'lead_sponsor_class': lead_sponsor_class,
        'collaborators': collaborators_str
    }


# ============================================================================
# GEMINI AI FUNCTIONS (for transformation)
# ============================================================================

def initialize_gemini_models() -> Dict[str, Dict[str, Any]]:
    """
    Initialize Gemini API client and config with multiple models, one per column.
    
    Returns:
        Dict[str, Dict]: Dictionary mapping column names to a dict of client and config
    """
    global _ACTUAL_GEMINI_MODEL
    
    ai_config = CONFIG.get('ai_processing', {})
    api_key_env = ai_config.get('api_key_env', 'GEMINI_API_KEY')
    gemini_api_key = os.getenv(api_key_env, '')
    
    if not gemini_api_key:
        print(f"❌ {api_key_env} environment variable not set")
        print(f"   Set it with: export {api_key_env}='your-api-key'")
        return {}
    
    try:
        vertexai.init(project="project-0b52e79a-4960-471d-9d8", location="us-central1")
        
        model_name = ai_config.get('model', 'gemini-2.5-flash')
        _ACTUAL_GEMINI_MODEL = model_name
        
        # Get column configurations
        ai_config = CONFIG.get('ai_processing', {})
        columns = ai_config.get('columns', [])
        
        models = {}
        
        # If columns are defined, create a model context for each
        if (not columns):
            print("❌ No columns to process. AI processing is required.")
            sys.exit(1)
            
        for col in columns:
            col_name = col.get('name')
            if (not col_name):
                print(f"❌ Column name is required. Column: {col}")
                sys.exit(1)
                
            system_instruction = col.get('system_instruction', '')
            
            col_model = GenerativeModel(
                model_name=model_name,
                system_instruction=[system_instruction] if system_instruction else None
            )
            
            models[col_name] = {
                'model': col_model
            }
            print(f"✅ Initialized model for column '{col_name}'")
        
        print(f"✅ Gemini API initialized (model: {model_name}) - {len(models)} columns")
        
        return models
        
    except Exception as e:
        print(f"❌ Failed to initialize Gemini API: {e}")
        return {}


def get_gemini_response(model_ctx: Dict[str, Any], row_prompt: str) -> Optional[str]:
    """
    Get response from Gemini API for a single row.
    
    Args:
        model_ctx (Dict): Dictionary with model
        row_prompt (str): Row-specific prompt (context already set via system instruction)
        
    Returns:
        Optional[str]: AI response or None if failed
    """
    try:
        # Generate content using the new SDK
        model = model_ctx['model']
        response = model.generate_content(row_prompt)

        # Extract text from response
        if response and response.text:
            return response.text.strip()
        else:
            return None
            
    except Exception as e:
        print(f"⚠️  Gemini API error: {e}")
        return None


def process_study_with_ai(model_ctx: Dict[str, Any], study_data: Dict[str, Any]) -> Optional[str]:
    """
    Process a single study with AI to determine a column value.
    
    Args:
        model_ctx (Dict): Dictionary with client, model_name, and config
        study_data (Dict[str, Any]): Study data dictionary
        
    Returns:
        Optional[str]: AI-determined value or None if failed
    """
    # Format the row prompt with study data
    ai_config = CONFIG.get('ai_processing', {})
    row_prompt_template = ai_config.get('row_prompt_template', '')
    row_prompt = row_prompt_template.format(**study_data)

    # Get AI response (context already set via system instruction)
    result = get_gemini_response(model_ctx, row_prompt)
    
    # Add small delay to respect rate limits
    api_delay = ai_config.get('api_delay', 0.5)
    if api_delay > 0:
        time.sleep(api_delay)
    
    return result


def transform_studies_with_ai(studies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform studies by adding AI-determined column values for all configured columns.
    
    Args:
        studies (List[Dict[str, Any]]): List of study data dictionaries
        
    Returns:
        List[Dict[str, Any]]: Studies with AI-determined values added
    """
    ai_config = CONFIG.get('ai_processing', {})
    if not ai_config.get('enabled', True):
        print("🤖 AI processing is disabled (enabled: false/FALSE in config)")
        return studies

    # Initialize models for all columns
    models = initialize_gemini_models()
    if not models:
        print("❌ Failed to initialize Gemini API. AI processing is required.")
        sys.exit(1)
    
    ai_config = CONFIG.get('ai_processing', {})
    columns = ai_config.get('columns', [])
    if (not columns):
        print("❌ No columns to process. AI processing is required.")
        sys.exit(1)
        
    output_config = CONFIG.get('output', {})
    filename = output_config.get('csv_filename', 'clinical_trials_filtered.csv')
    
    existing_ai_data = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    nct_id = row.get('nct_id')
                    if nct_id:
                        # Store all AI columns
                        ai_vals = {col.get('name'): row.get(col.get('name')) for col in columns if row.get(col.get('name')) and row.get(col.get('name')) != 'N/A'}
                        if ai_vals:
                            existing_ai_data[nct_id] = ai_vals
            print(f"📖 Loaded existing AI data for {len(existing_ai_data)} trials from {filename}")
        except Exception as e:
            print(f"⚠️ Could not load existing CSV to resume: {e}")

    # Determine which studies to process
    max_new_calls = ai_config.get('max_rows', 250)
    
    model_name = ai_config.get('model', 'gemini-2.5-flash')
    
    print(f"\n🤖 Transforming studies with Gemini AI (Max {max_new_calls} new AI evaluations)...")
    print(f"   Model: {model_name}")
    print("-" * 60)
    
    processed_studies = []
    remaining_studies = []
    new_processed_count = 0
    quota_exceeded = False

    try:
        for i, study in enumerate(studies):
            nct_id = study.get('nct_id', 'Unknown')
            
            # Check if we already have it
            if nct_id in existing_ai_data:
                for col_name, val in existing_ai_data[nct_id].items():
                    study[col_name] = val
                processed_studies.append(study)
                continue
                
            if new_processed_count >= max_new_calls or quota_exceeded:
                remaining_studies.append(study)
                continue
                
            print(f"  [{new_processed_count+1}/{max_new_calls} new] Processing {nct_id}...", end=' ', flush=True)
            
            for col in columns:
                col_name = col.get('name')
                model = models[col_name]
                ai_value = process_study_with_ai(model, study)
                
                if ai_value is None:
                    # Assumes None means API error/timeout/quota
                    print("⚠️ Quota Exceeded or API Error! Stopping AI processing.")
                    quota_exceeded = True
                    study[col_name] = "N/A"
                else:
                    study[col_name] = ai_value
            
            if not quota_exceeded:
                print("✓")
            
            processed_studies.append(study)
            new_processed_count += 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️ User interrupted! Saving {len(processed_studies)} processed studies so far...")
        # Move remaining to remaining_studies
        remaining_studies.extend(studies[len(processed_studies) + len(remaining_studies):])
    
    # Add remaining studies without AI transformation
    for study in remaining_studies:
        for col in columns:
            col_name = col.get('name')
            if col_name not in study:
                study[col_name] = 'N/A'
        processed_studies.append(study)
    
    print(f"\n✅ AI transformation complete:")
    print(f"   Resumed from CSV: {len(processed_studies) - len(remaining_studies) - new_processed_count} studies")
    print(f"   New Processed with AI: {new_processed_count} studies")
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
        'nct_id', 'brief_title', 'official_title', 'conditions', 'overall_status',
        'minimum_age', 'maximum_age', 'Applicable', 'study_type', 'start_date',
        'primary_completion_date', 'completion_date', 'first_posted_date', 'last_update_date',
        'gender', 'healthy_volunteers', 'brief_summary', 'detailed_description', 'criteria',
        'start_year', 'location', 'sites_detail', 'site_count_type', 'has_canadian_site', 'phase', 'enrollment', 'results_link',
        'publications', 'pubmed_publications',
        'intervention_types', 'intervention_names', 'intervention_descriptions',
        'primary_outcomes', 'secondary_outcomes',
        'allocation', 'intervention_model', 'primary_purpose', 'masking', 'arm_groups',
        'funder_info', 'lead_sponsor', 'lead_sponsor_class', 'collaborators',
        'Health Canada Approval', 'FDA Approved Status', 'Approval Consistency', 'FDA Pregnancy & Lactation Safety',
        'FDA Pregnancy Status', 'FDA Lactation Status', 'Cited in FDA Label'
    ]
    
    # Add all AI columns from config
    ai_config = CONFIG.get('ai_processing', {})
    columns = ai_config.get('columns', [])
    
    if columns:
        for col_config in columns:
            col_name = col_config.get('name')
            if col_name and col_name not in fieldnames:
                fieldnames.append(col_name)
    
    try:
        output_config = CONFIG.get('output', {})
        max_rows = output_config.get('max_rows_per_file', 3000)

        # Split studies into chunks
        total_studies = len(studies)
        for i in range(0, total_studies, max_rows):
            chunk = studies[i:i + max_rows]

            # Determine filename for this chunk
            if i == 0:
                current_filename = filename
            else:
                base, ext = os.path.splitext(filename)
                iteration = i // max_rows
                current_filename = f"{base}_{iteration}{ext}"

            with open(current_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(chunk)
            print(f"\n💾 Data saved to: {current_filename} ({len(chunk)} rows)")

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
    
    resume_from_csv = CONFIG.get('ctgov', {}).get('resume_from_csv', False)
    output_config = CONFIG.get('output', {})
    filename = output_config.get('csv_filename', 'clinical_trials_filtered.csv')
    
    transformed_studies = []
    
    if resume_from_csv and os.path.exists(filename):
        print(f"\n📂 Resuming entirely from existing CSV: {filename}")
        try:
            with open(filename, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                transformed_studies = [row for row in reader]
            print(f"📊 Loaded {len(transformed_studies)} studies directly from CSV")
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            sys.exit(1)
            
        # Apply row limits if needed
        ai_config = CONFIG.get('ai_processing', {})
        if ai_config.get('max_rows') is not None:
            transformed_studies = transformed_studies[:ai_config.get('max_rows')]
            
    else:
        # EXTRACT: Fetch all studies from API
        study_data = extract_clinical_trials()
        
        if not study_data:
            print("❌ Failed to extract data from API")
            sys.exit(1)
        
        # TRANSFORM: Transform raw studies to structured format
        raw_studies = study_data.get('studies', [])
        print(f"\n📊 Found {len(raw_studies)} total studies")

        # Limit processing if max_rows is set
        ai_config = CONFIG.get('ai_processing', {})
        if ai_config.get('debug_only_tuning_trials', False):
            tuning_trials = set(CONFIG.get('tuning_trials', []))
            raw_studies = [s for s in raw_studies if s.get('protocolSection', {}).get('identificationModule', {}).get('nctId') in tuning_trials]
            print(f"🔍 Debug mode: Filtered to {len(raw_studies)} tuning trials")
        elif ai_config.get('max_rows') is not None:
            raw_studies = raw_studies[:ai_config.get('max_rows')]
            print(f"📊 Limited to first {ai_config.get('max_rows')} studies for testing")
        
        for i, study in enumerate(raw_studies):
            if i % 10 == 0:
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
