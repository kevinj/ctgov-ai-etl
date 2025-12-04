#!/Users/kevinjones/src/clinical-trials/venv/bin/python3
"""
Quick script to filter clinical_trials_filtered.csv to only include
NCT_IDs that exist in clinical_trials_comparisons.csv
"""

import csv

# DEBUG MODE: Set to True to only process a specific NCT_ID
DEBUG_MODE = True
DEBUG_NCT_ID = "NCT05103982"

# File paths
COMPARISON_CSV = "clinical_trials_comparisons.csv"
INPUT_CSV = "clinical_trials_filtered.csv"
OUTPUT_CSV = "clinical_trials_filtered.csv"  # Overwrite the original

# Load NCT_IDs from comparison CSV
if DEBUG_MODE:
    print(f"🐛 DEBUG MODE: Only processing {DEBUG_NCT_ID}")
    nct_ids = {DEBUG_NCT_ID}
else:
    print(f"Loading NCT_IDs from {COMPARISON_CSV}...")
    nct_ids = set()
    try:
        with open(COMPARISON_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nct_id = row.get('NCT_ID', '').strip()
                if nct_id:
                    nct_ids.add(nct_id)
        print(f"✅ Found {len(nct_ids)} NCT_IDs in comparison file")
    except Exception as e:
        print(f"❌ Error reading comparison CSV: {e}")
        exit(1)

# Filter the input CSV
print(f"\nFiltering {INPUT_CSV}...")
filtered_rows = []
total_rows = 0

try:
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        if not fieldnames:
            print("❌ No headers found in input CSV")
            exit(1)
        
        for row in reader:
            total_rows += 1
            nct_id = row.get('nct_id', '').strip()
            if nct_id in nct_ids:
                filtered_rows.append(row)
    
    print(f"✅ Filtered {len(filtered_rows)} rows from {total_rows} total rows")
    
    # Write filtered results
    print(f"\nWriting filtered data to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"✅ Successfully wrote {len(filtered_rows)} rows to {OUTPUT_CSV}")
    
except Exception as e:
    print(f"❌ Error processing CSV: {e}")
    exit(1)

print("\n✅ Done!")

