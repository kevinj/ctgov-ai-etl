import pandas as pd
import argparse
import os

def compare_spreadsheets(csv_path, excel_path, output_dir):
    # Load data
    print(f"Loading ETL CSV: {csv_path}...")
    try:
        df_csv = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please make sure you have run the ETL script first.")
        return

    print(f"Loading Excel Spreadsheet: {excel_path}...")
    try:
        df_excel = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Error: Could not find {excel_path}. Please check the path and try again.")
        return
    
    # Ensure NCT ID columns exist
    csv_id_col = 'nct_id'
    excel_id_col = 'NCT ID'
    
    if csv_id_col not in df_csv.columns:
        print(f"Error: Could not find '{csv_id_col}' in the CSV file. Available columns: {list(df_csv.columns)}")
        return
        
    if excel_id_col not in df_excel.columns:
        print(f"Error: Could not find '{excel_id_col}' in the Excel file. Available columns: {list(df_excel.columns)}")
        return
        
    # Standardize IDs (strip whitespace, uppercase) to ensure perfect matching
    df_csv['match_id'] = df_csv[csv_id_col].astype(str).str.strip().str.upper()
    df_excel['match_id'] = df_excel[excel_id_col].astype(str).str.strip().str.upper()
    
    # 1. Overlapping trials (Inner Join)
    # This keeps all columns from both spreadsheets, including the "yes or no" column!
    overlap_df = pd.merge(df_csv, df_excel, on='match_id', how='inner', suffixes=('_from_etl', '_from_excel'))
    overlap_df.drop('match_id', axis=1, inplace=True)
    
    # 2. Unique to CSV
    excel_ids = set(df_excel['match_id'])
    unique_csv_df = df_csv[~df_csv['match_id'].isin(excel_ids)].copy()
    unique_csv_df.drop('match_id', axis=1, inplace=True)
    
    # 3. Unique to Excel
    csv_ids = set(df_csv['match_id'])
    unique_excel_df = df_excel[~df_excel['match_id'].isin(csv_ids)].copy()
    unique_excel_df.drop('match_id', axis=1, inplace=True)
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    overlap_path = os.path.join(output_dir, "1_overlapping_trials.csv")
    unique_csv_path = os.path.join(output_dir, "2_unique_to_etl_script.csv")
    unique_excel_path = os.path.join(output_dir, "3_unique_to_excel_spreadsheet.csv")
    
    overlap_df.to_csv(overlap_path, index=False)
    unique_csv_df.to_csv(unique_csv_path, index=False)
    unique_excel_df.to_csv(unique_excel_path, index=False)
    
    print("\n✅ Comparison Complete!")
    print(f"📊 Overlapping Trials: {len(overlap_df)} -> Saved to: {overlap_path}")
    
    # Check for relevance column to create Yes/No lists
    relevance_col = None
    for col in overlap_df.columns:
        if "is this reference potentially relevant to our review" in str(col).lower():
            relevance_col = col
            break
            
    if relevance_col:
        is_yes = overlap_df[relevance_col].astype(str).str.strip().str.lower() == 'yes'
        is_no = overlap_df[relevance_col].astype(str).str.strip().str.lower() == 'no'
        
        yes_df = overlap_df[is_yes]
        no_df = overlap_df[is_no]
        
        yes_path = os.path.join(output_dir, "4_overlapping_relevant_yes.csv")
        no_path = os.path.join(output_dir, "5_overlapping_relevant_no.csv")
        
        yes_df.to_csv(yes_path, index=False)
        no_df.to_csv(no_path, index=False)
        
        print(f"   🟢 Relevant (Yes): {len(yes_df)} -> Saved to: {yes_path}")
        print(f"   🔴 Relevant (No): {len(no_df)} -> Saved to: {no_path}")
    else:
        print("\n⚠️ Note: Could not find the exact column 'is this reference potentially relevant to our review'. Yes/No lists were skipped.")
        
    print(f"📄 Trials unique to ETL script: {len(unique_csv_df)} -> Saved to: {unique_csv_path}")
    print(f"📄 Trials unique to Excel: {len(unique_excel_df)} -> Saved to: {unique_excel_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ETL CSV with an Excel spreadsheet")
    parser.add_argument("--csv", default="clinical_trials_vaccines.csv", help="Path to the ETL output CSV")
    parser.add_argument("--excel", required=True, help="Path to the Excel file")
    parser.add_argument("--outdir", default="comparison_results", help="Directory to save the outputs")
    
    args = parser.parse_args()
    compare_spreadsheets(args.csv, args.excel, args.outdir)
