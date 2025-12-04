"""
Example configuration for Gemini AI integration in clinical_trials_api.py

Copy these settings into the GEMINI AI CONFIGURATION section of clinical_trials_api.py
"""

# Example 1: Classify study focus area
GENERAL_CONTEXT = """
You are analyzing clinical trial data. These are interventional studies related to pregnancy,
conducted in Canada, with participants under 65 years old. 

Your task is to classify each study into one of these categories:
- "Prenatal Care": Studies focused on routine prenatal care
- "High-Risk Pregnancy": Studies for high-risk pregnancies
- "Pregnancy Complications": Studies addressing specific complications
- "Postpartum": Studies focused on postpartum care
- "Other": Studies that don't fit the above categories

Provide only the category name as your response.
"""

ROW_PROMPT_TEMPLATE = """
Classify this clinical trial:

NCT ID: {nct_id}
Title: {brief_title}
Summary: {brief_summary}
Eligibility Criteria: {criteria}

Category:
"""


# Example 2: Extract key intervention type
GENERAL_CONTEXT_2 = """
You are analyzing clinical trial interventions. Extract the primary intervention type
or drug name from the study information. Be concise and specific.
"""

ROW_PROMPT_TEMPLATE_2 = """
Extract the primary intervention from this study:

Title: {brief_title}
Summary: {brief_summary}
Description: {detailed_description}

Primary Intervention:
"""


# Example 3: Risk assessment
GENERAL_CONTEXT_3 = """
You are a clinical research expert. Assess the risk level of each pregnancy-related
clinical trial based on the study details. Respond with one of: "Low", "Medium", "High", or "Unknown".
"""

ROW_PROMPT_TEMPLATE_3 = """
Assess the risk level of this clinical trial:

Title: {brief_title}
Study Type: {study_type}
Age Range: {minimum_age} to {maximum_age}
Status: {overall_status}
Summary: {brief_summary}
Criteria: {criteria}

Risk Level:
"""


