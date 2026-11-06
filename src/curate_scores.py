"""
Curate cognitive scores and demographics.

Combines the per-subject questionnaire CSVs produced by load_questionnaires.py
into a single table with subject ID, age, gender, and total scores for two
screening instruments (an orientation test and a word-recall / clock-drawing
test). Subjects without both scores are dropped.

All labels and question texts come from configs/settings.yaml (gitignored) -
no dataset-specific details appear in this file.
"""

import glob
import os

import pandas as pd

from config import load_settings

# --- Main pipeline ------------------------------------------------------------

if __name__ == "__main__":
    settings = load_settings()
    cur_cfg = settings["curation"]

    input_dir = cur_cfg["input_dir"]
    output_file = cur_cfg["output_file"]

    demographics_label = cur_cfg["demographics_label"]
    age_question = cur_cfg["age_question"]
    gender_question = cur_cfg["gender_question"]
    orientation_label = cur_cfg["orientation_label"]
    word_recall_clock_label = cur_cfg["word_recall_clock_label"]

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    all_data = []

    for file in csv_files:
        df = pd.read_csv(file)

        if df.empty:
            print(f"Warning: The file {file} is empty.")
            continue

        # Extract subject ID
        subject_id = (
            df["SUBJECT_ID"].iloc[0]
            if "SUBJECT_ID" in df.columns and not df["SUBJECT_ID"].empty
            else "Unknown"
        )

        # Extract age
        age_row = df[
            (df["QUESTIONNAIRE"] == demographics_label)
            & (df["QUESTION"] == age_question)
        ]
        age = age_row["ANSWER"].iloc[0] if not age_row.empty else "Unknown"

        # Extract gender
        gender_row = df[
            (df["QUESTIONNAIRE"] == demographics_label)
            & (df["QUESTION"] == gender_question)
        ]
        gender = gender_row["ANSWER"].iloc[0] if not gender_row.empty else "Unknown"

        # Extract orientation test score
        orientation_rows = df[df["QUESTIONNAIRE"] == orientation_label]
        orientation_total = (
            orientation_rows["TOTAL"].max()
            if not orientation_rows.empty
            else "nil"
        )

        # Extract word-recall / clock-drawing score
        wrc_rows = df[df["QUESTIONNAIRE"] == word_recall_clock_label]
        wrc_total = wrc_rows["TOTAL"].max() if not wrc_rows.empty else "nil"

        all_data.append({
            "subject_id": subject_id,
            "age": age,
            "gender": gender,
            "orientation_score": orientation_total,
            "word_recall_clock_score": wrc_total,
        })

    result_df = pd.DataFrame(all_data)

    # Keep only subjects with both scores
    curated_df = result_df[
        (result_df["orientation_score"] != "nil")
        & (result_df["word_recall_clock_score"] != "nil")
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    curated_df.to_csv(output_file, index=False)
    print(f"Curated data saved to {output_file}")
