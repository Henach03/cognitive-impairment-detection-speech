"""
Load cognitive questionnaire results from a remote (restricted) data server.

Credentials are read from environment variables - never hardcoded:
    export DATA_SSH_HOST=...      # server hostname
    export DATA_SSH_USER=...      # SSH username
    export DATA_SSH_PASSWORD=...  # SSH password (or set DATA_SSH_KEY
                                   # for key-based auth)

Subject IDs are loaded from configs/subject_ids.json (gitignored).
Questionnaire filenames are mapped to generic labels via configs/settings.yaml
(gitignored), so no dataset contents appear in this repository.
"""

import json
import os
import posixpath

import numpy as np
import pandas as pd
from paramiko import AutoAddPolicy, SSHClient
from stat import S_ISREG

from config import load_settings, load_subject_ids

COLUMNS = [
    "SUBJECT_ID", "TIMESTAMP", "QUESTIONNAIRE",
    "QUESTION", "ANSWER", "SCORE", "TOTAL",
]


def connect_sftp():
    """Open an SSH/SFTP connection using credentials from environment variables."""
    ssh_host = os.environ["DATA_SSH_HOST"]
    ssh_user = os.environ["DATA_SSH_USER"]
    ssh_password = os.environ.get("DATA_SSH_PASSWORD")
    ssh_key = os.environ.get("DATA_SSH_KEY")

    client = SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.connect(
        ssh_host, username=ssh_user, password=ssh_password, key_filename=ssh_key
    )
    return client, client.open_sftp()


def parse_result_line(result, subject_id, label, loader_cfg):
    """Convert one JSON result record into a DataFrame row."""
    result_df = pd.DataFrame(columns=COLUMNS)
    result_df["SUBJECT_ID"] = subject_id
    result_df["TIMESTAMP"] = pd.to_datetime(
        result.get("date", "1970-01-01")[:10], utc=True  # default if missing
    )
    result_df["QUESTIONNAIRE"] = label

    if label == loader_cfg.get("word_recall_clock_label"):
        # Special case: word-recall + clock-drawing instrument
        result_df["QUESTION"] = [result.get("task_words"), "clock drawing"]
        result_df["ANSWER"] = "(see files)"
    else:
        result_df["QUESTION"] = result.get("questions")
        result_df["ANSWER"] = result.get("answers")
        # Drop unused rows for the orientation instrument
        if label == loader_cfg.get("orientation_label"):
            result_df.drop(
                loader_cfg.get("orientation_drop_rows", []),
                inplace=True, errors="ignore",
            )

    result_df["SCORE"] = result.get("scores", np.nan)
    result_df["TOTAL"] = result.get("total", np.nan)
    return result_df


def load_truth_data(subject_ids, remote_dir, settings):
    """Fetch questionnaire results for the given subjects over SFTP.

    Returns {subject_id: DataFrame} with columns SUBJECT_ID, TIMESTAMP,
    QUESTIONNAIRE (generic label), QUESTION, ANSWER, SCORE, TOTAL.
    """
    loader_cfg = settings["questionnaire_loader"]
    questionnaires = loader_cfg["questionnaires"]  # filename -> generic label

    client, sftp = connect_sftp()
    result_data = {}
    try:
        for subject_id in subject_ids:
            results = pd.DataFrame(columns=COLUMNS)
            subject_dir = posixpath.join(remote_dir, subject_id)

            for entry in sftp.listdir_attr(subject_dir):
                if not S_ISREG(entry.st_mode):
                    continue
                filename = entry.filename
                if filename not in questionnaires:
                    continue
                label = questionnaires[filename]

                file_path = posixpath.join(subject_dir, filename)
                with sftp.open(file_path, "r") as fh:
                    for line in fh:
                        result = json.loads(line)
                        result_df = parse_result_line(
                            result, subject_id, label, loader_cfg
                        )
                        results = pd.concat([results, result_df])

            result_data[subject_id] = results.sort_values(
                by=["TIMESTAMP", "QUESTIONNAIRE"]
            ).reset_index(drop=True)
    finally:
        sftp.close()
        client.close()

    return result_data


if __name__ == "__main__":
    settings = load_settings()
    ids = load_subject_ids()
    subject_ids = sorted(ids["ad"] + ids["non_ad"])

    loader_cfg = settings["questionnaire_loader"]
    data = load_truth_data(subject_ids, loader_cfg["remote_dir"], settings)

    output_dir = loader_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    for subject_id, df in data.items():
        out_path = os.path.join(output_dir, f"{subject_id}_results.csv")
        df.to_csv(out_path, index=False)

    print(f"Saved questionnaire results for {len(data)} subjects.")
