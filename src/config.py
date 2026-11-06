"""
Shared configuration loading.

Reads:
  - configs/settings.yaml     (copy from configs/settings.example.yaml)
  - configs/subject_ids.json  (copy from configs/subject_ids.example.json)

Both real files are gitignored; only the example placeholders are committed.
"""

import json
import os

import yaml

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs"
)
SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.yaml")
SUBJECT_IDS_PATH = os.path.join(CONFIG_DIR, "subject_ids.json")


def load_settings():
    """Load configs/settings.yaml."""
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_subject_ids():
    """Load configs/subject_ids.json.

    Returns a dict with keys 'ad' and 'non_ad' (lists of subject ID strings).
    """
    with open(SUBJECT_IDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
