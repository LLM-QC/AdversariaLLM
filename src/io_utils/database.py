"""
Database operations and MongoDB integration.

This module provides functions for connecting to MongoDB and performing
database operations for attack result management and analysis.
"""

import os
import glob

from omegaconf import OmegaConf
from pymongo import MongoClient
from pymongo.synchronous.database import Database

from .data_analysis import get_nested_value, normalize_value_for_grouping


def get_mongodb_connection() -> Database:
    """Get a MongoDB connection.

    Connects to MongoDB using connection details from a config file or environment
    variables. Falls back to a default localhost connection if not specified.
    """
    user = os.environ.get("MONGODB_USER")
    password = os.environ.get("MONGODB_PASSWORD")
    host = os.environ.get("MONGODB_HOST")
    mongo_uri = os.environ.get("MONGODB_URI", f"mongodb://{user}:{password}@{host}?authSource={user}")
    client = MongoClient(mongo_uri)
    db_name = os.environ.get("MONGODB_DB", user)
    return client[db_name]


def log_config_to_db(run_config, result, log_file):
    db = get_mongodb_connection()
    collection = db.runs

    idx = run_config.dataset_params.idx
    if idx is None:
        idx = [i for i in range(len(result.runs))]
    elif isinstance(idx, int):
        idx = [idx]

    for i in idx:
        run_config.dataset_params.idx = i
        config_data = {
            "config": OmegaConf.to_container(OmegaConf.structured(run_config), resolve=True),
            "log_file": log_file,
            "scored_by": []
        }
        # If a run with the same config already exists, replace it
        collection.replace_one(
            {"config": config_data["config"]},
            config_data,
            upsert=True
        )


def delete_orphaned_runs(dry_run: bool = False, direction: str = "both"):
    """
    Delete orphaned runs from the database.

    Args:
        dry_run: If True, only print what would be deleted without actually deleting
        direction: "db_only" - remove DB entries for missing files
                  "files_only" - remove files not tracked in DB
                  "both" - do both operations
    """
    db = get_mongodb_connection()

    if direction in ["db_only", "both"]:
        # Original functionality: remove DB entries for missing files
        items = db.runs.find()
        for item in items:
            log_file = item["log_file"]
            if not os.path.exists(log_file):
                print(f"Log file not found: {log_file}, deleting from database")
                if not dry_run:
                    db.runs.delete_one({"_id": item["_id"]})

    if direction in ["files_only", "both"]:
        tracked_files = set()
        items = db.runs.find({}, {"log_file": 1})  # Only fetch log_file field
        for item in items:
            tracked_files.add(item["log_file"])

        all_run_files = set(glob.glob("outputs/**/run.json", recursive=True))
        all_run_files_absolute = set(os.path.abspath(f) for f in all_run_files)

        # Find untracked files
        untracked_files = all_run_files_absolute - tracked_files
        print(f"Found {len(untracked_files)} untracked files")

        for untracked_file in sorted(list(untracked_files)):
            print(f"Untracked run file found: {untracked_file}")
            if not dry_run:
                print(f"Deleting untracked file: {untracked_file}")
                os.remove(untracked_file)


def check_match(doc_fragment, filter_fragment):
    """
    Recursively checks whether ``doc_fragment`` satisfies ``filter_fragment``.

    Supported filter types
    ----------------------
    * **primitive** (str/int/float/bool/None)  - exact equality
    * **iterable**  (list/tuple/set)           - *any* element of the
      iterable must match  ("gcg **or** autodan", …)
      If the document side is itself an iterable, **intersection ≥ 1** counts
      as a hit.
    * **dict**                                 - every key in the filter dict
      must be present in the document and its value must match recursively
      (this is the behaviour you already had).

    Examples
    --------
    filter_by = {
        "attack": ["gcg", "autodan"],           # <- match-any
    }
    """
    # --- 1. dict → recurse over its keys --------------------------------------
    if isinstance(filter_fragment, dict):
        if not isinstance(doc_fragment, dict):
            return False
        for k, v in filter_fragment.items():
            if k not in doc_fragment or not check_match(doc_fragment[k], v):
                return False
        return True  # every key matched

    # --- 2. iterable → "any of these values is fine" --------------------------
    if isinstance(filter_fragment, (list, tuple, set)):
        # doc side is also iterable  →  true if the two are the same
        if isinstance(doc_fragment, (list, tuple, set)):
            return filter_fragment == doc_fragment
            # return any(item in filter_fragment for item in doc_fragment)
        # doc side is a single value  →  true if it is one of the allowed ones
        return doc_fragment in filter_fragment

    # --- 3. primitive equality ------------------------------------------------
    return doc_fragment == filter_fragment


def get_filtered_and_grouped_paths(filter_by, group_by) -> dict[tuple[str], list[str]]:
    """
    Retrieves log paths from MongoDB filtered by criteria and grouped according to group_by.

    Args:
        filter_by (dict): Filtering criteria. Can contain nested dictionaries.
        group_by (list or tuple): List/tuple of keys to group by from the 'config' field.

    Returns:
        dict: A dictionary where keys are group identifiers (tuples of strings)
              and values are lists of log paths.
    """
    # Connect to MongoDB
    db = get_mongodb_connection()
    collection = db.runs

    # Use MongoDB's find() method to get all documents
    all_results = list(collection.find())

    # Filter in Python using the check_match helper for complex nested conditions
    if filter_by:
        filtered_results = [
            doc for doc in all_results
            if check_match(doc['config'], filter_by)
        ]
    else:
        filtered_results = all_results

    # --- Grouping ---
    if not group_by:
        return {("all",): [r["log_file"] for r in filtered_results if "log_file" in r]}

    grouped_results = {}
    for result in filtered_results:
        # Ensure the result has 'config' and 'log_file' before processing
        if "config" not in result or "log_file" not in result:
            continue  # Skip records missing essential fields

        config_data = result["config"]
        log_path = result["log_file"]

        # Create a group key based on the specified group_by fields
        group_key_parts = []
        for key_spec in group_by:
            if isinstance(key_spec, str):
                value = get_nested_value(config_data, [key_spec])
                normalized_value = normalize_value_for_grouping(value)
                group_key_parts.append(f"{key_spec}={normalized_value}")
            elif isinstance(key_spec, (list, tuple)):
                value = get_nested_value(config_data, key_spec)
                normalized_value = normalize_value_for_grouping(value)
                key_name = '.'.join(map(str, key_spec))
                group_key_parts.append(f"{key_name}={normalized_value}")
            else:
                group_key_parts.append(f"invalid_group_spec={key_spec}")

        # Use a tuple of sorted key parts for consistent group keys
        group_key_tuple = tuple(sorted(group_key_parts))

        # Add the log path to the appropriate group
        if group_key_tuple not in grouped_results:
            grouped_results[group_key_tuple] = []
        grouped_results[group_key_tuple].append(log_path)
    return grouped_results