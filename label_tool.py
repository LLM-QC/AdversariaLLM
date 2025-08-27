#!/usr/bin/env python3

import json
import os
import sys
import termios
import tty
from pathlib import Path

def load_json_file(filepath):
    """Load and parse a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json_file(filepath, data):
    """Save data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def get_single_keypress():
    """Get a single keypress without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ord(ch) == 3:  # Ctrl+C
            raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def get_user_input():
    """Get user input for labeling (h for harmful, s for safe)."""
    while True:
        print("Label as (h)armful or (s)afe? ", end='', flush=True)
        choice = get_single_keypress().lower()
        print(choice)  # Echo the character
        if choice in ['h', 's']:
            return choice
        print("Please enter 'h' for harmful or 's' for safe.")

def process_file(filepath, threshold):
    """Process a single JSON file for labeling."""
    print(f"\nProcessing: {filepath}")

    data = load_json_file(filepath)
    modified = False

    for run in data.get('runs', []):
        for step in run.get('steps', []):
            scores = step.get('scores', {})
            strong_reject = scores.get('strong_reject', {})
            p_harmful_list = strong_reject.get('p_harmful', [])

            if not p_harmful_list:
                continue

            completions = step.get('model_completions', [])
            if not completions:
                continue

            # Create list of (index, score) pairs above threshold, sorted by score descending
            items_above_threshold = [(i, score) for i, score in enumerate(p_harmful_list) if score > threshold]
            items_above_threshold.sort(key=lambda x: x[1], reverse=True)  # Sort by score, highest first

            if not items_above_threshold:
                continue

            # Process items in order of highest score first
            for rank, (original_index, current_score) in enumerate(items_above_threshold, 1):
                print(f"\n{'='*60}")
                print(f"Item {rank}/{len(items_above_threshold)} (original position {original_index+1}) - Current strongreject p_harmful score: {current_score}")
                print(f"{'='*60}")

                # Show original prompt's user message content
                user_message = run.get('original_prompt', [])
                if user_message:
                    user_message = user_message[0].get('content', '')
                    print(f"\nOriginal prompt:")
                    print("-" * 40)
                    print(user_message)
                    print("-" * 40)

                # Show corresponding model completion
                completion_idx = min(original_index, len(completions) - 1)  # Use last completion if fewer completions than scores
                print(f"\nModel generation {completion_idx + 1}:")
                print("-" * 40)
                print(completions[completion_idx][:1500] + ("..." if len(completions[completion_idx]) > 1500 else ""))
                print("-" * 40)

                choice = get_user_input()

                # Update the score based on user choice
                new_score = 1.0 if choice == 'h' else 0.0

                # Update only strong_reject scores at the original index to preserve order
                scores['strong_reject']['p_harmful'][original_index] = new_score

                modified = True
                print(f"Updated score at position {original_index+1} to: {new_score}")

                # If harmful, save and move to next file immediately
                if choice == 'h':
                    save_json_file(filepath, data)
                    print(f"File updated: {filepath}")
                    print("Marked as harmful - moving to next file")
                    return

    if modified:
        save_json_file(filepath, data)
        print(f"File updated: {filepath}")
    else:
        print(f"No items above threshold {threshold} in {filepath}")

def main():
    """Main function to run the labeling tool."""
    # Get threshold from user
    while True:
        try:
            threshold = float(input("Enter threshold (e.g., 0.5): "))
            break
        except ValueError:
            print("Please enter a valid number.")

    print(f"Using threshold: {threshold}")
    print("Only showing generations with strongreject p_harmful scores above {threshold}")

    # Directory containing the JSON files
    label_dir = Path("/ceph/ssd/staff/beyer/llm-quick-check/evaluate/label/num_return_sequences_100")

    if not label_dir.exists():
        print(f"Directory not found: {label_dir}")
        sys.exit(1)

    # Get all JSON files
    json_files = sorted(label_dir.glob("run_*.json"))

    if not json_files:
        print(f"No JSON files found in {label_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} files to process")

    try:
        for filepath in json_files:
            process_file(filepath, threshold)

    except KeyboardInterrupt:
        print("\nLabeling interrupted by user.")
        sys.exit(0)

    print("\nLabeling complete!")

if __name__ == "__main__":
    main()