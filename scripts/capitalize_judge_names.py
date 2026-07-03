import json
import os
import glob

# Folder containing the extracted JSON files
JSON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extracted_jsons")


def list_json_files(folder: str) -> list[str]:
    """Return sorted list of .json file paths in the folder."""
    pattern = os.path.join(folder, "*.json")
    files = sorted(glob.glob(pattern))
    return files


def capitalize_name(name: str) -> str:
    """Convert a name to title/capitalized case."""
    return name.title()


def process_file(filepath: str, save: bool = True) -> list[dict]:
    """
    Read a JSON file, capitalize judge names, and optionally save.
    Returns a list of change records with keys:
      - file, judge_index, old_name, new_name
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    judges = data.get("entities", {}).get("judges", [])
    changes = []

    for i, judge in enumerate(judges):
        old_name = judge.get("name", "")
        new_name = capitalize_name(old_name)

        if old_name != new_name:
            changes.append({
                "file": os.path.basename(filepath),
                "judge_index": i,
                "old_name": old_name,
                "new_name": new_name,
            })
            judge["name"] = new_name

    if changes and save:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return changes


def main():
    # Step 1: List all JSON files
    files = list_json_files(JSON_DIR)
    print(f"Found {len(files)} JSON files in {JSON_DIR}")

    if not files:
        print("No JSON files found.")
        return

    print()

    # Step 2: Process all files
    total_changes = 0
    files_modified = 0

    for filepath in files:
        changes = process_file(filepath, save=True)

        if changes:
            files_modified += 1
            total_changes += len(changes)
            print(f"Modified: {changes[0]['file']}")
            for c in changes:
                print(f"  Judge [{c['judge_index']}]: '{c['old_name']}' -> '{c['new_name']}'")

    # Step 3: Summary
    print()
    print(f"Done. Processed {len(files)} files.")
    print(f"Files modified: {files_modified}")
    print(f"Total judge name changes: {total_changes}")


if __name__ == "__main__":
    main()
