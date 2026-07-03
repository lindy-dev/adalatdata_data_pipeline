import json
import os
import glob

# Folder containing the extracted JSON files
JSON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extracted_jsons")

# Output file for the deduplicated judge names list
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_names.txt")


def list_json_files(folder: str) -> list[str]:
    """Return sorted list of .json file paths in the folder."""
    pattern = os.path.join(folder, "*.json")
    return sorted(glob.glob(pattern))


def extract_judge_names(json_dir: str) -> list[str]:
    """Extract all judge names from JSON files and return a deduplicated sorted list."""
    files = list_json_files(json_dir)
    print(f"Scanning {len(files)} JSON files...")

    names = set()
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        judges = data.get("entities", {}).get("judges", [])
        for judge in judges:
            name = judge.get("name", "").strip()
            if name:
                names.add(name)

    print(f"Found {len(names)} unique judge names.")
    return sorted(names)


def main():
    names = extract_judge_names(JSON_DIR)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")

    print(f"Saved {len(names)} unique names to {OUTPUT_FILE}")
    print()
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
