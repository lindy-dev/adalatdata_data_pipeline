import json
import os
import glob

# Folder containing the extracted JSON files
JSON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extracted_jsons"
)

# Canonical judge name mapping: incorrect/mismatched variants -> correct name
NAME_MAP: dict[str, str] = {
    # Initial spacing fixes
    "A.K. Sikri": "A. K. Sikri",
    "A.M. Khanwilkar": "A. M. Khanwilkar",
    "A.N. Joseph": "A. N. Joseph",
    "A.K. Misra": "A. K. Misra",
    "M.Y. Eqbal": "M. Y. Eqbal",
    "N.V. Ramana": "N. V. Ramana",
    "R.F. Nariman": "Rohinton F. Nariman",
    "R. F. Nariman": "Rohinton F. Nariman",
    "S.A. Bobde": "S. A. Bobde",
    "R. Dave": "Anil R. Dave",
    "T.S. Thakur": "T. S. Thakur",
    # OCR errors
    "Adarsii Kumar Goel": "Adarsh Kumar Goel",
    "Fakkir Mohamed Ibrahim Kalifulla": "Fakir Mohamed Ibrahim Kalifulla",
    "R. Banumatii": "R. Banumathi",
    # Name variants -> canonical
    "Kurian": "Kurian Joseph",
    "Kurian, J.": "Kurian Joseph",
    "Kuriyan Joseph": "Kurian Joseph",
    "Nariman": "Rohinton F. Nariman",
    "Rohinton Fali Nariman": "Rohinton F. Nariman",
    "Thakur": "T. S. Thakur",
    # Chandrachud variants
    "D.Y. Chandrachud": "Dr. D. Y. Chandrachud",
    "D. Y. Chandrachud": "Dr. D. Y. Chandrachud",
    "Dr. D.Y. Chandrachud": "Dr. D. Y. Chandrachud",
    # Prafulla C. Pant variants
    "Prallulla C. Pant": "Prafulla C. Pant",
    "Prathulla C. Pant": "Prafulla C. Pant",
    "Pratibha C. Pant": "Prafulla C. Pant",
    "Pratlulla C. Pant": "Prafulla C. Pant",
    "Pratulla C. Pant": "Prafulla C. Pant",
    "Pratulla P. Pant": "Prafulla C. Pant",
    "Pravulla C. Pant": "Prafulla C. Pant",
    "Prfulla C. Pant": "Prafulla C. Pant",
    "Prashanta Panta": "Prafulla C. Pant",
    "P. Pant": "Prafulla C. Pant",
    "P. C. Pant": "Prafulla C. Pant",
    "Prafula C. Pant": "Prafulla C. Pant",
}


def list_json_files(folder: str) -> list[str]:
    """Return sorted list of .json file paths in the folder."""
    pattern = os.path.join(folder, "*.json")
    return sorted(glob.glob(pattern))


def process_file(filepath: str) -> list[dict]:
    """
    Read a JSON file, normalize judge names, save back, and return change records.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    judges = data.get("entities", {}).get("judges", [])
    changes = []

    for i, judge in enumerate(judges):
        old_name = judge.get("name", "")
        if old_name in NAME_MAP:
            new_name = NAME_MAP[old_name]
            changes.append({
                "file": os.path.basename(filepath),
                "judge_index": i,
                "old_name": old_name,
                "new_name": new_name,
            })
            judge["name"] = new_name

    if changes:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return changes


def main():
    files = list_json_files(JSON_DIR)
    print(f"Found {len(files)} JSON files in {JSON_DIR}")
    print(f"Normalization map has {len(NAME_MAP)} entries.")
    print()

    total_changes = 0
    files_modified = 0

    for filepath in files:
        changes = process_file(filepath)

        if changes:
            files_modified += 1
            total_changes += len(changes)
            print(f"Modified: {changes[0]['file']}")
            for c in changes:
                print(f"  Judge [{c['judge_index']}]: '{c['old_name']}' -> '{c['new_name']}'")

    print()
    print(f"Done. Processed {len(files)} files.")
    print(f"Files modified: {files_modified}")
    print(f"Total judge name changes: {total_changes}")


if __name__ == "__main__":
    main()
