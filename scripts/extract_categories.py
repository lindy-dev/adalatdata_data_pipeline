import json
import os
from collections import Counter

json_dir = os.path.join(os.path.dirname(__file__), "extracted_jsons")
output_file = os.path.join(os.path.dirname(__file__), "categories.txt")

category_counts = Counter()
topic_counts = Counter()
file_count = 0

for filename in sorted(os.listdir(json_dir)):
    if not filename.endswith(".json"):
        continue
    filepath = os.path.join(json_dir, filename)
    with open(filepath) as f:
        data = json.load(f)
    file_count += 1
    topics = data.get("entities", {}).get("topics", [])
    for topic in topics:
        category = topic.get("category", "(none)")
        text = topic.get("text", "")
        category_counts[category] += 1
        topic_counts[text] += 1

print(f"Processed {file_count} JSON files")
print(f"Found {len(category_counts)} unique categories\n")

for category, count in category_counts.most_common():
    print(f"  {category}: {count}")

with open(output_file, "w") as f:
    for category, count in category_counts.most_common():
        f.write(f"{category}: {count}\n")

print(f"\nSaved to {output_file}")
