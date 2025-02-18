import os
import pickle

# Set paths
HUMAN_ANNOTATIONS_DIR = "/Users/adrian/Athena/data/General Data/human_annotated_tags"
ARTIFICIAL_PKL_PATH = "/Users/adrian/Athena/data/General Data/Tell2Design_artificial_all.pkl"

# Print a sample of human annotations
def check_human_annotations(directory):
    sample_count = 3
    print("\n=== 📝 Human Annotations (TXT Files) Sample ===")
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
            print(f"File: {filename} → Text: {text}")
            sample_count -= 1
            if sample_count == 0:
                break

# Print a sample of artificial annotations
def check_artificial_annotations(pkl_path):
    with open(pkl_path, "rb") as f:
        artificial_data = pickle.load(f)  # Should be a list of dicts
    print("\n=== 🤖 Artificial Annotations (PKL) Sample ===")
    for entry in artificial_data[:3]:  # Show first 3 samples
        print(f"Image ID: {entry['image_id']} → Text: {entry['description']}")

# Run checks
check_human_annotations(HUMAN_ANNOTATIONS_DIR)
check_artificial_annotations(ARTIFICIAL_PKL_PATH)


