import pandas as pd
human_annotations = pd.read_csv('/Users/adrian/Athena/data/Processed Data/human_annotations.csv')
floorplan_images = pd.read_csv('/Users/adrian/Athena/data/Processed Data/floorplan_images.csv')

# Print the lengths of both datasets
print(f"Number of annotations: {len(human_annotations)}")
print(f"Number of images: {len(floorplan_images)}")

# Check for unique identifiers
if 'tag_filename' in human_annotations.columns and 'tag_filename' in floorplan_images.columns:
    missing_in_images = set(human_annotations['tag_filename']) - set(floorplan_images['tag_filename'])
    missing_in_annotations = set(floorplan_images['tag_filename']) - set(human_annotations['tag_filename'])

    if missing_in_images:
        print("Annotations missing corresponding images:", missing_in_images)
    if missing_in_annotations:
        print("Images missing corresponding annotations:", missing_in_annotations)
else:
    print("No common identifier column found for matching.")

import os
image_dir = "path_to/image_directory"
for image_path in floorplan_images['image_path']:
    full_path = os.path.join(image_dir, image_path)
    if not os.path.exists(full_path):
        print(f"Missing file: {full_path}")


import torch
embeddings = torch.load('path_to/tokenized_human_annotations.pt')
assert len(embeddings) == len(human_annotations), "Mismatch between embeddings and annotations."
