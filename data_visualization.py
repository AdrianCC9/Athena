import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the processed CSV files
processed_data_dir = "Processed Data"
images_file = os.path.join(processed_data_dir, "floorplan_images.csv")
human_annotations_file = os.path.join(processed_data_dir, "human_annotations.csv")
artificial_annotations_file = os.path.join(processed_data_dir, "artificial_annotations.csv")

images_df = pd.read_csv(images_file)
human_annotations_df = pd.read_csv(human_annotations_file)
artificial_annotations_df = pd.read_csv(artificial_annotations_file)

# Step 1: Visualizing the Number of Images Available
plt.figure(figsize=(10, 6))
plt.bar(['Total Images'], [len(images_df)], color='skyblue')
plt.title('Total Number of Floor Plan Images')
plt.ylabel('Count')
plt.show(block=True)

# Step 2: Visualizing Human Annotations
plt.figure(figsize=(10, 6))
human_annotations_lengths = human_annotations_df['annotation'].apply(lambda x: len(x.split()))
plt.hist(human_annotations_lengths, bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Word Count in Human Annotations')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show(block=True)

# Step 3: Visualizing Artificial Annotations
plt.figure(figsize=(10, 6))
artificial_annotations_lengths = artificial_annotations_df['artificial_description'].apply(lambda x: len(str(x).split()))
plt.hist(artificial_annotations_lengths, bins=20, color='lightcoral', edgecolor='black')
plt.title('Distribution of Word Count in Artificial Annotations')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show(block=True)

# Step 4: Compare Human vs Artificial Annotation Word Count
plt.figure(figsize=(10, 6))
plt.boxplot([human_annotations_lengths, artificial_annotations_lengths], labels=['Human', 'Artificial'], patch_artist=True)
plt.title('Comparison of Annotation Word Counts')
plt.ylabel('Word Count')
plt.show(block=True)

# Step 5: Discussing Insights from Visualizations
# - The first plot shows the total number of images available for the floor plans.
# - The histogram for human annotations provides insight into the typical length of the descriptions provided by people.
# - The histogram for artificial annotations shows how verbose the artificial descriptions are.
# - The boxplot comparison highlights differences between human and artificial annotation word counts.

# Note: The `block=True` parameter ensures that the figures remain visible until closed manually,
# making it easier for you to observe the visualizations during analysis.

# Next Steps:
# We can expand the visualizations to include more in-depth analysis of the correlations between image features and annotations.
# If you need more visualizations or specific analysis, let me know!
