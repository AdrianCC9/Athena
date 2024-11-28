import pickle

artificial_annotations_file = "/Users/adrian/Documents/Tell2Design Data/General Data/Tell2Design_artificial_all.pkl"

with open(artificial_annotations_file, "rb") as f:
    artificial_annotations = pickle.load(f)

print(type(artificial_annotations))  # Check the type of data (e.g., list, dict, etc.)
print(artificial_annotations[0])  # Print the first element to examine the structure
