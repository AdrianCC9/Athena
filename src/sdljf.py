import torch

# Load the DataLoader object from the saved file
data_loader = torch.load('/Users/adrian/Athena/data/preprocessed_data.pt')

# Iterate through the DataLoader
for batch in data_loader:
    print(batch)
    break  # Print only the first batch to avoid clutter
