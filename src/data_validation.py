import os
import pandas as pd
import logging

# Paths to data
IMAGES_PATH = "/Users/adrian/Documents/Tell2Design Data/General Data/floorplan_image"
HUMAN_ANNOTATIONS_PATH = "/Users/adrian/Athena/data/Cleaned Data/cleaned_human_annotations.csv"
ARTIFICIAL_ANNOTATIONS_PATH = "/Users/adrian/Athena/data/Cleaned Data/cleaned_artificial_annotations.csv"
LOG_PATH = "/Users/adrian/Athena/data/validation_log.txt"

# Configure logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def load_data():
    """
    Loads image filenames and annotation data.
    """
    try:
        # Load image file names
        image_files = {os.path.basename(f) for f in os.listdir(IMAGES_PATH) if f.endswith(".png")}

        # Load annotations
        human_annotations = pd.read_csv(HUMAN_ANNOTATIONS_PATH)
        artificial_annotations = pd.read_csv(ARTIFICIAL_ANNOTATIONS_PATH)

        logging.info(f"Loaded {len(image_files)} images from {IMAGES_PATH}.")
        logging.info(f"Loaded {len(human_annotations)} human annotations.")
        logging.info(f"Loaded {len(artificial_annotations)} artificial annotations.")

        return image_files, human_annotations, artificial_annotations
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def validate_annotations(image_files, human_annotations, artificial_annotations):
    """
    Validates that annotations align with images and checks for duplicates or missing data,
    prioritizing artificial annotations if human annotations are missing.
    """
    results = {
        "images_without_annotations": [],
        "duplicate_human_annotations": False,
        "duplicate_artificial_annotations": False,
        "examples_images_without_annotations": []
    }

    human_image_ids = set(human_annotations["image_id"])
    artificial_image_ids = set(artificial_annotations["image_id"])

    # Check for images with no annotations at all
    for image in image_files:
        if image not in human_image_ids and image not in artificial_image_ids:
            results["images_without_annotations"].append(image)

    # Check for duplicates
    results["duplicate_human_annotations"] = human_annotations["image_id"].duplicated().any()
    results["duplicate_artificial_annotations"] = artificial_annotations["image_id"].duplicated().any()

    # Log examples of issues
    if results["images_without_annotations"]:
        results["examples_images_without_annotations"] = results["images_without_annotations"][:5]  # First 5 examples

    return results

def log_results(results):
    """
    Logs validation results with detailed messages about mismatches.
    """
    logging.info("Validation Results:")
    
    if results["images_without_annotations"]:
        logging.warning(f"{len(results['images_without_annotations'])} images have no annotations (neither human nor artificial).")
        logging.info(f"Examples of images without annotations: {results['examples_images_without_annotations']}")
    
    if results["duplicate_human_annotations"]:
        logging.warning("Duplicate entries found in human annotations.")
    if results["duplicate_artificial_annotations"]:
        logging.warning("Duplicate entries found in artificial annotations.")

    logging.info("Validation complete. See log above for details.")


if __name__ == "__main__":
    try:
        # Load data
        image_files, human_annotations, artificial_annotations = load_data()

        # Run validation
        results = validate_annotations(image_files, human_annotations, artificial_annotations)

        # Log results
        log_results(results)

        print("Validation complete. Check log file for details.")
    except Exception as e:
        logging.error(f"An error occurred during validation: {e}")
        print("An error occurred during validation. Check log file for details.")
