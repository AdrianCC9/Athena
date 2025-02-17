import os
from PIL import Image

def validate_data(image_dir, text_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    missing_txt = []
    unreadable_images = []
    empty_txt = []

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        txt_file = img_file.replace(".png", ".txt")
        txt_path = os.path.join(text_dir, txt_file)

        # Check matching .txt file
        if not os.path.exists(txt_path):
            missing_txt.append(txt_file)

        # Check if image is readable
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception:
            unreadable_images.append(img_file)

        # Check if .txt is empty
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    empty_txt.append(txt_file)

    if missing_txt or unreadable_images or empty_txt:
        if missing_txt:
            print("Missing .txt files for:")
            for f in missing_txt:
                print(" ", f)
        if unreadable_images:
            print("Unreadable .png files:")
            for f in unreadable_images:
                print(" ", f)
        if empty_txt:
            print("Empty .txt files:")
            for f in empty_txt:
                print(" ", f)
    else:
        print("All data is valid. No issues found.")

if __name__ == "__main__":
    image_folder = "/Users/adrian/Athena/data/General Data/floorplan_image"
    text_folder = "/Users/adrian/Athena/data/General Data/human_annotated_tags"
    validate_data(image_folder, text_folder)
