import cv2
import os


# Verify dimensions for every image in the folders
def check_image_dimensions(data_dir, splits, labels, corrupted_files):

    for split in splits:
        for label in labels:
            path = os.path.join(data_dir, split, label)
            for img_name in os.listdir(path):  
                img_path = os.path.join(path, img_name)
                with open(img_path, 'rb') as f:
                     check_chars = f.read()[-2:]
                if check_chars != b'\xff\xd9':
                    print(f"Warning: Unable to read image {img_name} in {split}/{label}. It may be corrupted.")
                    corrupted_files["files"].append(img_name)
                    corrupted_files["paths"].append(f"{split}/{label}")
                else:
                    img = cv2.imread(img_path)
                    if img.shape[:2] != (350, 350):
                        print(f"Image {img_name} in {split}/{label} has dimensions {img.shape[:2]} instead of 350x350.")