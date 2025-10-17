import os
import shutil
import subprocess
import argparse
import glob

def generate_box_file(image_path, output_base, tessdata_dir=None):
    """
    Runs Tesseract with Page Segmentation Mode 10 to generate a .box file for a single character.
    PSM 10 tells Tesseract to treat the image as a single character.
    """
    command = [
        "tesseract",
        image_path,
        output_base,
        "--psm", "10",
        "box"
    ]
    env = os.environ.copy()
    if tessdata_dir:
        env['TESSDATA_PREFIX'] = tessdata_dir

    try:
        # We run Tesseract, which will create a file at f"{output_base}.box"
        subprocess.run(command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {image_path} with Tesseract.")
        print(f"Tesseract stderr: {e.stderr.decode('utf-8')}")
        return False
    except FileNotFoundError:
        print("❌ Error: 'tesseract' command not found. Please ensure Tesseract is installed and in your PATH.")
        return False

def create_accurate_dataset(source_dir, dest_dir, tessdata_dir=None):
    """
    Creates a new dataset with accurately generated .box files.
    """
    print(f"Starting dataset processing...")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")

    os.makedirs(dest_dir, exist_ok=True)

    # Find all the .tif files in the source directory
    image_paths = glob.glob(os.path.join(source_dir, '*.tif'))
    if not image_paths:
        print("No .tif files found in the source directory.")
        return

    print(f"Found {len(image_paths)} images to process.")

    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Define paths for ground truth and output
        gt_path = os.path.join(source_dir, f"{base_name}.gt.txt")
        if not os.path.exists(gt_path):
            print(f"⚠️ Warning: Ground truth file not found for {base_name}.tif. Skipping.")
            continue

        # --- Create the new, accurate data ---
        
        # 1. Copy the original image and gt.txt file to the destination
        shutil.copy2(image_path, os.path.join(dest_dir, f"{base_name}.tif"))
        shutil.copy2(gt_path, os.path.join(dest_dir, f"{base_name}.gt.txt"))

        # 2. Use Tesseract to generate the new .box file in the destination directory
        output_base = os.path.join(dest_dir, base_name)
        if not generate_box_file(image_path, output_base, tessdata_dir=tessdata_dir):
            continue # Skip this file if Tesseract fails

        # 3. The .box file from Tesseract might have the wrong character.
        #    We need to replace it with the correct one from our .gt.txt file.
        try:
            with open(gt_path, 'r', encoding='utf-8') as f_gt:
                correct_char = f_gt.read().strip()

            box_path = f"{output_base}.box"
            with open(box_path, 'r+', encoding='utf-8') as f_box:
                lines = f_box.readlines()
                if not lines:
                    print(f"⚠️ Warning: Tesseract generated an empty box file for {base_name}. Skipping.")
                    continue
                
                # The line format is: <char> <left> <bottom> <right> <top> <page>
                parts = lines[0].split(' ')
                # Replace the character (first part) with the correct one
                parts[0] = correct_char
                
                # Go back to the beginning of the file and write the corrected line
                f_box.seek(0)
                f_box.write(' '.join(parts))
                f_box.truncate()

        except Exception as e:
            print(f"❌ Error correcting box file for {base_name}: {e}")

    print("\n✅ Dataset processing complete.")
    print(f"New dataset with accurate bounding boxes is ready at: {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate accurate Tesseract .box files for a character dataset.")
    parser.add_argument('--source', '-s', required=True, help="Source directory containing the original dataset (e.g., 'kaggle_dataset_dummy').")
    parser.add_argument('--dest', '-d', required=True, help="Destination directory to save the new, accurate dataset (e.g., 'kaggle_dataset_accurate').")
    parser.add_argument('--tessdata-dir', help="Path to the Tesseract tessdata directory (e.g., 'tesstrain/data').")
    args = parser.parse_args()
    
    create_accurate_dataset(args.source, args.dest, args.tessdata_dir)

if __name__ == '__main__':
    main()
