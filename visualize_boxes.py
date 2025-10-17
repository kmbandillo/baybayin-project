import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import urllib.request

def get_font(font_path="NotoSansTagalog-Regular.ttf", font_url="https://github.com/google/fonts/raw/main/ofl/notosanstagalog/NotoSansTagalog-Regular.ttf", size=12):
    """Downloads a font if not present and returns an ImageFont object."""
    if not os.path.exists(font_path):
        print(f"Downloading font: {font_path}...")
        try:
            urllib.request.urlretrieve(font_url, font_path)
            print("✅ Font downloaded successfully.")
        except Exception as e:
            print(f"❌ Failed to download font: {e}")
            return ImageFont.load_default() # Fallback
    try:
        return ImageFont.truetype(font_path, size)
    except IOError:
        print(f"⚠️ Failed to load font {font_path}. Using default.")
        return ImageFont.load_default()

def visualize_boxes(image_path, box_path, output_path):
    """
    Draws bounding boxes from a .box file onto an image and saves it.
    """
    try:
        # Open the TIFF image
        with Image.open(image_path) as img:
            # Convert to RGBA to allow drawing with color
            img = img.convert("RGBA")
            draw = ImageDraw.Draw(img)
            
            # Get image dimensions for parsing box file
            img_width, img_height = img.size
            font = get_font(size=48)
            padding = 4
            margin = 6

            # Read the box file
            with open(box_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) < 5:
                        continue
                        
                    char = parts[0]
                    # Box coordinates are (left, bottom, right, top)
                    # Tesseract's origin is bottom-left, Pillow's is top-left.
                    left, bottom, right, top = map(int, parts[1:5])
                    
                    # Convert from Tesseract's bottom-left origin to Pillow's top-left origin
                    pillow_top = img_height - top
                    pillow_bottom = img_height - bottom
                    # Normalize to Pillow's expectation (top <= bottom)
                    if pillow_top > pillow_bottom:
                        pillow_top, pillow_bottom = pillow_bottom, pillow_top
                    
                    # Draw the rectangle
                    draw.rectangle([left, pillow_top, right, pillow_bottom], outline="red", width=4)

                    # Measure text to position label and background precisely
                    text_bbox = draw.textbbox((0, 0), char, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    # Default position: slightly above the bounding box
                    text_x = left
                    text_y = pillow_top - text_height - (2 * padding) - margin

                    # Keep label within horizontal bounds
                    if text_x + text_width + (2 * padding) > img_width:
                        text_x = img_width - text_width - (2 * padding)
                        text_x = max(text_x, 0)

                    # Try placing the label below the box if top placement clips
                    if text_y < 0:
                        candidate_y = pillow_bottom + margin
                        if candidate_y + text_height + (2 * padding) <= img_height:
                            text_y = candidate_y
                        else:
                            # Last resort: keep label near the box without exceeding bounds
                            text_y = max(min(pillow_top, img_height - text_height - (2 * padding)), 0)

                    # Draw background rectangle with padding
                    background_bbox = [
                        text_x,
                        text_y,
                        text_x + text_width + (2 * padding),
                        text_y + text_height + (2 * padding)
                    ]
                    draw.rectangle(background_bbox, fill=(0, 0, 0, 220))

                    # Draw the character label in bright yellow
                    draw.text(
                        (text_x + padding, text_y + padding),
                        char,
                        fill="yellow",
                        font=font
                    )

            # Save the output image
            img.save(output_path, "PNG")
            print(f"✅ Visualization saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='Visualize Tesseract .box files on their corresponding images.')
    parser.add_argument('--dataset', '-d', required=True, help='Path to the dataset directory (e.g., kaggle_dataset_dummy).')
    parser.add_argument('--sample', '-s', required=True, help='The base name of the sample to visualize (e.g., "a_1", "ba_10").')
    parser.add_argument('--output-dir', '-o', default='visualizations', help='Directory to save the output images.')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    image_file = f"{args.sample}.tif"
    box_file = f"{args.sample}.box"
    
    image_path = os.path.join(args.dataset, image_file)
    box_path = os.path.join(args.dataset, box_file)
    output_path = os.path.join(args.output_dir, f"{args.sample}_visualization.png")
    
    visualize_boxes(image_path, box_path, output_path)

if __name__ == '__main__':
    main()
