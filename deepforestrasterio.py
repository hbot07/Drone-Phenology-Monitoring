import os
from deepforest import main
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import rasterio


model = main.deepforest()
model.use_release()

# Path to the folder containing images
image_folder = "/Users/gauravrajput/Desktop/IITDelhi/research/all/odm_orthophoto"  # Replace with your actual folder name
# Path to save annotated images
output_folder = "/Users/gauravrajput/Desktop/IITDelhi/research/output"
os.makedirs(output_folder, exist_ok=True)

# Path to your orthomosaic
orthomosaic_path = '/Users/gauravrajput/Desktop/IITDelhi/research/all/odm_orthophoto/odm_orthophoto.tif'  # Replace with your actual path

# Load the orthomosaic
try:
    ortho_src = rasterio.open(orthomosaic_path)
except rasterio.RasterioIOError as e:
    print(f"Error opening orthomosaic: {e}")
    exit()  # Exit the script if the orthomosaic cannot be opened


# Define a function to transform pixel coordinates to lat/lon
def pixel_to_lonlat(x, y, dataset):
    lon, lat = dataset.xy(y, x)  # Note the order: row, col
    return lon, lat


# Set up font (use default system font if specific fonts are unavailable)
try:
    font = ImageFont.truetype("arial.ttf", size=15)
except IOError:
    font = ImageFont.load_default()

# Iterate through all files in the folder
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
        image_path = os.path.join(image_folder, image_file)

        # Predict tree crowns
        try:
            predictions = model.predict_image(path=image_path, return_plot=False)  # return_plot=False is important
        except Exception as e:
            print(f"Error predicting image {image_file}: {e}")
            continue  # skip to the next image

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Create a drawable object
        draw = ImageDraw.Draw(image)

        # Total number of detected objects
        total_objects = len(predictions)

        # Annotate each prediction
        for _, row in predictions.iterrows():
            score = row.get("score", 0)  # Confidence score
            if(score>0.01):

                xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

                # Calculate the center pixel coordinates
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2

                # Transform pixel coordinates to lat/lon
                lon_center, lat_center = pixel_to_lonlat(center_x, center_y, ortho_src)

                
                label = row.get("label", "Tree")  # Default label if not available

                # Bounding box
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="white", width=3)

                # Label text with score and coordinates
                text = f"{label} ({score:.2f})\n({lon_center:.5f}, {lat_center:.5f})"

                # Calculate text size to center it within the bounding box
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                text_position = (center_x - text_width / 2, center_y - text_height / 2)

                draw.text(text_position, text, fill="yellow", font=font)

        # Save the annotated image
        output_path = os.path.join(output_folder, f"annotated_{image_file}")
        image.save(output_path)

        # Display additional information (optional)
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Detected Objects: {total_objects}", fontsize=20, color="black")
        plt.show()

print(f"Annotated images saved to {output_folder}")

ortho_src.close()  # Close the orthomosaic file