import os
import subprocess
import shutil
import json
import sys

def update_exif_with_xmp(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    print(f"Target folder '{target_folder}' is prepared.")

    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

    print(f"Found {len(image_files)} image files in source folder.")

    for image in image_files:
        source_image_path = os.path.join(source_folder, image)
        target_image_path = os.path.join(target_folder, image)

        # Copy the file to the target path before modifying EXIF
        shutil.copy2(source_image_path, target_image_path)
        
        print(f"Processing image: {target_image_path}")

        try:
            # Extract GPS data from XMP
            result = subprocess.run(
                ['exiftool', '-json', '-XMP:GPSLatitude', '-XMP:GPSLongitude', target_image_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"ExifTool Output: {result.stdout}")
            if result.stderr:
                print(f"ExifTool Error: {result.stderr}")

            metadata = json.loads(result.stdout)
            if metadata and len(metadata) > 0 and 'GPSLatitude' in metadata[0] and 'GPSLongitude' in metadata[0]:
                latitude = metadata[0]['GPSLatitude']
                longitude = metadata[0]['GPSLongitude']

                # Write GPS data to EXIF
                subprocess.run(
                    ['exiftool', '-overwrite_original',
                     f'-GPSLatitude={latitude}',
                     f'-GPSLongitude={longitude}',
                     f'-GPSLatitudeRef={latitude[-1]}',  # Assumes the latitude ends with 'N' or 'S'
                     f'-GPSLongitudeRef={longitude[-1]}',  # Assumes the longitude ends with 'E' or 'W'
                     target_image_path
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f"Processed and updated {image} successfully.")
            else:
                print(f"No GPS data found for {image}.")
        except Exception as e:
            print(f"Error processing {image}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <source_folder> <target_folder>")
        sys.exit(1)

    source_folder = sys.argv[1]
    target_folder = sys.argv[2]
    update_exif_with_xmp(source_folder, target_folder)
