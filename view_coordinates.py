import subprocess
import json

def extract_lat_lon_from_xmp(file_path):
    try:
        result = subprocess.run(
            ["exiftool", "-json", "-XMP:GPSLatitude", "-XMP:GPSLongitude", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Parse the output JSON
        metadata = json.loads(result.stdout)
        
        if metadata and len(metadata) > 0:
            gps_data = metadata[0]
            latitude = gps_data.get("GPSLatitude")
            longitude = gps_data.get("GPSLongitude")
            print(latitude)
            print(longitude)
            
            if latitude is not None and longitude is not None:
                return (latitude), (longitude)
    except Exception as e:
        print(f"Error extracting XMP metadata: {e}")
    
    return None

# Example usage
file_path = "/Users/hbot07/Downloads/images/dji_fly_20250127_132208_0001_1737965745987_photo.jpg"  # Replace with your image path
coords = extract_lat_lon_from_xmp(file_path)

if coords:
    print(f"Latitude: {coords[0]}, Longitude: {coords[1]}")
else:
    print("Could not extract latitude and longitude.")
