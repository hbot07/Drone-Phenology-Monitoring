import rasterio

# Load the orthomosaic
img_path = '/Users/hbot07/Downloads/all/odm_orthophoto/odm_orthophoto.tif'
with rasterio.open(img_path) as dataset:
    width = dataset.width  # Number of pixels along width
    height = dataset.height  # Number of pixels along height
    transform = dataset.transform  # Geospatial transform for scaling
    resolution = dataset.res  # Pixel resolution (meters per pixel)

    # Compute the real-world dimensions
pixel_resolution_x, pixel_resolution_y = resolution  # Meters per pixel
width_meters = width * pixel_resolution_x
height_meters = height * pixel_resolution_y
total_area_m2 = width_meters * height_meters  # in square meters
total_area_ha = total_area_m2 / 10_000  # Convert to hectares

print(f"Real-world size: {width_meters}m x {height_meters}m")
print(f"Total area covered: {total_area_m2:.2f} square meters ({total_area_ha:.2f} hectares)")


print(f"Orthomosaic size: {width} x {height} pixels")
print(f"Pixel resolution: {resolution} meters per pixel")