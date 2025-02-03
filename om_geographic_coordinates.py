import rasterio
import matplotlib.pyplot as plt

# Open the georeferenced orthomosaic image
with rasterio.open('/Users/hbot07/Downloads/all/odm_orthophoto/odm_orthophoto.tif') as src:
    # Read the data
    image = src.read(1)

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap='gray')

    # Set interval size for annotations to ensure they are not too crowded
    interval_x = src.width // 3
    interval_y = src.height // 3

    # Loop over the image at regular intervals
    for x in range(0, src.width, interval_x):
        for y in range(0, src.height, interval_y):
            # Convert pixel coordinates to world coordinates
            lon, lat = src.xy(y, x)
            # Annotate the image with geographic coordinates
            ax.annotate(f'Lon: {lon:.6f}\nLat: {lat:.6f}', xy=(x, y), xytext=(15, 15),
                        textcoords='offset points', arrowprops=dict(arrowstyle='->', color='white'),
                        fontsize=8, color='white', backgroundcolor='black')
            
            # Print the coordinates to console for verification
            print(f'Pixel ({x}, {y}) -> Lon: {lon:.6f}, Lat: {lat:.6f}')

    # Enhancements for better visualization
    plt.axis('off')
    plt.title('Orthomosaic with Annotated Geographic Coordinates')
    plt.show()

    # Optionally, save the annotated image
    fig.savefig('annotated_image.png', dpi=300, bbox_inches='tight')
