# script to extract coordinates from om
import rasterio
import matplotlib.pyplot as plt

# Open the georeferenced orthomosaic image
with rasterio.open('/Users/hbot07/Downloads/all/odm_orthophoto/odm_orthophoto.tif') as src:
    # Read the data
    image = src.read(1)

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap='gray')

    # Set interval size for annotations
    interval_x = src.width // 3
    interval_y = src.height // 3

    # Loop over the image at regular intervals
    for x in range(0, src.width, interval_x):
        for y in range(0, src.height, interval_y):
            # Convert pixel coordinates to world coordinates
            lon, lat = src.xy(y, x)
            # Annotate the image
            ax.annotate(f'({lon:.2f}, {lat:.2f})', xy=(x, y), xytext=(10, 10),
                        textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=6, color='yellow', backgroundcolor='black')
            
            # Print the coordinates
            print(f'Pixel ({x}, {y}) -> Lon: {lon:.2f}, Lat: {lat:.2f}')

    # Show the result
    plt.axis('off')
    plt.title('Orthomosaic with Annotated Coordinates')
    plt.show()

    # Optionally, save the annotated image
    fig.savefig('annotated_image.png', dpi=300, bbox_inches='tight')
