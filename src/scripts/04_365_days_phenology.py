import ee
import math

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()


inputAssetPath = input("Enter File Path of GEE Asset containing harmonic coefficients (e.g., 'users/project_id/asset_name'):").strip()
outputAssetPath = input("Enter File Path for output GEE Asset (e.g., 'users/project_id/asset_name'):").strip()
ecoId = int(input("Enter Ecoregion ID (e.g., 297): "))

# Load the image
image = ee.Image(inputAssetPath)

# This part is constant for every pixel. We calculate it once.
days = ee.List.sequence(1, 365)
omega = 2.0 * math.pi / 365.0

# Convert days to radians for the math
timeRadians = ee.Array(days).multiply(omega)

# Calculate the 5 harmonic terms for all 365 days
# Equation: y = Const + (sin_ann * sin(wt)) + (cos_ann * cos(wt)) ...
ones = ee.Array(ee.List.repeat(1, 365))
sin_ann = timeRadians.sin()
cos_ann = timeRadians.cos()
sin_sem = timeRadians.multiply(2).sin()
cos_sem = timeRadians.multiply(2).cos()

# Stack these into a matrix of shape [5, 365]
# Rows: 5 (matching your coefficients)
# Cols: 365 (representing the days)
basisMatrix = ee.Array.cat([
    ones,
    sin_ann,
    cos_ann, 
    sin_sem, 
    cos_sem
], 1).transpose()

# Convert to a constant Image so we can multiply it with your data
basisImage = ee.Image.constant(basisMatrix)

# --- 2. Prepare Your Coefficient Image ---

# Ensure we select the bands in the EXACT same order as the matrix above
# We typically exclude 't' (trend) for the annual seasonality shape
coeffBands = ['constant', 'sin_ann', 'cos_ann', 'sin_sem', 'cos_sem']

coeffArray = image.select(coeffBands).toArray()

# Reshape into a 2D Matrix (Shape: [1, 5]) so it acts as a "Row Vector"
# We create a constant image defining the new shape [1, 5]
shapeImage = ee.Image(ee.Array([1, 5]))
coeffMatrix = coeffArray.arrayReshape(shapeImage, 2)

# --- 3. Calculate the 365-Dimensional Vector ---

# Generate band names using Python list comprehension (cleaner than server-side map)
band_names = [f'day_{i}' for i in range(1, 366)]

# Multiply: (1x5) * (5x365) = (1x365)
reconstructedTimeSeries = coeffMatrix.matrixMultiply(basisImage) \
    .arrayProject([1]) \
    .arrayFlatten([band_names]) # Flattens the 1D array into 365 bands

# Result: 'reconstructedTimeSeries' is an image where 
# Band 0 contains day_1, Band 1 contains day_2, etc.

print('Reconstructed Vector shape calculated.')

# --- 4. Export ---

# regionToExport = image.geometry()
ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017");

# 2. Filter for Ecoregion ID 297
region297 = ecoregions.filter(ee.Filter.eq('ECO_ID', ecoId));
roi_boundary = region297.geometry();
regionToExport = roi_boundary

task = ee.batch.Export.image.toAsset(
    image=reconstructedTimeSeries.toFloat(),  # Essential: Reduces memory usage by half
    description='Phenology_Curve',
    assetId=outputAssetPath,
    region=regionToExport,
    scale=30,               # Ensure this matches your original Landsat/Sentinel scale
    maxPixels=1e13,         # Allows exporting large areas
    pyramidingPolicy={'.default': 'mean'} # Optimization for viewing at different zoom levels
)

task.start()
print(f"365 Days Phenology Image Export task started. Check your GEE Task Manager.")