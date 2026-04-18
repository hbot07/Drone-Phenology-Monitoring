import ee
import geemap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import re

# =========================================================
# 1. INITIALIZE EARTH ENGINE
# =========================================================
try:
    ee.Initialize(project='mayankg1908')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='mayankg1908')

# =========================================================
# 2. HELPER FUNCTIONS (From previous work)
# =========================================================
undisturbed_forest_mask = ee.Image("users/mayankg1908/habitat_mapping/img_undisturbed_forest_india_1985_2022")

# --- A. HLS Cloud Masking & Merging ---
def mask_and_rename(image, band_nir):
    fmask = image.select('Fmask')
    mask = (fmask.bitwiseAnd(1<<1).eq(0).And(fmask.bitwiseAnd(1<<2).eq(0))
            .And(fmask.bitwiseAnd(1<<3).eq(0)).And(fmask.bitwiseAnd(1<<4).eq(0)))
    return image.updateMask(mask).updateMask(undisturbed_forest_mask).select(['B4', band_nir]).rename(['RED', 'NIR']).copyProperties(image, ['system:time_start'])


def Get_HLSL30_HLSS30_ImageCollections(inputStartDate, inputEndDate, roi_boundary):
    L30 = ee.ImageCollection("NASA/HLS/HLSL30/v002").filterDate(inputStartDate, inputEndDate).filterBounds(roi_boundary).map(lambda img: mask_and_rename(img, 'B5'))
    S30 = ee.ImageCollection("NASA/HLS/HLSS30/v002").filterDate(inputStartDate, inputEndDate).filterBounds(roi_boundary).map(lambda img: mask_and_rename(img, 'B8A'))
    return L30, S30

# --- B. NDVI & Time Series Compositing ---
def addNDVI(image):
    return image.addBands(image.normalizedDifference(['NIR', 'RED']).rename('NDVI')).float()

def Get_NDVI_image_datewise(harmonized_LS_ic):
    def get_NDVI_datewise(date):
        # Convert the date string to an ee.Date
        target_date = ee.Date(date)
        startDateT = target_date.advance(-8, 'day')
        endDateT = target_date.advance(8, 'day')

        # 1. Filter and Calculate Median
        filtered = harmonized_LS_ic.select(['NDVI']).filterDate(startDateT, endDateT)
        median_img = filtered.median()

        # 2. SAFETY FIX: Create a "Dummy" image
        # If median_img is empty (0 bands), this dummy provides the structure needed.
        dummy = ee.Image.constant(0).selfMask().rename('NDVI').toFloat()

        # 3. Combine them
        # overwrite=False is crucial. It keeps valid data if median_img has it.
        # We immediately .select(['NDVI']) to ensure we don't get duplicate bands (like NDVI_1)
        safe_result = median_img.addBands(dummy, None, False).select(['NDVI'])

        return safe_result.set('system:time_start', target_date.millis())
        
    return get_NDVI_datewise

def Get_LS_16Day_NDVI_TimeSeries(inputStartDate, inputEndDate, harmonized_LS_ic):
  startDate = datetime.strptime(inputStartDate,"%Y-%m-%d")
  endDate = datetime.strptime(inputEndDate,"%Y-%m-%d")

  date_list = pd.date_range(start=startDate, end=endDate, freq='16D').tolist()
  date_list = ee.List( [datetime.strftime(curr_date,"%Y-%m-%d") for curr_date in date_list] )

  LSC =  ee.ImageCollection.fromImages(date_list.map(Get_NDVI_image_datewise(harmonized_LS_ic)))

  return LSC

def Get_NDVI_TS_Image(startDate, endDate, roi_boundary):
  buffered_start = ee.Date(startDate).advance(-8, 'day')
  buffered_end = ee.Date(endDate).advance(8, 'day')  
  L30, S30 = Get_HLSL30_HLSS30_ImageCollections(buffered_start, buffered_end, roi_boundary)
  merged = ee.ImageCollection(L30.merge(S30))
  merged = merged.map(addNDVI)
  LSC = Get_LS_16Day_NDVI_TimeSeries(startDate, endDate, merged)
  final_LSMC_NDVI_TS = LSC.select(['NDVI'])

  return final_LSMC_NDVI_TS


import math

# Harmonic terms
def addHarmonics(img):
    tYear = img.date().difference(ee.Date('2014-01-01'), 'year')
    tImg = ee.Image(tYear).float().multiply(2 * math.pi).rename('t')
    
    return img.addBands([
        ee.Image(1).double().rename('constant'),
        tImg,
        tImg.sin().rename('sin_ann'),
        tImg.cos().rename('cos_ann'),
        tImg.multiply(2).sin().rename('sin_sem'),
        tImg.multiply(2).cos().rename('cos_sem')
    ])

def calculate_harmonic_coefficients():
    ecoId = int(input("Enter Ecoregion ID: "))
    print("Calculate Harmonic Coefficients...")
    print(f"Processing Ecoregion ID: {ecoId}")
    assetName = input("Enter Asset Name (e.g., 'img_features_eco297'): ")
    GeeFolder = 'users/mayankg1908/habitat_mapping'

    #
    startDate = '2014-01-01'
    endDate = '2024-12-31'
    resolve_ds = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    ecoregion_geomtery = resolve_ds.filter(ee.Filter.eq('ECO_ID', ecoId)).geometry()

    # Get NDVI collection and apply harmonics
    ndvi_ts = Get_NDVI_TS_Image(startDate, endDate, ecoregion_geomtery)
    harmonic_coll = ndvi_ts.map(addHarmonics)

    # Harmonic Regression
    independents = ['constant', 't', 'sin_ann', 'cos_ann', 'sin_sem', 'cos_sem']
    dependent = 'NDVI'

    regression = harmonic_coll.select(independents + [dependent]) \
        .reduce(ee.Reducer.linearRegression(len(independents), 1))

    # Flatten to multi-band image
    coefficients = regression.select('coefficients') \
        .arrayProject([0]) \
        .arrayFlatten([independents])

    # Clip to ecoregion and visualize
    pheno_coefficients = coefficients.select(['constant', 't', 'sin_ann', 'cos_ann', 'sin_sem', 'cos_sem']).clip(ecoregion_geomtery) 
    Map = geemap.Map(center=[20.5937, 78.9629], zoom=5)
    Map.addLayer(pheno_coefficients, {}, 'Phenology Coefficients')
    Map

    # Exporting the coefficients image as an GEE Asset
    task = ee.batch.Export.image.toAsset(
        image=pheno_coefficients,
        description= assetName,
        assetId= f'{GeeFolder}/{assetName}',
        region=ecoregion_geomtery,
        scale=30, 
        maxPixels=1e13,
        pyramidingPolicy={'constant': 'mean', '.default': 'mean'}
    )

    task.start()

    print(f"Export for {assetName} started successfully!\n Check the Earth Engine Code Editor Tasks tab to monitor its progress.")

calculate_harmonic_coefficients()