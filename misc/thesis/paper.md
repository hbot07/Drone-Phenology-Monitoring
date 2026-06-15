# Tracking Individual Tree Phenology from Repeated UAV RGB Orthomosaics Using Graph-Based Crown Association and Consensus Crowns

## Abstract

## Keywords

## 1. Introduction

### 1.1 Background and Motivation

### 1.2 Individual-Tree Phenology from UAV RGB Imagery

### 1.3 From Crown Detection to Temporal Tree Identity

### 1.4 Scope of This Paper

### 1.5 Contributions

### 1.6 Paper Organization

## 2. Related Work

### 2.1 UAV-Based Vegetation and Tree Monitoring

### 2.2 Individual Tree Crown Detection and Delineation

### 2.3 Multi-Temporal Orthomosaic Alignment

### 2.4 Object Association and Tracking-by-Detection

### 2.5 RGB-Based Phenology Metrics

### 2.6 Field Labels and Crown-Level Ecological Interpretation

## 3. Study Area and Data

### 3.1 IIT Delhi Campus Study Sites

#### 3.1.1 LHC Site

#### 3.1.2 SIT Site

### 3.2 UAV Data Acquisition

### 3.3 Orthomosaic Generation

### 3.4 Crown Annotation and Field-Verified Labels

## 4. Methodology

### 4.1 Workflow Overview

### 4.2 Multi-Threshold Crown Delineation

#### 4.2.1 Detectree2 Crown Segmentation

#### 4.2.2 Tiled and Buffered Inference

#### 4.2.3 Multi-Threshold Crown Stores

#### 4.2.4 Crown Cleaning and Geospatial Export

### 4.3 Residual Orthomosaic Alignment

#### 4.3.1 Alignment Problem

#### 4.3.2 Phase Cross-Correlation Alignment

#### 4.3.3 Tiled Alignment and Robust Shift Aggregation

#### 4.3.4 Applying Alignment to Crown Geometries

### 4.4 Graph-Based Crown Association

#### 4.4.1 Crown Observations as Temporal Graph Nodes

#### 4.4.2 Candidate Edge Construction

#### 4.4.3 Edge Features and Similarity Scoring

#### 4.4.4 Crown Association Cases

#### 4.4.5 Split, Merge, and Ambiguous Associations

### 4.5 Temporal Chain Extraction

#### 4.5.1 Full Chains

#### 4.5.2 Branching Chains and Backbones

#### 4.5.3 Partial Chains

#### 4.5.4 Gap Filling with Lower-Threshold Crowns

### 4.6 Consensus Crown Generation

#### 4.6.1 Motivation for Stable Crown Geometry

#### 4.6.2 Medoid Consensus Crown

#### 4.6.3 Consensus Crown Deduplication

#### 4.6.4 Consensus Crown Export

### 4.7 Crown-Level Phenology Feature Extraction

#### 4.7.1 Crown Crop Extraction

#### 4.7.2 RGB Chromatic Coordinates

#### 4.7.3 HSV Vegetation Fraction

#### 4.7.4 Texture and Sharpness Features

#### 4.7.5 Observation Quality Control

### 4.8 Image-Derived Phenology State Estimation

#### 4.8.1 Temporal Feature Interpolation

#### 4.8.2 Deciduousness Scoring

#### 4.8.3 Per-Date Phenophase Assignment

#### 4.8.4 Flowering and Color-Trait Indices

### 4.9 Field Label Integration

#### 4.9.1 Crown IDs and Field-Verified Labels

#### 4.9.2 QField-Based Crown Validation

#### 4.9.3 Species and Trait Label Joins

### 4.10 Interactive Visualization and Quality Assurance

#### 4.10.1 Tracking Viewer

#### 4.10.2 Crown Crop Review

#### 4.10.3 Google Earth and Field Review Exports

## 5. Experiments and Evaluation

### 5.1 Crown Detection Evaluation

### 5.2 Orthomosaic Alignment Evaluation

### 5.3 Crown Tracking Evaluation

#### 5.3.1 Internal Tracking Diagnostics

#### 5.3.2 Manual Tracking Validation Subset

#### 5.3.3 Chain Completeness and Partial-Chain Coverage

### 5.4 Consensus Crown Evaluation

### 5.5 Phenology Feature Evaluation

### 5.6 Field Label and Species Analysis

### 5.7 Visualization-Based Quality Assurance

## 6. Results

### 6.1 Crown Detection Results

### 6.2 Alignment Results

### 6.3 Tracking and Chain Assembly Results

### 6.4 Consensus Crown Results

### 6.5 Crown-Level Phenology Results

### 6.6 Field-Verified Species and Trait Results

### 6.7 Interactive Visualization and Error Analysis

## 7. Discussion

### 7.1 Tree Identity as the Core Unit of UAV Phenology

### 7.2 Role of Multi-Threshold Detection

### 7.3 Impact of Orthomosaic Alignment

### 7.4 Graph Association Under Split-Merge Ambiguity

### 7.5 Consensus Crowns for Stable Temporal Sampling

### 7.6 Field Labels and Crown-Level Ecological Interpretation

### 7.7 Operational Considerations

### 7.8 Limitations and Future Work

## 8. Conclusion

## Acknowledgements

## References

## Appendix A. Pipeline Configuration

## Appendix B. Additional Tracking Diagnostics

## Appendix C. Additional Phenology Visualizations

## Appendix D. Dataset and Date Inventory

### D.1 UAV Orthomosaic Date Inventory

### D.2 LHC Dataset

### D.3 SIT Dataset

### D.4 Additional Site Datasets

### D.5 Field and Crown Label Dataset

### D.6 Pipeline Input and Output Dataset Manifest

### D.7 Dataset Versioning and Reproducibility Notes
