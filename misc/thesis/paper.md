# Tracking Individual Tree Phenology from Repeated UAV RGB Orthomosaics Using Graph-Based Crown Association and Consensus Crowns

## Abstract

## Keywords

## 1. Introduction

### 1.1 Background and Motivation

#### 1.1.1 Problem 1: Single-Orthomosaic Crown Detection Is Unreliable

#### 1.1.2 Problem 2: Phenology Labels as Training Data for Satellite Models

### 1.2 Examples of Unreliable Single-Orthomosaic Crown Detection

### 1.3 From Crown Detection to Temporal Tree Phenology

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

#### 3.4.1 QField Annotation Pilot and Its Limitations

#### 3.4.2 Paper-Form Field Labeling Workflow

#### 3.4.3 Crown-ID Orthomosaic Printouts for In-Field Identification

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

#### 4.3.3 Tiled Alignment and Robust Cumulative Shift Aggregation

#### 4.3.4 Applying Alignment to Crown Geometries

### 4.4 Graph-Based Crown Association

#### 4.4.1 Crown Observations as Temporal Graph Nodes

#### 4.4.2 Candidate Edge Construction

#### 4.4.3 Edge Features and Similarity Scoring

#### 4.4.4 Crown Association Cases: One-to-One, Containment, and Nearby

#### 4.4.5 Split, Merge, and Ambiguous Associations

### 4.5 Temporal Chain Extraction

#### 4.5.1 Full Chains

#### 4.5.2 Branching Chains and Backbones

#### 4.5.3 Partial Chains

#### 4.5.4 Gap Filling via Multi-Threshold Candidates

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

#### 4.9.2 Paper-Form to Crown-ID Label Joins

#### 4.9.3 Species and Trait Label Joins

### 4.10 Interactive Visualization and Quality Assurance

#### 4.10.1 Tracking Viewer

#### 4.10.2 Crown Crop Review

#### 4.10.3 Google Earth and Field Review Exports

## 5. Evaluation and Results

### 5.1 Crown Detection Evaluation

### 5.2 Orthomosaic Alignment Evaluation

### 5.3 Crown Tracking Evaluation

#### 5.3.1 Internal Tracking Diagnostics

#### 5.3.2 Manual Tracking Validation Subset

#### 5.3.3 Chain Completeness and Partial-Chain Coverage

### 5.4 Consensus Crown Evaluation

### 5.5 Phenology Feature Evaluation

### 5.6 Visualization-Based Error Analysis

## 6. Seasonal Phenology Mapping

### 6.1 Flowering-Color Labeling per Orthomosaic (Red / Yellow / White / None)

### 6.2 Leaf-Shed Phenophase Progression Across the Season

### 6.3 Species-Resolved Phenology Patterns

### 6.4 Illustrative Crown-Level Trajectories

## 7. Conclusion

## Acknowledgements

## References

## Appendix A. Pipeline Configuration

## Appendix B. Additional Tracking Diagnostics

## Appendix C. Additional Phenology Visualizations

## Appendix D. Dataset and Date Inventory (UAV and Field Data)

### D.1 UAV Orthomosaic Date Inventory

### D.2 LHC Dataset

### D.3 SIT Dataset

### D.4 Additional Site Datasets

### D.5 Field and Crown Label Dataset

### D.6 Pipeline Input and Output Dataset Manifest

### D.7 Dataset Versioning and Reproducibility Notes
