# **Drone-Based Landscape Monitoring System**

## **Project Overview**
This project aims to develop an automated system for monitoring the IIT Delhi campus landscape using drones. The system will:
- Capture aerial images using drones.
- Generate an orthomosaic (stitched map) with GPS coordinates.
- Detect trees using deep learning (DeepForest).
- Track trees over time for phenology analysis (leaf loss, flowering, fruiting, etc.).
- Classify tree species and analyze environmental impacts.
- Compare ground data with satellite images.

---

## **✅ Completed Steps**

### **1. Capturing Drone Images**
- Planned flight paths for the **DJI Pro** drone.
- Captured high-resolution aerial images of the IITD campus.

### **2. Creating Orthomosaic**
- Used **OpenDroneMap** to stitch drone images into a single orthomosaic.
- Ensured the generated map contains **GPS coordinates**.

### **3. Tree Detection Using DeepForest**
- Used **DeepForest** to identify tree crowns from the orthomosaic.
- Extracted tree coordinates in latitude/longitude format.
- Generated an **annotated image** with detected trees.
- Optimized detection by filtering low-confidence results.

### **4. Automating Data Processing**
- Created a script to:
  - Load orthomosaic images.
  - Predict tree locations.
  - Convert pixel positions to GPS coordinates.
  - Save results as annotated images.