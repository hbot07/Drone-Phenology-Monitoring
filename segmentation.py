import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from segment_anything import sam_model_registry, SamPredictor
import cv2
import os
from tqdm import tqdm

# Configuration
SAM_CHECKPOINT = "/content/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 256  # Adjust based on available memory
OVERLAP = 128  # Overlap between chunks to avoid edge artifacts

def load_sam_model(checkpoint_path, model_type, device):
    """Load the SAM model with error handling."""
    try:
        print(f"Loading SAM model to {device}...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        return None

def verify_file(file_path):
    """Verify if the file exists and can be opened."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with rasterio.open(file_path) as src:
            print(f"File verified: {file_path}")
            print(f"Dimensions: {src.width}x{src.height}, Bands: {src.count}")
            return True
    except Exception as e:
        print(f"Error opening file: {e}")
        return False

def process_orthomosaic(input_path, output_path, predictor):
    """Process orthomosaic and save as a JPG file."""
    if not verify_file(input_path):
        return False
    
    try:
        with rasterio.open(input_path) as src:
            height, width = src.shape
            final_image = np.zeros((height, width, 3), dtype=np.uint8)  # Create an empty RGB image
            
            total_chunks = ((height + CHUNK_SIZE - 1) // CHUNK_SIZE) * ((width + CHUNK_SIZE - 1) // CHUNK_SIZE)
            with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
                for i in range(0, height, CHUNK_SIZE):
                    for j in range(0, width, CHUNK_SIZE):
                        chunk_window = Window(
                            max(j - OVERLAP, 0),
                            max(i - OVERLAP, 0),
                            min(CHUNK_SIZE + 2 * OVERLAP, width - max(j - OVERLAP, 0)),
                            min(CHUNK_SIZE + 2 * OVERLAP, height - max(i - OVERLAP, 0))
                        )

                        try:
                            image = src.read(window=chunk_window)
                            
                            if image.shape[0] > 3:
                                image = image[:3]  # Convert to RGB

                            image = np.moveaxis(image, 0, -1)  # Convert to HWC format
                            
                            processed_chunk = process_chunk(image, predictor)
                            
                            start_y = OVERLAP if i > 0 else 0
                            start_x = OVERLAP if j > 0 else 0
                            end_y = processed_chunk.shape[0] - OVERLAP if i + CHUNK_SIZE < height else processed_chunk.shape[0]
                            end_x = processed_chunk.shape[1] - OVERLAP if j + CHUNK_SIZE < width else processed_chunk.shape[1]

                            processed_chunk = processed_chunk[start_y:end_y, start_x:end_x]

                            final_image[i:i+processed_chunk.shape[0], j:j+processed_chunk.shape[1]] = processed_chunk
                            
                        except Exception as e:
                            print(f"Error processing chunk at ({j}, {i}): {e}")
                        
                        pbar.update(1)

            # Save as JPG
            cv2.imwrite(output_path.replace(".tif", ".jpg"), final_image)
            print(f"Processing complete. Output saved to: {output_path.replace('.tif', '.jpg')}")
            return True

    except Exception as e:
        print(f"Error processing orthomosaic: {e}")
        return False


def process_chunk(chunk, predictor):
    """Process a single chunk of the orthomosaic."""
    original_shape = chunk.shape[:2]
    
    # Ensure chunk is in the correct format
    if chunk.dtype != np.uint8:
        # Normalize to 0-255 range
        chunk = cv2.normalize(chunk, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Resize for SAM processing
    max_size = 1024  # Maximum size for SAM processing
    scale = min(max_size/original_shape[0], max_size/original_shape[1])
    if scale < 1:
        new_height = int(original_shape[0] * scale)
        new_width = int(original_shape[1] * scale)
        processed_img = cv2.resize(chunk, (new_width, new_height))
    else:
        processed_img = chunk
    
    # Generate automatic points for tree detection
    # This is a simple approach - you might want to use a more sophisticated method
    # for actual tree detection, such as a detector that generates bounding boxes
    points = detect_trees(processed_img)
    
    # Generate masks using SAM
    predictor.set_image(processed_img)
    composite_mask = np.zeros((processed_img.shape[0], processed_img.shape[1]), dtype=bool)
    
    for point in points:
        # For each point, predict mask
        masks, scores, _ = predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array([1]),  # 1 for foreground
            multimask_output=True
        )
        
        # Use the best mask
        best_mask_idx = np.argmax(scores)
        composite_mask |= masks[best_mask_idx]
    
    # Resize mask back to original size
    if scale < 1:
        mask = cv2.resize(
            composite_mask.astype(np.uint8),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        mask = composite_mask
    
    # Apply mask to original chunk
    masked_chunk = chunk.copy()
    masked_chunk[~mask] = 0  # Set non-tree pixels to black
    
    return masked_chunk

def detect_trees(image):
    """Simple tree detection using color thresholding."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define green color range
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([80, 255, 255])
    
    # Create mask for green areas
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = 100  # Minimum area for a tree
    tree_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Get centroids of contours
    points = []
    for cnt in tree_contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append([cx, cy])
    
    return points

def main():
    # Configuration
    input_path = "/content/sample_data/odm_orthophoto.tif"
    output_path = "/content/sample_data/trees_detected.tif"
    
    # Load SAM model
    predictor = load_sam_model(SAM_CHECKPOINT, SAM_MODEL_TYPE, DEVICE)
    if predictor is None:
        print("Failed to load SAM model. Exiting.")
        return
    
    # Process orthomosaic
    process_orthomosaic(input_path, output_path, predictor)

if __name__ == "__main__":
    main()