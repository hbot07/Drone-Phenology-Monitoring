

# # Example usage
# if __name__ == "__main__":
#     # Path to your orthomosaic
#     orthomosaic_path = "/Users/gauravrajput/Desktop/IITDelhi/research/all/odm_orthophoto/odm_orthophoto.tif"
    
#     # Output folder
#     output_folder = "/Users/gauravrajput/Desktop/IITDelhi/research"
    
#     # Run tree segmentation with combined approach
#     segment_trees_from_orthomosaic(
#         orthomosaic_path=orthomosaic_path,
#         output_folder=output_folder,
#         method="combined"  # Try "deeplab", "ndvi", "clustering", or "combined"
#     )

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Function to build a U-Net with VGG16 pre-trained encoder
def build_pretrained_unet(input_size=(256, 256, 3), weights='imagenet'):
    # Get the VGG16 model as encoder
    vgg16 = VGG16(weights=weights, include_top=False, input_shape=input_size)
    
    # Create a dict of the feature maps we'll use in the decoder
    vgg_layers = dict([(layer.name, layer.output) for layer in vgg16.layers])
    
    # Input layer
    inputs = vgg16.input
    
    # Encoder layers (already created from VGG16)
    # We just need to identify the layers we want to use
    block1_conv2 = vgg_layers['block1_conv2']  # 256x256
    block2_conv2 = vgg_layers['block2_conv2']  # 128x128
    block3_conv3 = vgg_layers['block3_conv3']  # 64x64
    block4_conv3 = vgg_layers['block4_conv3']  # 32x32
    block5_conv3 = vgg_layers['block5_conv3']  # 16x16
    
    # Freeze the encoder layers
    for layer in vgg16.layers:
        layer.trainable = False
    
    # Decoder path
    u6 = UpSampling2D((2, 2))(block5_conv3)
    u6 = concatenate([u6, block4_conv3])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, block3_conv3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, block2_conv2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, block1_conv2])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    # Output layer (binary mask for tree segmentation)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Function to preprocess image for VGG16
def preprocess_input(img):
    # Convert to float
    img = img.astype(np.float32)
    # Subtract ImageNet mean (VGG16 was trained with this preprocessing)
    mean = [123.68, 116.779, 103.939]
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    return img

# Function to load and preprocess orthomosaic image
def load_orthomosaic(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB to remove alpha channel
    
    # Get original dimensions
    width, height = image.size
    
    if width > target_size[0] or height > target_size[1]:
        print(f"Original image size: {width}x{height}, will process in tiles")
        return image, width, height
    else:
        # For small images, just resize
        image = image.resize(target_size)
        img_array = img_to_array(image)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0), width, height

# Function to create tiles from large orthomosaic
def create_tiles(image, target_size=(256, 256), overlap=32):
    width, height = image.size
    tiles = []
    positions = []
    
    for y in range(0, height, target_size[1] - overlap):
        if y + target_size[1] > height:
            y = max(0, height - target_size[1])
            
        for x in range(0, width, target_size[0] - overlap):
            if x + target_size[0] > width:
                x = max(0, width - target_size[0])
                
            tile = image.crop((x, y, x + target_size[0], y + target_size[1])).convert('RGB')  # Convert to RGB to remove alpha channel
            tile_array = img_to_array(tile)
            tile_array = preprocess_input(tile_array)
            tiles.append(tile_array)
            positions.append((x, y))
            
            if x + target_size[0] == width:
                break
                
        if y + target_size[1] == height:
            break
            
    return np.array(tiles), positions

# Function to merge prediction tiles back into a full image
def merge_predictions(predictions, positions, original_size, target_size=(256, 256)):
    full_prediction = np.zeros((original_size[1], original_size[0]))
    count_map = np.zeros((original_size[1], original_size[0]))
    
    for pred, (x, y) in zip(predictions, positions):
        pred = pred.squeeze()
        for i in range(target_size[1]):
            for j in range(target_size[0]):
                if y + i < original_size[1] and x + j < original_size[0]:
                    full_prediction[y + i, x + j] += pred[i, j]
                    count_map[y + i, x + j] += 1
    
    # Average overlapping predictions
    count_map[count_map == 0] = 1  # Avoid division by zero
    full_prediction = full_prediction / count_map
    
    return full_prediction

# Function to download a pre-trained tree segmentation model or load local model
def get_pretrained_model(model_path=None, target_size=(256, 256, 3)):
    # If a specific model path is provided, try to load it
    if model_path and os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
    
    # If no model is available, create a model with pre-trained VGG16 encoder
    print("Creating U-Net with pre-trained VGG16 encoder")
    model = build_pretrained_unet(input_size=target_size, weights='imagenet')
    
    print("WARNING: This model has a pre-trained encoder but the decoder is not trained for tree segmentation.")
    print("For optimal results, fine-tune this model with labeled tree data.")
    
    return model

# Main function to segment trees in an orthomosaic image
def segment_trees(orthomosaic_path, model_path=None, output_path='tree_segmentation.png', target_size=(256, 256)):
    # Get model (either loaded or created with pre-trained encoder)
    model = get_pretrained_model(model_path, target_size=(*target_size, 3))
    
    # Load and process the orthomosaic
    print(f"Processing orthomosaic: {orthomosaic_path}")
    image, width, height = load_orthomosaic(orthomosaic_path, target_size)
    
    # If the image is already small enough
    if isinstance(image, np.ndarray) and image.shape[0] == 1:
        print("Processing small image...")
        prediction = model.predict(image)
        segmentation = (prediction[0, :, :, 0] > 0.05).astype(np.uint8) * 255
        
    else:  # For larger images, process in tiles
        print("Processing large image in tiles...")
        tiles, positions = create_tiles(image, target_size)
        print(f"Created {len(tiles)} tiles for processing")
        predictions = model.predict(tiles)
        segmentation = merge_predictions(predictions, positions, (width, height), target_size)
        segmentation = (segmentation > 0.5).astype(np.uint8) * 255
    
    # Save the segmentation result
    plt.figure(figsize=(10, 10))
    plt.imshow(segmentation, cmap='gray')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Tree segmentation saved to {output_path}")
    return segmentation

# Function to fine-tune the model if you have labeled data
def finetune_model(image_dir, mask_dir, model_path=None, 
                  output_model_path='tree_segmentation_finetuned.h5', 
                  target_size=(256, 256), 
                  epochs=50, 
                  batch_size=8):
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.model_selection import train_test_split
    
    # Load images and masks
    print("Loading training data...")
    images = []
    masks = []
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))])
    
    for img_file, mask_file in zip(image_files, mask_files):
        # Load and preprocess image
        img = load_img(os.path.join(image_dir, img_file), target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        images.append(img_array)
        
        # Load and preprocess mask
        mask = load_img(os.path.join(mask_dir, mask_file), 
                         target_size=target_size, color_mode='grayscale')
        mask_array = img_to_array(mask) / 255.0  # Normalize to [0,1]
        masks.append(mask_array)
    
    X = np.array(images)
    y = np.array(masks)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get the model
    model = get_pretrained_model(model_path, target_size=(*target_size, 3))
    
    # Unfreeze some of the encoder layers for fine-tuning
    # Usually we keep early layers frozen and train later layers
    if hasattr(model, 'layers'):
        # Find VGG16 layers
        vgg_layers = [layer for layer in model.layers if any(block in layer.name for block in ['block4', 'block5'])]
        
        # Unfreeze later blocks while keeping early blocks frozen
        for layer in vgg_layers:
            layer.trainable = True
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(output_model_path, save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=8, monitor='val_loss', restore_best_weights=True)
    ]
    
    # Train model
    print(f"Fine-tuning model for {epochs} epochs...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print(f"Model fine-tuning complete. Saved to {output_model_path}")
    return model

# Example usage
if __name__ == "__main__":
    # Example for segmenting a single orthomosaic
    # segment_trees('path_to_your_orthomosaic.tif', model_path='path_to_trained_model.h5')
    
    # Example for fine-tuning on labeled data
    # finetune_model('path_to_training_images', 'path_to_training_masks')
    
    # Demo with a single image (you'll need to replace with your actual path)
    orthomosaic_path = "/Users/gauravrajput/Desktop/IITDelhi/research/all/odm_orthophoto/odm_orthophoto.tif"  # Replace with your orthomosaic path
    
    # If you have a pre-trained model
    model_path = None  # Replace with your model path if available
    
    # If the orthomosaic path exists, run segmentation
    if os.path.exists(orthomosaic_path):
        segment_trees(orthomosaic_path, model_path)
    else:
        print("Please specify a valid path to your orthomosaic image to run the segmentation.")
        print("Example usage:")
        print("    segment_trees('your_orthomosaic.tif', 'your_trained_model.h5')")