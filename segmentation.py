

# # # Example usage
# # if __name__ == "__main__":
# #     # Path to your orthomosaic
# #     orthomosaic_path = "/Users/gauravrajput/Desktop/IITDelhi/research/all/odm_orthophoto/odm_orthophoto.tif"
    
# #     # Output folder
# #     output_folder = "/Users/gauravrajput/Desktop/IITDelhi/research"
    
# #     # Run tree segmentation with combined approach
# #     segment_trees_from_orthomosaic(
# #         orthomosaic_path=orthomosaic_path,
# #         output_folder=output_folder,
# #         method="combined"  # Try "deeplab", "ndvi", "clustering", or "combined"
# #     )

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array

# # --- (All the functions from the provided code: build_pretrained_unet, preprocess_input, load_orthomosaic, create_tiles, merge_predictions, get_pretrained_model, finetune_model) ---
# # Function to build a U-Net with VGG16 pre-trained encoder
# def build_pretrained_unet(input_size=(256, 256, 3), weights='imagenet'):
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate
#     from tensorflow.keras.optimizers import Adam
#     from tensorflow.keras.applications import VGG16

#     # Get the VGG16 model as encoder
#     vgg16 = VGG16(weights=weights, include_top=False, input_shape=input_size)

#     # Create a dict of the feature maps we'll use in the decoder
#     vgg_layers = dict([(layer.name, layer.output) for layer in vgg16.layers])

#     # Input layer
#     inputs = vgg16.input

#     # Encoder layers (already created from VGG16)
#     # We just need to identify the layers we want to use
#     block1_conv2 = vgg_layers['block1_conv2']  # 256x256
#     block2_conv2 = vgg_layers['block2_conv2']  # 128x128
#     block3_conv3 = vgg_layers['block3_conv3']  # 64x64
#     block4_conv3 = vgg_layers['block4_conv3']  # 32x32
#     block5_conv3 = vgg_layers['block5_conv3']  # 16x16

#     # Freeze the encoder layers
#     for layer in vgg16.layers:
#         layer.trainable = False

#     # Decoder path
#     u6 = UpSampling2D((2, 2))(block5_conv3)
#     u6 = concatenate([u6, block4_conv3])
#     c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
#     c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

#     u7 = UpSampling2D((2, 2))(c6)
#     u7 = concatenate([u7, block3_conv3])
#     c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
#     c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

#     u8 = UpSampling2D((2, 2))(c7)
#     u8 = concatenate([u8, block2_conv2])
#     c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
#     c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

#     u9 = UpSampling2D((2, 2))(c8)
#     u9 = concatenate([u9, block1_conv2])
#     c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
#     c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

#     # Output layer (binary mask for tree segmentation)
#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

#     model = Model(inputs=[inputs], outputs=[outputs])
#     model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

#     return model

# # Function to preprocess image for VGG16
# def preprocess_input(img):
#     # Convert to float
#     img = img.astype(np.float32)

#     # Subtract ImageNet mean (VGG16 was trained with this preprocessing)
#     mean = [123.68, 116.779, 103.939]
#     img[..., 0] -= mean[0]
#     img[..., 1] -= mean[1]
#     img[..., 2] -= mean[2]

#     return img

# # Function to load and preprocess orthomosaic image
# def load_orthomosaic(image_path, target_size=(256, 256)):
#     image = Image.open(image_path).convert('RGB')  # Convert to RGB to remove alpha channel

#     # Get original dimensions
#     width, height = image.size

#     if width > target_size[0] or height > target_size[1]:
#         print(f"Original image size: {width}x{height}, will process in tiles")
#         return image, width, height

#     else:
#         # For small images, just resize
#         image = image.resize(target_size)
#         img_array = img_to_array(image)
#         img_array = preprocess_input(img_array)
#         return np.expand_dims(img_array, axis=0), width, height

# # Function to create tiles from large orthomosaic
# def create_tiles(image, target_size=(256, 256), overlap=32):
#     width, height = image.size
#     tiles = []
#     positions = []

#     for y in range(0, height, target_size[1] - overlap):
#         if y + target_size[1] > height:
#             y = max(0, height - target_size[1])

#         for x in range(0, width, target_size[0] - overlap):
#             if x + target_size[0] > width:
#                 x = max(0, width - target_size[0])

#             tile = image.crop((x, y, x + target_size[0], y + target_size[1])).convert('RGB')  # Convert to RGB to remove alpha channel
#             tile_array = img_to_array(tile)
#             tile_array = preprocess_input(tile_array)
#             tiles.append(tile_array)
#             positions.append((x, y))

#             if x + target_size[0] == width:
#                 break

#         if y + target_size[1] == height:
#             break

#     return np.array(tiles), positions

# # Function to merge prediction tiles back into a full image
# def merge_predictions(predictions, positions, original_size, target_size=(256, 256)):
#     full_prediction = np.zeros((original_size[1], original_size[0]))
#     count_map = np.zeros((original_size[1], original_size[0]))

#     for pred, (x, y) in zip(predictions, positions):
#         pred = pred.squeeze()

#         for i in range(target_size[1]):
#             for j in range(target_size[0]):
#                 if y + i < original_size[1] and x + j < original_size[0]:
#                     full_prediction[y + i, x + j] += pred[i, j]
#                     count_map[y + i, x + j] += 1

#     # Average overlapping predictions
#     count_map[count_map == 0] = 1  # Avoid division by zero
#     full_prediction = full_prediction / count_map

#     return full_prediction

# # Function to download a pre-trained tree segmentation model or load local model
# def get_pretrained_model(target_size=(256, 256, 3)):
#     # If no model is available, create a model with pre-trained VGG16 encoder
#     print("Creating U-Net with pre-trained VGG16 encoder")
#     model = build_pretrained_unet(input_size=target_size, weights='imagenet')
#     print("WARNING: This model has a pre-trained encoder but the decoder is not trained for tree segmentation.")
#     print("For optimal results, fine-tune this model with labeled tree data.")
#     return model

# # Function to fine-tune the model if you have labeled data
# def finetune_model(image_dir, mask_dir, model_path=None,
#                    output_model_path='tree_segmentation_finetuned.h5',
#                    target_size=(256, 256),
#                    epochs=50,
#                    batch_size=8):
#     from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#     from sklearn.model_selection import train_test_split
#     from tensorflow.keras.preprocessing.image import load_img, img_to_array
#     import numpy as np
#     import os

#     # Load images and masks
#     print("Loading training data...")
#     images = []
#     masks = []
#     image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))])
#     mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))])

#     for img_file, mask_file in zip(image_files, mask_files):
#         # Load and preprocess image
#         img = load_img(os.path.join(image_dir, img_file), target_size=target_size)
#         img_array = img_to_array(img)
#         img_array = preprocess_input(img_array)
#         images.append(img_array)

#         # Load and preprocess mask
#         mask = load_img(os.path.join(mask_dir, mask_file),
#                         target_size=target_size, color_mode='grayscale')
#         mask_array = img_to_array(mask) / 255.0  # Normalize to [0,1]
#         masks.append(mask_array)

#     X = np.array(images)
#     y = np.array(masks)

#     # Split data
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Get the model
#     model = get_pretrained_model(model_path, target_size=(*target_size, 3))

#     # Unfreeze some of the encoder layers for fine-tuning
#     # Usually we keep early layers frozen and train later layers
#     if hasattr(model, 'layers'):
#         # Find VGG16 layers
#         vgg_layers = [layer for layer in model.layers if
#                       any(block in layer.name for block in ['block4', 'block5'])]

#         # Unfreeze later blocks while keeping early blocks frozen
#         for layer in vgg_layers:
#             layer.trainable = True

#     # Setup callbacks
#     callbacks = [
#         ModelCheckpoint(output_model_path, save_best_only=True, monitor='val_loss'),
#         EarlyStopping(patience=8, monitor='val_loss', restore_best_weights=True)
#     ]

#     # Train model
#     print(f"Fine-tuning model for {epochs} epochs...")
#     model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         batch_size=batch_size,
#         epochs=epochs,
#         callbacks=callbacks
#     )

#     print(f"Model fine-tuning complete. Saved to {output_model_path}")
#     return model

# # Modified segment_trees function
# def segment_trees(orthomosaic_path,output_path='tree_segmentation.png', target_size=(256, 256), overlay_color=(0, 256, 0, 128)):
#     try:
#         # Get model (either loaded or created with pre-trained encoder)
#         model = get_pretrained_model(target_size=(*target_size, 3))

#         # Load and process the orthomosaic
#         print(f"Processing orthomosaic: {orthomosaic_path}")
#         image, width, height = load_orthomosaic(orthomosaic_path, target_size)

#         # Perform prediction
#         if isinstance(image, np.ndarray) and image.shape[0] == 1:
#             print("Processing small image...")
#             prediction = model.predict(image)
#             segmentation = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255  # Threshold at 0.5
#         else:
#             print("Processing large image in tiles...")
#             tiles, positions = create_tiles(image, target_size)
#             print(f"Created {len(tiles)} tiles for processing")
#             predictions = model.predict(tiles)
#             segmentation = merge_predictions(predictions, positions, (width, height), target_size)
#             segmentation = (segmentation > 0.5).astype(np.uint8) * 255  # Threshold at 0.5

#         # Load the original image
#         original_image = Image.open(orthomosaic_path).convert('RGBA')

#         # Resize segmentation to original image size
#         segmentation_img = Image.fromarray(segmentation).convert('L').resize((original_image.width, original_image.height), resample=Image.NEAREST)
#         segmentation_img = segmentation_img.convert('RGBA')

#         # Create overlay color
#         overlay_color = overlay_color  # RGBA: Green with 50% transparency

#         # Apply the specified color to the segmentation mask
#         colored_mask = Image.new('RGBA', (original_image.width, original_image.height), overlay_color)

#         # Composite the colored mask onto the original image using the segmentation as an alpha mask
#         overlaid_image = Image.composite(colored_mask, original_image, segmentation_img.split()[0])

#         # Save the overlaid image
#         overlaid_image.save(output_path)
#         print(f"Tree segmentation overlaid on the original image and saved to {output_path}")

#         return overlaid_image

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None


# # Example usage
# if __name__ == "__main__":
#     # Replace with the actual path to your orthomosaic
#     orthomosaic_path = "/Users/gauravrajput/Desktop/IITDelhi/research/all/odm_orthophoto/odm_orthophoto.tif"  # Replace with your orthomosaic path
#     # Replace with your trained model path if available
#     output_path = "segmented_image.png"

#     if os.path.exists(orthomosaic_path):
#         overlaid_image = segment_trees(orthomosaic_path, output_path)
#         if overlaid_image:
#             overlaid_image.show()  # Display the image (optional)
#     else:
#         print("Please specify a valid path to your orthomosaic image to run the segmentation.")
#         print("Example usage:")
#         print(" segment_trees('your_orthomosaic.tif', 'your_trained_model.h5')")


import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet101

class TreeSegmenter:
    def __init__(self, weights_path=None):
        # Initialize DeepLabV3+ with pretrained ResNet101 backbone
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the model
        self.model = deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
        
        # Modify the classifier to output binary segmentation (trees vs background)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        
        # Load custom weights if provided
        if weights_path and os.path.exists(weights_path):
            print(f"Loading pretrained weights from {weights_path}")
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        # Set model to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Define transformations for input images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def segment_tiles(self, input_image_path, output_path, tile_size=512, overlap=64):
        """
        Process large orthomosaic by tiling it and then stitching results.
        
        Args:
            input_image_path: Path to the orthomosaic image
            output_path: Path to save the segmentation result
            tile_size: Size of tiles to process
            overlap: Overlap between adjacent tiles to avoid edge artifacts
        """
        # Open input image
        original_img = Image.open(input_image_path).convert('RGB')  # Ensure image is in RGB format
        width, height = original_img.size
        
        # Create empty mask for the output
        output_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate effective stride
        stride = tile_size - overlap
        
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Extract tile
                end_x = min(x + tile_size, width)
                end_y = min(y + tile_size, height)
                
                # Adjust starting position if we're at the boundary
                start_x = max(0, end_x - tile_size)
                start_y = max(0, end_y - tile_size)
                
                tile = original_img.crop((start_x, start_y, end_x, end_y))
                
                # Process tile
                tile_mask = self.segment_image(tile)
                
                # Place result in the output mask
                output_mask[start_y:end_y, start_x:end_x] = tile_mask
                
                print(f"Processed tile at ({start_x}, {start_y})")
        
        # Create a mask image from the output mask
        mask_img = Image.fromarray(output_mask * 255).convert('L')
        
        # Composite the original image with the mask
        segmented_img = Image.composite(original_img, Image.new('RGB', original_img.size, (0, 0, 0)), mask_img)
        
        # Save the output
        segmented_img.save(output_path)
        print(f"Segmentation completed and saved to {output_path}")
        
        return segmented_img
    
    def segment_image(self, image):
        """
        Segment a single image or tile.
        
        Args:
            image: PIL Image
            
        Returns:
            Binary mask with 1 for trees, 0 for background
        """
        # Ensure image is in RGB format
        image = image.convert('RGB')
        
        # Prepare image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)['out']
            prediction = torch.sigmoid(output) > 0.5
        
        # Convert to numpy array
        mask = prediction.squeeze().cpu().numpy().astype(np.uint8)
        
        return mask

    def fine_tune(self, train_dataset, val_dataset, epochs=10, batch_size=8, learning_rate=1e-4):
        """
        Fine-tune the model on your own data.
        
        Args:
            train_dataset: PyTorch dataset for training
            val_dataset: PyTorch dataset for validation
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        from torch.utils.data import DataLoader
        import torch.optim as optim
        
        # Prepare dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Set model to training mode
        self.model.train()
        
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)['out']
                loss = criterion(outputs, targets.unsqueeze(1))
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss = train_loss / len(train_loader.dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)['out']
                    loss = criterion(outputs, targets.unsqueeze(1))
                    
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Set back to training mode
            self.model.train()
        
        # Save the fine-tuned model
        torch.save(self.model.state_dict(), 'tree_segmentation_finetuned.pth')
        print("Fine-tuning complete. Model saved to tree_segmentation_finetuned.pth")

# Example usage
if __name__ == "__main__":
    # Initialize the model
    segmenter = TreeSegmenter(weights_path='tree_segmentation_weights.pth')
    
    # Process an example orthomosaic
    input_path = "/Users/gauravrajput/Desktop/IITDelhi/research/all/odm_orthophoto/odm_orthophoto.tif"
    output_path = "/Users/gauravrajput/Desktop/IITDelhi/research/output/segmented_image.png"
    
    # Segment the orthomosaic
    segmenter.segment_tiles(input_path, output_path)
    
    # Optionally, show the result
    result = Image.open(output_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    plt.title("Tree Segmentation Result")
    plt.axis('off')
    plt.show()