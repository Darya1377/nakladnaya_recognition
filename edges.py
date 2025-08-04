import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(image_path, output_folder):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(clahe_img, (3, 3), 0)
    
    # Edge detection using Sobel
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel / np.max(sobel) * 255)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Combine edge detection with thresholding
    combined = cv2.bitwise_or(sobel, thresh)
    
    # Determine if we need to invert (black text on white background)
    height, width = combined.shape
    top = combined[0:height//10, 0:width]
    bottom = combined[9*height//10:height, 0:width]
    left = combined[0:height, 0:width//10]
    right = combined[0:height, 9*width//10:width]
    
    border_pixels = np.concatenate([top.flatten(), bottom.flatten(), 
                                  left.flatten(), right.flatten()])
    white_percentage = np.sum(border_pixels > 128) / len(border_pixels)
    
    # Invert if needed (to ensure black text on white background)
    if white_percentage > 0.5:
        final = cv2.bitwise_not(combined)
    else:
        final = combined
    
    # Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)
    
    # Save the processed image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, final)

def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            try:
                process_image(image_path, output_folder)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    input_folder = r"C:\Users\user\Desktop\img2text\test"  # Folder containing your original images
    output_folder = "new_test"  # Folder where processed images will be saved
    
    process_folder(input_folder, output_folder)
    print("Processing complete!")