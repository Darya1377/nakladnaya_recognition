import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(image_path, output_folder):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    blurred = cv2.GaussianBlur(clahe_img, (3, 3), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel / np.max(sobel) * 255)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    combined = cv2.bitwise_or(sobel, thresh)
    height, width = combined.shape
    top = combined[0:height//10, 0:width]
    bottom = combined[9*height//10:height, 0:width]
    left = combined[0:height, 0:width//10]
    right = combined[0:height, 9*width//10:width]
    
    border_pixels = np.concatenate([top.flatten(), bottom.flatten(), 
                                  left.flatten(), right.flatten()])
    white_percentage = np.sum(border_pixels > 128) / len(border_pixels)

    if white_percentage > 0.5:
        final = cv2.bitwise_not(combined)
    else:
        final = combined

    kernel = np.ones((2, 2), np.uint8)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, final)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            try:
                process_image(image_path, output_folder)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    input_folder = r"C:\Users\user\Desktop\img2text\test"
    output_folder = "new_test"
    
    process_folder(input_folder, output_folder)

    print("Processing complete!")
