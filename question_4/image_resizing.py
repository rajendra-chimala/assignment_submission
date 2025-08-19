import cv2 as cv
import os

def resize_image_operations(image_path, output_dir="resized_outputs"):
    
    image = cv.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    original_height, original_width = image.shape[:2]
    print(f"Original Dimensions: {original_width}x{original_height}")

    # Resize 50% smaller
    small = cv.resize(image, (0, 0), fx=0.5, fy=0.5)
    
    cv.imshow("50% Smaller Image", small)
    cv.imwrite(f"{output_dir}/resized_50_percent_smaller.jpg", small)

    # Resize 200% larger
    large = cv.resize(image, (0, 0), fx=2.0, fy=2.0)
   
    cv.imshow("200% Larger Image", large)
    cv.imwrite(f"{output_dir}/resized_200_percent_larger.jpg", large)

    #  300x300 
    fixed = cv.resize(image, (300, 300))
    cv.imwrite(f"{output_dir}/resized_fixed_300x300.jpg", fixed)

    # Maintain aspect ratio
    aspect_ratio = original_height / original_width
    new_height = int(300 * aspect_ratio)
    aspect_resized = cv.resize(image, (300, new_height))
    cv.imshow("Aspect Ratio Resized Image", aspect_resized)
    cv.imwrite(f"{output_dir}/resized_300_aspect_ratio.jpg", aspect_resized)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

    
resize_image_operations("sample_images\image0.jpg")

