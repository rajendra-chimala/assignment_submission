import cv2
import numpy as np
import argparse

def apply_filters(image, k_gauss, k_avg, k_median):
    k_gauss = max(3, k_gauss | 1)
    k_avg = max(3, k_avg | 1)
    k_median = max(3, k_median | 1)

    # Blurring
    gaussian = cv2.GaussianBlur(image, (k_gauss, k_gauss), 0)
    average = cv2.blur(image, (k_avg, k_avg))
    median = cv2.medianBlur(image, k_median)

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection
    canny = cv2.Canny(gray, 100, 200)

    # Sobel Edge Detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    return [image, gaussian, average, median, canny, sobel]

def stack_images(images, scale=0.5):
    img_row1 = cv2.hconcat([cv2.resize(img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                        (0, 0), fx=scale, fy=scale) for img in images[:3]])
    img_row2 = cv2.hconcat([cv2.resize(img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                        (0, 0), fx=scale, fy=scale) for img in images[3:]])
    return cv2.vconcat([img_row1, img_row2])

def main():
    parser = argparse.ArgumentParser(description="Image Filtering Tool")
    parser.add_argument('--image', required=True, help="Path to input image")
    parser.add_argument('--gauss', type=int, default=15, help="Gaussian blur kernel size (odd number)")
    parser.add_argument('--avg', type=int, default=10, help="Average blur kernel size (odd number)")
    parser.add_argument('--median', type=int, default=9, help="Median blur kernel size (odd number)")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print("Error: Image not found!")
        return

    results = apply_filters(image, args.gauss, args.avg, args.median)
    combined = stack_images(results)

    cv2.imshow("Image Filtering Results", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
