import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image

# Load and convert to grayscale
img_color = cv2.imread('img\image0.jpg')  
if img_color is None:
    print("[Error] Image not found. Make sure 'sample.jpg' exists.")
    exit()

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Represent grayscale image as a 2D matrix
gray_matrix = np.array(img_gray, dtype=np.float32)

# Extract 5x5 patches from the image and compute the covariance matrix
patches = image.extract_patches_2d(gray_matrix, (5, 5))
patches_reshaped = patches.reshape(patches.shape[0], -1) 

# Mean-center the data
mean_patch = np.mean(patches_reshaped, axis=0)
patches_centered = patches_reshaped - mean_patch

#Compute covariance matrix (features x features)
cov_matrix = np.cov(patches_centered, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  

alpha = 1.2  
transform_matrix = np.array([[alpha]])  

# Apply transformation
transformed_matrix = gray_matrix * transform_matrix
transformed_matrix = np.clip(transformed_matrix, 0, 255).astype(np.uint8)

#Display original and transformed images side-by-side
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(gray_matrix, cmap='gray')
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(transformed_matrix, cmap='gray')
plt.title(f'Transformed (alpha={alpha})')
plt.axis('off')
plt.tight_layout()
plt.show()

#Plot histograms of intensity distributions
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(gray_matrix.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('Histogram - Original')

plt.subplot(1, 2, 2)
plt.hist(transformed_matrix.ravel(), bins=256, range=[0, 256], color='orange')
plt.title('Histogram - Transformed')
plt.tight_layout()
plt.show()

