# pipeline_class.py

import os
import cv2
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class ImagePipeline:
    def __init__(self, input_dir, output_dir, standard_size=(512, 512)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.standard_size = standard_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise IOError("Haar cascade XML file not found!")
        os.makedirs(self.output_dir, exist_ok=True)
        self.summary = []
        self.collage_images = []

    def compute_edge_density(self, edge_img):
        return np.sum(edge_img > 0) / edge_img.size

    def compute_contrast(self, gray_img):
        return np.std(gray_img)

    def create_collage(self, images, rows, cols, title):
        h, w = images[0].shape[:2]
        collage = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for idx, img in enumerate(images):
            y, x = divmod(idx, cols)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            collage[y*h:(y+1)*h, x*w:(x+1)*w] = img
        cv2.imwrite(os.path.join(self.output_dir, title), collage)

    def process_images(self):
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for filename in image_files:
            filepath = os.path.join(self.input_dir, filename)
            original = cv2.imread(filepath)
            if original is None:
                print(f"[Warning] Skipping unreadable image: {filename}")
                continue

            resized = cv2.resize(original, self.standard_size)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            edges = cv2.Canny(blurred, 100, 200)
            edge_density = self.compute_edge_density(edges)

            faces = self.face_cascade.detectMultiScale(blurred, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

            mean_intensity = np.mean(blurred)
            contrast = self.compute_contrast(blurred)

            self.summary.append([filename, mean_intensity, contrast, edge_density, len(faces)])
            self.collage_images.extend([
                cv2.resize(original, (256, 256)),
                cv2.resize(edges, (256, 256))
            ])

            cv2.imwrite(os.path.join(self.output_dir, f"{filename}_gray.jpg"), gray)
            cv2.imwrite(os.path.join(self.output_dir, f"{filename}_edges.jpg"), edges)
            cv2.imwrite(os.path.join(self.output_dir, f"{filename}_faces.jpg"), resized)

        return self.summary, self.collage_images

    def analyze_features(self):
        summary_np = np.array([row[1:] for row in self.summary], dtype=np.float32)
        scaler = StandardScaler()
        summary_norm = scaler.fit_transform(summary_np)
        cov_matrix = np.cov(summary_norm.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        densities = summary_np[:, 2]
        max_density_idx = np.argmax(densities)
        most_edges_filename = self.summary[max_density_idx][0]

        return summary_np, eigenvalues, most_edges_filename

    def save_report(self, summary_np, eigenvalues, most_edges_filename):
        report_path = os.path.join(self.output_dir, "report.txt")
        with open(report_path, "w") as report:
            report.write(f"Image Analysis Report - {datetime.now()}\n")
            report.write("="*40 + "\n")
            for row in self.summary:
                report.write(f"File: {row[0]}\n")
                report.write(f"Mean Intensity: {row[1]:.2f}, Contrast: {row[2]:.2f}, Edge Density: {row[3]:.4f}, Faces: {row[4]}\n\n")
            report.write(f"Image with highest edge density: {most_edges_filename}\n\n")
            report.write("Eigenvalues of Feature Covariance Matrix:\n")
            report.write(", ".join(f"{val:.4f}" for val in eigenvalues))
        return report_path
