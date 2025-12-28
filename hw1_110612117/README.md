# HW1: Vehicle Detection and Classification

Author: Chung-Yu Chang (張仲瑜)  
Student ID: 110612117

## Introduction
This project focuses on vehicle detection and classification using two approaches: 
Task A leverages classical machine learning models with Scikit-learn, and Task B fine-tunes the YOLOv7 object detection model.

## Methodology

### Task A: Classical Machine Learning
- Data preprocessing: load images in grayscale, resize to 36 x 16 with `cv2.INTER_AREA`, and flatten to 576 features.
- Models compared:
	- K-Nearest Neighbors (KNN): $k = 1$, $p = 1$ (Manhattan distance).
	- Random Forest (RF): 100 estimators, entropy criterion.
	- AdaBoost (AB): 60 estimators.
	- Gradient Boosting (GB): 140 estimators.
- Parking space detection: use a perspective transform (`crop`) to extract parking spaces from video frames, then feed crops to the trained classifier for real-time status monitoring.

### Task B: Deep Learning (YOLOv7)
- Fine-tuning: transfer learning on YOLOv7-tiny.
- Training: 300 epochs with strong convergence.
- Evaluation: training accuracy 99.67%, False Positive Rate 0.0, False Negative Rate 0.0067.

## Implementation Details
- Environment: Python 3.9+, OpenCV, NumPy, Scikit-learn, PyTorch (for YOLOv7).
- Key scripts:
	- dataset.py: image loading, resizing, labeling.
	- model.py: `CarClassifier` for training and evaluating ML models.
	- detection.py: detection logic for video stream processing.

## Results & Conclusion
- Task A: Random Forest and Gradient Boosting provide robust baselines for structured feature classification.
- Task B: YOLOv7 fine-tuning significantly outperforms classical methods, achieving near-perfect accuracy after training.