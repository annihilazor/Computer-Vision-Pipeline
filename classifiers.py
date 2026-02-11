"""
SVM Classifiers and Bounding Box Regressors for R-CNN
"""
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Ridge
import pickle
import os
from typing import List, Dict, Tuple


class SVMClassifier:
    """Binary SVM classifier for each object class"""

    def __init__(self, kernel: str = 'linear', C: float = 1.0):
        """
        Args:
            kernel: SVM kernel type ('linear' or 'rbf')
            C: Regularization parameter
        """
        self.kernel = kernel
        self.C = C

        if kernel == 'linear':
            # LinearSVC is faster for linear kernel
            self.svm = LinearSVC(C=C, max_iter=10000, dual=True)
        else:
            self.svm = SVC(kernel=kernel, C=C, probability=True)

        self.is_fitted = False

    def train(self, features: np.ndarray, labels: np.ndarray):
        """
        Train SVM classifier

        Args:
            features: Feature matrix (N, feature_dim)
            labels: Binary labels (N,) - 1 for positive, 0 for negative
        """
        self.svm.fit(features, labels)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            features: Feature matrix (N, feature_dim)

        Returns:
            Predicted labels (N,)
        """
        if not self.is_fitted:
            raise ValueError("SVM not trained yet")
        return self.svm.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (decision function for LinearSVC)

        Args:
            features: Feature matrix (N, feature_dim)

        Returns:
            Confidence scores (N,)
        """
        if not self.is_fitted:
            raise ValueError("SVM not trained yet")

        if isinstance(self.svm, LinearSVC):
            # Use decision function as confidence score
            return self.svm.decision_function(features)
        else:
            # Use probability for SVC with rbf kernel
            return self.svm.predict_proba(features)[:, 1]

    def save(self, path: str):
        """Save SVM model"""
        with open(path, 'wb') as f:
            pickle.dump(self.svm, f)

    def load(self, path: str):
        """Load SVM model"""
        with open(path, 'rb') as f:
            self.svm = pickle.load(f)
        self.is_fitted = True


class MultiClassSVM:
    """One-vs-all SVM classifier for multiple classes"""

    def __init__(self, num_classes: int, class_names: List[str] = None,
                 kernel: str = 'linear', C: float = 1.0):
        """
        Args:
            num_classes: Number of object classes (including background)
            class_names: List of class names
            kernel: SVM kernel type
            C: Regularization parameter
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.kernel = kernel
        self.C = C

        # Create one SVM per class (excluding background)
        self.svms = {}
        for i, class_name in enumerate(self.class_names):
            if class_name != 'background':
                self.svms[class_name] = SVMClassifier(kernel=kernel, C=C)

    def train_class(self, class_name: str, features: np.ndarray,
                    labels: np.ndarray):
        """
        Train SVM for a specific class

        Args:
            class_name: Name of the class
            features: Feature matrix (N, feature_dim)
            labels: Binary labels (N,) - 1 for positive, 0 for negative
        """
        if class_name not in self.svms:
            raise ValueError(f"Unknown class: {class_name}")

        self.svms[class_name].train(features, labels)

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class for each feature

        Args:
            features: Feature matrix (N, feature_dim)

        Returns:
            predicted_classes: Class indices (N,)
            confidence_scores: Confidence for predicted class (N,)
        """
        num_samples = features.shape[0]
        scores = np.zeros((num_samples, len(self.svms)))

        # Get scores from each SVM
        for i, (class_name, svm) in enumerate(self.svms.items()):
            if svm.is_fitted:
                scores[:, i] = svm.predict_proba(features)

        # Get class with highest score
        predicted_classes = np.argmax(scores, axis=1)
        confidence_scores = np.max(scores, axis=1)

        return predicted_classes, confidence_scores

    def save(self, save_dir: str):
        """Save all SVMs"""
        os.makedirs(save_dir, exist_ok=True)

        for class_name, svm in self.svms.items():
            save_path = os.path.join(save_dir, f"svm_{class_name}.pkl")
            svm.save(save_path)

        # Save metadata
        metadata = {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'kernel': self.kernel,
            'C': self.C
        }
        with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

    def load(self, save_dir: str):
        """Load all SVMs"""
        # Load metadata
        with open(os.path.join(save_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        self.num_classes = metadata['num_classes']
        self.class_names = metadata['class_names']

        # Load SVMs
        for class_name in self.class_names:
            if class_name != 'background':
                svm_path = os.path.join(save_dir, f"svm_{class_name}.pkl")
                if os.path.exists(svm_path):
                    self.svms[class_name].load(svm_path)


class BBoxRegressor:
    """Bounding box regressor using Ridge Regression"""

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Regularization strength
        """
        self.alpha = alpha
        # Separate regressor for each coordinate (dx, dy, dw, dh)
        self.regressors = {
            'dx': Ridge(alpha=alpha),
            'dy': Ridge(alpha=alpha),
            'dw': Ridge(alpha=alpha),
            'dh': Ridge(alpha=alpha)
        }
        self.is_fitted = False

    def train(self, features: np.ndarray, proposals: np.ndarray,
              ground_truth: np.ndarray):
        """
        Train bounding box regressor

        Args:
            features: Feature matrix (N, feature_dim)
            proposals: Proposed boxes (N, 4) - [xmin, ymin, xmax, ymax]
            ground_truth: Ground truth boxes (N, 4) - [xmin, ymin, xmax, ymax]
        """
        # Compute regression targets
        targets = self._compute_targets(proposals, ground_truth)

        # Train separate regressor for each coordinate
        for i, coord in enumerate(['dx', 'dy', 'dw', 'dh']):
            self.regressors[coord].fit(features, targets[:, i])

        self.is_fitted = True

    def _compute_targets(self, proposals: np.ndarray,
                        ground_truth: np.ndarray) -> np.ndarray:
        """
        Compute regression targets (transformation from proposal to GT)

        Args:
            proposals: (N, 4) - [xmin, ymin, xmax, ymax]
            ground_truth: (N, 4) - [xmin, ymin, xmax, ymax]

        Returns:
            targets: (N, 4) - [dx, dy, dw, dh]
        """
        # Convert to center coordinates and dimensions
        p_x = (proposals[:, 0] + proposals[:, 2]) / 2
        p_y = (proposals[:, 1] + proposals[:, 3]) / 2
        p_w = proposals[:, 2] - proposals[:, 0]
        p_h = proposals[:, 3] - proposals[:, 1]

        g_x = (ground_truth[:, 0] + ground_truth[:, 2]) / 2
        g_y = (ground_truth[:, 1] + ground_truth[:, 3]) / 2
        g_w = ground_truth[:, 2] - ground_truth[:, 0]
        g_h = ground_truth[:, 3] - ground_truth[:, 1]

        # Compute targets (avoid division by zero)
        p_w = np.maximum(p_w, 1)
        p_h = np.maximum(p_h, 1)

        dx = (g_x - p_x) / p_w
        dy = (g_y - p_y) / p_h
        dw = np.log(g_w / p_w)
        dh = np.log(g_h / p_h)

        targets = np.stack([dx, dy, dw, dh], axis=1)
        return targets

    def predict(self, features: np.ndarray, proposals: np.ndarray) -> np.ndarray:
        """
        Predict refined bounding boxes

        Args:
            features: Feature matrix (N, feature_dim)
            proposals: Proposed boxes (N, 4) - [xmin, ymin, xmax, ymax]

        Returns:
            refined_boxes: (N, 4) - [xmin, ymin, xmax, ymax]
        """
        if not self.is_fitted:
            return proposals  # Return original if not trained

        # Predict transformations
        dx = self.regressors['dx'].predict(features)
        dy = self.regressors['dy'].predict(features)
        dw = self.regressors['dw'].predict(features)
        dh = self.regressors['dh'].predict(features)

        # Apply transformations to proposals
        p_x = (proposals[:, 0] + proposals[:, 2]) / 2
        p_y = (proposals[:, 1] + proposals[:, 3]) / 2
        p_w = proposals[:, 2] - proposals[:, 0]
        p_h = proposals[:, 3] - proposals[:, 1]

        # Predict refined box
        g_x = p_w * dx + p_x
        g_y = p_h * dy + p_y
        g_w = p_w * np.exp(dw)
        g_h = p_h * np.exp(dh)

        # Convert back to corner coordinates
        refined_boxes = np.zeros_like(proposals)
        refined_boxes[:, 0] = g_x - g_w / 2  # xmin
        refined_boxes[:, 1] = g_y - g_h / 2  # ymin
        refined_boxes[:, 2] = g_x + g_w / 2  # xmax
        refined_boxes[:, 3] = g_y + g_h / 2  # ymax

        return refined_boxes

    def save(self, path: str):
        """Save regressor"""
        with open(path, 'wb') as f:
            pickle.dump(self.regressors, f)

    def load(self, path: str):
        """Load regressor"""
        with open(path, 'rb') as f:
            self.regressors = pickle.load(f)
        self.is_fitted = True


class MultiClassBBoxRegressor:
    """Bounding box regressor with one regressor per class"""

    def __init__(self, class_names: List[str], alpha: float = 1.0):
        """
        Args:
            class_names: List of class names
            alpha: Regularization strength
        """
        self.class_names = class_names
        self.regressors = {}

        for class_name in class_names:
            if class_name != 'background':
                self.regressors[class_name] = BBoxRegressor(alpha=alpha)

    def train_class(self, class_name: str, features: np.ndarray,
                    proposals: np.ndarray, ground_truth: np.ndarray):
        """Train regressor for a specific class"""
        if class_name not in self.regressors:
            raise ValueError(f"Unknown class: {class_name}")

        self.regressors[class_name].train(features, proposals, ground_truth)

    def predict(self, class_name: str, features: np.ndarray,
                proposals: np.ndarray) -> np.ndarray:
        """Predict refined boxes for a specific class"""
        if class_name not in self.regressors:
            return proposals

        return self.regressors[class_name].predict(features, proposals)

    def save(self, save_dir: str):
        """Save all regressors"""
        os.makedirs(save_dir, exist_ok=True)

        for class_name, regressor in self.regressors.items():
            save_path = os.path.join(save_dir, f"bbox_reg_{class_name}.pkl")
            regressor.save(save_path)

    def load(self, save_dir: str):
        """Load all regressors"""
        for class_name in self.class_names:
            if class_name != 'background':
                reg_path = os.path.join(save_dir, f"bbox_reg_{class_name}.pkl")
                if os.path.exists(reg_path):
                    self.regressors[class_name].load(reg_path)
