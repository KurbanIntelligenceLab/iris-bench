"""
Small video classifier for dynamics type (dropped_ball, free_fall, led, pendulum, sliding_block, torricelli).
Uses a pretrained image backbone + temporal mean pooling. Train with scripts/train_video_classifier.py.
"""

from .dataset import DynamicsDataset, get_gt_dynamics_from_path, DYNAMICS_CLASSES
from .model import VideoDynamicsClassifier
from .predictor import load_classifier, predict_dynamics_from_npy

__all__ = [
    "DynamicsDataset",
    "get_gt_dynamics_from_path",
    "DYNAMICS_CLASSES",
    "VideoDynamicsClassifier",
    "load_classifier",
    "predict_dynamics_from_npy",
]
