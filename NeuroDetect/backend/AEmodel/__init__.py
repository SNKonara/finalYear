"""
Fraud Detection Autoencoder Package
For batch processing and real-time inference
"""

__version__ = "1.0.0"
__author__ = "SKonara"

from .preprocessor import DataPreprocessor
from .model import FraudAutoencoder, AutoencoderTrainer
from .detector import FraudDetector
from .pipeline import BatchProcessor, RealTimeProcessor

__all__ = [
    'DataPreprocessor',
    'FraudAutoencoder',
    'AutoencoderTrainer',
    'FraudDetector',
    'BatchProcessor',
    'RealTimeProcessor'
]