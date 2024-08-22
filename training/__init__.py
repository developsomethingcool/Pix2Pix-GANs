# training/__init__.py

from .evaluator import evaluate_model
from .trainer import train_model


# Optional: Define what is available when importing the package
__all__ = ['train_model', 'evaluate_model']
