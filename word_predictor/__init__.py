"""Offline English autocomplete: completes the word being typed, predicts the next word and learns
the user's own words. Pure Python, no third-party dependencies."""

from .predictor import DEFAULT_MODEL, Predictor

__all__ = ["DEFAULT_MODEL", "Predictor"]
