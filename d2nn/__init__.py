"""
d2nn – Diffractive Deep Neural Network (D²NN) and
       Fourier Diffractive Deep Neural Network (F-D²NN)

Optical neural network architectures that compute via light propagation
through stacked diffractive layers.  Both free-space (Angular Spectrum
Method) and Fourier-plane (4-f system) propagation are supported.

References
----------
Lin X. et al., "All-optical machine learning using diffractive deep neural
networks," Science, 361(6406), 1004–1008 (2018).
"""

from .layers import DiffractiveLayer, FourierDiffractiveLayer
from .models import D2NN, FourierD2NN
from .propagation import angular_spectrum_propagation, fourier_lens_propagation

__all__ = [
    "DiffractiveLayer",
    "FourierDiffractiveLayer",
    "D2NN",
    "FourierD2NN",
    "angular_spectrum_propagation",
    "fourier_lens_propagation",
]
