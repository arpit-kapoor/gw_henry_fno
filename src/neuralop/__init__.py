# Adapted from the neuraloperator library (https://github.com/neuraloperator/neuraloperator).
# Copyright (c) 2023 NeuralOperator developers. MIT License; see LICENSE in this directory.

"""
Neural Operator models for learning operators on function spaces.

This module contains Fourier Neural Operator (FNO) implementations.
"""

from .fno import FNO

__all__ = [
    'FNO',
]
