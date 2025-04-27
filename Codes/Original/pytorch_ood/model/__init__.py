"""
Models
******************

Frequently used Neural Network based Models


Computer Vision
==================

Wide ResNet
-------------

..  autoclass:: pytorch_ood.model.WideResNet
    :members:

Vision Transformer
---------------------
..  autoclass:: pytorch_ood.model.VisionTransformer
    :members:


Natural Language Processing
==============================

GRU Classifier
---------------------
..  autoclass:: pytorch_ood.model.GRUClassifier
    :members:


Modules
=================
Neural Network modules frequently used in OOD Detection.


Class Centers
---------------------
..  autoclass:: pytorch_ood.model.ClassCenters
    :members:


..  autoclass:: pytorch_ood.model.RunningCenters
    :members:


"""

from .lenet import LeNet
