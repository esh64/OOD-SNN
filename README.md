# Boosting OOD Detection in Biomedical Data with Siamese Neural Networks

## Overview

Out-of-Distribution (OOD) detection is critical for ensuring the reliability of machine learning models, particularly in sensitive domains such as biomedicine. In this work, we propose a novel OOD detection framework based on Siamese Neural Networks (SNNs) trained with triplet loss, aiming to generate discriminative feature spaces where known-class samples are tightly clustered and unknown samples are pushed apart. We introduce the "one-from-each" triplet mining strategy, which selects hard negatives from each class individually, enhancing the diversity and representativeness of the training triplets.

A committee of traditional OOD detectors—including k-Nearest Neighbors, Gaussian Mixture Models, Isolation Forests, and One-Class SVMs—is applied on the learned feature spaces for class-specific OOD detection. Extensive experiments are conducted on diverse biomedical datasets, including OrganSMNIST, TissueMNIST, OCTMNIST, Genomics OOD, and Voice Pathology. Results demonstrate that our method significantly improves AUROC compared to baseline and standard classifiers, with gains of up to 28.9% on OrganSMNIST. While the proposed approach consistently outperforms classical baselines, the results also highlight the difficulty of OOD detection in certain datasets, such as voice pathology. These findings emphasize the potential of feature-embedding strategies for robust OOD detection and point towards promising directions for future research.

This repository contains the full implementation of the experiments described in the work:  
**Feature-Embedding OOD Detection in Biomedical Data via Siamese Neural Networks**.

## Repository Structure

The `Codes` folder contains four main subfolders:

- **CreateDataset**  
  Contains scripts to create the datasets from original sources.

- **Original**  
  Contains scripts to train a neural network classifier and apply state-of-the-art OOD detection methods.  
  Simply run the script `trainTestClassifier.sh` to execute the experiments and generate CSV files with metrics for each fold.

- **SNN**  
  Contains scripts to train a Siamese Neural Network (SNN) and apply traditional OOD detectors.  
  Simply run the script `trainDetectors.sh` to execute the experiments. CSV files containing the metrics for each fold will be created.  
  This folder implements the SNN approach using the "One-From-Each" triplet mining strategy.

- **SNNb**  
  Similar to the `SNN` folder but implements the "One-From-All" triplet mining approach instead.

## Citation

If you use this work or find it useful for your research, please cite:

Honorato, E.S., Dalalana, G.J.P. (2026). Boosting OOD Detection in Biomedical Data with Siamese Neural Networks. In: de Freitas, R., Furtado, D. (eds) Intelligent Systems. BRACIS 2025. Lecture Notes in Computer Science(), vol 16179. Springer, Cham. https://doi.org/10.1007/978-3-032-15987-8_13
