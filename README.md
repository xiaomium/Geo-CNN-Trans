# Geo-CNN-Trans

Overview
Geo CNN-Trans is a deep learning architecture specifically designed for interpretable 3D mineral prospectivity mapping. It combines 3D convolution with the Transformer model and introduces a spatial continuity loss. This repository demonstrates the full implementation applied to the geochemical data of the Zaozigou mining area, enabling a deep understanding of mineralization patterns and model - based geological interpretations.

Key Feature
The 3D cuboid data is first input into the 3D CNN and then fed into the Transformer model. End-to-end training is performed using a combination of classification loss and spatial continuity loss.

Input data
Please ensure that your data format is in npy
data: 3D cube data
label: Known classes or "-1" for unknowns
mask: Valid regions of the 3D grid data
center:Spatial location of the data
