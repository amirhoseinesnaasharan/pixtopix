# pixtopix

# Sketch-to-Image Translation Project Summary

This project implements a sketch-to-image translation model using a U-Net-based generator and a CNN-based discriminator. The goal of this model is to convert sketches into realistic images.

## Key Points of the Project:

1. **Data Preparation**:
   - Utilizes the `SketchDataset` class for loading and transforming images.
   - Splits the dataset into training and validation sets.

2. **Model Architecture**:
   - A U-Net generator with a ResNet34 encoder pre-trained on ImageNet.
   - A CNN-based discriminator.

3. **Training and Validation**:
   - Uses Mean Squared Error (MSE) loss for adversarial loss and L1 loss for pixel-wise reconstruction.
   - Trains the model using PyTorch Lightning for GPU acceleration support.

4. **Metrics**:
   - Computes Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) to evaluate the quality of generated images.

This project is suitable for those interested in deep learning and image processing and can serve as a foundation for similar projects.
