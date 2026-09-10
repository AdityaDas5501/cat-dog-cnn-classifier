# Cats vs. Dogs Image Classifier

A binary image classification model built from scratch using PyTorch. This project implements a custom Convolutional Neural Network (CNN) to distinguish between images of cats and dogs.

## Dataset
**Source:** [Kaggle Dogs vs. Cats Competition](https://www.kaggle.com/c/dogs-vs-cats/data)

*Note: The dataset is not included in this repository due to file size constraints. To run this notebook, download `train.zip` and `test1.zip` from the link above and extract them into your working directory.*

## Tech Stack
* **Framework:** PyTorch (with Multi-GPU support via `DataParallel`)
* **Data Processing:** Torchvision (`v2` transforms), Pillow (PIL)
* **Evaluation:** Scikit-learn (Classification report, Confusion matrix)
* **Visualization:** Matplotlib

## Model Architecture
The custom CNN processes `128x128` RGB images through the following pipeline:
1. **Conv Block 1:** 2D Convolution (32 filters, 3x3) + ReLU + MaxPool (2x2)
2. **Conv Block 2:** 2D Convolution (64 filters, 3x3) + ReLU + MaxPool (2x2)
3. **Conv Block 3:** 2D Convolution (128 filters, 3x3) + ReLU + MaxPool (2x2)
4. **Fully Connected:** Flattened to 128x16x16 -> Linear (256) + ReLU -> Linear (2 outputs)

## Data Preprocessing
Images are dynamically loaded using a custom PyTorch `Dataset` and processed with:
* Resizing to `128x128` pixels
* Conversion to 32-bit floating-point tensors
* Normalization to a mean and standard deviation of `0.5` across all channels

## Training
* **Optimizer:** Adam
* **Loss Function:** Cross-Entropy Loss
* **Batch Size:** 32 (with CPU multi-threading `num_workers=4` and `pin_memory`)
* **Epochs:** 30

## Evaluation
The model's performance is validated on a 20% holdout set, evaluated using standard accuracy metrics, a precision/recall classification report, and a confusion matrix.
