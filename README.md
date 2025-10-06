# GenderDetector
This is a lightweight image classification project built with PyTorch 2.x, designed to train a Convolutional Neural Network (CNN) on a dataset of ~18,000 images.Images used for mode training are excerpted from the CelebA dataset.The trained model are capable of identifying gender from facial photos.
# Project Structure
The project is organized to manage data, models, and code in a modular way:
After downloading the CelebA dataset, you need to unzip the `Img\img_align_celeba.zip` file contained in it.
#### Dataset Directory
- **CelebA/**: Contains the CelebA dataset and related subdirectories:
  - `Anno/`: Stores annotation files (e.g., `list_attr_celeba.txt` for attribute labels).
  - `Eval/`: Intended for evaluation-related resources (e.g., metrics or evaluation scripts).
  - `Img/`: Manages image data:
    - `img_align_celeba/`: Holds original aligned face images from the CelebA dataset.
    - `Train/`, `Val/`, `Test/`: Subdirectories with images split for training, validation, and testing.
    - `train_txt.txt`, `val_txt.txt`, `test_txt.txt`: Generated text files that map image paths to gender labels for each data subset.

#### Core Code Files
- `cnn.py`: Defines the Convolutional Neural Network (CNN) architecture for gender detection.
- `data_loader.py`: Handles data loading, batching, and preprocessing for training, validation, and testing.
- `model_setup.py`: Configures the model (e.g., initialization, device assignment, and setup of optimizers/loss functions).
- `train.py`: Main script for training the model, including the training loop and model checkpointing.
- `test.py`: Script for evaluating the trained model on the test dataset.
- `pic_partition.py`: Splits the original CelebA images into `Train/`, `Val/`, and `Test/` subsets (used to generate the image folders and label files).
- `pic_label.py`: Processes and generates gender label mappings for images.
- `loss_graph.py`: Generates visualizations (e.g., loss curves) during or after training.
- `load.py`: Helper functions for loading data or pre-trained models.
- `best_model.pth`: Saved weights of the best-performing model from training.

#### Documentation
- `README.txt` / `README.md`: Provides an overview of the project, setup instructions, and usage guidelines.

#### Requirement
- requirement.txt : List all the Python packages (and their specific versions if necessary) required by a project.

