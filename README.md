# DeepTwoSampleTest-GradCAM
## Overview
This project visualizes a Two-Sample Test using GradCAM, employing a Variational Autoencoder (VAE) and the dSprites dataset. The VAE is trained on one class from dSprites, and GradCAM is used to generate heatmaps which visualises most importnat features that distinguish between different classes.

## Project Structure
- **train.py**: Script to train the VAE model.
- **model.py**: Contains the VAE architecture.
- **visualise_two_sample_test.py**: Code to visualize the two-sample test using GradCAM.
- **utils.py**: Utility functions for plotting and overlaying heatmaps.
- **overlay_heatmap.py**: Script for overlaying heatmaps on images.
- **statistical_test.py**: Statistical test to check if two groups are statistically different.
- **gradcam.py**: Generates heatmaps of differences between two groups.
- **dataloader.py**: Data loader for train, validation, and test datasets.
- **masking_feature.py**: Masks the most important features using GradCAM.
- **applying_mask.py**: Applies the generated mask to the dataset.
- **/notebooks/**: Jupyter notebooks for model checks, clustering, and heatmap visualization.

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mjavan/DeepTwoSampleTest-GradCAM.git
   cd DeepTwoSampleTest-GradCAM

2. **Install dependencies**:
   `pip install -r requirements.txt`

3. **Train the Model**:
   `python src/train.py`

4. **Visualize the Two-Sample Test**:
   `python src/visualise_two_sample_test.py`

5. **Additional Scripts**:
   - Generate and overlay heatmaps using `overlay_heatmap.py`
   - Check the statistical differences between two groups using `statistical_test.py`
  
## Dependencies
 - PyTorch
 - NumPy
 - Matplotlib
 - Scikit-learn
 - Seaborn

## Results
 - Visualizations will be stored in the `results/ folder` (you can set this up in your code).
 - Heatmaps will show differences between two groups in test set based on GradCAM.

