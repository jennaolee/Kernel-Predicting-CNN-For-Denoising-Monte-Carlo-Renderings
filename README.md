# Kernel-Predicting-CNN-For-Denoising-Monte-Carlo-Renderings
## CMPT 469/722 Rendering and Visual Computing for AI - Final Project 
### Submitted By: Asmita Srivastava & Jenna Lee
A replication of Kernel-Predicting Convolutional Neural Networks (KPCN) for Denoising Monte Carlo Renderings paper by Bako et. al. This project includes a baseline KPCN model and a finetuned variant for specular and diffuse component separation in Python, and the methods have been implemented following the [research paper](https://studios.disneyresearch.com/2017/07/20/kernel-predicting-convolutional-networks-for-denoising-monte-carlo-renderings/).

## Repository Structure

### Notebooks
- **`469_baseline_CNN.ipynb`** - The baseline KPCN model implementation and training/evaluation pipeline
- **`specular+diffuse_CNN.ipynb`** - Finetuned KPCN model trained on separated specular and diffuse rendering components
- **`specular_diffuse.ipynb`** - Same as specular+diffuse_CNN.ipynb but with caching implementation for faster training; not recommended for Colab execution due to increased runtime

### Code Files
- **`kpcn_model.py`** - Core model architecture used by the baseline KPCN model; required for Colab execution (to be uploaded to Colab environment prior to running the baseline notebook)
- **`view_exr.py`** - Utility script for loading and visualizing EXR image files

### Data & Checkpoints
- **`baseline model checkpoints/`** - Pre-trained model weights for the baseline KPCN model
- **`spec+diff model checkpoints/`** - Pre-trained model weights for the specular+diffuse finetuned model
- **`results/`** - Training plots and results for the baseline model

## Running the Code

All code is designed to run on **Google Colab** with Google Drive mounted for data and checkpoint access.

### Setup Instructions

1. **Mount Google Drive in Colab:**
   ```python
   from google.colab import drive
   drive.mount('/content/gdrive')
   ```

2. **Access Data & Checkpoints:**
   - Data files and model checkpoint folders will be provided via a shared Google Drive link
   - Mount the drive in your Colab notebook to access these resources

3. **Upload Required Files (if needed):**
   - When running the code on Colab, upload `kpcn_model.py` to your Colab environment for successful execution of 469_Baseline_CNN.ipynb.
   - This file is essential for the baseline KPCN model execution

### Running the Notebooks

To replicate the report results:

1. Open the desired notebook in Google Colab:
   - For baseline results: use `469_Baseline_CNN.ipynb`
   - For finetuned results: use `specular+diffuse_CNN.ipynb`

2. **Run all cells EXCEPT the training cell:**
   - Training typically takes **6-10 hours** depending on the CNN model complexity
   - All other cells (data loading, evaluation, visualization) can be executed for replicating the results mentioned in the report.

3. **Using Pre-trained Checkpoints (if access through the Google Drive link is unsuccessful):**
   - Pre-trained model weights are included in the codebase:
     - Baseline model: `baseline model checkpoints/kpcn_best.pt`
     - Specular+Diffuse model: `spec+diff model checkpoints/` (latest checkpoint)
   - Load these into the notebooks with minor code modifications instead of training from scratch

### Alternative: Training from Scratch

If you wish to train the models:
- Include and run the training cell in the notebook
- Allow 6-10 hours for full training depending on the model on A100 GPU on Google Colab (takes longer on less powerful GPUs, ie. T4)
- Ensure sufficient GPU quota on Google Colab

## Notes

- Model checkpoints can alternatively be loaded directly from the codebase with minimal code changes if you prefer not to use the Google Drive link
- Caching implementation in `specular_diffuse.ipynb` significantly increases runtime; use `specular+diffuse_CNN.ipynb` instead
- Training plots and baseline results are saved in the `results/` folder
