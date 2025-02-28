**DCASE 2025 Sound Event Localization and Detection (SELD) Baseline**

## Project Structure

* `main.py` script serves as the entry point for the project. It coordinates all other scripts and executes the workflow.
* `data_generator.py` script is responsible for generating data and labels for training and evaluation.
* `extract_features.py` script extracts relevant features from the raw data (audio, visuals and labels(accdoa or multiaccdoa format)) to be used for model training.
* `inference.py` script handles model inference, allowing predictions on eval data.
* `loss.py` script defines singleaccdoa and multiaccdoa(adpit) loss functions used during training.
* `metrics.py` script implements different evaluation metrics to assess model performance.
* `model.py` script defines the seld model architecture.
* `parameters.py` script contains all hyperparameters and configurations. If a user needs to modify parameters, they should update them here.
* `utils.py` script includes various utility functions used throughout the project.
