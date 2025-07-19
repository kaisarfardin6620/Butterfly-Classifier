🦋 Butterfly Species Classification

Classify 40 species of butterflies using deep learning techniques, including transfer learning, data augmentation, and hyperparameter optimization. This project evolves from a basic pipeline to advanced tuning using MobileNetV2.

📂 Dataset

This project uses the Butterfly Images (40 Species) dataset from Kaggle.

Expected directory structure after extraction:

butterflies_v2/
├── butterflies and moths.csv
├── train/
├── test/
└── valid/
🔁 For butterfly_augmented.py, rename the directory to butterflies/.

🏗️ Project Structure

This repository includes four main scripts, each designed to progressively improve classification performance:

1. butterfly.py - Basic Single-Stage Training
Loads and preprocesses dataset

Applies basic data augmentation

Builds a MobileNetV2-based transfer learning model

Trains all layers in one go

Uses EarlyStopping and ReduceLROnPlateau callbacks

Outputs classification report, confusion matrix, and performance plots

Saves trained model as mobilenetv2_butterfly_classifier.keras

2. butterfly_2stage.py - Two-Stage Transfer Learning
Stage 1: Train classification head (freeze base)

Stage 2: Fine-tune full model with a small learning rate

Includes ModelCheckpoint and TensorBoard support

Merges training history for complete visualization

3. butterfly_augmented.py - Dataset Balancing via Augmentation
Detects class imbalance

Augments underrepresented classes to match majority class

Saves augmented images to respective class folders

Updates butterflies and moths.csv file

⚠️ Back up your dataset before running, as this script modifies original data

4. butterfly_hyperparameter.py - Hyperparameter Optimization
Builds on butterfly_2stage.py architecture

Uses KerasTuner to explore:

Dense units

Dropout rates

L2 regularization

Learning rates

Label smoothing

Implements stratified K-fold cross-validation

Trains final model on full data with best parameters

Saves model as final_tuned_mobilenetv2_butterfly_classifier.keras

🚀 Getting Started

✅ Requirements

bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn Pillow keras-tuner

📦 Dataset Setup

Download the dataset from Kaggle and extract to project root:

butterflies_v2/
├── butterflies and moths.csv
├── train/
├── test/
└── valid/

🧠 Running Scripts

bash
python butterfly.py
python butterfly_2stage.py
python butterfly_augmented.py
python butterfly_hyperparameter.py
📊 Outputs
Each script produces:

Training logs

Plots of:

Sample images

Class distributions

Accuracy/loss curves

Confusion matrix

Trained models saved to root directory
