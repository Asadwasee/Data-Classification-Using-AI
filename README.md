# DecodeLabs Project 2: Data Classification Using AI

This repository contains the complete implementation of **Project 2: Data Classification Using AI** as part of the DecodeLabs curriculum. The project demonstrates a complete machine learning pipeline adhering to the **IPO (Input-Process-Output) Framework**.

## Project Overview
The core objective of this project is to build a Supervised Learning classification pipeline. It loads a benchmark dataset, performs feature scaling, splits data for validation, implements the K-Nearest Neighbors (KNN) algorithm, tunes hyperparameters using the Elbow Method, and provides a comparative analysis using an alternative algorithm (Decision Tree Classifier).

## IPO Framework Breakdown

### 1. Input Phase (Data & Preprocessing)
- **Dataset:** The standard Iris Benchmark Dataset (150 samples, 3 balanced classes, 4 structural features).
- **Feature Scaling:** Applied `StandardScaler` to ensure the mean is 0 and variance is 1, neutralizing scaling bias across features.

### 2. Process Phase (Model Training & Tuning)
- **Data Partitioning:** Implemented an 80/20 train-test split with row shuffling to ensure structural integrity and prevent order bias.
- **Primary Model:** Trained a **K-Nearest Neighbors (KNN)** classifier ($K=5$).
- **Hyperparameter Tuning:** Integrated **The Elbow Method** iterating through $K$ values (1 to 14) to monitor and plot the optimal error rate trend.
- **Alternative Solution:** Implemented a **Decision Tree Classifier** to fulfill the unique experimentation requirement.

### 3. Output Phase (Validation & Performance)
- Both the KNN and Decision Tree models achieved exceptional performance on the unseen test set:
  - **Accuracy:** 100% (1.00 Score)
  - Perfect precision, recall, and F1-score across all three classes (*Setosa*, *Versicolor*, *Virginica*).
- **Custom Inference:** Included an inference pipeline for real-time user inputs. A test input of `[[5.1, 3.5, 1.4, 0.2]]` successfully classified the flower as **SETOSA**.

## Technologies Used
- Python
- Google Colab / Jupyter Notebooks
- Scikit-Learn (Data splitting, Scaling, KNN, Decision Tree, Metrics)
- Pandas & NumPy (Data manipulation)
- Matplotlib (Hyperparameter error plotting)

## Conclusion & Next Steps
With the tabular data classification milestone successfully reached, the next phase of this track transitions from structured data into **Deep Learning and Computer Vision (CNNs)** for multi-class image preprocessing and classification.
