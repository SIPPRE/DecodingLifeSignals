# -*- coding: utf-8 -*-
"""
NeuroRock EEG Feature Classification Script

Loads features extracted per window and trains a basic classifier
to distinguish between the four emotional categories (HAP, LAP, HAN, LAN).
Evaluates performance and plots a confusion matrix.

Usage:
    python classify_features.py --subject <SUBJECT_ID> [options]

Example:
    python classify_features.py --subject s01 --feature-file s01_features_win1.0s.csv

Options:
    --feature-file  Path to the input CSV feature file (required).
    --output-dir    Directory to save output plots (default: .)
    --test-size     Proportion of data for the test set (default: 0.2)
    --random-state  Random state for reproducibility (default: 42)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.model_selection import train_test_split # Or GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer

def run_classification(feature_file, output_dir, test_size=0.2, random_state=42):
    """Loads features, trains, tests a classifier, and saves results."""

    print(f"\n--- Starting Classification for Feature File: {feature_file} ---")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(feature_file)
        print(f"Successfully loaded features. Shape: {df.shape}")
        if df.empty:
            print("Error: Feature file is empty.")
            return
    except FileNotFoundError:
        print(f"Error: Feature file not found at {feature_file}")
        return
    except Exception as e:
        print(f"Error loading feature file: {e}")
        return

    # --- 2. Prepare Data (X and y) ---
    # Define target variable
    target_col = 'category'
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in the feature file.")
        return

    # Define metadata columns to exclude from features
    metadata_cols = ['event_id', 'category', 'clip_id', 'title', 'artist',
                     'original_event_sample', 'window_start_time', 'window_number']
    # Identify actual feature columns
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    if not feature_cols:
        print("Error: No feature columns identified.")
        return

    print(f"Using {len(feature_cols)} features: {feature_cols[:5]}...") # Print first 5

    X = df[feature_cols]
    y_labels = df[target_col]

    # Encode string labels to numerical labels
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    class_names = le.classes_ # Get the original class names (e.g., ['HAP', 'HAN', ...])
    print(f"Target classes: {class_names}")

    # --- 3. Handle Missing Values (Imputation) ---
    # Check for NaNs/Infs - compute_psd can sometimes produce them
    print(f"Checking for NaN/Inf values in features...")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("NaN or Inf values found. Applying SimpleImputer (strategy='median')...")
        # Replace Inf with NaN first
        X = X.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        # Convert back to DataFrame to keep column names (optional, but good practice)
        X = pd.DataFrame(X, columns=feature_cols)
        print("Imputation complete.")
    else:
        print("No NaN/Inf values found.")


    # --- 4. Split Data ---
    # Option 1: Simple Random Split (Easy but potentially leaky for time series)
    print(f"Splitting data randomly ({1-test_size:.0%} train, {test_size:.0%} test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y # Stratify helps keep class proportions
    )

    # Option 2: Group Split (Better practice to avoid data leakage)
    # Keeps all windows from the same song together in either train or test set.
    # Uncomment the following lines to use GroupShuffleSplit instead of random split.
    # print(f"Splitting data by song/clip ({1-test_size:.0%} train, {test_size:.0%} test using GroupShuffleSplit)...")
    # if 'clip_id' not in df.columns:
    #     print("Error: 'clip_id' column needed for group split, but not found.")
    #     return
    # groups = df['clip_id']
    # gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    # train_idx, test_idx = next(gss.split(X, y, groups))
    # X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # y_train, y_test = y[train_idx], y[test_idx]
    # print(f"Group split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    # print(f"Clips in Train set: {np.unique(groups.iloc[train_idx])}")
    # print(f"Clips in Test set: {np.unique(groups.iloc[test_idx])}")

    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # --- 5. Scale Features ---
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use transform only on test data

    # --- 6. Train Classifier ---
    print("Training Support Vector Classifier (SVC)...") # <--- Changed print message
    # Using SVC
    # model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    # model = LinearDiscriminantAnalysis()
    # Setting kernel='rbf' (default) and C=1 (default) is common.
    # Add probability=True if you need predict_proba later, but it slows training.
    model = SVC(C=10, kernel='rbf', random_state=random_state) # <--- Use this line

    model.fit(X_train_scaled, y_train) # Fit SVC model
    print("Training complete.")


    # --- 7. Test & Evaluate ---
    print("\n--- Evaluation Results ---")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # --- 8. Plot Confusion Matrix ---
    print("\nPlotting Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=le.transform(class_names)) # Use numerical labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
    plt.tight_layout()

    # Save and show plot
    cm_filename = os.path.join(output_dir, f"{os.path.basename(feature_file).replace('.csv', '')}_confusion_matrix.png")
    try:
        fig.savefig(cm_filename)
        print(f"Saved confusion matrix to: {cm_filename}")
        plt.show()
    except Exception as e:
        print(f"Error saving/showing confusion matrix plot: {e}")
    finally:
        plt.close(fig)

    print("\n--- Classification Complete ---")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier on NeuroRock EEG features.")
    parser.add_argument('--feature-file', type=str, required=True,
                        help='Path to the input CSV feature file (e.g., s01_features_win1.0s.csv)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output plots (default: current directory)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data for the test set (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output plots will be saved to: {args.output_dir}")

    # Run the classification workflow
    run_classification(args.feature_file, args.output_dir, args.test_size, args.random_state)