import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import time
from joblib import dump

# Load preprocessed data
X_train = np.load('./data/X_train_preprocessed.npy')
y_train = np.load('./data/y_train.npy')
X_test1 = np.load('./data/X_test1_preprocessed.npy')
y_test1 = np.load('./data/y_test1.npy')

# Define SVC model
svc = SVC(kernel='poly', C=10, random_state=42)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'C': [1, 10, 100],
    'kernel': ['poly'],
    'degree': [2, 3, 4]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)

# Train model and record time
start_time = time.time()
grid_search.fit(X_train, y_train)
training_time = time.time() - start_time

# Get best model
best_svc = grid_search.best_estimator_

# Save the best model
dump(best_svc, './Save_model/best_svc_model01.joblib')
print("Best model saved as 'best_svc_model.joblib'")

# Evaluate model on test1 and record inference time
start_time = time.time()
test1_predictions = best_svc.predict(X_test1)
inference_time = time.time() - start_time
test1_accuracy = accuracy_score(y_test1, test1_predictions)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"Test1 accuracy: {test1_accuracy:.4f}")
print(f"Training time: {training_time:.2f} seconds")
print(f"Inference time: {inference_time:.2f} seconds")