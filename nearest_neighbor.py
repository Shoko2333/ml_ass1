import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load preprocessed data
X_train = np.load('./data/X_train_preprocessed.npy')
y_train = np.load('./data/y_train.npy')
X_test1 = np.load('./data/X_test1_preprocessed.npy')
y_test1 = np.load('./data/y_test1.npy')
X_test2 = np.load('./data/X_test2_preprocessed.npy')

# Define parameter grid for GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters:", grid_search.best_params_)

# Make predictions using the best model
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test1, y_pred))

# Calculate and print accuracy
accuracy = accuracy_score(y_test1, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Visualize accuracy for different k values
k_range = range(1, 20)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores = knn.score(X_test1, y_test1)
    k_scores.append(scores)

plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.title('Accuracy vs. K Value')
plt.grid(True)
plt.show()

print("finish")