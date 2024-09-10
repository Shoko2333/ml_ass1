import numpy as np
import pandas as pd
from joblib import load
import time

# Load the saved model
loaded_model = load('./Save_model/best_svc_model01.joblib')
print("Best model loaded from 'best_svc_model.joblib'")

# Load test2 data
X_test2 = np.load('./data/X_test2_preprocessed.npy')

# Predict on test2 and record inference time
start_time = time.time()
test2_predictions = loaded_model.predict(X_test2)
inference_time = time.time() - start_time

print(f"Inference time for test2: {inference_time:.2f} seconds")

# Create submission file
submission = pd.DataFrame({'id': range(0, len(test2_predictions)), 'label': test2_predictions})
submission.to_csv('test_output.csv', index=False)

print("Submission file 'test_output.csv' has been created.")