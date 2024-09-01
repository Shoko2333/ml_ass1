import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the training data
data_train_df = pd.read_csv('./data/train.csv')
# Separate features and labels for training data
X_train = data_train_df.loc[:, "v1":"v784"].to_numpy()
y_train = data_train_df.label.to_numpy()

# Load the test1 data
data_test1_df = pd.read_csv('./data/test1.csv')
# Separate features and labels for test1 data
X_test1 = data_test1_df.loc[:, "v1":"v784"].to_numpy()
y_test1 = data_test1_df.label.to_numpy()

# Load the test2 data
data_test2_df = pd.read_csv('./data/test2.csv')
# Separate features for test data
X_test2 = data_test2_df.loc[:, "v1":"v784"].to_numpy()

# Normalization
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test1_normalized = scaler.transform(X_test1)  # Use the same scaler for test1 data
X_test2_normalized = scaler.transform(X_test2)  # Use the same scaler for test2 data

# Dimensionality Reduction using PCA
pca = PCA(n_components=100)  # Reduce the dimensions to 100
X_train_pca = pca.fit_transform(X_train_normalized)
X_test1_pca = pca.transform(X_test1_normalized)  # Use the same PCA model for test1 data
X_test2_pca = pca.transform(X_test2_normalized)  # Use the same PCA model for test2 data

# Print the shape of the data before and after preprocessing
print(f"Original training data shape: {X_train.shape}")
print(f"Training data shape after PCA: {X_train_pca.shape}")
print(f"Original test1 data shape: {X_test1.shape}")
print(f"Test1 data shape after PCA: {X_test1_pca.shape}")
print(f"Original test2 data shape: {X_test2.shape}")
print(f"Test2 data shape after PCA: {X_test2_pca.shape}")

# Save the preprocessed data for later use
np.save('./data/X_train_preprocessed.npy', X_train_pca)
np.save('./data/y_train.npy', y_train)
np.save('./data/X_test1_preprocessed.npy', X_test1_pca)
np.save('./data/y_test1.npy', y_test1)
np.save('./data/X_test2_preprocessed.npy', X_test2_pca)


print("finish")

# # ------------------------------code from task---------------------------
# # # load the training data
# # # print(os.listdir("./data"))
# # pd.set_option('display.max_columns', 10)
#
# # Load the data
# # train.csv including feature and label using for training model.
# data_train_df = pd.read_csv('./data/train.csv')
#
# # print out the first 5 rows of the training dataframe
# # print(data_train_df.head())
#
# # Selecting input feature
# data_train_feature = data_train_df.loc[:, "v1":"v784"].to_numpy() # X
#
# # Selecting output lable
# data_train_label = data_train_df.label.to_numpy() # y
#
# import matplotlib.pyplot as plt
# data_train_feature = data_train_feature.reshape((data_train_feature.shape[0], 28, 28))
# plt.figure(figsize=(3,3))
# plt.imshow(data_train_feature[0], cmap=plt.get_cmap('gray'))
# plt.title("class " + str(data_train_label[0]))
# plt.show()
#
# # Loading the testing data
# # test2.csv includes 5000 samples used for label prediction. Test samples do not have labels.
# data_test_df = pd.read_csv('./data/test2.csv', index_col=0)
#
# # print out the first 5 rows of the test dataframe
# # print(data_test_df.head())
#
# print("finish")












