from Models.utils import *
import matplotlib.pyplot as plt
import numpy as np
from Models.Preprocessing.PCA import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Models.NeuralNet import neuralNetworkMultiHidden
from Models.SimpleNet import neuralNetworkSingleHidden

iterations=100
learning_rate=0.02
X,y,x_validate,y_validate,x_test,y_test=SplitDataset("E:\CH512\Dataset\data.csv",0.7,0.3,0)
data=pd.read_csv("E:\CH512\Dataset\data.csv")
data_X=data.drop("y", axis=1)
print(data_X.columns)
pca=PCA(2)
pca.fit(X)

print('Components:\n', pca.filtered_components)
print('Explained variance ratio:\n', pca.explained_variance_ratio)

cum_explained_variance = np.cumsum(pca.explained_variance_ratio)
print('Cumulative explained variance:\n', cum_explained_variance)

X_pca = pca.transformData(X) # Apply dimensionality reduction to X.
print('Transformed data shape:', X_pca.shape)

x_test_pca=pca.transformData(x_validate)



##declaring the model here
print(X_pca.shape[1])
layer_sizes = [X_pca.shape[1], 3, 1]  # Input layer, Hidden layer 1, Output layer
model = neuralNetworkSingleHidden(X_pca.shape[1],4,1,0.001)

# Train the neural network
model.train(X_pca, y)

y_pred=model.predict(x_test_pca)

    
y_pred=y_pred.reshape(-1)
    #result
# MSE LOSS 0.028388499570481444
# AIC 11.123542321057162
# r2 -0.025298600741392008
# BIC 20.276481459153608
r2=r_squared(y_validate,y_pred)
aic=AIC(y_pred,y_validate,x_test_pca)
bic=BIC(y_pred,y_validate,x_test_pca)
loss=MSELoss(y_pred,y_validate)
print("MSE LOSS",loss)
print("AIC",aic)
print("r2",r2)
print("BIC",bic)

plt.figure(figsize=(8, 6))
# plt.scatter(range(len(y_validate)), y_validate, color='blue',label='Real Values')
# plt.scatter(range(len(y_pred)), y_pred, color='red',label='Predicted Values')
# plt.title('Scatter Plot of Predicted vs. Real Values for PCA+Singlelayer')
# plt.ylabel('Real Values (y_validate)')
# plt.xlabel('Predicted Values (y_pred)')
plt.plot(y_pred-y_validate)
plt.title('Error Plot for PCA+Neural Network Single hidden layer')
plt.ylabel('Error')
plt.xlabel('Index')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join("Results", 'PCA_NN_Single_error.png'))
plt.show()