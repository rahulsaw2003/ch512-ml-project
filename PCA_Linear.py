from Models.utils import *
import matplotlib.pyplot as plt
import numpy as np
from Models.Preprocessing.PCA import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Models.LinearModel import LinearRegression


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

model=LinearRegression(iterations,learning_rate)
model.fit(X_pca,y)
y_pred=model.predict(X_pca)
    
y_pred=y_pred.reshape(-1)
error=y_pred-y
    #result
# MSE LOSS 0.044719823733125205
# AIC 10.214676783420057
# BIC 21.061813671938392
aic=AIC(y_pred,y,X_pca)
bic=BIC(y_pred,y,X_pca)
loss=MSELoss(y_pred,y)
print("MSE LOSS",loss)
print("AIC",aic)
print("BIC",bic)
# plt.plot(error)
plt.figure(figsize=(8, 6))
# plt.scatter(range(len(y)), y, color='blue',label='Real Values')
# plt.scatter(range(len(y_pred)), y_pred, color='red',label='Predicted Values')
# plt.title('Scatter Plot of Predicted vs. Real Values for PCA+LINEAR')
# plt.ylabel('Real Values (y_validate)')
# plt.xlabel('Predicted Values (y_pred)')
plt.plot(y_pred-y)
plt.title('Error Plot for PCA+LINEAR')
plt.ylabel('Error')
plt.xlabel('Index')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join("Results", 'PCA_Linear_error.png'))
plt.show()

