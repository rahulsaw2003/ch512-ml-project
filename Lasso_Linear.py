import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Models.Preprocessing.Lasso import LassoRegression
from Models.Preprocessing.GridSearchCV import *
from Models.LinearModel import LinearRegression
from Models.utils import *
import os

def main():
    #defining the parameters
    iterations=100
    learning_rate=0.02
    data=pd.read_csv("E:\CH512\Dataset\data.csv")
    #print(data)

    x_train,y_train,x_validate,y_validate,x_test,y_test=SplitDataset("E:\CH512\Dataset\data.csv",0.7,0.3,0)
    params = np.arange(0.00001, 10, 0.005)
    LassoModel=LassoRegression(iterations,learning_rate,0.1)
    GridSearch=GridsearchCV(LassoModel,params=params)
    min_mse, best_param=GridSearch.Search(x_train,y_train,x_validate,y_validate)

    #calling the model with the best parameter
    BestLassoModel=LassoRegression(iterations,learning_rate,best_param)
    BestLassoModel.fit(x_train,y_train,False,0)
    features=data.drop("y", axis=1).columns
    print("Features: {}".format(features.values))

    #getting the importance factor 
    importance_factors=BestLassoModel.ImportanceFactor()
    importance_factors=importance_factors.reshape(-1)
    print("importance_factors: ",importance_factors)
    indices = np.where(importance_factors >0.02)[0]
    print("indices: ",indices)
    #cleaning the data
    features_to_keep = data.columns[indices]
    data_filtered = data[features_to_keep]
    data_filtered['y'] = data['y'].values
    print(data_filtered)


    #now getting the mse loss for the updated features
    x1_train,y1_train,x1_validate,y1_validate,x1_test,y1_test=SplitDataset_DataFrame(data_filtered,0.7,0.3,0)

    FinalModel=LinearRegression(iterations,learning_rate)
    FinalModel.fit(x1_train,y1_train)
    y_final_pred=FinalModel.predict(x1_validate)

    #Results
    # MSE LOSS 0.03241180293154075
    # AIC 16.858465269020208    
    # BIC 39.74081311426133
    final_loss=MSELoss(y_final_pred,y1_validate)
    print("final model mse loss: ",final_loss)
    aic=AIC(y_final_pred,y1_validate,x1_validate)
    bic=BIC(y_final_pred,y1_validate,x1_validate)
    print("MSE LOSS",final_loss)
    print("AIC",aic)
    print("BIC",bic)

    

    # Plot the scatter plot of predicted vs. real values
    
    plt.figure(figsize=(8, 6))
    # print(y_final_pred)
    y_final_pred=np.array(y_final_pred)
    y1_validate=np.array(y1_validate)
    plt.plot(y_final_pred.flatten()-y1_validate.flatten())
    plt.title('Error Plot for Lasso+LINEAR')
    plt.ylabel('Error')
    plt.xlabel('Index')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("Results", 'Lasso-Linear_error.png'))
    plt.show()


if __name__ == "__main__":
    main()