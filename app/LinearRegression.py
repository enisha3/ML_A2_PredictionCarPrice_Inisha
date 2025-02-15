from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib as plt
import mlflow

class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)

# 1. Gradient Descent Configuration
# Learning rate (lr), momentum (momentum), and method (batch, mini-batch, or stochastic) suggest an optimization process.
# prev_step = 0 is likely used to track previous weight updates when using momentum-based optimization.

# 2. Regularization
# regularization indicates that some form of L1 (Lasso) or L2 (Ridge) regularization is applied.

# 3. Polynomial Features
# If polynomial=True, the model applies polynomial transformation to input features, expanding them up to degree.

# 4. Cross-Validation (cv)
# cv=kfold suggests k-fold cross-validation, which helps in model validation.

# 5. Weight Initialization (weight)
# weight='zeros' suggests the model initializes weights with zeros, but it might support other strategies like random initialization.
            
    def __init__(self, regularization, lr=0.001, method='batch', num_epochs=500, batch_size=50, cv=kfold, polynomial= True, degree= 3, weight='zeros', momentum=0.0):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization
        self.weight = weight
        self.degree = degree
        self.momentum = momentum
        self.polynomial = polynomial
        self.prev_step = 0
    
    # Why Use MSE?
    # MSE penalizes large errors more than small errors because of squaring.
    # It's a smooth function (differentiable), making it useful for gradient-based optimization (e.g., gradient descent).
    # Commonly used in regression problems to measure how well a model fits the data.
    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    # Why Use R² Score?
    
    # The R² (R-squared) score, or coefficient of determination, is used to evaluate regression models by measuring how well the model's predictions explain the variance in the actual data.
    def r2(self, ytrue, ypred):
        ss_res = np.sum((ytrue - ypred) ** 2)  # Residual Sum of Squares (RSS) - Measures the total error in the model's predictions. Lower RSS = better model fit.
        ss_tot = np.sum((ytrue - np.mean(ytrue)) ** 2)  # Total Sum of Squares (TSS) - Measures the total variance in the actual data. Higher TSS = more variability in data.
        r2score = 1 - (ss_res / ss_tot) #If RSS is small, it means the model fits well, and R² is close to 1. If RSS is large, the model performs poorly, and R² approaches 0.
        return r2score
    
    # function to compute average mse for all kfold_scores
    def avgMse(self):
        return np.sum(np.array(self.kfold_scores))/len(self.kfold_scores)
    
    # function to compute average r2 for all kfold_scores
    def avgr2(self):
        return np.sum(np.array(self.kfold_r2))/len(self.kfold_r2)
    
    def fit(self, X_train, y_train):

        # Ensures that feature names are stored for later use.
        self.columns = X_train.columns

        if self.polynomial == True: # If True, it means the user wants to apply polynomial feature transformation to the dataset.
            X_train = self._transform_features(X_train) 
        # Calls a function _transform_features(X_train), which likely generates polynomial features (e.g., x^2, x^3, etc.) to improve model performance for non-linear relationships.
        # The transformed X_train replaces the original version.
            print(X_train.shape)
            print("Using Polynomial")
        else:
            X_train = X_train.to_numpy()
            print("Using Linear")
        # If self.polynomial is False, the dataset remains unchanged except that it is converted into a NumPy array.
        # This ensures consistency when passing data to machine learning models that require NumPy arrays instead of pandas DataFrames.

        y_train = y_train.to_numpy() #Converts the target variable (y_train) into a NumPy array for compatibility with ML models.

        #create a list of kfold scores and r2
        self.kfold_scores = list()
        self.kfold_r2 = list()
        
        #reset val lossß
        self.val_loss_old = np.inf

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            if(self.weight == 'zeros'):
                self.theta = np.zeros(X_cross_train.shape[1]) # This condition checks if the self.weight attribute is set to 'zeros'. If so, it initializes the model's parameters (theta) to be a vector of zeros.
            elif(self.weight == 'xavier'):
                # number of samples
                m = X_cross_train.shape[0] # calculates the number of samples in the training set.

                # calculating the range for the weight
                lower, upper = -(1.0 / np.sqrt(m)), (1.0 / np.sqrt(m)) # The range within which the weights will be initialized. This range depends on the size of the dataset (specifically the number of samples).
                num = np.random.rand(X_cross_train.shape[1]) # Generates random numbers between 0 and 1 for each feature.
                # The numbers are scaled and shifted to lie within the lower and upper bounds.
                # This helps to create weights that are small but not too close to zero, preventing issues with gradients during training.
                scaled = lower + num * (upper - lower)
                self.theta = scaled
            else:
                print("Weight Initialization Method Is Invalid")
                return
            
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx].reshape(1, ) # (1, ) ==> (1, ) ==> (m, )
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    # The code is tracking the model's performance on the cross-validation data during each training epoch.
                    # Validation loss (MSE) and validation R² score are logged at each epoch using MLflow to monitor how well the model is generalizing.
                    yhat_val = self.predict(X_cross_val) # This line uses the model's predict method to make predictions on the cross-validation data (X_cross_val).
                    val_loss_new = self.mse(y_cross_val, yhat_val) # This line calculates the mean squared error (MSE) between the true labels (y_cross_val) and the predicted labels (yhat_val) for the cross-validation set.
                    val_r2_new = self.r2(y_cross_val, yhat_val) # This line calculates the R² (coefficient of determination) score between the true labels (y_cross_val) and the predicted labels (yhat_val) for the cross-validation data.
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch) # This line logs the calculated validation loss (MSE) (val_loss_new) into MLflow, a platform for managing machine learning experiments.
                    mlflow.log_metric(key="val_r2", value=val_r2_new, step=epoch) # Similar to the previous line, this logs the validation R² score (val_r2_new) into MLflow for the current epoch.
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                self.kfold_r2.append(val_r2_new)
                print(f"Fold {fold}: MSE:{val_loss_new}")
                print(f"Fold {fold}: r2:{val_r2_new}")
        
    def _transform_features(self, X):
        # Transform input features to include polynomial degree --> highest degree is taken
        X_poly = np.column_stack([X ** (self.degree)])        
        return X_poly
                 
    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]

        if self.regularization:
            grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)
        else:
            grad = (1/m) * X.T @ (yhat - y)

        if(self.momentum >= 0 and self.momentum <= 1):
            #momentum implemented 
            gra = self.lr * grad
            self.theta = self.theta - gra + self.momentum * self.prev_step
            self.prev_step = gra
        else:
            #mommentum range is between (0, 1)
            self.theta = self.theta - self.lr * grad
        return self.mse(y, yhat)
    
    def predict(self, X):
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    def features(self, X, features):
        return plt.barh(X.columns, features)
    
    def feature_importance(self, width=5, height=10):
        
    # Create a DataFrame with coefficients and feature names, if available
        if self.theta is not None and self.columns is not None:
            coefs = pd.DataFrame(data=self.theta, columns=['Coefficients'],index=self.columns) 
            coefs.plot(kind="barh", figsize=(width, height)) # Create a horizontal bar plot
            plt.title("Feature Importance") # Set the title of the plot
            plt.show()  # Display the plot
        else:
            print("Coefficients or feature names are not available to create the graph.")


class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
class Lasso(LinearRegression):
    
    def __init__(self, method, lr, l,momentum, weight, polynomial, degree):
        self.regularization = LassoPenalty(l)
        
        super().__init__(self.regularization, lr, method, momentum=momentum, weight=weight, polynomial=polynomial, degree=degree)
    
    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)

class Ridge(LinearRegression):
    
    def __init__(self, method, lr, l,momentum, weight, polynomial, degree):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, momentum=momentum, weight=weight, polynomial=polynomial, degree=degree)

    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)
   
class ElasticNet(LinearRegression):
    
    def __init__(self, method, lr, l,  momentum, weight, polynomial, degree,l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr, method, momentum=momentum, weight=weight, polynomial=polynomial, degree=degree)

    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)
    
class Normal(LinearRegression):  
    def __init__(self, method, lr, l, momentum, weight, polynomial, degree):
        self.regularization = None
        super().__init__(self.regularization, lr, method, momentum=momentum, weight=weight, polynomial=polynomial, degree=degree)
   
    
    def avgMSE(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)