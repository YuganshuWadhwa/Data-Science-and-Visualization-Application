import numpy as np
import pandas as pd
import joblib
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor


class Regression:
    """
    This class allows to read a Pandas Dataframe and train different regression models on the given data.
    At the beginning the dataframe has to be read in when creating an instance of the class.
    Afterwards a target dimension is chosen and the data is divided into training and test sections.
    One of the provided regression types can be selected and the model can be trained.
    Information on the accuracy of the model is provided by means of various graphics and key figures.
    Via a method it is possible to enter new parameters and to output the prediction of the model.

    This class was build by Team 4.

    """
    
    ############### Load Pandas Dataframe ###############
    
    def __init__(self, Data):
        
        """
        Initialization of an instance of the Regression class

        :param dataframe: Give input as Pandas Dataframe
        :type dataframe: pd.DataFrame
        :raises Exception: When no Pandas Dataframe is given into the function
        :return: None
        
        """
       #For testing taking data from csv 
        self.data = Data
        if isinstance(Data, pd.DataFrame):
            self.dataframe = Data
            self.dataframe_shape = Data.shape
        else:
            raise Exception("No pandas dataframe was given!")
        
    ############### Drop Columns for Regression ###############
    
    def dropColumns(self, label_drop='date'):
        
        """
        Deletes a specified column within the dataframe.

        :param label_drop: Target Columns to drop
        :type label_drop: str
        :return: None
        
        """
        self.dataframe=self.dataframe.drop(label_drop, axis=1)
        print("Column: ", label_drop, " is deleted.")
    
    ############### Dividing Data into Training and Test ###############
    
    def split_train_test(self,label_target="lights", tolerence = 2, rows_to_keep = 16000, testsize=0.2, random_state=42, 
                         deleting_na=True, scaling=True, deleting_duplicates=True,remove_noise = True, cols_to_keep = []):
        
        """
        Method to preprocess the data. Sets a target column and splits the given dataframe into test and training data.
        Allows to delete NA columns, the scaling of the dataset (centering and scaling to unit variance) and deleting duplicates.

        :param label_target: Sets the target column for the regression
        :type label_target: str
        :param tolerence: Maximum tolerance level for the noise in data
        :type tolerence: float, optional
        :param rows_to_keep: Number of rows to keep in the dataframe after preprocessing
        :type rows_to_keep: int, optional
        :param testsize: Represents the proportion of the dataset to include in the test split, defaults to 0.2
        :type testsize: float, optional
        :param random_state: Controls the shuffling applied to the data before applying the split,
        defaults to 42
        :type random_state: int, optional
        :param deleting_na: Remove missing values, defaults to True
        :type deleting_na: bool, optional
        :param scaling: Scale the data using MinMaxScaler, defaults to True
        :type scaling: bool, optional
        :param deleting_duplicates: Deletes duplicates, defaults to True
        :type deleting_duplicates: bool, optional
        :param remove_noise: Remove the noisy data points from the dataframe, defaults to True
        :type remove_noise: bool, optional
        :param cols_to_keep: List of columns to keep in the dataframe, defaults to []
        :type cols_to_keep: list, optional
        :return: None
        
        """
        self.dropColumns()
        self.cols_to_keep = cols_to_keep
        self.dataframe = self.dataframe[self.cols_to_keep + [label_target]]           
        self.dataframe = self.dataframe.head(rows_to_keep)

        # set column targeted by the user as target label for the whole instance
        if (label_target in self.dataframe.columns.values):
            print("The target label is set as: ", label_target)
            self.label_target = label_target

        # set last column as target label for the whole instance (if no user input or wrong user input)
        else:
            print("Error: No valid label name!")
            label_target = self.dataframe.columns.values[len(self.dataframe.columns.values) - 1]
            self.label_target = label_target
            print("As default the last column of dataframe is placed as target label: ", label_target)

        if deleting_na:
            self.dataframe = self.dataframe.dropna()
            # st.write("Data has been preprocessed!"
        
        if deleting_duplicates:
            self.dataframe.drop_duplicates(keep='first', inplace=True)
            # st.write("Duplicates have been deleted!")
        
        if scaling:
            self.scaler = MinMaxScaler(feature_range=(0,5))                
            self.scaler = self.scaler.fit(self.dataframe)
            self.dataframe = pd.DataFrame(self.scaler.transform(self.dataframe), columns=self.dataframe.columns)

        if remove_noise:
                self.max_vals = tolerence
                self.threshold = self.max_vals
                print('Threshold = ' , self.max_vals)
                z_scores = np.abs(stats.zscore(self.dataframe[self.cols_to_keep  + [label_target]]))
                self.dataframe = self.dataframe[(z_scores < self.threshold).all(axis=1)]

        self.x = self.dataframe.drop(label_target, axis=1)
        self.y = self.dataframe[[label_target]]
        self.y = np.ravel(self.y)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=testsize,random_state=random_state)

        print("The given data is seperated into test and training data for the given Target Label")
        print("Train data size for x:", self.x_train.shape)
        print("Train data size for y:", self.y_train.shape)

        self.data_splitted = True

    ############### Create the Regression Model ###############
    
    def build_regression(self, regression_name="Support Vector Machine Regression", **args):
        
        """
        Builds a specified Regression Model with the given Training Data.

        :param regression_name: Name of the chosen Regression Model
        :type regression_name: str = 'Support Vector Machine Regression ','Elastic Net Regression ','Ridge Regression ','Linear Regression ', 'Stochastic Gradient Descent Regression '
        :param args: Arguments depending on the chosen Regression Model.
        :return: self.regression, params
        
        """
        self.regression_name=regression_name
        
        # check wether data is already preprocessed and therefore already splitted (if this is not the case, it is not possible to proceed)
        
        if self.data_splitted:
            
        # Checks which Regression Model is selected and calls the designated method. Serves as an interface between user selection and the model building.
        # The method returns a trained model and stores it in the class instance (self.regression).
        # Key figures as RMSE and r2 score are returned to the user.
           
            if self.regression_name == "Support Vector Machine Regression":
                self.regression = self.svm_regression(**args)                    
            elif self.regression_name == "Gradient Boosting Regression":
                self.regression = self.Gradient_boosting_regression(**args)
            elif self.regression_name == "Decision Tree":
                self.regression = self.decisionTree(**args)  
            else:
                raise Exception(
                    "Regression was not found! Avialable options are Support Vector Machine Regression (SVM), Gradient Boosting Regression, Decission Tree.")
            self.y_pred = self.regression.predict(self.x_test)
            self.y_pred2 = self.regression.predict(self.x_train)

            self.mse_test = [ round(mean_squared_error(self.y_test, self.y_pred), 3)]
            self.mse_train =[ round(mean_squared_error(self.y_train, self.y_pred2), 3)]
            self.r2_test =  [ round(r2_score(self.y_test, self.y_pred), 3)]
            self.r2_train = [ round(r2_score(self.y_train, self.y_pred2), 3)]            
            
            print('MSE errors', self.mse_test)
            print("For Test Data")
            print('R2 errors', self.r2_test)
            print("For Training Data")
            print('MSE errors', self.mse_train)
            print('R2 errors', self.r2_train)
        else:
            print("The data has to be splitted in train and test data befor a regression can be build (use the split_train_test method).")
    
    def Gradient_boosting_regression(self, alpha= 0.1, max_depth=10, min_samples_leaf=10, n_estimators=200, learning_rate=0.2):
        
        """
        Method to perform Gradient Boosting Regression on the preprocessed data.

        :param alpha: The regularization parameter alpha, defaults to 0.1
        :type alpha: float, optional
        :param max_depth: The maximum depth of the individual regression estimators, defaults to 10
        :type max_depth: int, optional
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node, defaults to 10
        :type min_samples_leaf: int, optional
        :param n_estimators: The number of boosting stages to perform, defaults to 200
        :type n_estimators: int, optional
        :param learning_rate: Learning rate shrinks the contribution of each estimator by learning_rate, defaults to 0.2
        :type learning_rate: float, optional
        :return: None
        
        """
        
        eln = GradientBoostingRegressor( alpha=alpha, max_depth=max_depth, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, learning_rate=learning_rate)
        return eln.fit(self.x_train, self.y_train)
    
    def svm_regression(self, kernel='rbf', degree=3, svmNumber=100, epsilon=0.1, maxIterations=-1):
        
        """
        Method to perform SVM Regression on the preprocessed data.

        :param kernel: Specifies the kernel type to be used in the algorithm, defaults to 'rbf'
        :type kernel: str = 'linear', 'poly', 'rbf', 'sigmoid', optional
        :param degree: Degree of the polynomial kernel function, defaults to 3
        :type degree: int, optional
        :param svmNumber: C-Support Vector Regression, defaults to 100
        :type svmNumber: int, optional
        :param epsilon: Specifies the epsilon-tube within which no penalty is associated in the training loss function, defaults to 0.1
        :type epsilon: float, optional
        :param maxIterations: Hard limit on iterations within solver, or -1 for no limit, defaults to -1
        :type maxIterations: int, optional
        :return: None
        
        """
        svm = SVR(kernel=kernel, degree=degree, C=svmNumber,epsilon=epsilon ,max_iter=maxIterations)
        return svm.fit(self.x_train, self.y_train)
    
    def decisionTree(self,splitter='best', max_depth=200, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=1500):
        
        """
        Method to perform Decision Tree Regression on the preprocessed data.

        :param splitter: The strategy used to choose the split at each node, defaults to 'best'
        :type splitter: str, optional
        :param max_depth: The maximum depth of the tree, defaults to 200
        :type max_depth: int, optional
        :param min_samples_split: The minimum number of samples required to split an internal node, defaults to 2
        :type min_samples_split: int, optional
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node, defaults to 1
        :type min_samples_leaf: int, optional
        :param max_leaf_nodes: Grow a tree with max_leaf_nodes in best-first fashion, defaults to 1500
        :type max_leaf_nodes: int, optional
        :return: None
        
        """
        clf = DecisionTreeRegressor( splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf = min_samples_leaf , max_leaf_nodes=max_leaf_nodes )
        return clf.fit(self.x_train, self.y_train)
    
    ############### Plot test Data in 2D Graph with regression ###############
    
    def plot_test_data(self):
        
        """
        Method to plot the test data against the predicted data using a scatter plot and a line plot.
        The first plot shows the relationship between y_test and y_pred, with a best fit line and a scatter plot of the actual data points.
        The second plot shows the first 50 data points of y_test and y_pred, with the actual data plotted in blue and the predicted data plotted in red.

        :return: None
        """
        try:
            fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
            ax1.scatter(self.y_test, self.y_pred, label='data', color='blue')
            max_val = np.max(np.abs(self.y))
            max_val = round(max_val*1.3) 
            ax1.set_xlim(0, max_val)
            ax1.set_ylim(0, max_val)
            line_x = np.arange(self.y_test.min(), self.y_test.max(), 0.1)
            line_y = line_x
            ax1.plot(line_x, line_y, color='red', label='best fit line')
            ax1.set_title('y_test vs predictions')
            ax1.set_xlabel("y_test")
            ax1.set_ylabel("y_pred")          
            ax1.legend()
            ax1.grid(True)

            ax2.plot(self.y_test[:50], color='blue', label='Real data')
            ax2.plot(self.y_pred[:50], color='red', label='Predicted data')
            ax2.set_title('y_test vs predictions')
            ax2.set_ylim(0, max_val)
            ax2.legend()
            ax2.grid(True)

            self.fig_test = fig
            # plt.show()      
            
        except:
            print("Something went wrong. Check your Inputs.")
    
    ############ Plot train Data in 2D Graph with regression ###############
    
    def plot_train_data(self):
        
        """
        Plots the actual training data vs predicted training data using two subplots:
        (1) Scatter plot of y_train vs y_pred2 with a best fit line
        (2) Line plot of the first 50 actual and predicted training data points
        
        :return: None
        """
        try:
            fig, (ax3,ax4) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))            
            ax3.scatter(self.y_train, self.y_pred2, label='data',color='blue')
            max_val = np.max(np.abs(self.y))
            max_val = round(max_val*1.3) 
            ax3.set_xlim(0, max_val)
            ax3.set_ylim(0, max_val)
            line_x = np.arange(self.y_train.min(), self.y_train.max(), 0.1)
            line_y = line_x
            ax3.plot(line_x, line_y, color='red', label='best fit line')
            ax3.set_title('y_train vs predictions')
            ax3.set_xlabel("y_train")
            ax3.set_ylabel("y_pred")          
            ax3.legend()         
            ax3.grid(True)

            ax4.plot(self.y_train[:50], color='blue', label='Real data')
            ax4.plot(self.y_pred2[:50], color='red', label='Predicted data')
            ax4.set_title('y_train vs predictions')
            ax4.set_ylim(0, max_val)     
            ax4.legend()
            ax4.grid(True)
            
            self.fig_train = fig
            # plt.show()           
        
        except:
            print("Something went wrong. Check your Inputs.")

    ############### Predict User Inputs ###############
    
    def regression_function(self, user_input):
        
        """
        Feeds user inputs into the Regression Model and outputs the Prediction.

        :param user_input: Give input as Pandas Dataframe
        :type user_input: pd.DataFrame
        :return: model prediction
        """
        if isinstance(user_input, pd.DataFrame):
            user_input = user_input 
            print("User Input is given as Pandas Dataframe!")
            #Check wether number of user inputs is correct 
            if len(user_input.columns) == (len(self.dataframe.columns)-1): 
                print("Number of Input variables fits the function!") 
                result = round(self.regression.predict(user_input)[0], 2)                
                self.predicted_result = result
                return result 
            else:
                print('Input data should be in this formate ',self.cols_to_keep)
        else:
            print( "User Input is not correct. Check wether the Input is converted as a Pandas Dataframe and the length of the input frame is correct."
                  "The active dataframe allows ", (len(self.dataframe.columns) - 1),
                  " input variables. The predicted colum is ", self.label_target)
        return None   



#### Below code is to check how the code is working 
#d = pd.read_csv('filtered_to_munam.csv') # dataframe
#classa = Regression(d)

#f='Appliances'
#A = classa.split_train_test(f)

#AA = classa.build_regression()

#classa.plot_test_data()

#classa.plot_train_data()

#user_input = pd.read_csv('UserInput - Copy.csv') # dataframe
#classa.regression_function(user_input)

