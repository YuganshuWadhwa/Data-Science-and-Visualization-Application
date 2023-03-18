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
    
    ############### Load Pandas Dataframe ###############
    def __init__(self, Data):
       #For testing taking data from csv 
        self.data = Data
        if isinstance(Data, pd.DataFrame):
            self.dataframe = Data
            self.dataframe_shape = Data.shape
            #self.dropColumns()
        else:
            raise Exception("No pandas dataframe was given!")
        
    ############### Drop Columns for Regression ###############
    def dropColumns(self, label_drop='date'):
        self.dataframe=self.dataframe.drop(label_drop, axis=1)
        print("Column: ", label_drop, " is deleted.")
    
    ############### Dividing Data into Training and Test ###############
    def split_train_test(self,label_target="lights", tolerence = 2, rows_to_keep = 16000, testsize=0.2, random_state=42, deleting_na=True, scaling=True, deleting_duplicates=True,remove_noise = True, cols_to_keep = []):
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
            self.regression_name=regression_name
            if self.data_splitted:
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
        eln = GradientBoostingRegressor( alpha=alpha, max_depth=max_depth, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, learning_rate=learning_rate)
        return eln.fit(self.x_train, self.y_train)
    
    def svm_regression(self, kernel='rbf', degree=3, svmNumber=100, epsilon=0.1, maxIterations=-1):
        svm = SVR(kernel=kernel, degree=degree, C=svmNumber,epsilon=epsilon ,max_iter=maxIterations)
        return svm.fit(self.x_train, self.y_train)
    
    def decisionTree(self,splitter='best', max_depth=200, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=1500):
        clf = DecisionTreeRegressor( splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf = min_samples_leaf , max_leaf_nodes=max_leaf_nodes )
        return clf.fit(self.x_train, self.y_train)
    
    ############### Plot test Data in 2D Graph with regression ###############s
    def plot_test_data(self):
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
        if isinstance(user_input, pd.DataFrame):
            user_input = user_input 
            print("User Input is given as Pandas Dataframe!")
            #Check wether number of user inputs is correct 
            if len(user_input.columns) == (len(self.dataframe.columns)-1): 
                print("Number of Input variables fits the function!") 
                result = round(self.regression.predict(user_input)[0], 2)                
                user_input= np.array(user_input)
                result = np.array(result)
                self.predicted_result = result
                self.final_prediction = user_input + result
                print('The prediction for the target label ', self.label_target, '" is ' )#self.final_prediction.iloc[:, [-1],'.'])              
                return result 
            else:
                print('Input data should be in this formate ',self.cols_to_keep)
        else:
            print( "User Input is not correct. Check wether the Input is converted as a Pandas Dataframe and the length of the input frame is correct."
                  "The active dataframe allows ", (len(self.dataframe.columns) - 1),
                  " input variables. The predicted colum is ", self.label_target)
        return None   

