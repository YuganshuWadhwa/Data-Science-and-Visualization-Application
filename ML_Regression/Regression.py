import array
import numpy as np
import pandas as pd
from sklearn.svm import NuSVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import NuSVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
import seaborn as sns 
import streamlit as st

class Regression:
    
    ############### Load Pandas Dataframe ###############
    
    def __init__(self, dataframe):
       #For testing taking data from csv 
        self.data = dataframe
        if isinstance(self.data, pd.DataFrame):
            self.dataframe_1 = self.data
            self.dataframe_shape = self.data.shape
            self.dropColumns()
        else:
            raise Exception("No pandas dataframe was given!")

    ############### Drop Columns for Regression ###############

    def dropColumns(self, label_drop='date'):

        self.dataframe=self.dataframe_1.drop(label_drop, axis=1)

        print("Column: ", label_drop, " is deleted.")

        
    ############### Dividing Data into Training and Test ###############
      
    def split_train_test(self,label_target, testsize=0.3, random_state=1, deleting_na=False, scaling=False, deleting_duplicates=False):

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
                # st.write("Data has been preprocessed!")

            if scaling:
                scaler = StandardScaler()
                self.dataframe = pd.DataFrame(scaler.fit_transform(self.dataframe), columns=self.dataframe.columns)
                # self.dataframe = scaler.transform(dataframe)
                # st.write("Data has been rescalled!")

            if deleting_duplicates:
                self.dataframe.drop_duplicates(keep='first', inplace=True)
                # st.write("Duplicates have been deleted!")

            self.x = self.dataframe.drop(label_target, axis=1)
            self.y = self.dataframe[[label_target]]
            self.y = np.ravel(self.y)

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=testsize,random_state=random_state)

            self.data_splitted = True

        
    ############### Create the Regression Model ###############

    def build_regression(self, regression_name="Support Vector Machine Regression", **args):
            self.regression_name=regression_name
            if self.data_splitted:
                if regression_name == "Support Vector Machine Regression":
                    self.regression = self.svm_regression(**args)
                elif regression_name == "Polynomial Regression":
                    self.regression = self.polynomial_regression(**args)
                elif regression_name == "Ridge Regression":
                    self.regression = self.ridge_regression(**args)
                elif regression_name == "Multiple Linear Regression":
                    self.regression = self.linear_regression(**args)
                elif regression_name == "Huber Regression":
                    self.regression = self.huber_regression(**args)
                else:
                    raise Exception(
                        "Regression was not found! Avialable options are Support Vector Machine Regression (SVM), Polynomial Regression, Ridge Regression, Multiple Linear Regression, Robust Regression.")
               
                if regression_name == "Polynomial regression":
                    self.y_pred = self.regression.predict(self.poly.fit_transform(self.x_test))
                else:
                    self.y_pred = self.regression.predict(self.x_test)
#Prams includes method name, MSE and r2_score .. need to show with graph 
                params = [str(regression_name),
                          round(mean_squared_error(self.y_test, self.y_pred), 4),
                          round(r2_score(self.y_test, self.y_pred), 4)]
                self.params=params
                return self.regression, params
                
            else:
                print("The data has to be splitted in train and test data befor a regression can be build (use the split_train_test method).")


      
    def svm_regression(self, kernel='rbf', degree=3, svmNumber=0.5, maxIterations=-1):
        svm = NuSVR(kernel=kernel, degree=degree, nu=svmNumber, max_iter=maxIterations)
        return svm.fit(self.x_train, self.y_train)
    
    def polynomial_regression(self,degree=4):
        self.poly = PolynomialFeatures(degree)
        X_train_poly = self.poly.fit_transform(self.x_train)
        self.poly.fit(X_train_poly, self.y_train)
        self.regressor = LinearRegression()
        return self.regressor.fit(X_train_poly, self.y_train)

    def ridge_regression(self, max_iter=15000, solver='auto'):
        clf = Ridge(max_iter=max_iter, solver=solver)
        return clf.fit(self.x_train, self.y_train)
    
    def linear_regression(self):
        reg = LinearRegression()
        return reg.fit(self.x_train, self.y_train)
        
    def huber_regression(self, max_iter=15000):
        huber = HuberRegressor(max_iter=max_iter)
        return huber.fit(self.x_train, self.y_train)
     
    ############### Plot Data in 2D Graph with regression ###############
    
    def plot_regression_1(self):
        try:
            fig, ax = plt.subplots()

            ax.scatter(self.y_test, self.y_pred, label='data')
            
            x_values = [self.y_test.min(), self.y_test.max()]
            y_values = [self.y_pred.min(), self.y_pred.max()]
            
            max_val = np.max(np.abs(self.y))

            ax.set_xlim(-5, max_val*0.8)
            ax.set_ylim(-5, max_val*0.8)

            if self.regression_name == "Polynomial regression":
                ax.plot(np.unique(self.y_test), np.poly1d(np.polyfit(self.y_test, self.y_pred, 1))(np.unique(self.y_test)),color='red', label='fittedmodel')
                ax.set_title('Polynomial Regression')

            elif self.regression_name == "Ridge Regression ":
                ax.plot(self.y_test,np.poly1d(np.polyfit(self.y_test, self.y_pred, 1.9))(self.y_test),"r", label='fittedmodel')
                ax.set_title('Ridge Regression')

            else:
                ax.plot(x_values,y_values,'r--', lw=2, label='fittedmodel')
                ax.set_title('Regression')
           

            ax.set_xlabel("y_test")
            ax.set_ylabel("y_pred")
            
            ax.legend()
            self.fig = fig   
     
        except:
            print("Something went wrong. Check your Inputs.")
    
    
        
        
    ############### Predict User Inputs ###############

    def regression_function(self, user_input):
            # check wether user input is given as Pandas Dataframe
            if isinstance(user_input, pd.DataFrame):
                user_input = user_input
                st.write("User Input is given as Pandas Dataframe!")                    # extra info
                # Check wether number of user inputs is correct
                if len(user_input.columns) == (len(self.dataframe.columns)-1):
                    st.write("Number of Input variables fits the function!")               # extra info
                    self.result = round(self.regression.predict(user_input)[0], 2)
                    st.write('The prediction for the target label "', self.label_target, '" is ', self.result, '.')    # actual result
                    return self.result

                else:
                    st.write('Length of Entered Data (' + str(len(user_input.columns)) + ') is unequal to the Lenght of Training Data (' + str(len(self.dataframe.columns)-1) + ')')
            else:
                st.write(
                    "User Input is not correct. Check wether the Input is converted as a Pandas Dataframe and the length of the input frame is correct."
                    "The active dataframe allows ", (len(self.dataframe.columns) - 1),
                    " input variables. The predicted colum is ", self.label_target)             # warning
            # default return value if function did not execute successfully
            return None
    
    ############### Get Pandas Dataframe Description ###############
     
    def get_dataframe_head(self):
        """Return the head of the dataframe.

        :return: Dataframe Head
        """

        return self.dataframe.head()