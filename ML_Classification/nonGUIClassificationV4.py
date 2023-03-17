import pandas as pd
from sklearn.model_selection import train_test_split


class Classification:

    def __init__(self,userIN_data,userIN_selected_column,userIN_from=-1,userIN_to=0):
        
        self.input_data = userIN_data

        self.user_selected_column = userIN_selected_column

        self.userIN_from,self.userIN_to = userIN_from,userIN_to

        self.data_train, self.data_test = pd.DataFrame, pd.DataFrame

        self.X_train,self.y_train,self.X_test,self.y_test = pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
        
        self.get_num_unique_values()

        self.train_testsplit()

        
        

    #Splits data into train and test data sets --> Returned as DataFrame        

    def train_testsplit(self, test_size=0.3):

        self.data_train, self.data_test = train_test_split(self.input_data, test_size=test_size, random_state=109) #have test_size and maybe random state as input variable



        #Splits Labels from rest of the data and returns train and test data. IT IS ASSUMED THAT LABELS ARE AT THE END OF THE DATA SET
        self.y_test = self.data_test[self.user_selected_column]    #labels are split from rest of the data
        self.y_train = self.data_train[self.user_selected_column]
        self.data_test = self.data_test.drop(self.user_selected_column,axis=1)
        self.data_train = self.data_train.drop(self.user_selected_column,axis=1)

        if self.userIN_from == -1:              #if default is selected use the whole rest of dataframe
            self.X_train = self.data_train
            self.X_test = self.data_test   
        else:
            self.X_train = self.data_train.iloc[:,self.userIN_from:self.userIN_to]              # if range is selected use selected range
            self.X_test = self.data_test.iloc[:,self.userIN_from:self.userIN_to] 
        

        return self.X_train,self.y_train,self.X_test,self.y_test
    
    def get_num_unique_values(self):

        self.num_unique = self.input_data[self.user_selected_column].nunique()      #get number of unique values of dependent variable
    
        self.class_labels = self.input_data[self.user_selected_column].unique()     #get list of labels of unique values of dependent variable


#
#data = read_data('divorce.csv')
#Result = Classification(data,userIN_selected_column='Class')

#print(Result.y_test)

#print("HEEEEEEEREE:")
#print(Result.X_test)