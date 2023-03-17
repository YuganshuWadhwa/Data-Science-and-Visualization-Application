import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from ML_Classification.nonGUIClassificationV4 import Classification

class LogisticRegressionClassifier(Classification):


    def __init__(self,userIN_data,userIN_selected_column,userIN_from=-1,userIN_to=0,input_solver='lbfgs',userIN_max_iter=100) :
        super().__init__(userIN_data,userIN_selected_column,userIN_from=-1,userIN_to=0)
        
        self.solver = input_solver
        self.userIN_max_iter = userIN_max_iter        
        
          

    def perform_LogisticRegression(self,userIN_penalty='l2',userIN_random_state=None): 

        #runs automatically in the current state of this code, will train the model using the given input
        #if you want to change this(refering to line 22) just call this method and pass the output to get_results and get_userinput_prediction
        #not every input for penalty runs with every solver 
        
        if self.solver == 'sag' or self.solver == 'saga' or self.solver == 'liblinear':     
            try:
                self.clf = LogisticRegression(solver = self.solver,penalty=userIN_penalty,max_iter=self.userIN_max_iter,random_state=userIN_random_state,l1_ratio=0.5) 
                self.clf.fit(self.X_train, self.y_train)
            except ValueError:
                print("THE CHOSEN COMBINATION OF SOLVER AND PENALTY IS NOT ACCEPTED!")            
        else:
            try: 
                self.clf = LogisticRegression(solver = self.solver,penalty=userIN_penalty,max_iter=self.userIN_max_iter) 
                self.clf.fit(self.X_train, self.y_train)
            except ValueError:
                print("THE CHOSEN COMBINATION OF SOLVER AND PENALTY IS NOT ACCEPTED!")       

         #trained model is returned and assigned to self.model, this can be used for further operations, see self.model.predict in line 33 below

    def get_results(self):      #returns barplot to evaluate the model, to access display self.bragraph results
#TRAINING RESULTS:
        self.y_pred_train = self.clf.predict(self.X_train)                     
        
        self.fltAccuracy_train = metrics.accuracy_score(self.y_train, self.y_pred_train)
        self.fltPrecision_train = metrics.precision_score(self.y_train, self.y_pred_train,average='micro')
        self.fltRecall_train = metrics.recall_score(self.y_train, self.y_pred_train,average='micro')
        self.fltFScore_train = metrics.f1_score(self.y_train, self.y_pred_train,average='micro')

        self.train_results = [self.fltAccuracy_train,self.fltPrecision_train,self.fltRecall_train,self.fltFScore_train]
#TESTING RESULTS:

        self.y_pred = self.clf.predict(self.X_test)                     
        
        self.fltAccuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        self.fltPrecision = metrics.precision_score(self.y_test, self.y_pred,average='micro')
        self.fltRecall = metrics.recall_score(self.y_test, self.y_pred,average='micro')
        self.fltFScore = metrics.f1_score(self.y_test, self.y_pred,average='micro')

        self.test_results = [self.fltAccuracy,self.fltPrecision,self.fltRecall,self.fltFScore]

        bar_labels = ['Accuracy','Precision','Recall','F-Score']

#MAKE THE BARPLOT

        half = np.arange(len(bar_labels))
        width = 0.33

        fig = plt.figure(figsize=(20, 4))
        self.barplot = plt.title("Train/Test Results")
        self.barplot = plt.ylim(0,1.2)
        self.barplot = plt.xticks(half, bar_labels) #label on x axis     

#Labels on the bar, 2nd value adjusts orientation of label height
        for i in range(len(half)):
            plt.text(half[i] - width/2, self.train_results[i] +0.05 , 'Train: '+str(round(self.train_results[i],2)), ha='center')
            plt.text(half[i] + width/2, self.test_results[i] +0.05, 'Test: '+str(round(self.test_results[i],2)), ha='center')
        
        colors = [(0.6, 0.2, 0.2), (0.1, 0.6, 0.6), (0.2, 0.7, 0.2), (0.1, 0.1, 0.6)] #less bright colors, different rgb values
        self.barplot = plt.bar(half-width/2,self.train_results,width = width,color = colors,edgecolor = 'black')
        self.barplot = plt.bar(half+width/2,self.test_results,width = width,color = colors,edgecolor = 'black')
        
        self.bargraph_results = fig


    def get_userinput_prediction(self, user_testdata): #takes pd dataframe and returns predicted classes as str, to access just print(self.strPredictedClass)
        
        try:
            self.y_pred_userinput = self.clf.predict(user_testdata)    # 
            self.strPredictedClass = 'Predicted user class: ' + " ".join(map(str,self.y_pred_userinput))
        except ValueError:
            print("VALUE ERROR:MAKE SURE YOUR TESTDATA HAS THE SAME AMOUNT OF COLUMNS AS THE DATA YOU TRAINED YOUR MODEL ON!")


    def get_plot(self):
       
        #returns confusion matrix plot in figure
        #will give an error if you have not run get_results before this (it is assumed that these will always run together anyways), problem is that self.y_pred will be missing
        #in line 63, if you want to run get_results and get_plot independently just insert self.y_pred = self.model.predict(self.X_test) here
       self.y_pred_train = self.clf.predict(self.X_train)
       
       self.confMatrice= metrics.confusion_matrix(self.y_train, self.y_pred_train)
       self.df_confMatrice= pd.DataFrame(self.confMatrice, range(self.num_unique), range(self.num_unique))
       fig = plt.figure() 
       sn.set(font_scale=1.4) #this is the size on numbers on the side like x axis and y axis representators 
       self.heatmap = sn.heatmap(self.df_confMatrice, annot=True, annot_kws={"size": 16}, xticklabels=self.class_labels,yticklabels=self.class_labels) # font size of internal values
       self.heatmap.set_title('Training Result Confusion Matrix', fontdict={'fontsize':20}, pad=12);       
       self.fig_train = fig  


       self.confMatrice= metrics.confusion_matrix(self.y_test, self.y_pred)
       self.df_confMatrice= pd.DataFrame(self.confMatrice, range(self.num_unique), range(self.num_unique)) 
       fig = plt.figure() 
       sn.set(font_scale=1.4) #this is the size on numbers on the side like x axis and y axis representators 
       self.heatmap = sn.heatmap(self.df_confMatrice, annot=True, annot_kws={"size": 16}, xticklabels=self.class_labels,yticklabels=self.class_labels) # font size of internal values
       self.heatmap.set_title('Test Result Confusion Matrix', fontdict={'fontsize':20}, pad=12);       
       self.fig_test = fig      


# user_test = read_data('user_data_test.csv')
# data = read_data('divorce.csv')
# Results = LogisticRegressionClassifier(data,userIN_selected_column='Class',input_solver='saga')

# Results.perform_LogisticRegression(userIN_penalty='l1',userIN_random_state=None)

# Results.get_results()
# print(Results.strResults)

# Results.get_userinput_prediction(user_test)
# print(Results.strPredictedClass)



#Results.get_plot()
#plt.show()
            
        
