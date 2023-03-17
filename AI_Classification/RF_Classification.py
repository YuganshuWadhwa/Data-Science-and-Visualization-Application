import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from AI_Classification.Classification import Classification
from AI_Classification.Classification_Data import Classification_Data
import pickle


class RF_Classification(Classification):
    def __init__(self, data_obj: Classification_Data):
        """
        RandomForest-classification.
        :param data_obj: Classification_Data object
        """
        # super class initialization for data preprocessing
        super().__init__(data_obj)
        # variable initialization
        self.n = data_obj.trees
        self.predictions = None
        # run classifier
        self.run_classifier(data_obj)

    def run_classifier(self, data_obj):
        """
        Initialize the model, train (if not loaded) and evaluate on test data
        :param data_obj: Classification_Data object
        :return: data_obj with modified variables
        """
        # check if loaded model exists and matches type
        if data_obj.model is not None and isinstance(data_obj.model, RandomForestClassifier):
            self.model = data_obj.model
            print("Model loaded")
        else:
            # create new model
            self.model = self.train_model()
            data_obj.model = self.model
            print("Model created")
        # error handling for loaded model mismatch with selected data
        try:
            # make predictions
            self.predictions = self.model.predict(self.x_test)
            # get evaluation
            data_obj.accuracy_score = accuracy_score(self.y_test, self.predictions)
            data_obj.result_string = f"The random forest classifier has a {accuracy_score(self.y_train, self.model.predict(self.x_train)):.2%} accuracy on the training data.\n\n"
            data_obj.result_string += f"The random forest classifier has a {data_obj.accuracy_score:.2%} accuracy on the testing data.\n\n"
            data_obj.feature_importance_dict = dict(zip(self.x_test.columns, self.model.feature_importances_))
            self.feature_importance_for_chart = {}
            rest = 0.0
            for value in data_obj.feature_importance_dict.keys():
                if data_obj.feature_importance_dict[value] < 0.025:
                    rest += data_obj.feature_importance_dict[value]
                else:
                    self.feature_importance_for_chart[value] = data_obj.feature_importance_dict[value]
            if rest != 0.0:
                self.feature_importance_for_chart["Rest (features with\nindividually < 2,5%"] = rest
            self.plot(data_obj)
        except ValueError:
            data_obj.result_string = "The loaded model does not match the set parameters, please try again!"

    def train_model(self):
        """
        Initializes and trains the random forest
        :return: RandomForestClassifier
        """
        # initialize random forest
        forest = RandomForestClassifier(n_estimators=self.n)
        # fit data
        forest.fit(self.x_train, self.y_train)
        return forest

    def __str__(self):
        return "This method implements the random forest classification."

    def plot(self, data_obj):
        """
        Creates the plots
        :param data_obj: Classification_Data object
        :return: data_object with modified variables
        """
        # confusion matrix
        data_obj.confusion_matrix_test = Classification.plot_confusion_matrix(y_test=self.y_test, predictions=self.predictions, title="Confusion Matrix for Testing Data")
        data_obj.confusion_matrix_train = Classification.plot_confusion_matrix(self.y_train, self.model.predict(self.x_train), "Confusion Matrix for Training Data")
        # feature importance pie chart
        fig, ax = plt.subplots()
        ax.pie(self.feature_importance_for_chart.values(), labels=self.feature_importance_for_chart.keys(), autopct='%1.1f%%', pctdistance=0.8)
        ax.axis('equal')
        plt.title("Feature importances")
        data_obj.feature_importance = fig


def main():
    # import test data
    data = pd.read_csv("./Data/divorce.csv", delimiter=";")
    data_obj = Classification_Data(data=data, trees=100)
    # Create classifier class
    filename = './model.sav'
    #data_obj.model = pickle.load(open(filename, 'rb'))

    classifier = RF_Classification(data_obj)
    #pickle.dump(data_obj.model, open(filename, 'wb'))

    plt.show()
    print(data_obj.result_string)

if __name__ == "__main__":
    main()
