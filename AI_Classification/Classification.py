import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from AI_Classification.Classification_Data import Classification_Data


class Classification:
    """
    Class for all classification methods
    """
    def __init__(self, data_obj: Classification_Data):
        """
        Initialize the class, preprocessing of data, split into x and y, train and test, encode variables if needed
        :param data_obj: Classification_Data object
        """
        # initialize necessary variables
        self.evidence, self.labels, self.model = pd.DataFrame, pd.DataFrame, None
        self.data = data_obj.data.dropna()
        data_obj.data = self.encode()
        self.test_size = data_obj.test_size

        # split the dataset into evidence and labels
        self.split_evidence_labels(data_obj)

        # split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.evidence, self.labels, test_size=self.test_size
        )

    def encode(self):
        """
        Encodes variables that are not integer or float format
        :return: converted dataframe
        """
        label_encoder = LabelEncoder()
        for value in self.data.select_dtypes(include=["object"]).columns.values:
            self.data[value] = label_encoder.fit_transform(self.data[value])
        return self.data

    def split_evidence_labels(self, data_obj):
        """
        Splits given dataset into evidence and labels
        :param data_obj: Classification_Data object
        """
        if data_obj.x_labels is None:
            if data_obj.y_label is None:
                self.evidence = self.data.iloc[:, :-1]
            else:
                self.evidence = self.data.drop(columns=[data_obj.y_label])
        else:
            self.evidence = self.data[data_obj.x_labels]
        if data_obj.y_label is None:
            self.labels = self.data[self.data.columns[-1]]
            data_obj.y_label = self.data.columns[-1]
        else:
            self.labels = self.data[data_obj.y_label].subtract(self.data[data_obj.y_label].min())

    def __str__(self):
        """
        Returns a string with infos about the used methods and the achieved results
        :return: string
        """
        return f"This is a Classification Superclass used for data preprocessing and evaluating and plotting results"

    @staticmethod
    def plot_confusion_matrix(y_test, predictions, title):
        """
        Generates a confusion matrix with given labels and predictions
        :param y_test: real labels
        :param predictions: predicted labels
        :param title: Title for the plot
        :return: matplotlib subplot
        """
        conf_matrix = confusion_matrix(y_test, predictions)
        conf_matrix = pd.DataFrame(conf_matrix)
        fig, ax = plt.subplots()
        ax = sn.heatmap(conf_matrix, annot=True)
        plt.title(title)
        return fig

