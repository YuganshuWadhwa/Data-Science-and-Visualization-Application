import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from AI_Regression.Regression_Data import Regression_Data
import matplotlib.pyplot as plt


class Regression:
    """
    Class for all regression methods
    """
    def __init__(self, data_obj: Regression_Data):
        """
        :param data: dataframe containing the dataset
        :param test_size: share of data that is used for testing. Default: 0.2
        """
        # initialize necessary variables
        self.features, self.target, self.model = pd.DataFrame, pd.DataFrame, None
        self.data = data_obj.data.dropna()
        self.process_data(data_obj)
        self.test_size = data_obj.test_size

        # split the dataset into evidence and labels
        self.split_data(data_obj)

        # scale data
        if data_obj.scale is True:
            self.x_scaler = MinMaxScaler()
            self.y_scaler = MinMaxScaler()
            self.target = pd.DataFrame(self.target)
            self.features[self.features.columns] = self.x_scaler.fit_transform(self.features[self.features.columns])
            self.target[self.target.columns] = self.y_scaler.fit_transform(self.target[self.target.columns])
        else:
            self.x_scaler = None
            self.y_scaler = None


        # split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=self.test_size
        )

    def process_data(self, data_obj):
        """
        Drop non-numeric columns from the dataframe, tries to find a column with date and converts it to usable format
        :param data_obj: Regression_Data object
        :return: modified self.data and data_obj.result_string to inform user of processing
        """
        objects = self.data.select_dtypes(include=["object", "datetime64"])
        dropped_columns = []
        converted_columns = []
        for column in objects.columns:
            try:
                objects[column] = pd.to_datetime(objects[column])
                converted_columns.append(column)
                print(f"Column converted: {column}")
            # pandas parsererror can't specifically be caught
            except Exception:
                dropped_columns.append(column)
                print(f"Not converted: {column}")
        self.data = self.data.drop(columns=dropped_columns)
        objects = objects.drop(columns=dropped_columns)
        if data_obj.x_labels is not None:
            for val in dropped_columns:
                if val not in data_obj.x_labels:
                    dropped_columns.remove(val)
        if dropped_columns:
            data_obj.result_string += f"The following columns couldn't be used due to being non-numeric and were " \
                                      f"dropped: {str(dropped_columns).strip('[]')}\n\n"
        date_objects = pd.DataFrame()
        for column in objects.columns:
            date_objects['year'] = objects[column].dt.year
            date_objects['month'] = objects[column].dt.month
            date_objects['day'] = objects[column].dt.day
            date_objects['hour'] = objects[column].dt.hour
            date_objects['minute'] = objects[column].dt.minute
        for column in date_objects.columns:
            if len(np.unique(date_objects[column])) == 1:
                date_objects = date_objects.drop(columns=[column])
        for column in date_objects.columns:
            self.data[column] = date_objects[column]
        self.data = self.data.drop(columns=converted_columns)
        if data_obj.x_labels is not None:
            for val in converted_columns:
                if val not in data_obj.x_labels:
                    converted_columns.remove(val)
                else:
                    for name in date_objects.columns:
                        data_obj.x_labels.append(name)
        if converted_columns:
            data_obj.result_string += f"The following column was converted: {str(converted_columns).strip('[]')}\n\n"
            for value in converted_columns:
                if data_obj.x_labels is not None:
                    data_obj.x_labels.remove(value)

    def split_data(self, data_obj):
        """
        Splits given dataset into evidence and labels, requires labels to be last column of dataframe
        """
        if data_obj.x_labels is None:
            if data_obj.y_label is None:
                self.features = self.data.iloc[:, :-1]
            else:
                self.features = self.data.drop(columns=[data_obj.y_label])
        else:
            self.features = self.data[data_obj.x_labels]
        if data_obj.y_label is None:
            self.target = self.data[self.data.columns[-1]]
            data_obj.y_label = self.data.columns[-1]
        else:
            self.target = self.data[data_obj.y_label]

    def __str__(self):
        """
        Returns a string with infos about the used methods and the achieved results
        :return:
        """
        return f"This is a Regression Superclass"

    @staticmethod
    def print_results(data_obj):
        """
        Adds the results to the result_string for the GUI
        :param data_obj: Regression_Data object
        :return: modified data_obj
        """
        data_obj.result_string += f"The regressors R2_Score is {data_obj.r2_score:.2f}.\n\n" \
                                 f"The mean abs. error is {data_obj.mean_abs_error:.3f}.\n\n" \
                                 f"The mean squared error is {data_obj.mean_sqr_error:.3f}."

    @staticmethod
    def plot_predictions(y_scaler, y_test, predictions, data_obj, train_test):
        """
        Plots the predicted and real values against each other. Plots as many values as the user selected
        :return: modified data_obj
        """
        if data_obj.scale is True:
            # reverse transform
            y_test = pd.DataFrame(y_test)
            predictions = pd.DataFrame(predictions)
            y_test = pd.DataFrame(y_scaler.inverse_transform(y_test[y_test.columns]))
            predictions = pd.DataFrame(y_scaler.inverse_transform(predictions[predictions.columns])).to_numpy()

        fig, ax = plt.subplots()
        plt.plot(y_test.to_numpy()[0:data_obj.n_values], color='red', label='Real data')
        plt.plot(predictions[0:data_obj.n_values], color='blue', label='Predicted data')
        plt.legend()
        plt.grid()
        if train_test == "test":
            plt.title('Prediction on testing set')
            data_obj.prediction = fig
        else:
            plt.title('Prediction on training set')
            data_obj.prediction_train = fig

        fig, ax = plt.subplots()
        plt.plot(y_test.to_numpy(), predictions, 'ro', label="Predicted Values")
        plt.legend()
        plt.grid()
        plt.axis('square')
        plt.axline([0, 0], [1, 1], label="Optimal", color="black")
        plt.legend()
        if train_test == "test":
            plt.title('y_test vs predictions')
            data_obj.prediction_y_test = fig
        else:
            plt.title('y_train vs predictions')
            data_obj.prediction_y_train = fig
