import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from AI_Classification.Classification import Classification
from AI_Classification.Classification_Data import Classification_Data


class NN_Classification(Classification):
    def __init__(self, data_obj: Classification_Data):
        """
        Neural network classification.
        :param data_obj: Classification_Data object
        """
        # initialize superclass for data preprocessing
        super().__init__(data_obj)

        # get the number of output categories from the dataset
        self.output_categories = data_obj.data[data_obj.y_label].nunique()
        # check if imported model exists and has correct type
        if data_obj.model is not None and isinstance(data_obj.model, tf.keras.models.Sequential):
            self.model = data_obj.model
            print("Model loaded")
        else:
            # create the neural network
            self.get_model(data_obj)
            # train the neural network
            if data_obj.validation_split is False:
                self.history = self.model.fit(self.x_train, self.y_train, epochs=data_obj.training_epochs)
            else:
                self.history = self.model.fit(self.x_train, self.y_train, validation_split=0.2,
                                              epochs=data_obj.training_epochs)
            data_obj.model = self.model
            print("Model created")

        # error handling for loaded model mismatch with selected data
        try:
            # predictions for confusion matrix
            self.predictions = self.model.predict(self.x_test)
            data_obj.accuracy_score = accuracy_score(self.y_test, tf.argmax(self.predictions, 1))

            # creating the text output (accuracies training and testing)
            data_obj.result_string = f"The neural network classifier has a " \
                                     f"{accuracy_score(self.y_train, tf.argmax(self.model.predict(self.x_train), 1)):.2%} " \
                                     f"accuracy on the training data.\n\n"
            data_obj.result_string += f"The neural network classifier has a " \
                                      f"{data_obj.accuracy_score:.2%} accuracy on the testing data.\n\n"
            self.plot(data_obj)
        except ValueError:
            data_obj.result_string = "The loaded model does not match the set parameters, please try again!"

    def get_model(self, data_obj):
        """
        Initialize the neural network
        :param data_obj: Classification_Data object
        :return: tf.keras.Sequential model (via self.model)
        """
        self.model = tf.keras.models.Sequential([tf.keras.Input(shape=(self.evidence.shape[1]))])
        for layer in data_obj.hidden_layers:
            self.model.add(tf.keras.layers.Dense(layer, activation=data_obj.activation_func))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(self.output_categories, activation='softmax'))
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def plot(self, data_obj):
        """
        Creates the plots
        :param data_obj: Classification_Data object
        :return: data_object with modified variables
        """

        try:
            fig = plt.figure()
            plt.plot(self.history.history["accuracy"])
            plt.plot(self.history.history["val_accuracy"])
            plt.plot(self.history.history["loss"])
            plt.plot(self.history.history["val_loss"])
            plt.legend(["training accuracy", "testing accuracy", "train-loss", "test-loss"], loc="best")
            plt.xlabel("epoch")
            plt.ylabel("accuracy/loss")
            plt.title("model accuracy & loss")
            plt.grid()
            data_obj.accuracy_per_epoch = fig
        except AttributeError:
            # Loaded model has no history
            data_obj.accuracy_per_epoch = None
        except KeyError:
            # if validation split is disabled
            data_obj.accuracy_per_epoch = None

        # convert predictions from percentages to labels
        conf_predictions = tf.argmax(self.predictions, 1)
        data_obj.confusion_matrix_test = Classification.plot_confusion_matrix(self.y_test, conf_predictions,
                                                                              "Confusion Matrix for Testing Data")
        data_obj.confusion_matrix_train = Classification.plot_confusion_matrix(self.y_train, tf.argmax(
            self.model.predict(self.x_train), 1), "Confusion Matrix for Training Data")


def main(file):
    data = pd.read_csv(file, delimiter=";")
    data_obj = Classification_Data(data=data)
    file = './keras_model.h5'

    #data_obj.model = tf.keras.models.load_model(file)
    classifier = NN_Classification(data_obj)

    # saving model to zip folder
    #data_obj.model.save(file)

    plt.show()
    print(data_obj.result_string)


if __name__ == "__main__":
    main("./Data/divorce.csv")
