from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Regression_Data:
    # INPUTS
    # data
    data: pd.DataFrame
    # Share/Percentage of Data used for testing
    test_size: float = 0.2
    # with x_labels, if None all will be used
    x_labels: list[str] = None
    # title of column with labels for regression, if None last column will be used
    y_label: str = None
    # number of values to use from the testing set for plotting
    n_values: int = 50
    # number and nodes for hidden layers as array 3 layers with 64 nodes each: [64, 64, 64]
    hidden_layers: list[int] = (64, 64)
    # number of training epochs
    training_epochs: int = 10
    # activation functions from tf.keras.activations
    activation_func: str = "relu"
    # Number of trees
    trees: int = 100
    # wether the data will be scaled
    scale: bool = True

    model: None = None

    # OUTPUTS
    # Plots
    prediction: plt.Figure = plt.figure(figsize=(10, 6))
    prediction_y_test: plt.figure = plt.figure(figsize=(10, 6))
    prediction_train: plt.Figure = plt.figure(figsize=(10, 6))
    prediction_y_train: plt.figure = plt.figure(figsize=(10, 6))
    loss_per_epoch: plt.Figure = plt.figure(figsize=(10, 6))
    feature_importance: plt.Figure = plt.figure(figsize=(10, 6))

    # text based
    result_string: str = ""
    r2_score: float = None
    mean_abs_error: float = None
    mean_sqr_error: float = None
    feature_importance_dict: dict = None

    def __str__(self):
        return "This is a Dataclass for Regression."
