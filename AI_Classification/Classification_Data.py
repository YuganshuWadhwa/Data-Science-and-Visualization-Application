from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Classification_Data:
    # INPUTS
    # data
    data: pd.DataFrame
    # Share of Data that is used for testing
    test_size: float = 0.2
    # Labels of columns used as input, if None all will be used
    x_labels: list[str] = None
    # Label of column used as Output/Classification Categories, if None last column will be used
    y_label: str = None
    # number and nodes for hidden layers as array. Ex: 3 layers with 64 nodes each: [64, 64, 64], more nodes and
    # layers lead to longer training times and more accurate results
    hidden_layers: list[int] = (64, 64)
    # number of training epochs, higher number leads to longer training time
    training_epochs: int = 10
    # activation functions from tf.keras.activations, some might work better than others
    activation_func: str = "relu"
    # Whether during the training a part of the data will already be used for testing after each epoch,
    # needed for accuracy/loss per epoch graphs
    validation_split: bool = True
    # Number of trees in the forest, higher number leads to higher training times
    trees: int = 100

    # classifier model, type must match chosen class or model will be ignored
    model: None = None

    # OUTPUTS
    # Plots
    confusion_matrix_train: plt.Figure = plt.figure(figsize=(10, 6))
    confusion_matrix_test: plt.Figure = plt.figure(figsize=(10, 6))
    accuracy_per_epoch: plt.Figure = plt.figure(figsize=(10, 6))
    feature_importance: plt.Figure = plt.figure(figsize=(10, 6))

    # text based
    result_string: str = ""
    feature_importance_dict: dict = None
    accuracy_score: float = 0


