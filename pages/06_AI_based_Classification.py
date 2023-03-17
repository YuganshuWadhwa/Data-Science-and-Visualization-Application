import streamlit as st
from zipfile import ZipFile
import shutil
import keras
import pickle

# import necessary class definitions from relevant packages
from GUI.GUI_Class import GUI_class
from AI_Classification.Classification_Data import Classification_Data
from AI_Classification.NN_Classification import NN_Classification
from AI_Classification.RF_Classification import RF_Classification


# Setup for page configuration
st.set_page_config(
	page_title = 'AI based Classification',
	layout = 'wide'
	)


GUI_data = st.session_state['GUI_data']

header_cont = st.container()


with header_cont :
	st.markdown("<h2 style = 'text-align : center; color : #0077b6;'> CLASSIFICATION USING ARTIFICIAL INTELLIGENCE </h2>", unsafe_allow_html = True)

	st.markdown("<h5 style = 'text-align : center; color : #023e8a;'> Neural Networks &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Random Forest</h5>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <br><b>Classification is a supervised machine learning process by which a trained algorithm predicts the class (target, label or category) of a new set of input data.</b> </div>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <br>Depending on the number of classes, a classification problem can be either binary or multi-class. The most common classification problems are – speech recognition, face detection, handwriting recognition, document classification, etc. Common classification algorithms include: K-nearest neighbor, decision trees, naive bayes, random forest, support vector machine and artificial neural networks. <br><br></div>", unsafe_allow_html = True)

	st.markdown(':bulb: <small> <i> :orange[Classification models work best with Classification type datasets.] </i> </small>', unsafe_allow_html = True)

	st.markdown('# ')


test_col_1, dummy_col_1, test_col_2 = st.columns([1, 0.2, 1])

method = test_col_1.selectbox(label = '**Select Classification Method** :', options = ['Neural Networks', 'Random Forest'], index = 0)
train_or_upload = test_col_2.selectbox(label = '**Train New Model** or **Upload Pre-trained Model** :', options = ['Train New', 'Upload'], index = 0)


method_cont = st.container()


if method == 'Neural Networks' :

	with method_cont :
		st.markdown("<div style ='text-align: justify;'><br><b>Artificial Neural Network (ANN)</b> is a supervised learning process by which computer programs train themselves to recognize patterns in a data to be able to predict the outcome for a new set of similar data. A neural network contains layers of interconnected nodes, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. ANN works based on forward and backward propagation. </div>", unsafe_allow_html = True)

		st.markdown('# ')

		st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

		st.markdown('''

		- **X-Label(s)** : columns used as input (if None then all columns will be used)
		- **Y-Label** : column used as output/classification category (by default, the last column will be used)
		- **Test Size** : share of data used for testing
		- **Hidden Layers** : number of hidden layers, and nodes for each hidden layer
		- **Training Epochs** : number of training epochs
		- **Activation Function**
		- **Validation Split** : signifies whether during the training, a part of the data will already be used for testing after each epoch
			
		''', unsafe_allow_html = True)

		st.markdown('# ')

		st.dataframe(GUI_data.data)

		st.markdown('# ')
		st.markdown('# ')


		if train_or_upload == 'Train New' :

			inputs_cont = st.container()

			with inputs_cont :

				test_col_3, dummy_col_1, test_col_4, dummy_col_2, test_col_5 = st.columns([1, 0.2, 1, 0.2, 1])

				x_labels = test_col_3.multiselect(label = 'Select **X-Labels** :', options = list(GUI_data.data.columns), default = None, help = 'If None is selected, then all columns are used')
				if x_labels == [] : x_labels = None

				y_label = test_col_4.selectbox(label = 'Select **Y-Label** :', options = list(GUI_data.data.columns), index = len(list(GUI_data.data.columns))-1)
				test_size = (test_col_5.slider(label = 'Select **Test Size** (% of Total Data) :', min_value = 20, max_value = 80, value = 20, step = 10)) / 100.

				st.markdown('# ')

				st.markdown('<small> Select the **Number of Nodes** for each **Hidden Layer** : </small>', unsafe_allow_html = True)

				test_col_6, dummy_col_1, test_col_7 = st.columns([1, 0.2, 1])

				layer_1 = test_col_6.slider(label = '*Layer 1*', min_value = 32, max_value = 4096, value = 64, help = 'Cannot be `0` since atleast 1 hidden layer is required')
				layer_2 = test_col_7.slider(label = '*Layer 2*', min_value = 0, max_value = 4096, value = 64, help = 'If `0` is passed then hidden layer is not created')
				layer_3 = test_col_6.slider(label = '*Layer 3*', min_value = 0, max_value = 4096, value = 0, help = 'If `0` is passed then hidden layer is not created')
				layer_4 = test_col_7.slider(label = '*Layer 4*', min_value = 0, max_value = 4096, value = 0, help = 'If `0` is passed then hidden layer is not created')
				layer_5 = test_col_6.slider(label = '*Layer 5*', min_value = 0, max_value = 4096, value = 0, help = 'If `0` is passed then hidden layer is not created')

				hidden_layers = [layer_1, layer_2, layer_3, layer_4, layer_5]

				while 0 in hidden_layers : hidden_layers.remove(0)

				st.markdown('# ')

				test_col_8, dummy_col_1, test_col_9, dummy_col_2, test_col_10 = st.columns([1, 0.2, 1, 0.2, 1])

				training_epochs = test_col_8.slider(label = 'Select **Number of Training Epochs** :', min_value = 1, max_value = 200, value = 10)
				activation_func = test_col_9.selectbox(label = 'Select **Activation Function** :', options = ['elu', 'relu', 'linear', 'sigmoid', 'hard_sigmoid', 'softmax', 'softplus', 'tanh', 'exponential', 'gelu', 'selu', 'softsign', 'swish'], index = 1)
				test_col_10.markdown('<small> Checkbox for **Validation Split** : </small>', unsafe_allow_html = True)
				validation_split = test_col_10.checkbox(label = 'vs', value = True, label_visibility = 'collapsed')


				data_obj = Classification_Data(data = GUI_data.data, 
											test_size = test_size,
											x_labels = x_labels,
											y_label = y_label,
											hidden_layers = hidden_layers,
											training_epochs = training_epochs,
											activation_func = activation_func,
											validation_split = validation_split)

				classifier = NN_Classification(data_obj)

				def save_keras_model(data_obj) :
					dir = './keras_model_nn_classification'
					filename = './keras_model_nn_classification'
					data_obj.model.save(filename)
					shutil.make_archive(filename, 'zip', dir)

				st.markdown('# ')

				save_keras_model(data_obj)

				st.warning('Trained Model Saved as .zip File')

				st.markdown('# ')

				st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)
				st.markdown(data_obj.result_string, unsafe_allow_html = True)

				st.markdown('# ')

				res_col_1, dummy_col_1, res_col_2 = st.columns([1, 0.1, 1])

				res_col_1.pyplot(data_obj.confusion_matrix_train)
				res_col_2.pyplot(data_obj.confusion_matrix_test)

				if validation_split :
					st.markdown('# ')
					res_col_3, dummy_col_1, res_col_4 = st.columns([1, 0.1, 1])
					res_col_3.pyplot(data_obj.accuracy_per_epoch)
				else :
					pass


		elif train_or_upload == 'Upload' :

			inputs_cont = st.container()

			with inputs_cont :

				test_col_3, dummy_col_1, test_col_4, dummy_col_2, test_col_5 = st.columns([1, 0.2, 1, 0.2, 1])

				x_labels = test_col_3.multiselect(label = 'Select **X-Labels** :', options = list(GUI_data.data.columns), default = None, help = 'If None is selected, then all columns are used')
				if x_labels == [] : x_labels = None

				y_label = test_col_4.selectbox(label = 'Select **Y-Label** :', options = list(GUI_data.data.columns), index = len(list(GUI_data.data.columns))-1)
				test_size = (test_col_5.slider(label = 'Select **Test Size** (% of Total Data) :', min_value = 20, max_value = 99, value = 99, step = 1)) / 100.

				st.markdown('# ')

				zip_name = (st.file_uploader(label = 'Upload a pre-trained **Keras Model** as a **.zip file** :', type = ['zip']))

				if zip_name is not None :

					zip_name = zip_name.name
					dir = './keras_model_nn_classification'

					with ZipFile(zip_name, 'r') as zip:
						zip.extractall(path=dir)

					model = keras.models.load_model(dir)

					data_obj = Classification_Data(data = GUI_data.data, 
												test_size = test_size,
												x_labels = x_labels,
												y_label = y_label,
												model = model)

					classifier = NN_Classification(data_obj)


					st.markdown('# ')

					st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)
					st.write(data_obj.result_string)

					st.markdown('# ')

					res_col_1, dummy_col_1, res_col_2 = st.columns([1, 0.2, 1])

					res_col_1.pyplot(data_obj.confusion_matrix_train)
					res_col_2.pyplot(data_obj.confusion_matrix_test)

				else : 
					pass




if method == 'Random Forest' :

	with method_cont :
		st.markdown("<div style ='text-align: justify;'><br><b>Random Forest</b> is a supervised ensemble learning method for classification and regression that operates by constructing a multitude of decision trees at training time. For classification tasks, the decision of the majority of the trees is chosen by the random forest as the final decision. For regression tasks, the mean or average prediction of the individual trees is returned. </div>", unsafe_allow_html = True)

		st.markdown('# ')

		st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

		st.markdown('''

		- **X-Label(s)** : columns used as input (if None then all columns will be used)
		- **Y-Label** : column used as output/classification category (by default, the last column will be used)
		- **Test Size** : share of data used for testing
		- **Trees** : number of trees in the forest

		''', unsafe_allow_html = True)

		st.markdown('# ')

		st.dataframe(GUI_data.data)

		st.markdown('# ')
		st.markdown('# ')


		if train_or_upload == 'Train New' :

			inputs_cont = st.container()

			with inputs_cont :

				test_col_3, dummy_col_1, test_col_4 = st.columns([1, 0.2, 1])
				test_col_5, dummy_col_2, test_col_6 = st.columns([1, 0.2, 1])

				x_labels = test_col_3.multiselect(label = 'Select **X-Labels** :', options = list(GUI_data.data.columns), default = None, help = 'If None is selected, then all columns are used')
				if x_labels == [] : x_labels = None

				y_label = test_col_4.selectbox(label = 'Select **Y-Label** :', options = list(GUI_data.data.columns), index = len(list(GUI_data.data.columns))-1)
				test_size = (test_col_5.slider(label = 'Select **Test Size** (% of Total Data) :', min_value = 20, max_value = 80, value = 20, step = 10)) / 100.
				trees = test_col_6.slider(label = 'Select **Number of Trees** :', min_value = 1, max_value = 10000, value = 100, step = 1, help = 'Higher number leads to higher training times')

				st.markdown('# ')

				data_obj = Classification_Data(data = GUI_data.data, 
											test_size = test_size,
											x_labels = x_labels,
											y_label = y_label,
											trees = trees)

				classifier = RF_Classification(data_obj)


				filename = 'model_rf_classification.sav'

				pickle.dump(data_obj.model, open(filename, 'wb'))

				st.warning('Trained Model Saved as .sav File')

				st.markdown('# ')

				st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)

				st.write(data_obj.result_string)

				st.markdown('# ')

				res_col_1, dummy_col_1, res_col_2 = st.columns([1, 0.1, 1])

				res_col_1.pyplot(data_obj.confusion_matrix_train)
				res_col_2.pyplot(data_obj.confusion_matrix_test)

				res_col_3, dummy_col_1, res_col_4 = st.columns([1, 0.1, 1])
				res_col_3.pyplot(data_obj.feature_importance)



		elif train_or_upload == 'Upload' :

			inputs_cont = st.container()

			with inputs_cont :

				test_col_3, dummy_col_1, test_col_4, dummy_col_2, test_col_5 = st.columns([1, 0.2, 1, 0.2, 1])

				x_labels = test_col_3.multiselect(label = 'Select **X-Labels** :', options = list(GUI_data.data.columns), default = None, help = 'If None is selected, then all columns are used')
				if x_labels == [] : x_labels = None

				y_label = test_col_4.selectbox(label = 'Select **Y-Label** :', options = list(GUI_data.data.columns), index = len(list(GUI_data.data.columns))-1)
				test_size = (test_col_5.slider(label = 'Select **Test Size** (% of Total Data) :', min_value = 20, max_value = 99, value = 99, step = 1)) / 100.

				st.markdown('# ')

				model_name = (st.file_uploader(label = 'Upload a **Pre-trained Model** as a **.sav file** :', type = ['sav']))

				if model_name is not None :

					model_name = model_name.name

					model = pickle.load(open(model_name, 'rb'))

					data_obj = Classification_Data(data = GUI_data.data, 
												test_size = test_size,
												x_labels = x_labels,
												y_label = y_label,
												model = model)

					classifier = RF_Classification(data_obj)


					st.markdown('# ')

					st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)
					st.write(data_obj.result_string)

					st.markdown('# ')

					res_col_1, dummy_col_1, res_col_2 = st.columns([1, 0.1, 1])

					res_col_1.pyplot(data_obj.confusion_matrix_train)
					res_col_2.pyplot(data_obj.confusion_matrix_test)

					st.markdown('# ')

					res_col_3, dummy_col_1, res_col_4 = st.columns([1, 0.1, 1])
					res_col_3.pyplot(data_obj.feature_importance)