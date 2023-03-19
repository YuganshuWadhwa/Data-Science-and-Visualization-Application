import streamlit as st
from zipfile import ZipFile
import shutil
import keras
import pickle

# import necessary class definitions from relevant packages
from GUI.GUI_Class import GUI_class
from AI_Regression.RF_Regression import RF_Regression
from AI_Regression.Regression_Data import Regression_Data
from AI_Regression.NN_Regression import NN_Regression


# Setup for page configuration
st.set_page_config(
	page_title = 'AI based Regression',
	layout = 'wide'
	)


header_cont = st.container()


with header_cont :
	st.markdown("<h2 style = 'text-align : center; color : #0077b6;'> REGRESSION USING ARTIFICIAL INTELLIGENCE </h2>", unsafe_allow_html = True)

	st.markdown("<h5 style = 'text-align : center; color : #023e8a;'> Neural Networks &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Random Forest </h5>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <br><b>Regression is a statistical method used to estimate the relationships between a dependent variable and one or more independent variables.</b> </div>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <br> It is used to predict the values of a dependent variable based on the values of one or more independent variables. Regression analysis is widely used in many fields, including economics, finance, engineering, and science. <br><br></div>", unsafe_allow_html = True)

	st.markdown(':bulb: <small> <i> :orange[Regression models work best with Time-series type datasets.] </i> </small>', unsafe_allow_html = True)

	st.markdown('# ')



# providing the choice to use original dataframe or processed dataframe
working_df_choice = st.selectbox(label = 'df', options = ['Select Dataset to Proceed :', 'Original Dataset', 'Processed Dataset after performing Outlier Recognition, Interpolation and Smoothening'], index = 0, label_visibility = 'collapsed')

if working_df_choice == 'Original Dataset' :
	# loading original dataframe from cache
	working_df = st.session_state['GUI_data'].data
	st.dataframe(working_df)
	st.markdown('# ')


elif working_df_choice == 'Processed Dataset after performing Outlier Recognition, Interpolation and Smoothening' :

	try :
		# loading processed dataframe from cache
		working_df = st.session_state['smoothed_df']
		st.dataframe(working_df)
		st.markdown('# ')

	except :
		# exception handling
		st.markdown(':exclamation: <small> <i> :orange[Please generate processed data using previous pages, or select "Original Dataset" option] </i> </small>', unsafe_allow_html = True)
		working_df = None
		
else :
	working_df = None



if working_df is not None :

	test_col_1, dummy_col_1, test_col_2 = st.columns([1, 0.2, 1])

	# selection of method
	method = test_col_1.selectbox(label = '**Select Regression Method** :', options = ['Neural Networks', 'Random Forest'], index = 0)
	train_or_upload = test_col_2.selectbox(label = '**Train New Model** or **Upload Pre-trained Model** :', options = ['Train New', 'Upload'], index = 0)


	method_cont = st.container()


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
			st.markdown('# ')


			if train_or_upload == 'Train New' :

				inputs_cont = st.container()

				with inputs_cont :

					test_col_3, dummy_col_1, test_col_4 = st.columns([1, 0.2, 1])
					test_col_5, dummy_col_2, test_col_6 = st.columns([1, 0.2, 1])

					x_labels = test_col_3.multiselect(label = 'Select **X-Labels** :', options = list(working_df.columns), default = None, help = 'If None is selected, then all columns are used')
					if x_labels == [] : x_labels = None

					y_label = test_col_4.selectbox(label = 'Select **Y-Label** :', options = list(working_df.columns), index = len(list(working_df.columns))-1)
					test_size = (test_col_5.slider(label = 'Select **Test Size** (% of Total Data) :', min_value = 20, max_value = 80, value = 20, step = 10)) / 100.
					trees = test_col_6.slider(label = 'Select **Number of Trees** :', min_value = 1, max_value = 10000, value = 100, step = 1, help = 'Higher number leads to higher training times')


					if 20 < (test_size * len(working_df)) :
						n_values = int(st.slider(label = 'Select Number of Datapoints to Plot :', min_value = 20., max_value = test_size * len(working_df), value = 50., step = 1.))

					else :
						n_values = int(test_size * len(working_df))


					st.markdown('# ')

					# initializing dataclass object with user inputs
					data_obj = Regression_Data(data = working_df, 
												test_size = test_size,
												x_labels = x_labels,
												y_label = y_label,
												trees = trees,
												n_values = n_values)

					regressor = RF_Regression(data_obj)

					# generating and saving trained model
					filename = 'model_rf_regression.sav'

					# auto-saving the model in local system
					pickle.dump(data_obj.model, open(filename, 'wb'))

					st.warning('Trained Model Saved as .sav File')

					st.markdown('# ')

					st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)

					st.write(data_obj.result_string)

					st.markdown('# ')

					st.markdown('# ')

					res_col_1, dummy_col_1, res_col_2 = st.columns([1, 0.1, 1])
					res_col_1.pyplot(data_obj.prediction)
					res_col_2.pyplot(data_obj.prediction_train)

					st.markdown('# ')

					res_col_3, dummy_col_1, res_col_4 = st.columns([1, 0.1, 1])
					res_col_3.pyplot(data_obj.prediction_y_test)
					res_col_4.pyplot(data_obj.prediction_y_train)

					st.markdown('# ')

					res_col_5, dummy_col_1, res_col_6 = st.columns([1, 0.1, 1])
					res_col_5.pyplot(data_obj.feature_importance)



			# functionality to upload pre-trained model
			elif train_or_upload == 'Upload' :

				inputs_cont = st.container()

				with inputs_cont :

					test_col_3, dummy_col_1, test_col_4, dummy_col_2, test_col_5 = st.columns([1, 0.2, 1, 0.2, 1])

					x_labels = test_col_3.multiselect(label = 'Select **X-Labels** :', options = list(working_df.columns), default = None, help = 'If None is selected, then all columns are used')
					if x_labels == [] : x_labels = None

					y_label = test_col_4.selectbox(label = 'Select **Y-Label** :', options = list(working_df.columns), index = len(list(working_df.columns))-1)
					test_size = (test_col_5.slider(label = 'Select **Test Size** (% of Total Data) :', min_value = 20, max_value = 99, value = 99, step = 1)) / 100.

					st.markdown('# ')
					# load pre-trained model
					model_name = (st.file_uploader(label = 'Upload a **Pre-trained Model** as a **.sav file** :', type = ['sav']))

					if model_name is not None :

						model_name = model_name.name

						model = pickle.load(open(model_name, 'rb'))

						data_obj = Regression_Data(data = working_df, 
													test_size = test_size,
													x_labels = x_labels,
													y_label = y_label,
													model = model)

						classifier = RF_Regression(data_obj)


						st.markdown('# ')

						st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)
						# displaying outputs on the interface
						st.write(data_obj.result_string)

						st.markdown('# ')

						res_col_1, dummy_col_1, res_col_2 = st.columns([1, 0.1, 1])
						res_col_1.pyplot(data_obj.prediction)
						res_col_2.pyplot(data_obj.prediction_train)

						st.markdown('# ')

						res_col_3, dummy_col_1, res_col_4 = st.columns([1, 0.1, 1])
						res_col_3.pyplot(data_obj.prediction_y_test)
						res_col_4.pyplot(data_obj.prediction_y_train)

						st.markdown('# ')

						res_col_5, dummy_col_1, res_col_6 = st.columns([1, 0.1, 1])
						res_col_5.pyplot(data_obj.feature_importance)




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
			- **Scale** : signifies whether the data will be scaled or not
				
			''', unsafe_allow_html = True)

			st.markdown('# ')
			st.markdown('# ')


			if train_or_upload == 'Train New' :

				inputs_cont = st.container()

				with inputs_cont :

					test_col_3, dummy_col_1, test_col_4, dummy_col_2, test_col_5 = st.columns([1, 0.2, 1, 0.2, 1])

					x_labels = test_col_3.multiselect(label = 'Select **X-Labels** :', options = list(working_df.columns), default = None, help = 'If None is selected, then all columns are used')
					if x_labels == [] : x_labels = None

					y_label = test_col_4.selectbox(label = 'Select **Y-Label** :', options = list(working_df.columns), index = len(list(working_df.columns))-1)
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
					test_col_10.markdown('<small> Checkbox to **Scale** the data : </small>', unsafe_allow_html = True)
					scale = test_col_10.checkbox(label = 's', value = True, label_visibility = 'collapsed')

					if 20 < (test_size * len(working_df)) :
						n_values = int(st.slider(label = 'Select Number of Datapoints to Plot :', min_value = 20., max_value = test_size * len(working_df), value = 50., step = 1.))

					else :
						n_values = int(test_size * len(working_df))


					data_obj = Regression_Data(data = working_df, 
												test_size = test_size,
												x_labels = x_labels,
												y_label = y_label,
												n_values = n_values,
												hidden_layers = hidden_layers,
												training_epochs = training_epochs,
												activation_func = activation_func,
												scale = scale)

					regressor = NN_Regression(data_obj)

					def save_keras_model(data_obj) :
						'''
						function to generate a .zip file for trained model and save in the local system
						'''
						dir = './keras_model_nn_regression'
						filename = 'keras_model_nn_regression'
						data_obj.model.save(dir)
						shutil.make_archive(filename, 'zip', dir)

					st.markdown('# ')

					save_keras_model(data_obj)

					st.warning('Trained Model Saved as .zip File')

					st.markdown('# ')

					st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)
					st.markdown(data_obj.result_string, unsafe_allow_html = True)

					st.markdown('# ')

					res_col_1, dummy_col_1, res_col_2 = st.columns([1, 0.1, 1])
					res_col_1.pyplot(data_obj.prediction)
					res_col_2.pyplot(data_obj.prediction_train)

					st.markdown('# ')

					res_col_3, dummy_col_1, res_col_4 = st.columns([1, 0.1, 1])
					res_col_3.pyplot(data_obj.prediction_y_test)
					res_col_4.pyplot(data_obj.prediction_y_train)

					if scale :
						st.markdown('# ')
						res_col_5, dummy_col_1, res_col_6 = st.columns([1, 0.1, 1])
						res_col_5.pyplot(data_obj.loss_per_epoch)
					else :
						pass

					

			elif train_or_upload == 'Upload' :

				inputs_cont = st.container()

				with inputs_cont :

					test_col_3, dummy_col_1, test_col_4, dummy_col_2, test_col_5 = st.columns([1, 0.2, 1, 0.2, 1])

					x_labels = test_col_3.multiselect(label = 'Select **X-Labels** :', options = list(working_df.columns), default = None, help = 'If None is selected, then all columns are used')
					if x_labels == [] : x_labels = None

					y_label = test_col_4.selectbox(label = 'Select **Y-Label** :', options = list(working_df.columns), index = len(list(working_df.columns))-1)
					test_size = (test_col_5.slider(label = 'Select **Test Size** (% of Total Data) :', min_value = 20, max_value = 99, value = 99, step = 1)) / 100.

					st.markdown('# ')

					# load pre-trained model
					zip_name = (st.file_uploader(label = 'Upload a pre-trained **Keras Model** as a **.zip file** :', type = ['zip']))

					if zip_name is not None :

						zip_name = zip_name.name
						dir = './keras_model_nn_regression'

						with ZipFile(zip_name, 'r') as zip:
							zip.extractall(path=dir)

						model = keras.models.load_model(dir)

						data_obj = Regression_Data(data = working_df, 
													test_size = test_size,
													x_labels = x_labels,
													y_label = y_label,
													model = model)

						classifier = NN_Regression(data_obj)


						st.markdown('# ')

						st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)
						st.write(data_obj.result_string)

						st.markdown('# ')

						res_col_1, dummy_col_1, res_col_2 = st.columns([1, 0.1, 1])
						res_col_1.pyplot(data_obj.prediction)
						res_col_2.pyplot(data_obj.prediction_train)

						st.markdown('# ')

						res_col_3, dummy_col_1, res_col_4 = st.columns([1, 0.1, 1])
						res_col_3.pyplot(data_obj.prediction_y_test)
						res_col_4.pyplot(data_obj.prediction_y_train)

					else : 
						pass



else : 
	pass