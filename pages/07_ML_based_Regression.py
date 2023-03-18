import streamlit as st
import pandas as pd
import numpy as np

# import necessary class definitions from relevant packages
from GUI.GUI_Class import GUI_class
from ML_Regression.Regression import Regression


# Setup for page configuration
st.set_page_config(
	page_title = 'ML based Regression',
	layout = 'wide'
	)


header_cont = st.container()


with header_cont :
	st.markdown("<h2 style = 'text-align : center; color : #0077b6;'> REGRESSION </h2>", unsafe_allow_html = True)

	st.markdown("<h5 style = 'text-align : center; color : #023e8a;'> SVM &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Gradient Boosting &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Decision Tree </h5>", unsafe_allow_html = True)

	st.markdown(':bulb: <small> <i> :orange[Regression models work best with Time-series type datasets.] </i> </small>', unsafe_allow_html = True)

	st.markdown( 'Regression is a supervised machine learning technique that involves predicting a continuous output variable based on one or more input variables. In regression, a model is trained on a dataset of known input-output pairs to learn the relationship between the inputs and the output. Once the model is trained, it can be used to make predictions on new input data where the output is unknown.   ')
    
	st.markdown( 'In general, the goal of regression is to minimize the difference between the predicted output of the model and the actual output of the dataset. This difference is typically measured using a loss function such as mean squared error or mean absolute error. The regression algorithm then adjusts the parameters of the model to minimize this loss function and improve the accuracy of the predictions. ') 
	
	st.markdown( 'Regression is commonly used in a variety of applications such as finance, economics, marketing, and engineering, among others. It is a powerful tool for predicting future trends and outcomes based on historical data and can provide valuable insights for decision-making.' )

# providing the choice to use original dataframe or processed dataframe
working_df_choice = st.selectbox(label = 'df', options = ['Select Dataset to Proceed :', 'Original Dataset', 'Processed Dataset after performing Outlier Recognition, Interpolation and Smoothening', 'Upload a file from drive' ], index = 0, label_visibility = 'collapsed')

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
		
elif working_df_choice == 'Upload a file from drive' :        

		# st.file_uploader		GUI widget to implement upload function
		uploaded_file = st.file_uploader(label = 'Upload Dataset in .csv Format', type = ['csv'])

		# detect delimiters in uploaded datasets
		if uploaded_file is not None :
			df = pd.read_csv('data/energydata_complete.csv', sep = ',')
			GUI_data = GUI_class(df, arg_filename = working_df_choice)
			st.session_state['GUI_data'] = GUI_data
			working_df = st.session_state['GUI_data'].data
			st.dataframe(working_df)
			GUI_data.showInfo()
			st.markdown('# ')


		else :
			df = pd.DataFrame()
			working_df = df
			GUI_data = GUI_class(df)
			st.session_state['GUI_data'] = GUI_data
        
else :
	working_df = None


if working_df is not None : 

	inputs_cont = st.container()

	with inputs_cont :

		# initializing class object
		Regression_object = Regression(working_df)

		st.markdown('# ')

		# user-inputs for generating training and test data
		st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Constructing Training and Testing Data : </h5>", unsafe_allow_html = True)
		st.markdown('#### ')


		test_col_4, dummy_col_2, test_col_5 = st.columns([2, 0.5, 2])

		test_col_4.markdown('Select ***Target Attribute*** from the dataset to ***Predict*** : ', unsafe_allow_html = True)
		label_target = test_col_4.selectbox(label = 'lt', options = list(working_df.columns), index = 1, label_visibility = 'collapsed')
		st.markdown('#### ')

		test_col_5.markdown('Select ***Percentage*** of complete data to be used for ***Testing*** :', unsafe_allow_html = True)
		testsize = test_col_5.slider(label = 'ts', min_value = 20, max_value = 70, value = 20, step = 10, label_visibility = 'collapsed')


		test_col_6, dummy_col_3 ,test_col_7, dummy_col_4, test_col_8, dummy_col_100, test_col_100 = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1])

		test_col_6.markdown('***Delete NULL values*** from the data', unsafe_allow_html = True)
		deleting_na = test_col_6.checkbox(label = 'dn', value = True, label_visibility = 'collapsed')
		st.markdown('# ')


		test_col_7.markdown('***Scaling***', unsafe_allow_html = True)
		scaling = test_col_7.checkbox(label = 's', value = True, label_visibility = 'collapsed')
		st.markdown('# ')


		test_col_8.markdown('***Delete Duplicates***', unsafe_allow_html = True)
		deleting_duplicates = test_col_8.checkbox(label = 'dd', value = True, label_visibility = 'collapsed')

		test_col_100.markdown('***Remove Noise***', unsafe_allow_html = True)
		remove_noise = test_col_100.checkbox(label = 'rn', value = True, label_visibility = 'collapsed')
		tolerence = test_col_100.slider(label = 'Select tolerence value ', min_value = 0.1, max_value = 10.0, value = 2.0, step = 0.1)

		test_col_101, dummy_col_101, test_col_102 = st.columns([1, 0.2, 1])

		rows_to_keep = test_col_101.slider(label = 'Select Rows to Keep', min_value = 0, max_value = len(working_df), value = len(working_df), step = 1)
		

		cols_to_keep = test_col_102.multiselect(label = 'Select Columns to Keep', options = list(working_df.columns), default = list(working_df.columns))


		# calling class method to generate training and testing data
		Regression_object.split_train_test(label_target = label_target, 
				                        tolerence = tolerence,
										rows_to_keep = rows_to_keep,
										testsize = testsize/100., 
										random_state = 1, 
										deleting_na = deleting_na, 
										scaling = scaling, 
										deleting_duplicates = deleting_duplicates,
										remove_noise = remove_noise,
										cols_to_keep = cols_to_keep)


		# selection of method
		choice_text, choice_box = st.columns([1, 2])

		choice_text.markdown('Select Regression Model :')
		method = choice_box.selectbox(label = '', label_visibility = 'collapsed', options = [ 'Decision Tree', 'Gradient Boosting Regression','Support Vector Machine Regression'], index = 0)


		if method == 'Support Vector Machine Regression' :
            
			st.markdown( ' SVM regression is a supervised machine learning algorithm used for predicting a continuous output variable. The SVM regression algorithm tries to minimize the sum of the residuals while keeping the magnitude of the residuals as small as possible. This is done by adding a penalty term to the objective function that controls the magnitude of the residuals.  ')

			test_col_9, dummy_col_5, test_col_10, dummy_col_6, test_col_11 = st.columns([1, 0.2, 1, 0.2, 1])
			test_col_103, dummy_col_7, test_col_12 = st.columns([1, 0.2, 1])

			kernel = test_col_9.selectbox(label = 'Select kernel', options = ['linear', 'poly', 'rbf', 'sigmoid'], index = 2)
			degree = test_col_10.slider(label = 'Select degree', min_value = 0, max_value =10, value = 3)
			svmNumber = test_col_11.slider(label = 'Select svmNumber', min_value = 0, max_value =500, value = 100, step = 10)

			maxIterations = test_col_12.slider(label = 'Select maxIterations', min_value = -1, max_value =15000, value = -1)
			epsilon = test_col_103.slider(label = 'Select epsilon', min_value = 0.1, max_value =1., value = 0.1, step = 0.1)


			# checkbox to begin training
			test_col_13, dummy_col_8, test_col_14 = st.columns([3, 0.2, 1.5])

			test_col_13.markdown(':bulb: <small> <i> :orange[Click the checkbox to build the regression model. This process can take a few minutes.<br>While changing the model parameters, uncheck the box, change the parameters, then check the box again.] </i> </small>', unsafe_allow_html = True)

			build_model = test_col_14.checkbox(label = 'Build Regression Model')

			if build_model :
				Regression_object.build_regression(regression_name = method,
											kernel = kernel, 
											degree = degree, 
											svmNumber = svmNumber, 
											epsilon = epsilon,
											maxIterations = maxIterations)


		elif method == 'Gradient Boosting Regression' :
			st.markdown( 'Gradient Boosting Regression is a powerful algorithm that is widely used for regression tasks in a variety of domains, including finance, healthcare, and marketing. It has several advantages over other regression algorithms, such as its ability to handle high-dimensional data and its robustness to outliers. However, gradient boosting regression can be sensitive to overfitting, especially if the hyperparameters are not carefully tuned. ')
			test_col_9, dummy_col_5, test_col_10, dummy_col_6, test_col_11 = st.columns([1, 0.2, 1, 0.2, 1])
			test_col_103, dummy_col_7, test_col_12 = st.columns([1, 0.2, 1])

			alpha = test_col_9.slider(label = 'Select alpha', min_value = 0.1, max_value = 1.0,value = 0.1, step = 0.1)
			max_depth = test_col_10.slider(label = 'Select max_depth', min_value = 10, max_value = 1000, value = 10, step = 5)
			min_samples_leaf = test_col_11.slider(label = 'Select min_samples_leaf', min_value = 0, max_value = 100, value = 10, step = 5)

			n_estimators = test_col_12.slider(label = 'Select n_estimators', min_value = 100, max_value = 1000, value = 200, step = 10)
			learning_rate = test_col_103.slider(label = 'Select learning_rate', min_value = 0.1, max_value =1.0 ,value = 0.2, step = 0.1)


			# checkbox to begin training
			test_col_13, dummy_col_8, test_col_14 = st.columns([3, 0.2, 1.5])

			test_col_13.markdown(':bulb: <small> <i> :orange[Click the checkbox to build the regression model. This process can take a few minutes.<br>While changing the model parameters, uncheck the box, change the parameters, then check the box again.] </i> </small>', unsafe_allow_html = True)

			build_model = test_col_14.checkbox(label = 'Build Regression Model')

			if build_model :
				Regression_object.build_regression(regression_name = method,
											alpha = alpha, 
											max_depth = max_depth, 
											min_samples_leaf = min_samples_leaf, 
											n_estimators = n_estimators,
											learning_rate = learning_rate)


		elif method == 'Decision Tree' :
			st.markdown(' Decision Tree Regression is a machine learning algorithm that creates a tree-like model to predict a continuous output variable. This algorithm divides the data into smaller subsets based on the values of the input features, and then fits a simple model to each subset. The splits are chosen based on the feature that maximizes the reduction in the variance of the output variable. The algorithm tries to minimize the sum of squared errors by selecting the feature that splits the data the most effectively.')	
	        
			test_col_10, dummy_col_6, test_col_11 = st.columns([1, 0.2, 1])
			test_col_103, dummy_col_7, test_col_12 = st.columns([1, 0.2, 1])

			max_depth = test_col_10.slider(label = 'Select max_depth', min_value = 10, max_value =1000, value = 200, step = 10)
			min_samples_leaf = test_col_11.slider(label = 'Select min_samples_leaf', min_value = 1, max_value =10,value = 1, step = 1)

			min_samples_split = test_col_12.slider(label = 'Select min_samples_split', min_value = 2, max_value =10,value = 2, step = 1)
			max_leaf_nodes = test_col_103.slider(label = 'Select max_leaf_nodes', min_value = 2, max_value = 2000,value = 1500, step = 100)


			# checkbox to begin training
			test_col_13, dummy_col_8, test_col_14 = st.columns([3, 0.2, 1.5])

			test_col_13.markdown(':bulb: <small> <i> :orange[Click the checkbox to build the regression model. This process can take a few minutes.<br>While changing the model parameters, uncheck the box, change the parameters, then check the box again.] </i> </small>', unsafe_allow_html = True)

			build_model = test_col_14.checkbox(label = 'Build Regression Model')
			
			if build_model :
				Regression_object.build_regression(regression_name = method,
											max_depth = max_depth, 
											min_samples_leaf = min_samples_leaf, 
											min_samples_split = min_samples_split,
											max_leaf_nodes = max_leaf_nodes)


	if build_model :
		result_cont = st.container()

		with result_cont:
			
			st.markdown('# ')
			st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)

			Regression_object.plot_test_data()
			Regression_object.plot_train_data()

			st.pyplot(Regression_object.fig_test)
			# Printing r2 error
			RegressionName = "".join(str(item) for item in Regression_object.regression_name)
			r2_test_string = ", ".join(str(item) for item in Regression_object.r2_test)
			st.markdown("<h7 style = 'text-align : left;'> R2 error for Test data  with "  + RegressionName + "  is =   " + r2_test_string + " </h7>"  ,unsafe_allow_html = True)
			st.markdown('#### ')

			st.pyplot(Regression_object.fig_train)
			# Printing r2 error
			r2_train_string = ", ".join(str(item) for item in Regression_object.r2_train)
			st.markdown("<h7 style = 'text-align : left'> R2 error for Training data  with "  + RegressionName + "  is =   " + r2_train_string + " </h7>"  ,unsafe_allow_html = True)
			st.markdown('#### ')

			st.markdown('# ')
			st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Test The Model  </h5>", unsafe_allow_html = True)
			
			InputFrame = ", ".join(str(item) for item in Regression_object.cols_to_keep)
			st.markdown("<h7 style = 'text-align : left; color : #0096c7; '> Data should be in following frame </h7>"  ,unsafe_allow_html = True)
			st.markdown("<h11 style = 'text-align : left; '>  "  + InputFrame + " </h11>"  ,unsafe_allow_html = True)


			test_file = st.file_uploader(label = 'Upload Test Dataset in .csv Format', type = ['csv'])

			if test_file is not None :
				test_data = pd.read_csv(test_file, sep=';|,', engine='python')
				st.write(test_data)
				Regression_object.regression_function(test_data)

			else :
				test_data = pd.DataFrame()
				Predicted_result1 = 0
				predicted_result = Predicted_result1

				

			#Predicted_result1 =  ", ".join(str(item) for item in Regression_object.predicted_result)
			Predicted_result1 = ", ".join(str(item) for item in np.atleast_1d(Regression_object.predicted_result))
			st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Prediction according to provided row  =  "  + Predicted_result1 + " </h5>"  ,unsafe_allow_html = True)
			st.markdown('#### ')
	