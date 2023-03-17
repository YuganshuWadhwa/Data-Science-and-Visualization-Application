import streamlit as st
import numpy as np
import pandas as pd

# import necessary class definitions from relevant packages
from GUI.GUI_Class import GUI_class
from ML_Classification.SVMClassV10 import SVM
from ML_Classification.KN_ClassificationV5 import KN_Classification
from ML_Classification.LogisticRegressionV5 import LogisticRegressionClassifier


# Setup for page configuration
st.set_page_config(
	page_title = 'ML based Classification',
	layout = 'wide'
	)


GUI_data = st.session_state['GUI_data']


header_cont = st.container()
inputs_cont = st.container()
method_cont = st.container()
result_cont = st.container()



with header_cont :
	st.markdown("<h2 style = 'text-align : center; color : #0077b6;'> CLASSIFICATION </h2>", unsafe_allow_html = True)

	st.markdown("<h5 style = 'text-align : center; color : #023e8a;'> Support Vector Machine &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; K-Nearest Neighbours &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Logistic Regression </h5>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <br><b>Classification is a supervised machine learning process of categorizing a given set of input data into classes based on one or more variables.</b> </div>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <br> A classification problem can be performed on structured and unstructured data to accurately predict whether the data will fall into predetermined categories. Classification in machine learning can require two or more categories of a given data set. Therefore, it generates a probability score to assign the data into a specific category, such as spam or not spam, yes or no, disease or no disease, red or green, male or female, etc. <br><br></div>", unsafe_allow_html = True)

	st.markdown(':bulb: <small> <i> :orange[Classification models work best with Classification type datasets.] </i> </small>', unsafe_allow_html = True)

	st.markdown('# ')

	st.dataframe(GUI_data.data)

	st.markdown('# ')
	st.markdown('# ')



with inputs_cont :

	test_col_1, dummy_col_1, test_col_2 = st.columns([1.5, 0.2, 3])

	userIN_selected_column = test_col_1.selectbox(label = 'Select the column containing the **dependent variable** :', options = list(GUI_data.data.columns), index = len(GUI_data.data.columns)-1)
	userIN_from, userIN_to = test_col_2.select_slider(label = 'Select the range of data to use as **independent variable** :', options = np.arange(-1, (len(GUI_data.data.columns)-1), 1), value = (-1, (len(GUI_data.data.columns)-2)), help = 'The default values on the slider indicate that complete dataset (other than the dependent variable) will be used as independent variable.')

	st.markdown('# ')
	st.markdown('# ')

	choice_text, choice_box = st.columns([1, 2])

	choice_text.markdown('Select Classification Method :')
	method = choice_box.selectbox(label = '', label_visibility = 'collapsed', options = ['Support Vector Machine', 'K-Nearest Neighbours', 'Logistic Regression'], index = 0)



if method == 'Support Vector Machine' :

	with method_cont :
			st.markdown("<div style ='text-align: justify;'><br><b>Support Vector Machine (SVM)</b> is a supervised machine learning algorithm used for both classification and regression. Though we say regression problems as well its best suited for classification. The objective of SVM algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points. The dimension of the hyperplane depends upon the number of features. If the number of input features is two, then the hyperplane is just a line. If the number of input features is three, then the hyperplane becomes a 2-D plane. </div>", unsafe_allow_html = True)

			st.markdown('# ')

			st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

			st.markdown('''

			- **Kernel** : used when it is difficult to classify the data with a straight line or plane 
				
			''', unsafe_allow_html = True)

			st.markdown('# ')

			test_col_3, dummy_col_1, test_col_4 = st.columns([1, 0.2, 1])

			input_kernel = test_col_3.selectbox(label = 'Select **kernel** for SVM :', options = ['linear', 'poly', 'rbf', 'sigmoid'], index = 2)
			userIN_max_iter = int(test_col_4.text_input(label = 'Input the **number of iterations** :', value = -1))


			if input_kernel == 'linear' :
				st.markdown('# ')
				userIN_gamma='scale'
				userIN_degree=3
				userIN_coef0=0.0
				pass

			elif input_kernel == 'poly' :
				st.markdown('# ')

				test_col_5, dummy_col_1, test_col_6, dummy_col_2, test_col_7 = st.columns([1, 0.2, 1, 0.2, 1])

				userIN_gamma = test_col_5.text_input(label = 'Input **gamma** value :', value = 'scale', help = 'Possible values for **Gamma** are: scale, auto, or a float value beteen 0 and 5')
				userIN_degree = test_col_6.slider(label = 'Select **degree** :', min_value = 1, max_value = 20, step = 1, value = 3)
				userIN_coef0 = test_col_7.slider(label = 'Select **coef0** value :', min_value = -5., max_value = 5., step = 0.1, value = 0.)

				try :
					userIN_gamma = round(float(userIN_gamma), 1)
				except :
					pass

				test_col_5.markdown('<small> <i> :blue[Gamma value selected : ] </i> %s </small>' % str(userIN_gamma), unsafe_allow_html = True)

			elif input_kernel == 'rbf' :
				st.markdown('# ')

				test_col_5, dummy_col_1, test_col_6, dummy_col_2, test_col_7 = st.columns([1, 0.2, 1, 0.2, 1])

				userIN_gamma = test_col_5.text_input(label = 'Input **gamma** value :', value = 'scale', help = 'Possible values for **Gamma** are: scale, auto, or a float value beteen 0 and 5')
				userIN_degree=3
				userIN_coef0=0.0

				try :
					userIN_gamma = round(float(userIN_gamma), 1)
				except :
					pass

				test_col_5.markdown('<small> <i> :blue[Gamma value selected : ] </i> %s </small>' % str(userIN_gamma), unsafe_allow_html = True)

			elif input_kernel == 'sigmoid' :
				st.markdown('# ')

				test_col_5, dummy_col_1, test_col_6, dummy_col_2, test_col_7 = st.columns([1, 0.2, 1, 0.2, 1])

				userIN_gamma = test_col_5.text_input(label = 'Input **gamma** value :', value = 'scale', help = 'Possible values for **Gamma** are: scale, auto, or a float value beteen 0 and 5')
				userIN_coef0 = test_col_6.slider(label = 'Select **coef0** value :', min_value = -5., max_value = 5., step = 0.1, value = 0.)
				userIN_degree=3

				try :
					userIN_gamma = round(float(userIN_gamma), 1)
				except :
					pass

				test_col_5.markdown('<small> <i> :blue[Gamma value selected : ] </i> %s </small>' % str(userIN_gamma), unsafe_allow_html = True)


			Classification_object = SVM(userIN_data = GUI_data.data,
                    			userIN_selected_column = userIN_selected_column,
                    			input_kernel = input_kernel,
                    			userIN_from = userIN_from,
                    			userIN_to = userIN_to,
                    			userIN_max_iter = userIN_max_iter)

			Classification_object.perform_SVM(userIN_gamma = userIN_gamma, 
                        		userIN_degree = userIN_degree,
                        		userIN_coef0 = userIN_coef0)



elif method == 'K-Nearest Neighbours' :

	with method_cont :
			st.markdown("<div style ='text-align: justify;'><br><b>K-Nearest Neighbors (KNN)</b> algorithm is a data classification method for estimating the likelihood that a data point will become a member of one group or the other, based on what group the data points nearest to it belong to. </div>", unsafe_allow_html = True)

			st.markdown('# ')

			st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

			st.markdown('''

			- **k** or **n_neighbours** : represents number of nearest neighbours used for classification 
				
			''', unsafe_allow_html = True)

			st.markdown('# ')

			k = st.slider(label = 'Select the value of **k** :', min_value = 1, max_value = 20, value = 5)

			Classification_object = KN_Classification(userIN_data = GUI_data.data,
                    			userIN_selected_column = userIN_selected_column,
                    			k = k)

			Classification_object.train_model()




elif method == 'Logistic Regression' :

	with method_cont :
			st.markdown("<div style ='text-align: justify;'><br>A <b>Logistic Regression</b> model predicts a dependent data variable by analysing the relationship between one or more existing independent variables. For example, a logistic regression could be used to predict whether a political candidate will win or lose an election or whether a high school student will be admitted or not to a particular college. These binary outcomes allow straightforward decisions between two alternatives. </div>", unsafe_allow_html = True)

			st.markdown('# ')

			st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

			st.markdown('''

			- **Solver** : algorithm used in the optimization problem
			- **Penalty** : penalty applies different methods of regularisation and helps with overfitting
			- **Random State** : uses a new random number generator seeded by the given integer; using an int will produce the same results across different calls (relevant for 'sag', 'saga' or 'liblinear' solver)
				
			''', unsafe_allow_html = True)

			st.markdown(':bulb: <small> <i> :orange[Tips while selecting the solver] </i> </small>', unsafe_allow_html = True)
			st.markdown('<small> :blue[&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For small datasets, `liblinear` is a good choice, whereas `sag` and `saga` are faster for large ones] </small>', unsafe_allow_html = True)
			st.markdown('<small> :blue[&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For multiclass problems, only `newton-cg`, `sag`, `saga` and `lbfgs` handle multinomial loss] </small>', unsafe_allow_html = True)
			st.markdown('<small> :blue[&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `liblinear` is limited to one-versus-rest schemes] </small>', unsafe_allow_html = True)
			st.markdown('<small> :blue[&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `newton-cholesky` is a good choice for `n_samples >> n_features`, especially with one-hot encoded categorical features with rare categories. Note that it is limited to binary classification and the one-versus-rest reduction for multiclass classification. Be aware that the memory usage of this solver has a quadratic dependency on `n_features` because it explicitly computes the Hessian matrix. ] </small>', unsafe_allow_html = True)

			st.markdown('# ')

			test_col_3, dummy_col_1, test_col_4 = st.columns([1, 0.2, 1])

			input_solver = test_col_3.selectbox(label = 'Select **solver** for logistic regression :', options = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], index = 0)
			userIN_max_iter = int(test_col_4.text_input(label = 'Input the **number of iterations** :', value = 100))

			st.markdown('# ')

			test_col_5, dummy_col_1, test_col_6 = st.columns([1, 0.2, 1])
			

			if input_solver == 'lbfgs' :
				userIN_penalty = test_col_5.selectbox(label = 'Select **penalty** :', options = ['l2', None], index = 0)
				userIN_random_state=None

			elif input_solver == 'liblinear' :
				userIN_penalty = test_col_5.selectbox(label = 'Select **penalty** :', options = ['l1', 'l2'], index = 1)
				userIN_random_state = test_col_6.slider(label = 'Select **random state** value :', min_value = 1, max_value = 150, step = 5)

			elif input_solver == 'newton-cg' :
				userIN_penalty = test_col_5.selectbox(label = 'Select **penalty** :', options = ['l2', None], index = 0)
				userIN_random_state=None

			elif input_solver == 'newton-cholesky' :
				userIN_penalty = test_col_5.selectbox(label = 'Select **penalty** :', options = ['l2', None], index = 0)
				userIN_random_state=None

			elif input_solver == 'sag' :
				userIN_penalty = test_col_5.selectbox(label = 'Select **penalty** :', options = ['l2', None], index = 0)
				userIN_random_state = test_col_6.slider(label = 'Select **random state** value :', min_value = 1, max_value = 150, step = 5)

			elif input_solver == 'saga' :
				userIN_penalty = test_col_5.selectbox(label = 'Select **penalty** :', options = ['elasticnet', 'l1', 'l2', None], index = 2)
				userIN_random_state = test_col_6.slider(label = 'Select **random state** value :', min_value = 1, max_value = 150, step = 5)


			Classification_object = LogisticRegressionClassifier(userIN_data = GUI_data.data,
                    			userIN_selected_column = userIN_selected_column,
                    			input_solver = input_solver,
                    			userIN_from = userIN_from,
                    			userIN_to = userIN_to,
                    			userIN_max_iter = userIN_max_iter)

			Classification_object.perform_LogisticRegression(userIN_penalty = userIN_penalty, 
                        		userIN_random_state = userIN_random_state)




with result_cont :
	st.markdown('# ')
	st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)

	Classification_object.get_results()
	# st.write(Classification_object.strResults)

	Classification_object.get_plot()

	res_col_1, dummy_col_1, res_col_2 = st.columns([1, 0.2, 1])

	res_col_1.pyplot(Classification_object.fig_train)
	res_col_2.pyplot(Classification_object.fig_test)

	st.markdown('# ')
	st.pyplot(Classification_object.bargraph_results)

	st.markdown('# ')
	st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Test Classification Model : </h5>", unsafe_allow_html = True)

	test_file = st.file_uploader(label = 'Upload Test Dataset in .csv Format', type = ['csv'])

	if test_file is not None :
		test_data = pd.read_csv(test_file, sep=';|,', engine='python')
		st.write(test_data)
		Classification_object.get_userinput_prediction(user_testdata = test_data)
		st.write(Classification_object.strPredictedClass)

	else :
		test_data = pd.DataFrame()