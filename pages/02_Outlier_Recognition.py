import streamlit as st
import matplotlib.pyplot as plt

# import necessary class definitions from relevant packages
from GUI.GUI_Class import GUI_class
from Data_Manipulation.Outlier import Outliers_Recognization


# Setup for page configuration
st.set_page_config(
	page_title = 'Outlier Recognition',
	layout = 'wide'
	)


# loading cached object from the previous page
GUI_data = st.session_state['GUI_data']


header_cont = st.container()
inputs_cont = st.container()
result_cont = st.container()


with header_cont :
	st.markdown("<h2 style = 'text-align : center; color : #0077b6;'> OUTLIER RECOGNITION  </h2>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <b>Outlier detection is a process of identifying and removing observations or data points in a dataset that are significantly different from most of the other data points in the dataset.</b> </div>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <br>Outliers can arise due to various reasons, such as measurement errors, data entry errors, natural variations in the data, or even due to the presence of rare events in the data. </div>", unsafe_allow_html = True)

	st.markdown("<div style ='text-align: justify;'> <br>Outliers can have a significant impact on the statistical analysis and modeling of a dataset, as they can distort the estimation of parameters, affect the accuracy of predictive models, and reduce the reliability of the results. Therefore, identifying and handling outliers is a critical step in data analysis and modeling. </div>", unsafe_allow_html = True)

	st.markdown('# ')

	st.markdown(':bulb: <small> <i> :orange[Outlier Recognition works best with Time-series type datasets.] </i> </small>', unsafe_allow_html = True)

	st.markdown('# ')

	# st.dataframe		display dataframe on the interface
	st.dataframe(GUI_data.data)

	st.markdown('# ')
	st.markdown('# ')



with inputs_cont :
	choice_text, choice_box = st.columns([1, 2])

	# selection of method
	choice_text.markdown('Select Method to for Outlier Recognition :')
	method = choice_box.selectbox(label = '', label_visibility = 'collapsed', options = ['Z-Score Method', 'Modified Z-Score Method', 'Quantile Method', 'Isolation Forest Method'], index = 0)


	if method == 'Z-Score Method' :
		st.markdown("<div style ='text-align: justify;'><br>The <b>Z-Score Method</b> is a statistical approach used to identify outliers in a dataset. The z-score measures how many standard deviations a data point is from the mean of the data. A data point that has a z-score greater than a certain threshold (typically 2 or 3) is considered an outlier. </b></div>", unsafe_allow_html = True)

		st.markdown('# ')

		# st.latex		display mathematical expressions formatted as LaTeX
		st.latex(r'''

			z = \frac{x_{i} - \mu_{X}}{\sigma_{X} }

			''')
		
		st.markdown('where $x_{i}$ is the $i_{th}$ data point, $\mu_{X}$ is the mean, and $\sigma_{X}$ is the standard deviation of vector $X$.')

		st.markdown("<div style ='text-align: justify;'><br>A z-score of 2 indicates that the score is two standard deviations above the mean. If you set a threshold of 2 for identifying outliers, any score with a z-score greater than 2 would be considered an outlier. </b></div>", unsafe_allow_html = True)
		st.markdown("<div style ='text-align: justify;'>The z-score method is useful for identifying outliers in datasets where the data is normally distributed. However, it may not be appropriate for datasets with non-normal distributions or datasets with extreme outliers that can skew the mean and standard deviation. </b></div>", unsafe_allow_html = True)

		st.markdown('# ')

		st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

		st.markdown('''

				- **Working Column** : The column/attribute of the data frame, for which the user wants the plot after performing outlier recognition
				- **Threshold** : Z-Score threshold value

				''', unsafe_allow_html = True)

		st.markdown('# ')

		date_column = st.selectbox(label = 'Select Attribute from Dataset containing Date/Time :', options = list(GUI_data.data.columns), index = 0)

		test_col_1, dummy_col_1, test_col_2 = st.columns([1, 0.2, 1])

		WorkingColumnOptions = list(GUI_data.data.columns)
		WorkingColumnOptions.remove(date_column)

		Working_Column = test_col_1.selectbox(label = 'Select **Working Column** :', options = WorkingColumnOptions, index = 0, help = 'This is only used to display the plot. The selected method with the selected parameters work on complete dataframe.')
		
		# st.slider		slider widget
		threshold = test_col_2.slider(label = 'Select **Threshold** value :', min_value = 1, max_value = 10, value = 3)

		# initializing class (Outliers_Recognization) object using loaded object (GUI_data) from the cache
		Outlier_object = Outliers_Recognization(df = GUI_data.data,
												date_column = date_column)

		# calling relevant method for necessary computations
		Outlier_object.Z_score(threshold = threshold)

		# calling relevant method to generate resultant figures
		Outlier_object.plot_results(Working_Column = Working_Column)



	elif method == 'Modified Z-Score Method' :
		st.markdown("<div style ='text-align: justify;'><br>The <b>Modified Z-Score</b> is a statistical measure used to identify outliers in a dataset. It is a modification of the standard z-score, which is a measure of how many standard deviations a data point is away from the mean of a distribution. </div>", unsafe_allow_html = True)
		st.markdown("<div style ='text-align: justify;'><br>The modified z-score is calculated by dividing the difference between a data point and the median by the median absolute deviation (MAD). MAD is a robust measure of the spread of the data that is less sensitive to outliers than the standard deviation.  </div>", unsafe_allow_html = True)
		st.markdown("<div style ='text-align: justify;'><br>The formula for calculating the modified z-score is: </div>", unsafe_allow_html = True)

		st.latex(r'''

			mod\; z \; = \; 0.6745 \times \frac{x_{i} - median(X)}{MAD}

			''')
		
		st.markdown('where $x_{i}$ is the $i_{th}$ data point, $median(X)$ is the median of the dataset, and $MAD$ is the median absolute deviation of vector $X$.')

		st.markdown("<div style ='text-align: justify;'><br>A data point is considered an outlier if its modified z-score is greater than a certain threshold value. The threshold value is typically set to 3.5 or 4, although it can vary depending on the specific application. </div>", unsafe_allow_html = True)
		st.markdown("<div style ='text-align: justify;'><br>The advantage of using the modified z-score over the standard z-score is that it is more robust to outliers. The MAD is less sensitive to extreme values than the standard deviation, which means that the modified z-score is less likely to misclassify outliers as non-outliers or vice versa. However, one potential limitation of the modified z-score is that it is less effective for identifying outliers in datasets with a small sample size.   </div>", unsafe_allow_html = True)

		st.markdown('# ')

		st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

		st.markdown('''

				- **Working Column** : The column/attribute of the data frame, for which the user wants the plot after performing outlier recognition
				- **Threshold** : Modified Z-Score threshold value

				''', unsafe_allow_html = True)

		st.markdown('# ')

		date_column = st.selectbox(label = 'Select Attribute from Dataset containing Date/Time :', options = list(GUI_data.data.columns), index = 0)

		test_col_1, dummy_col_1, test_col_2 = st.columns([1, 0.2, 1])

		WorkingColumnOptions = list(GUI_data.data.columns)
		WorkingColumnOptions.remove(date_column)

		Working_Column = test_col_1.selectbox(label = 'Select **Working Column** :', options = WorkingColumnOptions, index = 0, help = 'This is only used to display the plot. The selected method with the selected parameters work on complete dataframe.')
		threshold = test_col_2.slider(label = 'Select **Threshold** value :', min_value = 1, max_value = 10, value = 3)

		Outlier_object = Outliers_Recognization(df = GUI_data.data,
												date_column = date_column)

		Outlier_object.Modified_Z_score(threshold = threshold)

		Outlier_object.plot_results(Working_Column = Working_Column)



	elif method == 'Quantile Method' :
		st.markdown("<div style ='text-align: justify;'><br><b>Quantile Method</b> is a statistical technique that divides a set of data into smaller, equally sized groups or subsets based on their rank or position in the data set. Quantiles are the points in a distribution that divide it into equal proportions. </div>", unsafe_allow_html = True)
		st.markdown("<div style ='text-align: justify;'><br>To use the quantile method to remove outliers, a lower and upper quantile are calculated based on the desired percentage of the data to keep. Any data points outside this range are considered outliers and are removed from the dataset. For example, if the lower quantile is set to 10% and the upper quantile is set to 90%, the resulting dataset will contain only the middle 80% of the original data. </div>", unsafe_allow_html = True)
		st.markdown("<div style ='text-align: justify;'><br>One advantage of the quartile method is that it is less sensitive to extreme values than the range or standard deviation methods, which makes it more suitable for datasets with outliers. However, it can be less effective for detecting outliers in datasets with a small number of observations or with a skewed distribution. </div>", unsafe_allow_html = True)

		st.markdown('# ')

		st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

		st.markdown('''

				- **Working Column** : The column/attribute of the data frame, for which the user wants the plot after performing outlier recognition
				- **Q1** : Lower Quantile value
				- **Q2** : Upper Quantile value 

				''', unsafe_allow_html = True)

		st.markdown('# ')

		date_column = st.selectbox(label = 'Select Attribute from Dataset containing Date/Time :', options = list(GUI_data.data.columns), index = 0)

		test_col_1, dummy_col_1, test_col_2, dummy_col_2, test_col_3 = st.columns([1, 0.2, 1, 0.2, 1])

		WorkingColumnOptions = list(GUI_data.data.columns)
		WorkingColumnOptions.remove(date_column)

		Working_Column = test_col_1.selectbox(label = 'Select **Working Column** :', options = WorkingColumnOptions, index = 0, help = 'This is only used to display the plot. The selected method with the selected parameters work on complete dataframe.')
		Q1 = test_col_2.slider(label = 'Select **Q1** value :', min_value = 0., max_value = 0.49, value = 0.25, step = 0.05)
		Q2 = test_col_3.slider(label = 'Select **Q2** value :', min_value = 0.5, max_value = 1., value = 0.75, step = 0.05)

		Outlier_object = Outliers_Recognization(df = GUI_data.data,
												date_column = date_column)

		Outlier_object.Quantile(Q1 = Q1, Q2 = Q2)

		Outlier_object.plot_results(Working_Column = Working_Column)



	elif method == 'Isolation Forest Method' :
		st.markdown("<div style ='text-align: justify;'><br><b>Isolation Forest</b> is a machine learning algorithm used for outlier detection. It is a tree-based algorithm that isolates outliers by partitioning the dataset into subsets using binary trees. </div>", unsafe_allow_html = True)
		st.markdown("<div style ='text-align: justify;'><br>The function takes in a contamination parameter, which determines the percentage of data points that are expected to be outliers. This parameter is used to set the threshold score for classifying data points as outliers. </div>", unsafe_allow_html = True)
		st.markdown("<div style ='text-align: justify;'><br>The advantages of the Isolation Forest algorithm are its ability to handle large datasets, its computational efficiency, and its ability to handle both high-dimensional and low-dimensional data. It also does not require any assumptions about the distribution of the data and can handle mixed data types. </div>", unsafe_allow_html = True)
		st.markdown("<div style ='text-align: justify;'><br>However, one potential limitation of the Isolation Forest algorithm is that it may struggle to identify outliers in datasets with clusters of anomalies or where the anomalies are located in dense regions of the data. </div>", unsafe_allow_html = True)

		st.markdown('# ')

		st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Parameters : </h5>", unsafe_allow_html = True)

		st.markdown('''

				- **Working Column** : The column/attribute of the data frame, for which the user wants the plot after performing outlier recognition
				- **Contamination** : The amount of contamination of the data set, i.e. the proportion of outliers in the data set

				''', unsafe_allow_html = True)

		st.markdown('# ')

		date_column = st.selectbox(label = 'Select Attribute from Dataset containing Date/Time :', options = list(GUI_data.data.columns), index = 0)

		test_col_1, dummy_col_1, test_col_2 = st.columns([1, 0.2, 1])

		WorkingColumnOptions = list(GUI_data.data.columns)
		WorkingColumnOptions.remove(date_column)

		Working_Column = test_col_1.selectbox(label = 'Select **Working Column** :', options = WorkingColumnOptions, index = 0, help = 'This is only used to display the plot. The selected method with the selected parameters work on complete dataframe.')
		contamination = test_col_2.slider(label = 'Select **Contamination** value :', min_value = 0., max_value = 1., value = 0.05, step = 0.05)

		Outlier_object = Outliers_Recognization(df = GUI_data.data,
												date_column = date_column)

		Outlier_object.Isolation_Forest(contamination = contamination)

		Outlier_object.plot_results(Working_Column = Working_Column)



with result_cont :
	st.markdown('# ')
	st.markdown('# ')

	st.markdown("<h5 style = 'text-align : left; color : #0096c7;'> Results : </h5>", unsafe_allow_html = True)

	st.markdown('# ')
	st.markdown('**Complete Dataframe after performing Outlier Recognition :**', unsafe_allow_html = True)

	st.dataframe(Outlier_object.df_Without_Outliers)

	# st.download_button		download-button widget
	st.download_button(label = 'Download Filtered Data', 
					file_name = 'Filtered_Data.csv',
					data = Outlier_object.df_Without_Outliers.to_csv(), 
					mime = 'text/csv')

	st.markdown('# ')

	# display generated resultant figures
	# st.pyplot		display a matplotlib.pyplot figure
	st.pyplot(Outlier_object.lineplot)

	st.markdown('# ')

	res_col_1, dummy_col_1, res_col_2 = st.columns([1, 0.1, 1])
	st.set_option('deprecation.showPyplotGlobalUse', False)

	res_col_1.pyplot(Outlier_object.boxplot_original)					
	res_col_2.pyplot(Outlier_object.boxplot_filtered)

	# cache-ing relevant information to use in subsequent pages
	st.session_state['df_Without_Outliers'] = Outlier_object.df_Without_Outliers
	st.session_state['date_column'] = Outlier_object.date_column