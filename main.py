import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Instant ML",
    page_icon="media_files\\icon.png",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:shahdishank24@gmail.com',
        'Report a bug': "mailto:shahdishank24@gmail.com",
        'About': "# Make your model."
    }
)

lt = st.empty()
with lt.container():
	st.title("Instant ML")
	st.write("")

	img_path = "media_files\\home_img.svg"
	with open(img_path, 'r') as f:
		img = f.read()
	st.image(img, width=360)

	st.header("")
	st.markdown("""
	<p style='font-size:18px'>Introducing Instant ML, the revolutionary platform that lets anyone build powerful models in seconds.\
	Upload your data, and Instant ML takes care of the rest - no coding required.</p>
	""",unsafe_allow_html=True)



def get_data(df, target):
	y = df[target]
	X = df.drop(target, axis=1, inplace=False)
	return X,y

def params_clf(model_name):
	params = dict()
	if model_name == "Logistic Regression":
		params["solver"] = st.sidebar.selectbox(
			"solver",
			("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga")
			)
		params["penalty"] = st.sidebar.selectbox(
			"penalty",
			("l2", "l1", "elasticnet")
			)
		params["C"] = st.sidebar.slider("C", 0.01, 1.0, 0.9)
	elif model_name == "KNN":
		params["n_neighbors"] = st.sidebar.slider("n_neighbors", 2, 20, 5)
		params["weights"] = st.sidebar.selectbox(
			"weights",
			("uniform", "distance")
			)
		params["metric"] = st.sidebar.selectbox(
			"metric",
			("minkowski", "euclidean", "manhattan")
			)
	elif model_name == "SVM":
		params["C"] = st.sidebar.slider("C", 0.1, 100.0, 1.0)
		params["gamma"] = st.sidebar.select_slider(
			"gamma",
			options=[0.0001, 0.001, 0.01, 0.1, 1, 10]
			)
		params["kernel"] = st.sidebar.selectbox(
			"kernel",
			("rbf", "linear", "sigmoid", "poly")
			)
		params["degree"] = 3
		if params["kernel"] == "poly":
			params["degree"] = st.sidebar.slider("degree", 2, 6, 3)
	elif model_name == "Naive Bayes":
		# params["var_smoothing"] = np.log(st.sidebar.slider("var_smoothing", -9, 1, -9))
		pass
	elif model_name == "Decision Tree":
		params["max_depth"] = st.sidebar.slider("max_depth", 3, 15, 3)
		params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 3, 20, 3)
		params["min_samples_split"] = st.sidebar.select_slider(
			"min_samples_split",
			options = [8, 10, 12, 14, 16, 18, 20]
			)
		params["criterion"] = st.sidebar.selectbox(
			"criterion",
			("gini", "entropy")
			)
	elif model_name == "Random Forest":
		params["n_estimators"] = st.sidebar.slider("n_estimators", 25, 150, 100)
		params["max_depth"] = st.sidebar.slider("max_depth", 1, 10, 1)
		params["max_features"] = st.sidebar.selectbox(
			"max_features",
			("sqrt", "log2", "None")
			)
		params["max_leaf_nodes"] = st.sidebar.slider("max_leaf_nodes", 3, 9, 3)
	return params


def model_clf(model_name, params):
	model = None
	if model_name == "Logistic Regression":
		model = LogisticRegression(solver = params["solver"], penalty = params["penalty"], C = params["C"])
	elif model_name == "KNN":
		model = KNeighborsClassifier(n_neighbors = params["n_neighbors"], weights = params["weights"], metric = params["metric"])
	elif model_name == "SVM":
		model = SVC(C = params["C"], gamma = params["gamma"], kernel = params["kernel"], degree = params["degree"])
	elif model_name == "Naive Bayes":
		model = GaussianNB()
		st.sidebar.caption("No need to tune the Parameters")
		st.sidebar.write(model.get_params())
	elif model_name == "Decision Tree":
		model = DecisionTreeClassifier(criterion = params["criterion"], max_depth = params["max_depth"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
	elif model_name == "Random Forest":
		model = RandomForestClassifier(n_estimators = params["n_estimators"], max_leaf_nodes = params["max_leaf_nodes"], max_depth = params["max_depth"], max_features = params["max_features"])
	return model

auto = ""

def grid_search_cv_clf(model_name):
	model = None
	if model_name == "Logistic Regression":
		params = [{"solver" : ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], "penalty" : ["l2", "l1", "elasticnet"], "C" : [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]}]
		model = GridSearchCV(LogisticRegression(), params, cv = 5, scoring = 'accuracy')
	elif model_name == "KNN":
		params = [{"n_neighbors" : np.arange(2, 30, 1), "weights" : ['uniform', 'distance'], "metric" : ["minkowski", "euclidean", "manhattan"]}]
		model = GridSearchCV(KNeighborsClassifier(), params, cv = 5, scoring = 'accuracy')
	elif model_name == "SVM":
		params = [{"C" : [0.1, 1, 10, 100], "gamma" : [0.0001, 0.001, 0.01, 0.1, 1, 10], "kernel" : ["rbf", "linear", "sigmoid", "poly"], "degree" : [2, 3, 4, 5, 6]}]
		model = GridSearchCV(SVC(), params, cv = 5, scoring = 'accuracy')
	elif model_name == "Naive Bayes":
		params = [{"var_smoothing" : np.logspace(1, -9, 100)}]
		model = GridSearchCV(GaussianNB(), params, cv = 5, scoring = 'accuracy')
	elif model_name == "Decision Tree":
		params = [{"max_depth" : [3, 6, 9], "min_samples_split" : [8, 12, 16, 20], "min_samples_leaf" : [3, 6, 9, 12, 15], "criterion" : ["gini", "entropy"]}]
		model = GridSearchCV(DecisionTreeClassifier(), params, cv = 5, scoring = 'accuracy')
	elif model_name == "Random Forest":
		params = [{"n_estimators" : [25, 50, 100, 150], "max_depth" : [3, 6, 9], "max_features" : ["sqrt", "log2", "None"], "max_leaf_nodes" : [3, 6, 9]}]
		model = GridSearchCV(RandomForestClassifier(), params, cv = 5, scoring = 'accuracy')
	return model

model_select = ""

def classification():
	global model_select
	model_select = st.sidebar.selectbox(
	'Select a model',
	('Logistic Regression', 'KNN', 'SVM', 'Naive Bayes', 'Decision Tree', 'Random Forest')
	)
	tune_choice = st.sidebar.selectbox(
	'Hyperparameter Tuning',
	('Manually', 'Automatically')
	)
	if tune_choice == "Manually":
		params = params_clf(model_select)
		model = model_clf(model_select, params)
	else:
		model = grid_search_cv_clf(model_select)
		global auto
		auto = "auto"
	return model

def regression():
	global model_select
	model_select = st.sidebar.selectbox(
	'Select a model',
	('Linear Regression', 'KNN', 'Decision Tree', 'Random Forest')
	)

def show_data(df):
	st.subheader(f"Shape of the Dataset: {df.shape}")
	st.caption("Data Summary")
	st.dataframe(df.head(), hide_index=True)
	st.caption("Some Statistics")
	st.table(df.describe())


def stream_data(string):
    for word in string.split(" "):
        yield word + " "
        time.sleep(0.04)


def fetch_code(fname):
	with open(f"templetes\\{fname}.py", "r") as f:
		data = f.read()
	return data

def get_code(algo_type, f_var, params):
	if algo_type == "Classification":
		if model_select == "Logistic Regression":
			data = fetch_code("clf_logistic_reg")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], solver = params["solver"], penalty = params["penalty"], C = params["C"])
		elif model_select == "KNN":
			data = fetch_code("clf_knn")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_neighbors = params["n_neighbors"], weights = params["weights"], metric = params["metric"])
		elif model_select == "SVM":
			data = fetch_code("clf_svm")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], C = params["C"], gamma = params["gamma"], kernel = params["kernel"], degree = params["degree"])
		elif model_select == "Naive Bayes":
			data = fetch_code("clf_naive_bayes")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"])
		elif model_select == "Decision Tree":
			data = fetch_code("clf_decision_tree")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], criterion = params["criterion"], max_depth = params["max_depth"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
		elif model_select == "Random Forest":
			data = fetch_code("clf_random_forest")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_estimators = params["n_estimators"], max_leaf_nodes = params["max_leaf_nodes"], max_depth = params["max_depth"], max_features = params["max_features"])
		return data
	elif algo_type == "Regression":
		pass


def algorithm(df):
	if not df.empty:
		show_data(df)
		cols = ("select", )
		for i, j in enumerate(df.columns):
			cols = cols + (j,)
		target = st.sidebar.selectbox(
			'Select target value',
			cols,
			)
		if target != "select":
			X, y = get_data(df, target)
			if not X.empty:
				tst_size = st.sidebar.slider("Select the test size of the dataset to split", 0.1, 0.9, 0.2)
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tst_size, random_state = 101)
				st.subheader("Shape of...")
				st.write(f"X_train: {X_train.shape}")
				st.write(f"X_test: {X_test.shape}")
				st.write(f"y_train: {y_train.shape}")
				st.write(f"y_test: {y_test.shape}")

				algo_type = st.sidebar.selectbox(
					'Select an algorithm type',
					('Classification', 'Regression')
					)
				if algo_type == "Classification":
					start_time = time.time()
					model = classification()
					model.fit(X_train, y_train)
					end_time = time.time()
					time_taken = end_time - start_time
					y_pred = model.predict(X_test)

					if auto == "auto":
						params = model.best_params_
						st.sidebar.caption("Better Parameters")
						st.sidebar.write(model.best_params_)
						st.sidebar.caption("Average Score")
						st.sidebar.write(model.best_score_)
					else:
						params = model.get_params()

					st.markdown(
					"""
					---
					"""
					)

					st.sidebar.caption("Execution Time (in seconds)")
					st.sidebar.write(time_taken)

					# accuracy = accuracy_score(y_test, y_pred)
					train_score = model.score(X_train, y_train)
					test_score = model.score(X_test, y_test)
					# st.subheader(f"accuracy: {accuracy}")
					st.subheader(f"train accuracy: {train_score}")
					st.subheader(f"test accuracy: {test_score}")
					st.header("\n")
					# st.sidebar.write(list(model.cv_results_.keys()))
					cr = classification_report(y_test, y_pred)
					st.code(f"Classification Report: \n\n {cr}")
					cm = confusion_matrix(y_test, y_pred)
					st.code(f"Confusion Matrix: \n\n {cm}")

					st.subheader("")
					show = st.toggle("**Show Comparisons**", value=True)
					if show:
						count = st.slider("How many rows do you want to see", 1, 30, 5)
						col1, col2 = st.columns(2)
						with col1:
							# st.caption("Actual target values")
							st.dataframe(y_test.head(count), hide_index = True, use_container_width = True, column_config = {target : "Actual Target Values"})
						with col2:
							# st.caption("Predicted target values")
							st.dataframe(y_pred[:count], hide_index = True, use_container_width = True, column_config = {"value" : "Predicted Target Values"})

					st.header("")
					gen = st.toggle("**Generate Code**")
					if gen:
						format_variable = {"filename":filename, "target":target, "tst_size":tst_size}
						data = get_code(algo_type, format_variable, params)
						st.code(data)
						st.download_button(
						    label="Download Code",
						    data=data,
						    file_name=filename.replace('.csv', "") + "_" + model_select.replace(" ", "_") + ".py",
						    mime='text/python',
						    help="Download"
						)
				else:
					regression()


uploaded_file = st.sidebar.file_uploader("Upload the CSV file (separator must be coma)", type=['csv'])
if uploaded_file is not None:
	try:
		df = pd.read_csv(uploaded_file)
		global filename
		filename = uploaded_file.name
		lt.empty()
	except:
		st.sidebar.error("The File is empty or unable to read!")
		df = pd.DataFrame()
	finally:
		algorithm(df)