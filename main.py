import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

st.set_page_config(
    page_title="Instant ML",
    page_icon="media_files/icon.png",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'mailto:shahdishank24@gmail.com',
        'Report a bug': "mailto:shahdishank24@gmail.com",
        'About': "Make your model."
    }
)


lt = st.empty()
with lt.container():

	st.markdown("""
	<h1 style='text-align:center;'>Instant ML</h1>
	""", unsafe_allow_html=True)
	st.write("")

	col1, col2, col3 = st.columns([0.2, 0.5, 0.2])
	with col2:
		img_path = "media_files/home_img.svg"
		with open(img_path, 'r') as f:
			img = f.read()
		st.image(img, use_column_width=True)

	
	st.write("")
	st.write("")
	st.markdown("""
	<p style='font-size:20px; text-align:center'>
	Build Machine Learning models in seconds. Upload your data and <strong style='color:dodgerblue'>Get Started!<strong></p>
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
			("sqrt", "log2", None)
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
		params = [{"n_estimators" : [25, 50, 100, 150], "max_depth" : [3, 6, 9], "max_features" : ["sqrt", "log2", None], "max_leaf_nodes" : [3, 6, 9]}]
		model = GridSearchCV(RandomForestClassifier(), params, cv = 5, scoring = 'accuracy')
	return model


def params_reg(model_name):
	params = dict()
	if model_name == "Linear Regression":
		params["fit_intercept"] = st.sidebar.selectbox("fit_intercept", (True, False))
		params["copy_X"] = st.sidebar.selectbox("copy_X", (True, False))
	elif model_name == "Ridge Regression":
		params["alpha"] = st.sidebar.slider("alpha", 0.0, 10.0, 0.5)
		params["fit_intercept"] = st.sidebar.selectbox("fit_intercept", (True, False))
		params["solver"] = st.sidebar.selectbox("solver", ("auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"))
	elif model_name == "Lasso Regression":
		params["alpha"] = st.sidebar.slider("alpha", 0.0, 10.0, 0.5)
		params["fit_intercept"] = st.sidebar.selectbox("fit_intercept", (True, False))
		params["selection"] = st.sidebar.selectbox("selection", ("cyclic", "random"))
	elif model_name == "Elastic Net":
		params["alpha"] = st.sidebar.slider("alpha", 0.0, 10.0, 0.5)
		params["fit_intercept"] = st.sidebar.selectbox("fit_intercept", (True, False))
		params["l1_ratio"] = st.sidebar.slider("l1_ratio", 0.0, 1.0, 0.5)
	elif model_name == "KNN":
		params["n_neighbors"] = st.sidebar.slider("n_neighbors", 2, 20, 5)
		params["weights"] = st.sidebar.selectbox(
			"weights",
			("uniform", "distance")
			) 
	elif model_name == "SVM":
		params["C"] = st.sidebar.slider("C", 0.1, 100.0, 1.0)
		params["gamma"] = st.sidebar.selectbox(
			"gamma",
			("scale", "auto")
			)
		params["kernel"] = st.sidebar.selectbox(
			"kernel",
			("rbf", "linear", "sigmoid", "poly")
			)
		params["degree"] = 3
		if params["kernel"] == "poly":
			params["degree"] = st.sidebar.slider("degree", 2, 6, 3)
	elif model_name == "Decision Tree":
		params["criterion"] = st.sidebar.selectbox("criterion", ("squared_error", "friedman_mse", "absolute_error", "poisson"))
		params["splitter"] = st.sidebar.selectbox("splitter", ("best", "random"))
		params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 20, 1)
		params["min_samples_split"] = st.sidebar.select_slider(
			"min_samples_split",
			options = [2, 8, 10, 12, 14, 16, 18, 20]
			)
	elif model_name == "Random Forest":
		params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 200, 100)
		params["max_features"] = st.sidebar.selectbox(
			"max_features",
			("sqrt", "log2", None)
			)
		params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 20, 1)
		params["min_samples_split"] = st.sidebar.select_slider(
			"min_samples_split",
			options = [2, 8, 10, 12, 14, 16, 18, 20]
			)
	return params


def model_reg(model_name, params):
	model = None
	if model_name == "Linear Regression":
		model = LinearRegression(fit_intercept = params["fit_intercept"], copy_X = params["copy_X"])
	elif model_name == "Ridge Regression":
		model = Ridge(alpha = params["alpha"], fit_intercept = params["fit_intercept"], solver = params["solver"])
	elif model_name == "Lasso Regression":
		model = Lasso(alpha = params["alpha"], fit_intercept = params["fit_intercept"], selection = params["selection"])
	elif model_name == "Elastic Net":
		model = ElasticNet(alpha = params["alpha"], fit_intercept = params["fit_intercept"], l1_ratio = params["l1_ratio"])
	elif model_name == "KNN":
		model = KNeighborsRegressor(n_neighbors = params["n_neighbors"], weights = params["weights"])
	elif model_name == "SVM":
		model = SVR(C = params["C"], gamma = params["gamma"], kernel = params["kernel"], degree = params["degree"])
	elif model_name == "Decision Tree":
		model = DecisionTreeRegressor(criterion = params["criterion"], splitter = params["splitter"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
	elif model_name == "Random Forest":
		model = RandomForestRegressor(n_estimators = params["n_estimators"], max_features = params["max_features"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
	return model


def grid_search_cv_reg(model_name):
	model = None
	if model_name == "Linear Regression":
		params = [{"fit_intercept" : [True, False], "copy_X" : [True, False]}]
		model = GridSearchCV(LinearRegression(), params, cv = 5)
	elif model_name == "Ridge Regression":
		params = [{"alpha" : [0, 0.5, 1, 1.5, 2], "fit_intercept" : [True, False], "solver" : ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}]
		model = GridSearchCV(Ridge(), params, cv = 5)
	elif model_name == "Lasso Regression":
		params = [{"alpha" : [0, 0.5, 1, 1.5, 2], "fit_intercept" : [True, False], "selection" : ["cyclic", "random"]}]
		model = GridSearchCV(Lasso(), params, cv = 5)
	elif model_name == "Elastic Net":
		params = [{"alpha" : [0, 0.5, 1, 1.5, 2], "fit_intercept" : [True, False], "l1_ratio" : [0, 0.2, 0.5, 0.8, 1]}]
		model = GridSearchCV(ElasticNet(), params, cv = 5)
	elif model_name == "KNN":
		params = [{"n_neighbors" : np.arange(2, 20, 1), "weights" : ["uniform", "distance"]}]
		model = GridSearchCV(KNeighborsRegressor(), params, cv = 5)
	elif model_name == "SVM":
		params = [{"C" : [0.1, 1, 10, 100], "gamma" : ["scale", "auto"], "kernel" : ["rbf", "linear", "sigmoid", "poly"], "degree" : [2, 3, 4, 5, 6]}]
		model = GridSearchCV(SVR(), params, cv = 5)
	elif model_name == "Decision Tree":
		params = [{"splitter" : ["best", "random"], "min_samples_split" : [2, 5, 8, 12, 16, 20], "min_samples_leaf" : [1, 3, 6, 9, 12, 15], "criterion" : ["squared_error", "friedman_mse", "absolute_error", "poisson"]}]
		model = GridSearchCV(DecisionTreeRegressor(), params, cv = 5)
	elif model_name == "Random Forest":
		params = [{"n_estimators" : [50, 100, 150, 200], "max_features" : ["sqrt", "log2", None], "min_samples_split" : [2, 5, 8, 12, 16, 20], "min_samples_leaf" : [1, 3, 6, 9, 12, 15]}]
		model = GridSearchCV(RandomForestRegressor(), params, cv = 5)
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
	('Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net', 'KNN', 'SVM', 'Decision Tree', 'Random Forest')
	)
	tune_choice = st.sidebar.selectbox(
	'Hyperparameter Tuning',
	('Manually', 'Automatically')
	)
	if tune_choice == "Manually":
		params = params_reg(model_select)
		model = model_reg(model_select, params)
	else:
		model = grid_search_cv_reg(model_select)
		global auto
		auto = "auto"
	return model

def show_data(df):
	st.subheader(f"Shape of the Dataset: {df.shape}")
	st.caption("Data Summary")
	st.dataframe(df.head(), hide_index=True)
	st.caption("Some Statistics")
	st.table(df.describe())


def fetch_code(fname):
	with open(f"templetes/{fname}.py", "r") as f:
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
			if params["max_features"] is None:
				data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_estimators = params["n_estimators"], max_leaf_nodes = params["max_leaf_nodes"], max_depth = params["max_depth"], max_features = params["max_features"])
			else:
				max_f = "\""+params["max_features"]+"\""
				data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_estimators = params["n_estimators"], max_leaf_nodes = params["max_leaf_nodes"], max_depth = params["max_depth"], max_features = max_f)
	elif algo_type == "Regression":
		if model_select == "Linear Regression":
			data = fetch_code("reg_linear")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], fit_intercept = params["fit_intercept"], copy_X = params["copy_X"])
		elif model_select == "Ridge Regression":
			data = fetch_code("reg_ridge")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], alpha = params["alpha"], fit_intercept = params["fit_intercept"], solver = params["solver"])
		elif model_select == "Lasso Regression":
			data = fetch_code("reg_lasso")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], alpha = params["alpha"], fit_intercept = params["fit_intercept"], selection = params["selection"])
		elif model_select == "Elastic Net":
			data = fetch_code("reg_elastic_net")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], alpha = params["alpha"], fit_intercept = params["fit_intercept"], l1_ratio = params["l1_ratio"])
		elif model_select == "KNN":
			data = fetch_code("reg_knn")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_neighbors = params["n_neighbors"], weights = params["weights"])
		elif model_select == "SVM":
			data = fetch_code("reg_svm")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], C = params["C"], gamma = params["gamma"], kernel = params["kernel"], degree = params["degree"])
		elif model_select == "Decision Tree":
				data = fetch_code("reg_decision_tree")
				data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], criterion = params["criterion"], splitter = params["splitter"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
		elif model_select == "Random Forest":
				data = fetch_code("reg_random_forest")
				if params["max_features"] is None:
					data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_estimators = params["n_estimators"], max_features = params["max_features"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
				else:
					max_f = "\""+params["max_features"]+"\""
					data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_estimators = params["n_estimators"], max_features = max_f, min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
	return data

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
						st.sidebar.write(model.best_score_*100)
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
					st.subheader(f"train accuracy: {train_score*100:.4f} %")
					st.subheader(f"test accuracy: {test_score*100:.4f} %")
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
					start_time = time.time()
					model = regression()
					model.fit(X_train, y_train)
					end_time = time.time()
					time_taken = end_time - start_time
					y_pred = model.predict(X_test)

					if auto == "auto":
						params = model.best_params_
						st.sidebar.caption("Better Parameters")
						st.sidebar.write(model.best_params_)
						st.sidebar.caption("Average Score")
						st.sidebar.write(model.best_score_*100)
					else:
						params = model.get_params()


					st.markdown(
					"""
					---
					"""
					)

					st.sidebar.caption("Execution Time (in seconds)")
					st.sidebar.write(time_taken)
					
					train_score = model.score(X_train, y_train)
					test_score = model.score(X_test, y_test)
					mae = mean_absolute_error(y_test, y_pred)
					mse = mean_squared_error(y_test, y_pred)
					rmse = root_mean_squared_error(y_test, y_pred)
					r2 = r2_score(y_test, y_pred)

					st.subheader(f"train score: {train_score:.4f}")
					st.subheader(f"test score: {test_score:.4f}")
					st.subheader(f"Mean Absolute Error: {mae:.4f}")
					st.subheader(f"Mean Squared Error: {mse:.4f}")
					st.subheader(f"Root Mean Squared Error: {rmse:.4f}")
					st.subheader(f"R2 Score: {r2:.4f}")

					st.subheader("")
					show = st.toggle("**Show Comparisons**", value=True)
					if show:
						count = st.slider("How many rows do you want to see", 1, 30, 5)
						col1, col2 = st.columns(2)
						with col1:
							st.dataframe(y_test.head(count), hide_index = True, use_container_width = True, column_config = {target : "Actual Target Values"})
						with col2:
							st.dataframe(y_pred[:count], hide_index = True, use_container_width = True, column_config = {"value" : "Predicted Target Values"})

					st.subheader("")

					col = len(X_test.columns)
					col_select = st.slider("Select column for graph", 1, col, 1)
					fig = plt.figure()
					plt.scatter(X_test.iloc[:,col_select-1], y_test, color='b')
					plt.plot(X_test.iloc[:,col_select-1], y_pred, color ='g')
					plt.xlabel(f"X_test column {col_select}")
					plt.ylabel(f"y_test & y_pred")
					st.pyplot(fig)
					st.subheader("")
					fig2 = plt.figure()
					plt.scatter(y_test, y_pred, color = 'b')
					plt.xlabel("y_test")
					plt.ylabel("y_pred")
					st.pyplot(fig2)

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