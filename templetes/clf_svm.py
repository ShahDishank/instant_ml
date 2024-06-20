import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

df = pd.read_csv("{filename}")
X = df.drop("{target}", axis=1, inplace = False)
y = df["{target}"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state = 101)

model = SVC(C = {C}, gamma = {gamma}, kernel = "{kernel}", degree = {degree})
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"train accuracy: {{train_score*100:.4f}} %")
print(f"test accuracy: {{test_score*100:.4f}} %")

cr = classification_report(y_test, y_pred)
print(f"Classification Report: \n\n {{cr}}")
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n\n {{cm}}")

def intr_plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=[str(i) for i in range(cm.shape[1])], 
                    y=[str(i) for i in range(cm.shape[0])])
    fig.update_layout(title='Confusion Matrix')
    fig.show()

def intr_plot_class_distribution(y_pred):
    unique_classes, counts = np.unique(y_pred, return_counts=True)
    fig = px.bar(x=unique_classes, y=counts, labels={{'x': 'Class', 'y': 'Number of Instances'}})
    fig.update_layout(title='Class Distribution')
    fig.show()

def intr_plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {{roc_auc:.2f}})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    fig.show()

def intr_plot_precision_recall_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall curve'))
    fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
    fig.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_class_distribution(y_pred):
    unique_classes, counts = np.unique(y_pred, return_counts=True)
    fig = plt.figure(figsize=(10, 7))
    plt.bar(unique_classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Class Distribution')
    plt.xticks(unique_classes)
    plt.show()

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig = plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

interactive = True
n = y_test.nunique()
if interactive:
	intr_plot_confusion_matrix(y_test, y_pred)
	intr_plot_class_distribution(y_pred)
	if n == 2:
		intr_plot_roc_curve(y_test, y_pred)
		intr_plot_precision_recall_curve(y_test, y_pred)

else:
	plot_confusion_matrix(y_test, y_pred)
	plot_class_distribution(y_pred)
	if n == 2:
		plot_roc_curve(y_test, y_pred)
		plot_precision_recall_curve(y_test, y_pred)