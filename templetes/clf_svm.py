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

interactive = True
if interactive:
    intr_plot_confusion_matrix(y_test, y_pred)
    intr_plot_class_distribution(y_pred)
else:
    plot_confusion_matrix(y_test, y_pred)
    plot_class_distribution(y_pred)