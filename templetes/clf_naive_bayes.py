import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("{filename}")
X = df.drop("{target}", axis=1, inplace = False)
y = df["{target}"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state = 101)

model = GaussianNB()
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