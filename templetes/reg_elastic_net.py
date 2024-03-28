import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

df = pd.read_csv("{filename}")
X = df.drop("{target}", axis=1, inplace = False)
y = df["{target}"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state = 101)

model = ElasticNet(alpha = {alpha}, fit_intercept = {fit_intercept}, l1_ratio = {l1_ratio})
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"train score: {{train_score:.4f}}")
print(f"test score: {{test_score:.4f}}")
print(f"Mean Absolute Error: {{mae:.4f}}")
print(f"Mean Squared Error: {{mse:.4f}}")
print(f"Root Mean Squared Error: {{rmse:.4f}}")
print(f"R2 Score: {{r2:.4f}}")

column = 1
fig = plt.figure()
plt.scatter(X_test.iloc[:,column-1], y_test, color='b')
plt.plot(X_test.iloc[:,column-1], y_pred, color ='g')
plt.xlabel(f"X_test column {{column}}")
plt.ylabel(f"y_test & y_pred")
plt.show()
fig2 = plt.figure()
plt.scatter(y_test, y_pred, color = 'b')
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()