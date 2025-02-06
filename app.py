from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1
from sklearn import linear_model
model2 = linear_model.LinearRegression()
model2
import sklearn
model3 = sklearn.linear_model.LinearRegression()
model3
model = LinearRegression(fit_intercept=True)
model
import pandas as pd
data = pd.read_csv('student_mat.csv', sep=';')
data.head()
data.studytime.head()
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

model = LinearRegression(fit_intercept=True)


data = pd.read_csv('student_mat.csv', sep=';')


X1 = pd.DataFrame(data.studytime)
y1 = data.G3
data = pd.read_csv('student_mat.csv', sep=';')

# Create a new column 'G3_pass' based on the condition
data['G3_pass'] = (data['G3'] >= 10).astype(int)  # 1 for Pass, 0 for Fail

# Define features (X) and target (y)
X = data[['studytime', 'traveltime', 'G1', 'G2']]
y = data['G3_pass']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Create a LogisticRegression model
model = LogisticRegression()

# Train the model using the training data
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Generate the classification report
print(classification_report(y_test, y_pred))
