import pandas
from sklearn import linear_model

df = pandas.read_csv("multiple_linear_regression_dataset.csv")

X = df[['age', 'experience']]
y = df['income']

regr = linear_model.LinearRegression()
regr.fit(X, y)


#predict the income from age and years of experience
predicted_income = regr.predict([[24, 3]])

print("Predicted Income", predicted_income)
