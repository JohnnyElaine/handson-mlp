import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Download and prepare the data
data_root = "./data"
lifesat = pd.read_csv(f"{data_root}/lifesat/lifesat.csv")

# Visualize the data
sns.scatterplot(data=lifesat, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis((23_500, 62_500, 4, 9))
plt.show()

X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

X_prediction_puerto_rico = [[33_442.8]]
y_prediction = model.predict(X_prediction_puerto_rico)
print(
    f"Puerto Rico' GDP per capita in 2020 of {X_prediction_puerto_rico[0][0]} yields a predicted life satisfcation of {y_prediction[0][0]}."
)
