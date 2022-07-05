import pandas
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data= pandas.read_csv('iphone_data.csv')
plt.scatter(data['version'], data['price'])
plt.show()
model= LinearRegression()
model.fit(data[['version']],data[['price']])
print(model.predict([[30]]))