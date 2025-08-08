import pandas as pd
import matplotlib.pyplot as pt 
import sklearn.linear_model as lm
from sklearn.preprocessing import LabelEncoder
mydata = pd.read_csv("persons_weight_finder_app.csv")
le= LabelEncoder()
mydata["Gender_encoded"] = le.fit_transform(mydata[["Gender"]])
mydata["BodyType_encoded"] = le.fit_transform(mydata[["BodyType"]])
x=mydata[["Age","Gender_encoded","BodyType_encoded","Height"]]
y=mydata["Weight"]
model=lm.LinearRegression()
model.fit(x,y)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

print(model.predict([[25,2,3,170]]))  

