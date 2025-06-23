import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv")
print(data)
X = data[["SquareFootage","Bath","Bedrooms"]] #2D array
y = data["Price"] #1D array
print(X)
print(y)

model = LinearRegression() #Create model object - brain
model.fit(X,y) #Teaches model how footage relates to price
print(f"Coefficients: {model.coef_}")  # [per_sqft, per_bath, per_bedroom]
print(f"Intercept (base price): ${model.intercept_:,.2f}")


new_data = pd.DataFrame({
    "SquareFootage": [4500],
    "Bath": [4],
    "Bedrooms": [8]
})
prediction = model.predict(new_data)
print(f"Predicted price: ${prediction[0]:,.2f}")

