from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load data
data = pd.read_csv('data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

import matplotlib.pyplot as plt


# Resample data to monthly and yearly
monthly_data = data.resample('M').sum()
yearly_data = data.resample('Y').sum()

# Prepare data
X_monthly = monthly_data.drop('Sales', axis=1)
y_monthly = monthly_data['Sales']
X_yearly = yearly_data.drop('Sales', axis=1)
y_yearly = yearly_data['Sales']
# Create a bar chart for monthly sales data
plt.bar(monthly_data.index, monthly_data['Sales'])
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Monthly Sales')
plt.show()

# Train the models
monthly_regressor = LinearRegression()
monthly_regressor.fit(X_monthly, y_monthly)

yearly_regressor = LinearRegression()
yearly_regressor.fit(X_yearly, y_yearly)
# Compute mean squared error for monthly sales predictions
monthly_pred = monthly_regressor.predict(X_monthly)
monthly_mse = mean_squared_error(y_monthly, monthly_pred)
print('Monthly Sales MSE:', monthly_mse)

# Compute mean squared error for yearly sales predictions
yearly_pred = yearly_regressor.predict(X_yearly)
yearly_mse = mean_squared_error(y_yearly, yearly_pred)
print('Yearly Sales MSE:', yearly_mse)
# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monthly', methods=['GET', 'POST'])
def monthly():
    if request.method == 'POST':
        # Get the new data from the form
        year = int(request.form['year'])
        month = int(request.form['month'])

        # Create a DataFrame with the new data
        new_data = pd.DataFrame({'Year': [year], 'Month': [month]})

        # Use the trained model to predict the sales for the new data
        y_pred = monthly_regressor.predict(new_data)[0]

        # Return the predicted sales to the user
        return render_template('monthly.html', monthly_sales=y_pred)
    else:
        return render_template('monthly.html')

@app.route('/yearly', methods=['GET', 'POST'])
def yearly():
    if request.method == 'POST':
        # Get the new data from the form
        year = int(request.form['year'])

        # Create a DataFrame with the new data
        new_data = pd.DataFrame({'Year': [year]})

        # Use the trained model to predict the sales for the new data
        y_pred = yearly_regressor.predict(new_data)[0]

        # Return the predicted sales to the user
        return render_template('yearly.html', yearly_sales=y_pred)
    else:
        return render_template('yearly.html')

if __name__ == '__main__':
    app.run(debug=True)
