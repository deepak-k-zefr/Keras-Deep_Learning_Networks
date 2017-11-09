import pandas as pd
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler

# Train a model and make prediction on earnings
# Video-game dataset, y =Total earnings
# Data is scaled using min max scaler
# The mean squared error (MSE) for the test data set is: 0.0003009536094032228
# Earnings Prediction for Proposed Product - $260129.9568627


# Load training data set from CSV file
training_data_df =pd.read_csv("sales_data_training.csv")

# Load testing data set from CSV file
test_data_df =pd.read_csv("sales_data_training.csv")

# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
scaler =MinMaxScaler(feature_range=(0,1))

# Scale both the training inputs and outputs
training_data_df = scaler.fit_transform(training_data_df)
test_data_df = scaler.transform(test_data_df)


X = pd.DataFrame(training_data_df).drop('total_earnings', axis=1).values
Y = pd.DataFrame(training_data_df)[['total_earnings']].values

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

# Load the separate test data set
test_data_df = pd.read_csv("sales_data_test_scaled.csv")

X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

# Load the data we make to use to make a prediction
X = pd.read_csv("proposed_new_product.csv").values

# Make a prediction with the neural network
prediction = model.predict(X)

# Grab just the first element of the first prediction (since that's the only have one)
prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product - ${}".format(prediction))
