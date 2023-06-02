from hmmlearn import hmm
import numpy as np

# Define the HMM model
model = hmm.GaussianHMM(n_components=3)

# Define the observed weather states
weather_states = ['Sunny', 'Cloudy', 'Rainy']

# Define the observed weather data
weather_data = np.array([[0, 0, 1, 2, 1, 1, 2, 0, 0, 1, 2, 1]])

# Fit the model to the weather data
model.fit(weather_data.T)

# Predict the weather for the next day
predicted_weather = model.predict(weather_data.T)

# Convert the predicted weather states to their corresponding labels
predicted_weather_labels = [weather_states[state] for state in predicted_weather]

# Print the predicted weather for the next day
print("Predicted Weather:", predicted_weather_labels)
