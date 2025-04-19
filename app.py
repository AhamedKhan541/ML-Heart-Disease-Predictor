from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load and train the model (you can optimize by saving and loading it instead)
df = pd.read_csv("Data/heart_disease_data.csv")

X = df.drop(columns='target', axis=1)
Y = df['target']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=45)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(x) for x in request.form.values()]
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)

        if prediction[0] == 0:
            result = "✅ Good News! The patient doesn't have any heart disease."
        else:
            result = "⚠️ The patient might have heart disease. A doctor visit is recommended."

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
