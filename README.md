# Heart Disease Predictor

## Overview
This project implements a machine learning model to predict the likelihood of heart disease in patients based on medical attributes. The system uses logistic regression to analyze various health indicators and classify patients as either having heart disease or being healthy.

## Dataset
The dataset contains 303 patient records with 13 clinical features:
- **age**: Age in years
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: The slope of the peak exercise ST segment
- **ca**: Number of major vessels colored by fluoroscopy (0-4)
- **thal**: Thalassemia (0-3)

**Target Variable:**
- **target**: Presence of heart disease (1 = disease, 0 = no disease)

## Requirements
- Python 3.6+
- NumPy
- Pandas
- Scikit-learn

## Installation
```bash
pip install numpy pandas scikit-learn
```

## Usage
1. Clone the repository
2. Ensure the dataset is in the `Data/` directory
3. Run the Jupyter notebook or Python script

## Model Information
- **Algorithm**: Logistic Regression
- **Training/Testing Split**: 80% training, 20% testing
- **Feature Scaling**: StandardScaler
- **Training Accuracy**: 85.95%
- **Testing Accuracy**: 78.69%

## Prediction Example
```python
# Input data format (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
input_data = (37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2)

# Convert to numpy array and reshape
input_data = np.asarray(input_data).reshape(1, -1)

# Scale the input
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)

# Interpret result
if prediction[0] == 0:
    print("✅ Good News the patient doesn't have any heart disease")
else:
    print("⚠️ The Patient should visit the doctor")
```

## Future Improvements
- Implement feature selection techniques
- Try different classification algorithms
- Perform hyperparameter tuning
- Add more comprehensive evaluation metrics
- Create a user-friendly interface for predictions


## Acknowledgments
- The UCI Machine Learning Repository for providing the heart disease dataset, which is widely used in the machine learning community for benchmarking classification algorithms
- The scikit-learn development team for their excellent implementation of machine learning algorithms and preprocessing tools
- The open-source community for continuous improvements to data science libraries
- Healthcare professionals who provided domain expertise in interpreting cardiac health indicators
- All contributors to research on early detection of heart disease using machine learning techniques