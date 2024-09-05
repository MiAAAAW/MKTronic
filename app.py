from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

# Cargar el modelo entrenado para la predicción de fallas en máquinas
model_failure = load_model('machine_failure_prediction_model.h5')

# Cargar el modelo entrenado para la predicción del tipo de falla
model_type = load_model('machine_failure_type_prediction_model.h5')

# Definir una función para preprocesar los datos de entrada
def preprocess_input(temperature, process_temperature, torque, tool_wear):

    temp_diff = temperature - process_temperature
    
    # Devolver la entrada preprocesada como un arreglo de NumPy
    return np.array([[temperature, process_temperature, torque, tool_wear, temp_diff]])

# Definir una función para realizar predicciones de fallas
def predict_failure(input_data):
    # Usar el modelo de predicción de fallas en máquinas para hacer predicciones
    prediction = model_failure.predict(input_data)

    predicted_class = np.argmax(prediction)
    # Devolver la predicción
    return predicted_class

# Definir una función para realizar predicciones del tipo de falla
def predict_type(input_data):
    # Usar el modelo de predicción del tipo de falla para hacer predicciones
    prediction = model_type.predict(input_data)
    predicted_class = np.argmax(prediction)
    # Devolver la predicción
    return predicted_class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los valores de entrada del formulario
    temperature = float(request.form['temperature'])
    process_temperature = float(request.form['process_temperature'])
    torque = float(request.form['torque'])
    tool_wear = float(request.form['tool_wear'])
    
    # Preprocesar los datos de entrada
    input_data = preprocess_input(temperature, process_temperature, torque, tool_wear)
    
    # Realizar predicciones
    failure_prediction = predict_failure(input_data)
    type_prediction = predict_type(input_data)
    
    # Renderizar la plantilla de resultados con las predicciones
    return render_template('index.html', failure_prediction=failure_prediction, type_prediction=type_prediction)

if __name__ == '__main__':
    # Escalado de características
    app.run(debug=True)
