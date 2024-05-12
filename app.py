import os
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from keras.models import model_from_json

# Create flask app
app = Flask(__name__)

# Load model architecture from JSON file
with open("model_new.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Create model from loaded architecture
loaded_model = model_from_json(loaded_model_json)

# Load weights into the model
loaded_model.load_weights("model_new_weights.weights.h5")

# Ensure the 'uploads' directory exists
os.makedirs('uploads', exist_ok=True)

# Define a function to preprocess the uploaded image
def preprocess_image(file_path):
    # Open and resize the image
    image = Image.open(file_path)
    image = image.resize((150, 150))
    # Convert the image to grayscale
    image = image.convert('L')
    # Convert the image to numpy array
    image = np.array(image)
    # Reshape the image to match the input shape of the model
    image = image.reshape((1, 150, 150, 1))
    # Normalize the image
    image = image.astype('float32') / 255.0
    return image

# Define a function to make predictions
#def predict_image(image):
#    prediction = loaded_model.predict(image)
#    result = '1' if prediction[0][0] > 0.5 else '0'
#    return result
    
# Define a function to make predictions
def predict_image(image):
    prediction = loaded_model.predict(image)
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    # Map the class index to the corresponding label (assuming you have a list of class labels)
    class_labels = ["Class 0", "Class 1", "Class 2"]  # Replace with your actual class labels
    result = class_labels[predicted_class_index]
    return result

@app.route("/")
def home():
    return render_template("index.html", prediction_text=None)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return render_template("index.html", error_message="No file selected")
    
    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", error_message="No file selected")
    
    try:
        # Save the file to the 'uploads' folder
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        # Preprocess the uploaded image
        image = preprocess_image(file_path)
        # Make predictions
        prediction_result = predict_image(image)
        prediction_text = "The Type of Cancer Cell is {}".format(prediction_result)
        return render_template("index.html", prediction_text=prediction_text)
    except Exception as e:
        # Handle errors gracefully
        error_message = str(e)
        return render_template("index.html", error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
