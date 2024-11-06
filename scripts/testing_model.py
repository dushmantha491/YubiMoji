from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('C:/Users/IoTAIc_work01/Desktop/IOT2024/Yubi_Moji/model/asl_model.h5')  # Replace with the path to your saved model



# Load an image file that you want to classify
img_path = 'C:/Users/IoTAIc_work01/Desktop/IOT2024/Yubi_Moji/data/test/B_test.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(200, 200))  # Resize the image to the target size

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Expand the dimensions to match the model input shape (batch size, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

# Normalize the image (same as training)
img_array = img_array / 255.0


# Make a prediction using the trained model
predictions = model.predict(img_array)

# Get the class with the highest probability
predicted_class_index = np.argmax(predictions)

# Map the index to the corresponding ASL letter
class_names = ['A', 'B', 'C', 'D','Del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','Nothing', 'O', 'P', 'Q', 'R', 'S','Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
predicted_class = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class}")    
