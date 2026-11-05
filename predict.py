
import tensorflow as tf
import numpy as np
from pathlib import Path

# Load trained model
MODEL_PATH = "model/plant_disease_model.keras"
CLASS_PATH = "model/class_names.txt"

model = tf.keras.models.load_model(MODEL_PATH)

# Load disease names
with open(CLASS_PATH, "r") as file:
    class_names = file.read().splitlines()


def predict_disease(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        print("Error: Image file not found!")
        return

    # Load and resize image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(128, 128)
    )

    # Convert image into an array
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict disease
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    print("\n===== PLANT DISEASE PREDICTION =====")
    print("Predicted Category:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")
    print("===================================")


if __name__ == "__main__":
    image_path = input("Enter the image path: ").strip().strip('"')
    predict_disease(image_path)
