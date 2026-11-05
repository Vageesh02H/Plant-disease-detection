
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

# Project configuration
DATASET_DIR = Path("dataset")
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

print("TensorFlow version:", tf.__version__)

# Locate dataset split folders
def find_split(name):
    matches = [
        p for p in DATASET_DIR.rglob("*")
        if p.is_dir() and p.name.lower() == name
    ]
    return matches[0] if matches else None

train_dir = find_split("train")
val_dir = find_split("val") or find_split("validation")

if train_dir is None or val_dir is None:
    raise FileNotFoundError(
        "Could not find train and validation folders. "
        "Check your dataset structure."
    )

print("Training folder:", train_dir)
print("Validation folder:", val_dir)

# Load training images
train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)

# Load validation images
val_ds = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)

# Improve data-loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Build CNN model
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save trained model
Path("model").mkdir(exist_ok=True)
model.save("model/plant_disease_model.keras")

# Save class names
with open("model/class_names.txt", "w") as f:
    f.write("\n".join(class_names))

# Plot accuracy
Path("results").mkdir(exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Plant Disease Detection Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/accuracy.png")
plt.close()

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/loss.png")
plt.close()

print("\nTraining completed successfully!")
print("Model saved in model/plant_disease_model.keras")


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

# 1. Project settings
DATASET_DIR = Path("dataset/Dataset")
MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

print("TensorFlow version:", tf.__version__)

# 2. Load training, validation and testing datasets
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# 3. Check class names
class_names = train_ds.class_names
num_classes = len(class_names)

print("\nDisease categories:")
for name in class_names:
    print(name)

assert val_ds.class_names == class_names, "Validation classes do not match!"
assert test_ds.class_names == class_names, "Test classes do not match!"

# 4. Improve data-loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# 5. Build the CNN model
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Normalize pixel values
    layers.Rescaling(1.0 / 255),

    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax")
])

# 6. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

# 7. Train the model
print("\nStarting model training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 8. Evaluate using the separate test dataset
print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# 9. Save the trained model
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

model.save(MODEL_DIR / "plant_disease_model.keras")

with open(MODEL_DIR / "class_names.txt", "w") as file:
    file.write("\n".join(class_names))

print("\nModel and class names saved successfully!")

# 10. Plot accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(RESULTS_DIR / "accuracy.png")
plt.close()

# 11. Plot loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(RESULTS_DIR / "loss.png")
plt.close()

print("Training graphs saved in the results folder.")
print("\nPROJECT TRAINING COMPLETED!")
