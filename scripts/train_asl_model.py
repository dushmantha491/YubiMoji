# train_asl_model.py
from tensorflow import keras
from tensorflow.
keras.preprocessing.image import ImageDataGenerator

# Define paths and constants
train_dir = 'C:/Users/IoTAIc_work01/Desktop/IOT2024/Yubi_Moji/data/train'
val_dir = 'C:/Users/IoTAIc_work01/Desktop/IOT2024/Yubi_Moji/data/val'
image_size = (200, 200)
batch_size = 32

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)


val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)


from tensorflow.keras import layers, models # type: ignore

# Build a CNN model for image recognition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(29, activation='softmax')  # 26 classes for A-Z plus three for space,delete and nothing
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Training the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=epochs
)

# Save the model
model.save('C:/Users/IoTAIc_work01/Desktop/IOT2024/Yubi_Moji/model/asl_model.h5')
keras.saving.save_model(model, 'my_model.keras')
print("Model saved!")
