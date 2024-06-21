from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from src.constants import IMAGE_SIZE, CHANNELS

def build_cnn_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), num_output=1, learning_rate=0.001, dropout_rate=0.5):

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_output, activation='sigmoid' if num_output == 1 else 'softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy' if num_output == 1 else 'categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# Function to train a CNN model
def train_cnn_model(model, train_generator, validation_generator, epochs=20, steps_per_epoch=None, validation_steps=None):

    if steps_per_epoch is None:
        steps_per_epoch = train_generator.samples // train_generator.batch_size
    
    if validation_steps is None:
        validation_steps = validation_generator.samples // validation_generator.batch_size

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs
    )

    return history
