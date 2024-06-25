from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckPoint
from tensorflow.keras.applications import InceptionV3
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import seaborn as sns
import numpy as np
import os

from src.constants import IMAGE_SIZE, CHANNELS, EPOCHS

def build_cnn_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), num_output=1, learning_rate=0.001, dropout_rate=0.5):

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding = 'same' ,input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding = 'same', activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), padding = 'same', activation='relu'),
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


# Function to check if GPU is available
def check_gpu():
    devices = tf.config.list_physical_devices()
    print("\nAll Physical Devices:", devices)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        print("GPU details:", details)
        return True
    else:
        print("No GPU found.")
        return False


# Function to train a CNN model
def train_model(model, train_generator, validation_generator, epochs=EPOCHS, steps_per_epoch=None, validation_steps=None, batch_size=32, verbose=1):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_generator)

    if validation_steps is None:
        validation_steps = len(validation_generator)

    os.makedirs('Saved_Models', exist_ok=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    checkpoint = ModelCheckPoint('Saved_Models/best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks = [early_stopping, checkpoint]
    )

    return history



# Models evaluation
import matplotlib.pyplot as plt

def plot_training_validation_history(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'ro', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# Function to compare training and test evaluation metrics
def evaluate_train_test_performance(model, train_generator, test_generator):

    train_loss, train_accuracy = model.evaluate(train_generator, steps=train_generator.samples // train_generator.batch_size)
    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Train Loss: {train_loss:.4f}')

    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print(f'Test Loss: {test_loss:.4f}')

    test_generator.reset()
    y_true = test_generator.classes
    y_pred = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)
    y_pred = np.rint(y_pred).astype(int).flatten()

    print("\nClassification Report for Test Set:")
    print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Test Set')
    plt.show()


# Transfer Learning model
def create_inceptionv3_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), num_classes=1, learning_rate=0.001):

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x) if num_classes == 1 else Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy', metrics=['accuracy'])
    
    return model