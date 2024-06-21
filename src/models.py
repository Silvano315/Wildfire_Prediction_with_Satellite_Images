from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

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
