import medmnist
from medmnist import INFO
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
# Load BreastMNIST dataset
data_flag = 'breastmnist'
download = True
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# Define the transformations (resize to 128x128)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load training and test datasets
train_dataset = DataClass(split='train', transform=transform, download=download, size=128)
test_dataset = DataClass(split='test', transform=transform, download=download, size=128)
# Extract image data and labels (these will be NumPy arrays)
X_train = train_dataset.imgs
y_train = train_dataset.labels
X_test = test_dataset.imgs
y_test = test_dataset.labels

# Normalize and reshape the images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Add the channel dimension: (N, H, W, C) -> (N, 128, 128, 1)
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (N, 128, 128, 1)
X_test = np.expand_dims(X_test, axis=-1)    # Shape: (N, 128, 128, 1)

# Checking the shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
X_train shape: (546, 128, 128, 1)
X_test shape: (156, 128, 128, 1)
# Display first 5 training images
for i in range(5):
    plt.imshow(X_train[i].squeeze(), cmap='gray')  # .squeeze() to remove the channel dimension for display
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
    plt.show()
# Build the Keras CNN model
model = Sequential([
    Input(shape=(128, 128, 1)),  # Define the input shape (128x128 images with 1 channel)
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer=Adam(),
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

# Model summary
model.summary()
# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
# Make predictions on the test data
predictions = (model.predict(X_test[:5]) > 0.5).astype("int32")

# Display the first 5 predictions
for i in range(5):
    plt.imshow(X_test[i].reshape(128,128), cmap='gray')
    plt.title(f"Pred: {predictions[i][0]} | Actual: {y_test[i][0]}")
    plt.axis('off')
    plt.show()
