import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score


def cnnSvm():
    from tensorflow.keras import datasets, layers, models
    # Load your dataset (e.g., CIFAR-10)
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Preprocess the data
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Define and train the CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # Extract features from the trained CNN
    cnn_features_train = model.predict(train_images)
    cnn_features_test = model.predict(test_images)

    # Train an SVM classifier on the extracted features
    svm_classifier = svm.SVC()
    svm_classifier.fit(cnn_features_train, train_labels)

    # Make predictions using the SVM classifier
    svm_predictions = svm_classifier.predict(cnn_features_test)

    # Evaluate the SVM classifier's performance
    svm_accuracy = accuracy_score(test_labels, svm_predictions)
    print(f"SVM Classifier Accuracy: {svm_accuracy * 100:.2f}%")
        