🧠 Handwritten Digit Classification with TensorFlow & Keras

This project demonstrates how to build and train a simple Neural Network (NN) using TensorFlow and Keras on the popular MNIST dataset of handwritten digits (0–9).

The notebook walks through loading the dataset, preprocessing it, building a neural network model, training it, and evaluating its performance.

🚀 Features

Loads the MNIST dataset (70,000 grayscale images of handwritten digits).

Preprocesses data by normalizing pixel values (0–255 → 0–1).

Visualizes sample digits with Matplotlib.

Builds a Sequential neural network:

Flatten layer to reshape 28×28 images into vectors.

Dense(128, relu) hidden layer for learning features.

Dense(10, softmax) output layer for classification.

Compiles the model with:

Loss: sparse_categorical_crossentropy

Optimizer: Adam

Metric: Accuracy

Trains the model for 5 epochs with validation split.

Summarizes the model architecture.

📂 Project Structure
tensorflow.keras        # Original notebook file
tensorflow_keras_full.py # Extracted full Python script
README.md               # Documentation

🛠️ Requirements

Install dependencies using pip:

pip install tensorflow matplotlib

▶️ Usage

Run the script to train the model:

python tensorflow_keras_full.py


Or open the notebook (tensorflow.keras) in Jupyter / Google Colab for an interactive experience.

📊 Expected Output

Training accuracy should reach ~97–98% after 5 epochs.

Validation accuracy will be slightly lower (~96–97%).

Example visualization of handwritten digits.

📌 Notes

This is a beginner-friendly deep learning example.

You can tweak the number of layers, neurons, or epochs for better results.

For higher accuracy, try Convolutional Neural Networks (CNNs) instead of simple dense layers.
