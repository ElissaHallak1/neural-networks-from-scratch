# neural-networks-from-scratch
### Overview
This project implements a simple neural network from scratch in Python without using deep learning frameworks.
The goal of the project is to understand the core concepts behind neural networks such as forward propagation, backpropagation, and gradient descent.
This project was developed as part of the Deep Learning course at the University of Deusto.

### Features

The library includes implementations of:
* Linear layers (fully connected layers)
* Activation functions (Sigmoid, Tanh, ReLU)
* Loss Functions (MSE, Cross Entropy Loss)
* Optimization (Stochastic Gradient Descent (SGD))
* Multilayer Perceptron (MLP)

### Training Example: MNIST Classification
The implemented library was used to train a multilayer perceptron on the MNIST handwritten digit dataset.
Hidden layers use ReLU activation, while the output layer produces raw logits used with CrossEntropyLoss.

### Project structure
deustorch/
│
├── nn/
│   ├── activations.py
│   ├── linear.py
│   ├── loss.py
│   └── batchnorm.py
│
├── models/
│   └── mlp.py
│
├── optim/
│   └── sgd.py
│
└── main.py

### Installation/runnning the script
git clone https://github.com/your-username/deustorch.git
cd deustorch

Install dependencies:
pip install numpy scikit-learn

Running the script:
python main.py

This will load the MNIST dataset, train the neural network and print training loss and accuracy.





