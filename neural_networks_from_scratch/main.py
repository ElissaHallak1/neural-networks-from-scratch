import sys , os
sys . path . insert (0 , os . path . dirname ( os . path . abspath ( __file__ ) ) )

from sklearn.datasets import fetch_openml
import numpy as np
import deustorch . nn , deustorch . models , deustorch . optim
from deustorch.models.mlp import MLP4



### 1 - Load and preprocess data
mnist = fetch_openml("mnist_784", version = 1, as_frame = False, parser = "auto")
X_all = mnist.data.astype(np.float32)
y_all = mnist.target.astype(int)

X_train, X_test = X_all[:60000] / 255.0, X_all[60000:] / 255.0
y_train, y_test = y_all[:60000], y_all[60000:]

def one_hot(labels, num_classes=10):
    Y = np.zeros((len(labels), num_classes), dtype = np.float32)
    Y[np.arange(len(labels)), labels] = 1.0
    return Y 

y_train = one_hot(y_train)
y_test = one_hot(y_test)

print(np.shape(y_train))

### 2 - Build model
model = MLP4()

### 3 - Initialise weights with Kaiming initialisation
np.random.seed(42)
for layer in model.layers:
    fan_in = layer.W.shape[1]
    layer.W = np.random.randn(*layer.W.shape).astype(np.float32) * np.sqrt(2.0 / fan_in)
    layer.b = np.zeros_like(layer.b)

### 4 - Loss and optimizer
criterion = deustorch.nn.CrossEntropyLoss()
optimizer = deustorch.optim.SGD(model, lr = 0.1)

### 5 - Mini-batch training loop
BATCH_SIZE, NUM_EPOCHS = 128, 20

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_loss = 0.0
    num_batches = 0
    idx = np.random.permutation(len(X_train))

    for start in range(0, len(X_train), BATCH_SIZE):
        b = idx[start: start + BATCH_SIZE]
        Xb, Yb = X_train[b], y_train[b]
        
        logits = model.forward(Xb)
        loss = criterion.forward(A = logits, Y = Yb)

        dLdA = criterion.backward()

        model.backward(dLdA)

        optimizer.step()

        epoch_loss += loss
        num_batches += 1
        print(num_batches)
        

### 6 - Evaluate accuracy on train and test sets

train_logits = model.forward(X_train)
train_pred = np.argmax(train_logits, axis=1)
train_true = np.argmax(y_train, axis=1)

train_acc = np.mean(train_pred == train_true)
print("Train accuracy:", train_acc)

test_logits = model.forward(X_test)
test_pred = np.argmax(test_logits, axis=1)
test_true = np.argmax(y_test, axis=1)

test_acc = np.mean(test_pred == test_true)
print("Test accuracy:", test_acc)


avg_loss = epoch_loss / num_batches

print(f"Epoch {epoch}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")







