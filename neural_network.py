import pandas as pd
import numpy as np

# csv is in
# label 1x1 1x2 1x3 ... 28x28 form so
# we make labels a 1 x 60000 matrix and the rest a 784 x 60000 matrix.
raw_df = pd.read_csv('D:\pycharmBackUp\First-Neural-Network\dataset\mnist_train.csv')  # NOQA
labels = raw_df.iloc[:, 0].values  # 60000 x 1 np array
labels = np.eye(10)[labels]

values = raw_df.iloc[:, 1:].values  # 60000 x 784 --
values = values / 255.0  # converting values from 0-255 to 0.0-1.0

# initializing weights & biases
weights_hidden = np.random.uniform(-0.5, 0.5, (784, 10))
biases_hidden = np.zeros((10, 1))

weights_output = np.random.uniform(-0.5, 0.5, (10, 10))
biases_output = np.zeros((10, 1))

weights_hidden = np.transpose(weights_hidden)  # for matrix multiplication
learning_rate = 0.01
epochs = 4
correct = 0

for epoch in range(epochs):

    for i in range(1, 60000):
        label = labels[i]  # 1x10 np array
        image = values[i]  # 784x10 np array
        image = image.reshape(784, 1)  # turning image and label rows into matrices
        label = label.reshape(10, 1)

        # forward prop.
        hidden_neurons = weights_hidden @ image + biases_hidden
        # activation using sigmoid
        hidden_neurons = 1 / (1 + np.exp(-hidden_neurons))
        output_neurons = weights_output @ hidden_neurons + biases_output
        output_neurons = 1 / (1 + np.exp(-output_neurons))

        # error calculation
        delta_o = output_neurons - label
        # backprop
        weights_output += -learning_rate * delta_o @ hidden_neurons.T
        biases_output += -learning_rate * delta_o

        delta_h = weights_output @ delta_o * (hidden_neurons * (1 - hidden_neurons))
        weights_hidden += delta_h * - learning_rate @ image.T
        biases_hidden += -learning_rate * delta_h
        correct += int(np.argmax(output_neurons) == np.argmax(label))  # accuracy for each epoch
        
    # print(correct * 100 /60000)
    correct = 0

def main():
    while True:
        index = int(input("Enter a number (0 - 59999): "))
        img = values[index]

        img.shape += (1,)
        # forward prop (getting the output from index input)
        h_neurons = biases_hidden + weights_hidden @ img.reshape(784, 1)
        h_neurons = 1 / (1 + np.exp(-h_neurons))

        o_neurons = biases_output + weights_output @ h_neurons
        o_neurons = 1 / (1 + np.exp(-o_neurons))

        print(f"It should be a {o_neurons.argmax()} (it's actually a {np.argmax(labels[index])})")

if __name__ == '__main__':
    main()