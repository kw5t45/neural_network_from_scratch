import pandas as pd
import numpy as np


def get_images_and_labels():
    """

    csv is in
    label, 1x1, 1x2, 1x3, ... 28x28 form so
    we make labels a 1 x 60000 matrix and the rest a 784 x 60000 matrix.
    :return: values matrix sized 60000 x 784, labels np array sized 60000x1
    """
    raw_df = pd.read_csv('D:\pycharmBackUp\First-Neural-Network\dataset\mnist_train.csv')  # NOQA
    labels_ = raw_df.iloc[:, 0].values  # 60000 x 1 np array
    labels_ = np.eye(10)[labels_]

    values_ = raw_df.iloc[:, 1:].values  # 60000 x 784 --
    values_ = values_ / 255.0  # converting values from 0-255 to 0.0-1.0
    return values_, labels_


def initialize_params():
    """

    :return: values, labels, weights set to random values between -0.5 to 0.5, biases set to 0, 0.01 learning rate
    and setting correct variable = to 0 for accuracy measures.
    """

    values, labels = get_images_and_labels()

    weights_hidden = np.random.uniform(-0.5, 0.5, (784, 10))
    biases_hidden = np.zeros((10, 1))

    weights_output = np.random.uniform(-0.5, 0.5, (10, 10))
    biases_output = np.zeros((10, 1))

    weights_hidden = np.transpose(weights_hidden)  # for matrix multiplication
    epochs = 6
    learning_rate = 0.01
    correct = 0

    return values, labels, weights_hidden, biases_hidden, weights_output, biases_output, epochs, learning_rate, correct


def train_network(values, labels, weights_hidden, biases_hidden, weights_output, biases_output, epochs, learning_rate,
                  correct):
    """

    :return: trained weights and biases
    """
    for epoch in range(epochs):
        total_loss = 0
        for i in range(1, 60000):
            label = labels[i]  # 1x10 np array
            image = values[i]  # 784x10 np array
            image = image.reshape(784, 1)  # turning image and label rows into matrices
            label = label.reshape(10, 1)

            # forward prop.
            hidden_neurons = weights_hidden @ image + biases_hidden
            # activation using sigmoid
            hidden_neurons = np.clip(hidden_neurons, -500, 500) # overflow case
            hidden_neurons = 1 / (1 + np.exp(-hidden_neurons))
            output_neurons = weights_output @ hidden_neurons + biases_output
            output_neurons = 1 / (1 + np.exp(-output_neurons))

            # error calculation
            delta_o = output_neurons - label

            loss = 0.5 * np.sum((output_neurons - label) ** 2)  # Mean Squared Error
            total_loss += loss  # Accumulate loss for this epoch
            # backprop
            weights_output += -learning_rate * delta_o @ hidden_neurons.T
            biases_output += -learning_rate * delta_o

            delta_h = weights_output @ delta_o * (hidden_neurons * (1 - hidden_neurons))
            weights_hidden += delta_h * - learning_rate @ image.T
            biases_hidden += -learning_rate * delta_h
            correct += int(np.argmax(output_neurons) == np.argmax(label))  # accuracy for each epoch

        # print(correct * 100 /60000)
        correct = 0
    return weights_hidden, biases_hidden, weights_output, biases_output


def main():
    values, labels = get_images_and_labels()
    params: tuple = initialize_params()

    w_h, b_h, w_o, b_o = train_network(*params)  # unpacking parameters
    while True:
        index = int(input("Enter a number (0 - 59999): "))
        img = values[index]

        img.shape += (1,)
        # forward prop (getting the output from index input)
        h_neurons = b_h + w_h @ img.reshape(784, 1)
        h_neurons = 1 / (1 + np.exp(-h_neurons))

        o_neurons = b_o + w_o @ h_neurons
        o_neurons = 1 / (1 + np.exp(-o_neurons))

        print(f"It should be a {o_neurons.argmax()} (it's actually a {np.argmax(labels[index])})")


if __name__ == '__main__':
    main()
