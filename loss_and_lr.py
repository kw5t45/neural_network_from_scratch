import numpy as np
from neural_network import initialize_params
import matplotlib.pyplot as plt

learning_rates = [0.001, 0.01, 0.1, 1.0, 10.0]
loses: list[list[float]] = []
nested_epoch_accuracies: list[list[float]] = []  # for each learning rate
epochs = 5
# training the network for each learning_rate
for learning_rate in learning_rates:
    # unpacking params ----------------------------------------------------------dont need these two se we can test
    # different epochs-
    values, labels, weights_hidden, biases_hidden, weights_output, biases_output, not_used_epochs,\
        not_used_learning_rate, correct = initialize_params()
    avg_losses_for_learning_rate = []
    epoch_accuracies = []

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
            hidden_neurons = np.clip(hidden_neurons, -500, 500)
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

        epoch_accuracies.append(correct * 100 / 60000)
        correct = 0
        avg_loss = total_loss / 60000  # Divide by the number of training samples
        print(f'Epoch {epoch}: Average Loss = {avg_loss}')
        avg_losses_for_learning_rate.append(avg_loss)
    nested_epoch_accuracies.append(epoch_accuracies)

    loses.append(avg_losses_for_learning_rate)


def plot_epochs_and_accuracy(number_of_epochs: int, epoch_accuracies: list[list[float]], learning_rate_: float):
    # actual_number_of_epochs = number_of_epochs * 5  # for each of 5 learning rates
    plt.plot(list(range(number_of_epochs)), epoch_accuracies[learning_rates.index(learning_rate_)], marker='o')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy percentage %')
    plt.title('Accuracy per number of epochs')
    plt.grid()
    plt.show()


def plot_hyperparameter_losses(loses: list[list[float]], learning_rates: list):
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Loss vs. Learning rate')
    plt.plot(learning_rates, loses, marker='o')
    plt.xscale('log')  # Logarithmic scale for learning rates
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot_hyperparameter_losses(loses, learning_rates)
    plot_epochs_and_accuracy(epochs, nested_epoch_accuracies, 0.01)
