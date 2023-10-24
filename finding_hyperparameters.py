import numpy as np
from neural_network import get_images_and_labels, initialize_params, train_network
import matplotlib.pyplot as plt

learning_rates = [0.001, 0.01, 0.1, 1.0, 10.0]
loses = []
values, labels, weights_hidden, biases_hidden, weights_output, biases_output, epochs, learning_rate, correct = initialize_params()
epoch_accuracies = []
# for learning_rate in learning_rates:
# initializing weights & biases
avg_losses_for_learning_rate = []

epochs = 15

# to do : fix this mess of a file, use test data to test on main, do the math presentation


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
#loses.append(avg_losses_for_learning_rate)

def main():
    plt.plot([1, 2, 3, 4, 5, 6 ,7 ,8, 9, 10, 11, 12, 13, 14, 15], epoch_accuracies , marker='o')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy percentage %')
    #plt.xscale('log')  # Logarithmic scale for learning rates
    plt.title('Accuracy per number of epochs')
    #plt.plot(learning_rates, loses, marker='o')
    # plt.xlabel('Learning Rate')
    # plt.ylabel('Loss')
    # plt.xscale('log')  # Logarithmic scale for learning rates
    # plt.title('Learning Rate vs. Loss')
    plt.grid()
    plt.show()
if __name__ == '__main__':
    main()