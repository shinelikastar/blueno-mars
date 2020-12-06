import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
from preprocess import get_data
from preprocess import get_data_by_song
import copy

class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # initialize vocab_size, embedding_size
        self.vocab_size = vocab_size
        self.window_size = 20 # DO NOT CHANGE!
        self.embedding_size = 64
        self.batch_size = 64 

        # initialize embeddings and forward pass weights (weights, biases)
        self.E = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size], stddev=0.1))
        self.lstm = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        embeddings = tf.nn.embedding_lookup(self.E, inputs)

        output, mem_output, carry_output = self.lstm(embeddings, initial_state=initial_state)
        layer1out = self.dense1(output)
        layer2out = self.dense2(layer1out)
        
        return layer2out,(mem_output, carry_output)

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """

    prev_state = None
    to_remove = train_inputs.shape[0] % model.window_size
    train_inputs = train_inputs[:-to_remove]
    train_labels = train_labels[:-to_remove]
    train_inputs = np.reshape(train_inputs, (-1, model.window_size))
    train_labels = np.reshape(train_labels, (-1, model.window_size))
    for i in range(0, train_inputs.shape[0], model.batch_size):
        batch_inputs = train_inputs[i : i + model.batch_size, :]
        batch_labels = train_labels[i : i + model.batch_size, :]

        if batch_inputs.shape[0] < model.batch_size:
            continue

        with tf.GradientTape() as tape:
            probs, prev_state = model.call(batch_inputs, prev_state)
            loss = model.loss(probs, batch_labels)

        print("loss is " + str(loss))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    
    prev_state = None
    to_remove = test_inputs.shape[0] % model.window_size
    test_inputs = test_inputs[:-to_remove]
    test_labels = test_labels[:-to_remove]
    test_inputs = np.reshape(test_inputs, (-1, model.window_size))
    test_labels = np.reshape(test_labels, (-1, model.window_size))

    total_loss = 0
    num_batches = 0

    for i in range(0, test_inputs.shape[0], model.batch_size):
        batch_inputs = test_inputs[i : i + model.batch_size, :]
        batch_labels = test_labels[i : i + model.batch_size, :]
        if batch_inputs.shape[0] < model.batch_size:
            continue

        probs, prev_state = model.call(batch_inputs, prev_state)
        loss = model.loss(probs, batch_labels)

        total_loss += loss
        num_batches += 1

    return tf.math.exp(total_loss / num_batches)


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    #NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input[0].append(out_index)
        # next_input = [[out_index]]

    print(" ".join(text))


def main():
    # Pre-process and vectorize the data
    print("Loading data...")
    ## train_data, test_data, vocab = get_data("lowercase.txt", "lowercase.txt")

    train_data, vocab = get_data_by_song("data.txt")

    print(len(vocab))

    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see impossibly small perplexities.
    
    # Separate your train and test data into inputs and labels
    train_inputs = copy.copy(train_data)
    train_inputs = np.array(train_inputs[:-1])
    train_labels = copy.copy(train_data)
    train_labels = np.array(train_labels[1:])

    # test_inputs = copy.copy(test_data)
    # test_inputs = np.array(test_inputs[:-1])
    # test_labels = copy.copy(test_data)
    # test_labels = np.array(test_labels[1:])

    # initialize model and tensorflow variables
    model = Model(len(vocab))

    # Set-up the training step
    print("Training...")
    for i in range(5):
        train(model, train_inputs, train_labels)

    # Set up the testing steps
    print("Testing...")
    perp = test(model, test_inputs, test_labels)

    # Print out perplexity 
    print("Perplexity is: " + str(perp))

    generate_sentence("<|startoftext|>", 40, vocab, model)
    generate_sentence("<|startoftext|>", 40, vocab, model)
    generate_sentence("<|startoftext|>", 40, vocab, model)

    # BONUS: Try printing out various sentences with different start words and sample_n parameters 
    

if __name__ == '__main__':
    main()