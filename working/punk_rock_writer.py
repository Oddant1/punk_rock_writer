import tensorflow as tf
from tensorflow import keras
tf.enable_eager_execution()

import functools
import numpy as np
import os
import time


# Get the text file
def get_text(text_name):
    return open(text_name, 'rb').read().decode(encoding='utf-8')


# Extract the characters from the text
def get_vocab(text):
    return sorted(set(text))


# Map the characters to ints and vice versa
def get_mappings(vocab):
    return ({u:i for i, u in enumerate(vocab)}, np.array(vocab))


# Get the text mapped to ints
def get_text_as_int(char_to_index, text):
    return np.array([char_to_index[c] for c in text])


# Construct the neural network model
def build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE):

    # If the GPU is there use it otherwise build the model to use the CPU
    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNGRU
    else:
        rnn = functools.partial(tf.keras.layers.GRU,
              recurrent_activation='sigmoid')

    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
            batch_input_shape=[BATCH_SIZE, None]),
            rnn(rnn_units, return_sequences=True,
            recurrent_initializer='glorot_uniform', stateful=True),
            tf.keras.layers.Dense(vocab_size)])
    return model


# Offset every piece of text by one character to use as answers in training
def create_answer_data(chunk):

    # This is fed in during training
    input_text = chunk[:-1]
    # This is what the network wants to produce
    target_text = chunk[1:]
    return input_text, target_text


# Define loss
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits,
                                                           from_logits=True)


# Define checkpoint creation path and methods
def setup_checkpoints():

    # Name of checkpoint files
    checkpoint_prefix = os.path.join('./training_checkpoints',
                                     'checkpoint_{epoch}')

    # Make sure the checkpoints save properly
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    return checkpoint_prefix, checkpoint_callback


# Evaluation step (generating text from the trained model)
def generate_text(model, start_string, char_to_index, index_to_char, output_file=None):

    # Number of characters to generate
    num_generate= 1000

    # Vectorize strings
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # Remove batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Use a multinomial distribution to predict word returned by model
        # predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # Along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(index_to_char[predicted_id])

    if output_file == None:
        print()
        print(start_string + ''.join(text_generated))
    else:
        with open(output_file, 'w') as file__:
            file__.write(start_string + ''.join(text_generated))


# Save the generated model
def save_model(model, model_name, model_path):

    if model_path != None:
        model_name = model_path + '/' + model_name
    model.save(model_name)


# Run the saved model
def run_model(model, input, text_path, output_file=None):

    model = tf.keras.models.load_model(model)
    # Get needed data
    text = get_text(text_path)
    vocab = get_vocab(text)
    char_to_index, index_to_char = get_mappings(vocab)
    generate_text(model, input, char_to_index, index_to_char, output_file)


# Call functions
def main(text_name=None, num_epochs=0, model_name=None, model_prompt=None,
         model_path=None, output_file=None):

    if text_name == None:
        text_name = input('Please input a path to training data: ')

    # Get needed data
    text = get_text(text_name)
    vocab = get_vocab(text)
    char_to_index, index_to_char = get_mappings(vocab)

    vocab = get_vocab(text)
    char_to_index, index_to_char = get_mappings(vocab)
    text_as_int = get_text_as_int(char_to_index, text)

    # Max length we want for single input in characters
    seq_length = 50
    # Number of examples is the amount of text over the size of a sequence
    examples_per_epoch = len(text) // seq_length

    # Create a tensorflow dataset of our text mapped to ints
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    # Split that dataset into manageable batches to feed into the network
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    # Prepare answer data
    dataset = sequences.map(create_answer_data)

    # Size of the batches to be fed to the network in sequences
    # Batch size set to one for max training even though it takes longer
    BATCH_SIZE = 1
    # Calculate number of steps it will take for one epoch given each batch is
    # processed individually
    steps_per_epoch = examples_per_epoch // BATCH_SIZE

    # Size of buffer to shuffle the data in
    # (TF data is designed to work with possibly infinite sequences
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000
    # Shuffle the data around in the buffer then split it into batches
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,
                              drop_remainder=True)

    # Number of different characters used
    vocab_size = len(vocab)
    # Number of dimensions in embedding vector
    embedding_dim = 256
    # Number of units in network
    rnn_units = 1024

    # Build our model
    model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
    # Compile the model we've defined
    model.compile(optimizer=tf.train.AdamOptimizer(), loss=loss)

    # Define how to create checkpoints
    checkpoint_prefix, checkpoint_callback = setup_checkpoints()

    if num_epochs == 0:
        while True:
            try:
                num_epochs = int(input("Please enter an integer number of epochs: "))
            except ValueError:
                print('That was not a valid integer')
            else:
                break

    history = model.fit(dataset.repeat(), epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[checkpoint_callback])

    # Change batch size to one
    tf.train.latest_checkpoint('./training_checkpoints')
    model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE=1)
    model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
    model.build(tf.TensorShape([1, None]))

    if model_name == None:
        model_name = input('\nInput the name of the model: ')
    save_model(model, model_name, model_path)
    if model_prompt == None:
        model_prompt = input('Give the model some initial input: ')
    generate_text(model, model_prompt, char_to_index, index_to_char, output_file);


if __name__ == '__main__':
    main()
