from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time

def get_data():

    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.'
                                           'googleapis.com/download.tensorflow.'
                                           'org/data/shakespeare.txt')

    # Read, then decode for py2 compatibility
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    # Get each unique character and sort them
    vocab = sorted(set(text))

    # Create a dict correlating numbers to indices and reverse
    char_to_index = {u:i for i, u in enumerate(vocab)}
    index_to_char = np.array(vocab)

    text_as_int = np.array([char_to_index[c] for c in text])

    return (vocab, char_to_index, index_to_char, text_as_int, text)


def main():

    vocab, char_to_index, index_to_char, text_as_int, text = get_data()

    # Maximum length we want for single input in characters
    seq_length = 50
    examples_per_epoch = len(text) // seq_length

    # Create training data
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    # Split the data into batches of seq_length characters
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    # Get out input and answer data prepared
    dataset = sequences.map(split_input_target)

    # Shuffle the dataset into manageable sequences for training
    BATCH_SIZE = 64
    steps_per_epoch = examples_per_epoch // BATCH_SIZE

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Number of different characters
    vocab_size = len(vocab)

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    # Build the model
    model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

    # Test the model without training
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)

    # Get the prediction for the first sequnce
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    # Decode and print the prediction for the first sequence
    print('Input: \n', repr(''.join(index_to_char[input_example_batch[0]])))
    print()
    print('prediction: \n', repr(''.join(index_to_char[sampled_indices])))

    example_batch_loss  = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

    # Compile the model
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss=loss)

    # Save training checkpoints
    # Define directory to save checkpoints to
    checkpoint_dir = './training_checkpoints'
    # Name of checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, 'checkpoint_{epoch}')

    # Make sure the checkpoints do save properly
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    # Train for three epochs and track statistics
    EPOCHS = 3
    print()
    print('fitting')
    print()
    history = model.fit(dataset.repeat(), epochs=EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[checkpoint_callback])

    #Change batch size to one
    tf.train.latest_checkpoint(checkpoint_dir)
    model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    model.save('my_model01.h5')

# Evaluation step (generating text from the trained model)
def generate_text(model, start_string):

    # Number of characters to generate
    num_generate= 1000

    char_to_index, index_to_char = get_data()[1:3]

    # Vectorize strings
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    print(input_eval)

    # Empty string to store results
    text_generated = []

    # Low temperature results in more predictable text
    # High results in more surprising
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # Remove batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Using a multinomial distribution to predict the word returned bt the model
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # Along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(index_to_char[predicted_id])

    print(start_string + ''.join(text_generated))

# Shift each sequence one to the right to create a target
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE):

    # If the GPU is there use it otherwise build the model to use the CPU
    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNGRU
    else:
        import functools
        rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')

    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[BATCH_SIZE, None]),
            rnn(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True),
            tf.keras.layers.Dense(vocab_size)])
    return model

# Define loss
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Run the saved model
def run_model(model, input):
    new_model = tf.keras.models.load_model(model)
    generate_text(new_model, input)

if __name__ == '__main__':
    main()
    run_model()