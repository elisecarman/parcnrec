from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, early_processing
from CNN import Model
from preprocess import get_data

import os
import tensorflow as tf
import numpy as np
import random
import math


def train(model, train_inputs, train_labels):

    width = tf.shape(train_inputs)[1]
    height = tf.shape(train_inputs)[2]

    indices = tf.range(0, len(train_inputs), 1)
    indices = tf.random.shuffle(indices).numpy()

    shuffled_inputs = tf.gather(tf.reshape(train_inputs, tf.shape(train_inputs)), indices).numpy()

    shuffled_labels = tf.gather(train_labels, indices).numpy()

    shuffled_inputs = tf.image.random_flip_left_right(shuffled_inputs)

    size = int(len(train_inputs) / (model.batch_size))
    losses = np.empty(size)
    accuracies = np.empty(size)
    start = 0
    end = model.batch_size

    for i in range(size):

        with tf.GradientTape() as tape:
            inputs = tf.reshape(shuffled_inputs[start: end], (model.batch_size, width, height, 3))

            # normalization
            inputs = tf.cast(inputs, dtype=tf.float32)
            inputs = inputs / 255

            predictions = model.call(inputs, False)

            loss = model.loss(predictions, shuffled_labels[start: end])
            labels_chunk = shuffled_labels[start: end]
            accuracy = model.accuracy(predictions, labels_chunk)
            accuracies[i] = accuracy.numpy()
            losses[i] = loss

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        start += model.batch_size
        end += model.batch_size

    visualize_loss(losses)

    accur_final = np.sum(accuracies) / size
    print(accur_final)

    return losses


def test(model, test_inputs, test_labels):
    probabilities = model.call(test_inputs, True)
    accuracy = model.accuracy(probabilities, test_labels)
    return accuracy


def visualize_loss(losses):
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, probabilities, image_labels):
    cat = np.array(
        ["Art", "Greenthumb", "Festivals", "Volunteer", "Film", "Fitness",
         "Kid", "Holidays","Culture", "Community", "Wildlife","Education", "Seniors",
         "Recreation", "Food", "Park"])
    original = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    for i in image_inputs:
        i = np.reshape(i, (120, 160, 3))
    predicted_labels = tf.where(probabilities > 0.5, 1, 0)

    find_cat = np.multiply(predicted_labels, original)
    find_cat_true = np.multiply(image_labels, original)

    num_labels = probabilities.shape[0]
    string_labels = []
    string_labels_true = []
    for i in range(num_labels):
        short_list = []
        short_list_true = []
        for j in range(16):
            if find_cat[i][j] != 0:
                short_list.append(cat[find_cat[i][j]])

            if find_cat_true[i][j] != 0:
                index = find_cat_true[i][j]
                short_list_true.append(cat[int(index)])

        string_labels.append(short_list)
        string_labels_true.append(short_list_true)

    num_images = image_inputs.shape[0]

    fig, axs = plt.subplots(1, 2, figsize=(120, 160))
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")

    for ind, ax in enumerate(axs):
        ax.imshow(image_inputs[ind])  # cmap="Greys"
        ax.set(title="PL: {}\nAL: {}".format(string_labels[ind], string_labels_true[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

    plt.show()

def main():
    early_processing('./data/categories.csv', './data/images.csv', './data/labels.npy',
                     './data/inputs.npy')

    input_arr, input_arr_t, label_arr, label_arr_t = get_data("./data/inputs.npy", "./data/labels.npy")
    model = Model()
    epoch = 5

    for i in range(epoch):
        loss = train(model, input_arr, label_arr)

    accuracy = test(model, input_arr_t, label_arr_t)

    print("accuracy:")
    print(accuracy)

    total_size = len(input_arr)
    probs = model.call(input_arr)

    visualize_results(input_arr[total_size-2 :], probs[total_size-2 :],
                      label_arr[total_size-2 :])

    return


if __name__ == '__main__':
    main()