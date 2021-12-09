from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, early_processing
# from convolution import conv2d
from CNN import Model
from preprocess import get_data

import os
import tensorflow as tf
import numpy as np
import random
import math


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''



    width = tf.shape(train_inputs)[1]
    height = tf.shape(train_inputs)[2]

    indices = tf.range(0, len(train_inputs), 1)
    indices = tf.random.shuffle(indices).numpy()

    shuffled_inputs = tf.gather(tf.reshape(train_inputs, tf.shape(train_inputs)), indices).numpy()

    shuffled_labels = tf.gather(train_labels, indices).numpy()

    shuffled_inputs = tf.image.random_flip_left_right(shuffled_inputs)

    print(len(train_inputs))


    size = int(len(train_inputs) / (model.batch_size))
    print("size")
    print(size)
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

            #rint(predictions)
            loss = model.loss(predictions, shuffled_labels[start: end])
            #print("loss")
            #print(loss)
            labels_chunk = shuffled_labels[start: end]
            accuracy = model.accuracy(predictions, labels_chunk)
            accuracies[i] = accuracy.numpy()
            losses[i] = loss

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        start += model.batch_size
        end += model.batch_size
        #print("training batch")
        #print(i)

    visualize_loss(losses)

    accur_final = np.sum(accuracies) / size
    print(accur_final)

    return losses


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.

    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    probabilities = model.call(test_inputs, True)
    accuracy = model.accuracy(probabilities, test_labels)
    return accuracy


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
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

    print("labels in visualization")
    print(image_labels)

    print("input imagess")
    print(image_inputs)

    print("probabilities")
    print(probabilities)

    # From Assignment 1

    # how to do this with RGB
    for i in image_inputs:
        i = np.reshape(i, (120, 160, 3))
    # predicted_labels = np.argmax(probabilities, axis=1)
    predicted_labels = tf.where(probabilities > 0.5, 1, 0)

    find_cat = np.multiply(predicted_labels, original)
    print("black cat")
    print(find_cat)
    find_cat_true = np.multiply(image_labels, original)

    #print(probabilities)


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
                #print(index)
                short_list_true.append(cat[int(index)])

        string_labels.append(short_list)
        string_labels_true.append(short_list_true)
        #print(short_list)

    print(string_labels)
    num_images = image_inputs.shape[0]

    """
    #this prints only one image
    plt.imshow(images[0])
    plt.title('Input Image')
    plt.show()
    """

    "this prints multiple images"
    fig, axs = plt.subplots(1, 2, figsize=(120, 160))
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")

    for ind, ax in enumerate(axs):
        ax.imshow(image_inputs[ind])  # cmap="Greys"
        ax.set(title="PL: {}\nAL: {}".format(string_labels[ind], string_labels_true[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

    """for i in range(5):
        axs[i].imshow(image_inputs[i])
        # axs[i].set_title('PL:',predicted_labels[i])"""
    plt.show()


def visualize(image_input):
    image = np.reshape(image_input, (120, 160, 3))
    plt.imshow(image)
    plt.title('hello')
    plt.show()


def main():
    """early_processing('./data/categories.csv', './data/images.csv', './data/labels-little.npy',
                     './data/inputs-little.npy')
    """

    input_arr, input_arr_t, label_arr, label_arr_t = get_data("./data/inputs-little.npy", "./data/labels-little.npy")
    # input_arr, input_arr_t, label_arr, label_arr_t = get_data('d:/DeepLearning/FinalProj/data/inputs.npy','d:/DeepLearning/FinalProj/data/labels.npy')

    print(np.shape(input_arr))
    print(np.shape(label_arr))

    model = Model()
    epoch = 1

    for i in range(epoch):
        loss = train(model, input_arr, label_arr)
        print(i)

    accuracy = test(model, input_arr_t, label_arr_t)

    print("accuracy:")
    print(accuracy)


    total_size = len(input_arr)
    probs = model.call(input_arr)
    print("early")

    print(probs)

    query1 = 55
    query2 = 33
    """
    47 + 8
    25 + 3 or 8
    """

    #40 -? promising
    #25 -> very good 25

    #13
    #21
    visualize_results(input_arr[query1 : (query1 + 3)], probs[query1 : (query1 + 3)],
                      label_arr[query1 : (query1 + 3)])


    visualize_results(input_arr[query2 : (query2 + 3)], probs[query2 : (query2 + 3)],
                      label_arr[query2 : (query2 + 3)])


    return


if __name__ == '__main__':
    main()

""
""""
want sigmoid: range from 0 to 1, each representing probality that that image belongs to the class

softmax: 10 numbers that add up to 1. probability distribution
"""
