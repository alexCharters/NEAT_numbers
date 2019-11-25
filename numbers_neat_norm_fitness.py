from __future__ import print_function
import tensorflow as tf
import input_data
import os
import neat
import visualize
import statistics
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import to_categorical

images = []
labels = []
last_genome = None


def initialize_in_out():
    '''load images and their respective expected output (may want to perform convolution with keras,
but that may not be neccesary)'''
    global images
    global labels
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # print(data.train.images)
    print(x_train[0]/255)


    #print((np.array(images[0]).reshape((28, 28), order='A')*255).astype(int))
    im = Image.fromarray(x_train[0])
    im.save("yeeter.png")

    onehot_encoded = to_categorical(y_train)

    images = x_train/255
    labels = onehot_encoded


def eval_genome(genome, config):
    '''fitness function used in the neat algorithm'''
    global images
    global labels
    global last_genome
    num_pics = 300
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # if(last_genome is not None):
    #     print(genome.distance(last_genome, config.genome_config))
    counter = 0
    errors = []
    for image, label in zip(images, labels):
        if(counter < num_pics):
            temp = 10
            output = net.activate(image.reshape(784))
            if np.argmax(output) == np.argmax(label):
                temp = 1;
            else:
                temp = 0;
            errors.append(temp)
        counter += 1
    last_genome = genome
    return statistics.mean(errors)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    pe = neat.ThreadedEvaluator(100, eval_genome)
    winner = p.run(pe.evaluate, 100)
    pe.stop()

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    count = 0
    for image, label in zip(images, labels):
        if(count < 30):
            output = winner_net.activate(image.reshape(784))
            print("expected output {!r}, got {!r}".format(np.argmax(label), np.argmax(output)))
            count += 1

    node_names = {0: 'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    pe = neat.ThreadedEvaluator(4, eval_genome)
    winner = p.run(pe.evaluate, 10)
    pe.stop()


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.

    #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'configFile_strict')
    initialize_in_out()
    run(config_path)
