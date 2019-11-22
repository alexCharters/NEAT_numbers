from __future__ import print_function
import input_data
import os
import neat
import visualize

images = []
labels = []

def initialize_in_out():
    '''load images and their respective expected output (may want to perform convolution with keras,
but that may not be neccesary)'''
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print(data.train.images)
    images = data.train.images
    labels = data.train.labels


def eval_genomes(genomes, config):
    '''fitness function used in the neat algorithm'''
    for genome_id, genome in genomes:
        genome.fitness = 9.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for image, label in zip(images, labels):
            output = net.activate(image)
            genome.fitness -= (output[0] - label[0]) ** 2

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
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for image, label in zip(images, labels):
        output = winner_net.activate(image)
        print("input {!r}, expected output {!r}, got {!r}".format(image, label, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.


    #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'configFile')
    initialize_in_out()
    run(config_path)