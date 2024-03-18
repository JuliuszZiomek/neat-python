"""
Single-pole balancing experiment using a feed-forward neural network.
"""

import argparse
import multiprocessing
import os
import pickle
import random

import cart_pole
import neat
import visualize
import pyglet
pyglet.options["headless"] = True
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import gym.wrappers
import matplotlib.pyplot as plt
import numpy as np

from slimevolleygym import SlimeVolleyEnv

from multiprocessing import Pool

class ParallelEvaluator(object):
    def __init__(self, num_workers, timeout=None, competetive=False, mode=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function_pvp = eval_two_genomes
        self.eval_function_coop = eval_two_genome_coop
        self.eval_function_pve = eval_one_genome_oponent
        self.timeout = timeout
        self.pool = Pool(num_workers)
        self.mode = mode

        assert mode in ['pvp', 'pve', 'coop']

        if mode == 'pvp':
             self.eval_function = eval_two_genomes
        elif mode == 'pve':
             self.eval_function = eval_one_genome_oponent
        else:
             self.eval_function = eval_two_genome_coop

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config):
        jobs = []
        if self.mode in ['pvp']:
            random.shuffle(genomes)
            for genome_a, genome_b in zip(genomes[:len(genomes)//2], genomes[len(genomes)//2:]):
                jobs.append(self.pool.apply_async(self.eval_function, (genome_a[1], genome_b[1], config)))

            # assign the fitness back to each genome

            sum_outputs = 0.0
            for job, genome_a, genome_b in zip(jobs, genomes[:len(genomes)//2], genomes[len(genomes)//2:]):
                output = job.get(timeout=self.timeout)

                if type(output) == tuple and len(output) == 2:
                    genome_a[1].fitness = output[0]
                    genome_b[1].fitness = output[1]
                    sum_outputs += output[0] + output[1]
                else:
                    genome_a[1].fitness = output
                    genome_b[1].fitness = output

                    sum_outputs += output
            
            if len(genomes) % 2 != 0:

                for genome in genomes:
                    if genome[1].fitness is None:
                        genome[1].fitness = sum_outputs / (len(genomes) - 1)

        else:
            for genome in genomes:
                jobs.append(self.pool.apply_async(self.eval_function, (genome[1], config)))

            for job, genome_a in zip(jobs, genomes):
                    output_a = job.get(timeout=self.timeout)
                    genome_a[1].fitness = output_a


runs_per_net = 1
simulation_seconds = 60.0

action_table = [[0, 0, 0], # NOOP
                  [1, 0, 0], # LEFT (forward)
                  [1, 0, 1], # UPLEFT (forward jump)
                  [0, 0, 1], # UP (jump)
                  [0, 1, 1], # UPRIGHT (backward jump)
                  [0, 1, 0]] # RIGHT (backward)
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

# Use the NN network phenotype and the discrete actuator force function.
def eval_two_genomes(genome_a, genome_b, config):
    net_a = neat.nn.FeedForwardNetwork.create(genome_a, config)
    net_b = neat.nn.FeedForwardNetwork.create(genome_b, config)

    fitnesses_a = []
    fitnesses_b = []

    for runs in range(runs_per_net):
        sim = SlimeVolleyEnv()

        # Run the given simulation for up to num_steps time steps.
        fitness_a = 0.0
        fitness_b = 0.0
        observation = sim.reset()
        while True:
            
            action_a = random.choices(action_table, net_a.activate(sim.game.agent_right.getObservation()))[0]
            action_b = random.choices(action_table, net_b.activate(sim.game.agent_left.getObservation()))[0]

            # Apply action to the simulated 
            observation, reward, terminated, info = sim.step(action_a,action_b)

            if sim.game.ball.isColliding(sim.game.agent_right):
                    fitness_a += 1
            if sim.game.ball.isColliding(sim.game.agent_left):
                    fitness_b += 1

            if terminated:
                break

        fitnesses_a.append(fitness_a)
        fitnesses_b.append(fitness_b)

    # The genome's fitness is its worst performance across all runs.
    mean_a, mean_b = sum(fitnesses_a) / len(fitnesses_a), sum(fitnesses_b) / len(fitnesses_b)
    
    return mean_a, mean_b

def eval_two_genome_coop(genome_a, config):
    net_a = neat.nn.FeedForwardNetwork.create(genome_a, config)
    net_b = neat.nn.FeedForwardNetwork.create(genome_a, config)

    fitnesses_a = []
    fitnesses_b = []

    for runs in range(runs_per_net):
        sim = SlimeVolleyEnv()

        # Run the given simulation for up to num_steps time steps.
        fitness_a = 0.0
        fitness_b = 0.0
        observation = sim.reset()
        while True:
            
            action_a = random.choices(action_table, net_a.activate(sim.game.agent_right.getObservation()))[0]
            action_b = random.choices(action_table, net_b.activate(sim.game.agent_left.getObservation()))[0]

            # Apply action to the simulated cart-pole
            observation, reward, terminated, info = sim.step(action_a,action_b)

            if sim.game.ball.isColliding(sim.game.agent_right):
                    fitness_a += 1
            if sim.game.ball.isColliding(sim.game.agent_left):
                    fitness_b += 1

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.

            if terminated:
                break

        fitnesses_a.append(fitness_a)
        fitnesses_b.append(fitness_b)

    # The genome's fitness is its worst performance across all runs.
    mean_a, mean_b = sum(fitnesses_a) / len(fitnesses_a), sum(fitnesses_b) / len(fitnesses_b)
    
    return (mean_a + mean_b)


def eval_one_genome_oponent(genome_a, config):
    net_a = neat.nn.FeedForwardNetwork.create(genome_a, config)

    fitnesses_a = []
    fitnesses_b = []

    for runs in range(runs_per_net):
        sim = SlimeVolleyEnv()

        # Run the given simulation for up to num_steps time steps.
        fitness_a = 0.0

        observation = sim.reset()
        while True:
            
            action_a = action_table[np.argmax(net_a.activate(sim.game.agent_right.getObservation()))]
            

            # Apply action to the simulated cart-pole
            observation, reward, terminated, info = sim.step(action_a)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.

            fitness_a += reward

            if terminated:
                break

        fitnesses_a.append(fitness_a)

    # The genome's fitness is its worst performance across all runs.
    return sum(fitnesses_a) / len(fitnesses_a)

def run(mode):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = ParallelEvaluator(multiprocessing.cpu_count(), mode= mode)
    scores = []
    tournaments = 0
    while True:
        winner = pop.run(pe.evaluate, 1)
        tournaments += len(pop.population)
        visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.png")

        scores.append(eval_one_genome_oponent(winner, config))

        if len(scores) > 10:
            scores_smooth = np.convolve(np.array(scores), np.ones(10)/10, mode="valid")

            plt.clf()
            plt.plot(scores_smooth)
            plt.savefig("scores.png")
            plt.clf()


        print("Total tournaments:", tournaments)
        # Save the winner.
        with open('winner-feedforward', 'wb') as f:
            pickle.dump(winner, f)

    print(winner)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    args = parser.parse_args()

    run(args.mode)
