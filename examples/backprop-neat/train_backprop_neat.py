from functools import partial
import multiprocessing
import os

import jax
from generate_data import generateCircleData, generateXORData, generateGaussianData, generateSpiralData

import neat
import pickle
import matplotlib.pyplot as plt
from neat.graphs import feed_forward_layers
from neat.six_util import itervalues
import jax.numpy as jnp
from jax import grad, jit, vmap
import visualize

class AggregationFunctionSet(object):
    """Contains aggregation functions and methods to add and retrieve them."""

    def __init__(self):
        self.functions = {}
        self.add('product', lambda x: jnp.product(x, axis=-1))
        self.add('sum', lambda x: jnp.sum(x, axis=-1))
        self.add('max', lambda x: jnp.max(x, axis=-1))
        self.add('min', lambda x: jnp.min(x, axis=-1))
        self.add('maxabs', lambda x: jnp.max(jnp.abs(x), axis=-1))
        self.add('median', lambda x: jnp.median(x, axis=-1))
        self.add('mean', lambda x: jnp.mean(x, axis=-1))

    def add(self, name, function):
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise ValueError("No such aggregation function: {0!r}".format(name))

        return f

    def __getitem__(self, index):
        print("Use get, not indexing ([{!r}]), for aggregation functions".format(index),
                      DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return name in self.functions
    
class ActivationFunctionSet(object):
        def __init__(self):
            self.functions = {}
            self.add('sigmoid', jax.nn.sigmoid)
            self.add('tanh', jax.nn.tanh)
            self.add('sin', jnp.sin)
            self.add('gauss', lambda x: jnp.exp(-5.0 * jnp.max(-3.4, jnp.min(3.4, x))**2))
            self.add('relu', jax.nn.relu)
            self.add('softplus', jax.nn.softplus)
            self.add('identity', lambda x: x)
            self.add('inv', lambda x: 1/x)
            self.add('log', jnp.log)
            self.add('exp', jnp.exp)
            self.add('square', lambda x: x**2)
            self.add('cube', lambda x: x**3)

        def add(self, name, function):
            self.functions[name] = function

        def get(self, name):
            f = self.functions.get(name)
            if f is None:
                raise ValueError("No such activation function: {0!r}".format(name))

            return f

        def is_valid(self, name):
            return name in self.functions


class FeedForwardNetwork:
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = {key: 0.0 for key in inputs + outputs}

    def activate(self, inputs, params):
        w_ix = 0
        r_ix = 0
        b_ix = 0
        if len(self.input_nodes) != inputs.shape[1]:
            raise RuntimeError(f"Expected {len(self.input_nodes)} inputs, got {len(inputs)}")

        for no, v in enumerate(self.input_nodes):
            self.values[v] = inputs[:, no]

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * params[w_ix])
                w_ix += 1
            s = agg_func(jnp.concatenate([node_input.reshape(-1, 1) for node_input in node_inputs], axis=-1))
            self.values[node] = act_func(params[self.biases_start_ix + b_ix] + params[self.responses_start_ix + r_ix] * s)
            r_ix += 1
            b_ix += 1
        for i in self.output_nodes:
            if type(self.values[i]) == float:
                self.values[i] = jnp.ones(inputs.shape[0]) * self.values[i]

        return jnp.concatenate([self.values[i].reshape(-1, 1) for i in self.output_nodes], axis=-1)

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, jnp.array([cg.weight])))
                        node_expr.append("v[{}] * {:.7e})".format(inode, cg.weight))


                ng = genome.nodes[node]
                aggregation_function = AggregationFunctionSet().get(ng.aggregation)
                activation_function = ActivationFunctionSet().get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, jnp.array([ng.bias]), jnp.array([ng.response]), inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)


def get_loss(genome_a, config, data):
    net_a = FeedForwardNetwork.create(genome_a, config)

    # Create an ordering for the parameters, 
    # so that they can be stored in one list
    weights = []
    responses = []
    biases = []
    for node, act_func, agg_func, bias, response, links in net_a.node_evals:
        for i, w in links:
            weights.append(w)
        responses.append(response)
        biases.append(bias)

    net_a.responses_start_ix = len(weights)
    net_a.biases_start_ix = len(responses) + len(weights)
    params = jnp.array(weights + responses + biases)

    inp = jnp.array([[d.x, d.y] for d in data])
    out = jnp.array([d.l for d in data])
    
    def loss_fn(w): # cross-entropy loss
        pred = net_a.activate(inp, w)
        pred = jax.nn.sigmoid(pred)
        return jnp.mean((1- out) * jnp.log(pred + 1e-6) + (out) * jnp.log(1 - pred + 1e-6))
    grad_loss_fn = grad(loss_fn)

    lr = 0.01
    loss_history = []

    for it in range(100):
        params -= lr * grad_loss_fn(params)
        loss_history.append(loss_fn(params))

        if len(loss_history) > 1:
            # decrease lr if loss increasing
            if loss_history[-1] >= loss_history[-2]:
                lr /= 3
            
            if len(loss_history) > 10:
                # increase lr if uniformly decreasing
                if all(l1 > l2 for l1, l2 in zip(loss_history[-11:-1], loss_history[-10:])):
                    lr *= 1.5

                if all(l1 == l2 for l1, l2 in zip(loss_history[-11:-1], loss_history[-10:])):
                    break
    
    # set genomes weights to the learnt weights
    w_ix = 0
    r_ix = 0
    b_ix = 0
    for node, act_func, agg_func, bias, response, links in net_a.node_evals:
        for i, w in links:
            genome_a.connections[i, node].weight = params[w_ix]
            w_ix += 1
        genome_a.nodes[node].response = params[len(weights) + r_ix]
        genome_a.nodes[node].bias = params[len(weights) + len(responses)  + b_ix]
        r_ix += 1
        b_ix += 1

    if isinstance(loss_history[-1].item(), (int, float)):
        return - loss_history[-1].item()
    else:
        return 0.0


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

pop = neat.Population(config)
stats = neat.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.StdOutReporter(True))
data = generateCircleData(100, 0.1)
pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), partial(get_loss, data=data))
scores = []

tournaments = 0
while True:
    winner = pop.run(pe.evaluate, 1)
    print("Iter finished")
    visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.png")
    best_genomes = stats.best_unique_genomes(3)

    for n, g in enumerate(best_genomes):
        name = 'winner-{0}'.format(n)
        with open(name + '.pickle', 'wb') as f:
            pickle.dump(g, f)

        visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
        #visualize.draw_net(config, g, view=False, filename=name + "-net-pruned.gv", prune_unused=True)