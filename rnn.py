#!/usr/bin/env python

import logging
import numpy as np
import theano

from argparse import ArgumentParser
from theano import tensor
from skimage.transform import rotate

from blocks.bricks import MLP, Identity, Tanh, Logistic, Rectifier
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.algorithms import GradientDescent, Adam, CompositeRule, StepClipping
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from fuel.streams import DataStream
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from fuel.transformers import Mapping, Flatten
from blocks.graph import ComputationGraph
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop

from blocks_contrib.extensions import DataStreamMonitoringAndSaving
floatX = theano.config.floatX


mnist = MNIST('train', sources=['features'])
handle = mnist.open()
data = mnist.get_data(handle, slice(0, 50000))[0]
means = data.reshape((50000, 784)).mean(axis=0)


def autocorrentropy2(X, ksize=np.inf):
    b, t, d = X.shape
    V = np.zeros((b, t, d))
    for i in range(b):
        for j in range(t):
            if ksize in (np.inf, 0., np.nan):
                V[i, j, :] = (X[i, :(t-j), :] * X[i, j:, :]).sum(axis=0) / (t-j)
            else:
                V[i, j, :] = np.exp((-ksize * (X[i, :(t-j), :]-X[i, j:, :])**2)).sum(axis=0) / (t-j)
    return V


def allrotations(image, N):
    angles = np.linspace(0, 350, N)
    R = np.zeros((N, 784))
    for i in xrange(N):
        img = rotate(image, angles[i])
        R[i] = img.flatten()
    return R


def _meanize(n_steps):
    def func(data):
        newfirst = data[0] - means[None, :]
        Rval = np.zeros((n_steps, newfirst.shape[0], newfirst.shape[1]))
        for i, sample in enumerate(newfirst):
            Rval[:, i, :] = allrotations(sample.reshape((28, 28)), n_steps)
            Rval[10:, i, :] = 0.
            num = np.random.randint(0, n_steps, size=1)
            Rval[:, i, :] = np.roll(Rval[:, i, :], num, axis=0)
        Rval = Rval.astype(floatX)
        return (Rval[1:], Rval[:-1])
    return func


def main(save_to, num_epochs):
    batch_size = 128
    dim = 100
    n_steps = 20
    i2h1 = MLP([Identity()], [784, dim], biases_init=Constant(0.), weights_init=IsotropicGaussian(.001))
    h2o1 = MLP([Rectifier(), Logistic()], [dim, dim, 10],
               biases_init=Constant(0.), weights_init=IsotropicGaussian(.001))
    rec1 = SimpleRecurrent(dim=dim, activation=Tanh(), weights_init=Orthogonal())
    i2h1.initialize()
    h2o1.initialize()
    rec1.initialize()

    x = tensor.tensor3('features')
    y = tensor.lmatrix('targets')

    preproc = i2h1.apply(x)
    h1 = rec1.apply(preproc)
    x_hat = h2o1.apply(h1)
    cost = tensor.nnet.binary_crossentropy(x_hat, y).mean()
    # cost = CategoricalCrossEntropy().apply(y.flatten(), probs)
    cost.name = 'final_cost'

    cg = ComputationGraph([cost, ])

    mnist_train = MNIST("train", subset=slice(0, 50000))
    mnist_valid = MNIST("train", subset=slice(50000, 60000))
    mnist_test = MNIST("test")
    trainstream = Mapping(Flatten(DataStream(mnist_train,
                          iteration_scheme=SequentialScheme(50000, batch_size))),
                          _meanize(n_steps))
    validstream = Mapping(Flatten(DataStream(mnist_valid,
                                             iteration_scheme=SequentialScheme(10000,
                                                                               batch_size))),
                          _meanize(n_steps))
    teststream = Mapping(Flatten(DataStream(mnist_test,
                                            iteration_scheme=SequentialScheme(10000,
                                                                              batch_size))),
                         _meanize(n_steps))

    algorithm = GradientDescent(
        cost=cost, params=cg.parameters,
        step_rule=CompositeRule([Adam(), StepClipping(100)]))
    main_loop = MainLoop(
        algorithm,
        trainstream,
        extensions=[Timing(),
                    FinishAfter(after_n_epochs=num_epochs),
                    DataStreamMonitoring(
                        [cost, ],
                        validstream,
                        prefix="test"),
                    DataStreamMonitoringAndSaving(
                    [cost, ],
                    teststream,
                    [i2h1, h2o1, rec1],
                    'best_'+save_to+'.pkl',
                    cost_name=cost.name,
                    after_epoch=True,
                    prefix='valid'
                    ),
                    TrainingDataMonitoring(
                        [cost,
                         aggregation.mean(algorithm.total_gradient_norm)],
                        prefix="train",
                        after_epoch=True),
                    # Plot(
                    #     save_to,
                    #     channels=[
                    #         ['test_final_cost',
                    #          'test_misclassificationrate_apply_error_rate'],
                    #         ['train_total_gradient_norm']]),
                    Printing()])
    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="mnist.pkl", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    parser.add_argument("--flag", type=bool, default=False,
                        help="Use autocorrentropy or not?")
    parser.add_argument("--ksize", type=float, default=np.inf,
                        help="Kernel size of the autocorrentropy function.")
    args = parser.parse_args()
    main(args.save_to, args.num_epochs)
