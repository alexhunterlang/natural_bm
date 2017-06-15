# Natural Boltzmann Machines (Natural_BM)

Natural_BM is a Python library for Boltzmann Machines, which are probabilistic generative deep neural networks trained by unsupervised learning. This library reflects my ongoing research and will be frequently updated.

Currently, Natural_BM can be used to train things like:
* Restricted Boltzmann Machines (RBMs)
* Deep Restricted Boltzmann Machines (DBMs)

Features that will be coming soon include:
* Centered DBMs
* Natural DBMs (my new research development)

## Dependencies
This code is written in Python 3 and has only been tested on versions >= 3.4. Natural_BM requires Theano to perform backend computations.

## About the name:
Neural networks are trained by some form of gradient descent. However, gradient descent has certain pathologies such as depending on the specific parameterization of the weights. The natural gradient tries to remove this dependency by utilizing information about the underlying information metric of the parameters. Centered DBMs are a recent modification to help train neural networks that works by providing an approximation to the natural gradient. My research takes this another step and introduces a new DBM with is an even closer approximation to the natural gradient, hence the name Natural Boltzmann Machines.