"""A model organizes the training of a neural network.

The general structure, and especial the fit method, are similar to the keras
Model class.
"""

#%%
import copy
import numpy as np
import time
import warnings

import natural_bm.backend as B
import natural_bm.callbacks as cbks
from natural_bm.utils import merge_OrderedDicts
from natural_bm.callbacks import CSVLogger

#%%
def check_batches(size, batch_size):     
    """Checks batches on the first epoch to see if any data is missed """
    if np.mod(size, batch_size) > 0:
        warn = 'Batch size does not evenly divide into data. Remainders are ignored.'
        warnings.warn(warn)


#%%
def make_batches(size, batch_size, epoch=None):
    """Returns a list of batch indices (tuples of indices). """

    if epoch in [None, 0]:
        check_batches(size, batch_size)

    nb_batch = int(np.floor(size / float(batch_size)))
    batches = [(i * batch_size, min(size, (i + 1) * batch_size))
               for i in range(0, nb_batch)]

    return batches


#%%
class Model:
    """Class that handles the training of a neural network """
    def __init__(self, nnet, optimizer, trainer):

        self.nnet = nnet
        self.optimizer = optimizer
        self.trainer = trainer

        self.inputs = B.placeholder(shape=(None, self.nnet.layer_size_list[0]), name='x')
        self.loss_fn = trainer.loss_fn()
        loss = self.loss_fn(self.inputs)
        for part in self.nnet.parts:
            for pl in part.losses:
                loss += pl                
        self.loss = loss

        self.trainable_weights = self.nnet.trainable_weights
        self._updates = self.trainer.updates

    @property
    def _train_updates(self):
        training_updates = self.optimizer.get_updates(self.trainable_weights, self.loss)
        updates = merge_OrderedDicts(self._updates, training_updates)
        return updates

    def _make_function(self, index, data, updates, name):
        givens = {self.inputs: data[index]}
        fn = B.function([index],
                        self.loss,
                        updates=updates,
                        givens=givens,
                        name=name)

        return fn

    def _make_train_function(self):
        self.train_function = self._make_function(self.train_index,
                                                  self.train_data,
                                                  self._train_updates,
                                                  'train_function')

    def _make_validation_function(self):
        self.validation_function = self._make_function(self.valid_index,
                                                       self.validation_data,
                                                       self._updates,
                                                       'valid_function')

    def _make_test_function(self):
        self.test_function = self._make_function(self.test_index,
                                                 self.test_data,
                                                 self._updates,
                                                 'test_function')

    def _fit_loop(self,
                  f,
                  out_labels=None,
                  batch_size=100,
                  n_epoch=100,
                  callbacks=None,
                  val_f=None,
                  shuffle=True,
                  callback_metrics=None,
                  initial_epoch=0):
        """Abstract fit function for f.
        Assume that f returns a list, labeled by out_labels.
        
        # Arguments
            f: Backend function returning a list of tensors
            out_labels: list of strings, display names of
                the outputs of `f`
            batch_size: integer batch size
            n_epoch: number of times to iterate over the data
            callbacks: list of callbacks to be called during training
            val_f: Backend function to call for validation
            shuffle: whether to shuffle the data at the beginning of each epoch
            callback_metrics: list of strings, the display names of the metrics
                passed to the callbacks. They should be the
                concatenation of list the display names of the outputs of
                 `f` and the list of display names of the outputs of `f_val`.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
        # Returns
            `History` object.
        """

        time_start = time.time()

        do_validation = False
        n_valid_sample = 0
        if val_f:
            do_validation = True
            n_valid_sample = B.eval(self.validation_data.shape[0])

        index_array = np.arange(self.n_train_sample, dtype='int32')

        self.history = cbks.History()
        # CSVLogger needs to be second to last callback
        # otherwise AIS results are not recorded 
        callbacks = callbacks or []
        index_csv = None
        for i, cb in enumerate(callbacks):
            if isinstance(cb, CSVLogger):
                index_csv = i
        if index_csv is not None:
            cb_csv = callbacks.pop(index_csv)
            callbacks.append(cb_csv)
        callbacks = [cbks.BaseLogger()] + callbacks + [self.history]
        callbacks = cbks.CallbackList(callbacks)
        out_labels = out_labels or []
        callbacks.set_model(self)
        callbacks.set_params({
                            'batch_size': batch_size,
                            'n_epoch': n_epoch,
                            'n_sample': self.n_train_sample,
                            'do_validation': do_validation,
                            'metrics': callback_metrics or [],
                            })

        callbacks.on_train_begin()

        self.stop_training = False

        for epoch in range(initial_epoch, n_epoch):
            callbacks.on_epoch_begin(epoch)

            if shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(self.n_train_sample, batch_size, epoch)
            epoch_logs = {}
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)

                callbacks.on_batch_begin(batch_index, batch_logs)

                # actual training
                outs = f(batch_ids)
                if not isinstance(outs, list):
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                if batch_index == len(batches) - 1:  # last batch
                    # validation
                    if do_validation:
                        val_outs = self._valid_loop(val_f, n_valid_sample,
                                                    batch_size=batch_size)
                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]
                        # same labels assumed
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

            callbacks.on_epoch_end(epoch, epoch_logs)

            if self.stop_training:
                break

        # Tracks the timing of everything except train_end
        # Skips train_end otherwise timing can't be included in summary callback
        fit_total_time = time.time() - time_start
        fit_callback_time = callbacks.cb_time
        self.history.fit_total_time = fit_total_time
        self.history.fit_callback_time = fit_callback_time
        self.history.fit_train_time = fit_total_time - fit_callback_time
        
        callbacks.on_train_end()

        return self.history

    def _valid_loop(self, f, n_sample, batch_size=100):
        """Abstract method to loop over some data in batches.
        
        # Arguments
            f: Backend function returning a list of tensors.
            n_sample: integer of number of samples in data.
            batch_size: integer batch size.
        
        # Returns
            Scalar loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics).
        """

        outs = []
        batches = make_batches(n_sample, batch_size)
        index_array = np.arange(n_sample, dtype='int32')
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            batch_outs = f(batch_ids)
            if isinstance(batch_outs, list):
                if batch_index == 0:
                    for batch_out in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    outs[i] += batch_out * len(batch_ids)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += batch_outs * len(batch_ids)

        for i, out in enumerate(outs):
            outs[i] /= n_sample
        if len(outs) == 1:
            return outs[0]
        return outs

    def fit(self,
            x,
            batch_size=100,
            n_epoch=10,
            callbacks=None,
            validation_data=None,
            shuffle=True,
            initial_epoch=0):
        """Trains the model for a fixed number of epochs (iterations on a dataset).
        
        # Arguments
            x: Theano shared array of training data
            batch_size: integer. Number of samples per gradient update.
            n_epoch: integer, the number of times to iterate
                over the training data arrays.
            callbacks: list of callbacks to be called during training.
            validation_data: Theano shared array of data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
            shuffle: boolean, whether to shuffle the training data
                before each epoch.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
        
        # Returns
            A `History` instance. Its `history` attribute contains
            all information collected during training.
        """
        self.train_data = x
        self.n_train_sample = B.eval(x.shape[0])
        self.validation_data = validation_data

        # makes the generic indices to access data
        self.train_index = B.placeholder(shape=(batch_size,),
                                         dtype=B.intx(), name='train_index')

        # makes the training functions
        self._make_train_function()
        f = self.train_function

        # preps for validation
        out_labels = ['cost']
        if validation_data:
            self.valid_index = B.placeholder(shape=(batch_size,),
                                             dtype=B.intx(), name='valid_index')
            callback_metrics = copy.copy(out_labels) + ['val_' + n for n in out_labels]
            self._make_validation_function()
            val_f = self.validation_function
        else:
            callback_metrics = copy.copy(out_labels)
            val_f = None

        # delegate logic to _fit_loop
        return self._fit_loop(f, out_labels=out_labels,
                              batch_size=batch_size, n_epoch=n_epoch,
                              callbacks=callbacks,
                              val_f=val_f, shuffle=shuffle,
                              callback_metrics=callback_metrics,
                              initial_epoch=initial_epoch)

    def train_on_batch(self, x):
        """Runs a single gradient update on a single batch of data.
        # Arguments
            x: Numpy array of training data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named,
                you can also pass a dictionary
                mapping input names to Numpy arrays.
        # Returns
            Scalar training loss
            (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics).
        """

        # makes the generic indices to access data
        batch_size = B.eval(x.shape)[0]
        self.train_index = B.placeholder(shape=(batch_size,),
                                         dtype=B.intx(), name='train_index')
        self.train_data = x
        index = np.arange(batch_size)

        self._make_train_function()
        outputs = self.train_function(index)

        return outputs

    def predict_on_batch(self, x):
        """Runs a single gradient update on a single batch of data.
        # Arguments
            x: Numpy array of training data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named,
                you can also pass a dictionary
                mapping input names to Numpy arrays.
        # Returns
            Scalar training loss
            (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics).
        """

        # makes the generic indices to access data
        batch_size = B.eval(x.shape)[0]
        self.test_index = B.placeholder(shape=(batch_size,),
                                        dtype=B.intx(), name='test_index')
        self.test_data = x
        index = np.arange(batch_size)

        self._make_test_function()
        outputs = self.test_function(index)

        return outputs
