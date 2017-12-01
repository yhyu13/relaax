import numpy as np
import tensorflow as tf
import collections

from relaax.common.algorithms.lib import graph


def get_session():
    """Returns recently made Tensorflow session"""
    return tf.get_default_session()


def is_placeholder(x):
    return type(x) is tf.Tensor and len(x.op.inputs) == 0


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplemented()

    def make_feed_dict(data):
        """Given data input it to the placeholder(s)."""
        raise NotImplemented()


class PlacholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}


def function(inputs, outputs, updates=None, givens=None):
    """Just like Theano function. Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be fed to the input's placeholders and produces the values of the expressions
    in outputs.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).

    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()

            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    Parameters
    ----------
    inputs: [tf.placeholder or TfInput]
        list of input arguments
    outputs: [tf.Variable] or tf.Variable
        list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        for inpt in inputs:
            if not issubclass(type(inpt), TfInput):
                assert len(inpt.op.inputs) == 0,\
                    "inputs should all be placeholders of baselines.common.TfInput"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens
        self.check_nan = check_nan

    def _feed_input(self, feed_dict, inpt, value):
        if issubclass(type(inpt), TfInput):
            feed_dict.update(inpt.make_feed_dict(value))
        elif is_placeholder(inpt):
            feed_dict[inpt] = value

    def __call__(self, session, *args, **kwargs):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update the kwargs
        kwargs_passed_inpt_names = set()
        for inpt in self.inputs[len(args):]:
            inpt_name = inpt.name.split(':')[0]
            inpt_name = inpt_name.split('/')[-1]
            assert inpt_name not in kwargs_passed_inpt_names, \
                "this function has two arguments with the same name \"{}\"," \
                " so kwargs cannot be used.".format(inpt_name)
            if inpt_name in kwargs:
                kwargs_passed_inpt_names.add(inpt_name)
                self._feed_input(feed_dict, inpt, kwargs.pop(inpt_name))
            else:
                assert inpt in self.givens, "Missing argument " + inpt_name
        assert len(kwargs) == 0, "Function got extra arguments " + str(list(kwargs.keys()))
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        if session is not None:
            sess = session
        else:
            sess = get_session()
        results = sess.run(self.outputs_update, feed_dict=feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")
        return results


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=()):
        # self._sum = graph.GetVariable(dtype=tf.float64,
        #                               shape=shape,
        #                               initializer=tf.constant_initializer(0.0),
        #                               name="runningsum", trainable=False)
        self._sum = graph.Variable(np.zeros(shape=shape, dtype=np.float64),
                                   dtype=np.float64, name="runningsum", trainable=False).node

        sumsq = np.empty(shape=shape, dtype=np.float64)
        sumsq.fill(epsilon)
        self._sumsq = graph.Variable(sumsq,
                                     dtype=np.float64, name="runningsumsq", trainable=False).node

        self._count = graph.Variable(epsilon,
                                     dtype=np.float64, name="counter", trainable=False).node

        self.shape = shape

        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt(tf.maximum(tf.to_float(self._sumsq / self._count) - tf.square(self.mean), 1e-2))

        newsum = tf.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
        newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float64, name='var')
        newcount = tf.placeholder(shape=[], dtype=tf.float64, name='count')
        self.incfiltparams = function([newsum, newsumsq, newcount], [],
                                      updates=[tf.assign_add(self._sum, newsum),
                                               tf.assign_add(self._sumsq, newsumsq),
                                               tf.assign_add(self._count, newcount)])

    def update(self, session, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n * 2 + 1, 'float64')
        addvec = np.concatenate(
            [x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)], dtype='float64')])
        # np.sum(addvec, totalvec)
        totalvec += addvec
        self.incfiltparams(session, totalvec[0:n].reshape(self.shape),
                           totalvec[n:2 * n].reshape(self.shape), totalvec[2 * n])
