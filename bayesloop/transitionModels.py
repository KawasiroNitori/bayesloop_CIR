#!/usr/bin/env python
"""
Transition models refer to stochastic or deterministic models that describe how the time-varying parameter values of a
given time series model change from one time step to another. The transition model can thus be compared to the state
transition matrix of Hidden Markov models. However, instead of explicitly stating transition probabilities for all
possible states, a transformation is defined that alters the distribution of the model parameters in one time step
according to the transition model. This altered distribution is subsequently used as a prior distribution in the next
time step.
"""

from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.interpolation import shift
from scipy.stats import multivariate_normal
from scipy.stats import ncx2  # noncentral chi-square
from collections.abc import Iterable
from copy import deepcopy
from .exceptions import ConfigurationError, PostProcessingError

try:
    from inspect import getargspec
except ImportError:
    from inspect import getfullargspec as getargspec

class TransitionModel:
    """
    Parent class for transition models. All transition models inherit from this class. It is currently only used to
    identify transition models as such.
    """


class Static(TransitionModel):
    """
    Constant parameters over time. This trivial model assumes no change of parameter values over time.
    """
    def __init__(self):
        self.study = None
        self.latticeConstant = None
        self.hyperParameterNames = []
        self.hyperParameterValues = []
        self.prior = None
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Static/constant parameter values'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior(ndarray): Parameter distribution from current time step
            t(int): integer time step

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """
        return posterior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class GaussianRandomWalk(TransitionModel):
    """
    Gaussian parameter fluctuations. This model assumes that parameter changes are Gaussian-distributed. The standard
    deviation can be set individually for each model parameter.

    Args:
        name(str): custom name of the hyper-parameter sigma
        value(float, list, tuple, ndarray): standard deviation(s) of the Gaussian random walk for target parameter
        target(str): parameter name of the observation model to apply transition model to
        prior: hyper-prior distribution that may be passed as a(lambda) function, as a SymPy random variable, or
            directly as a Numpy array with probability values for each hyper-parameter value
    """
    def __init__(self, name='sigma', value=None, target=None, prior=None):
        if isinstance(value, (list, tuple)):  # Online study expects Numpy array of values
            value = np.array(value)

        self.study = None
        self.latticeConstant = None
        self.hyperParameterNames = [name]
        self.hyperParameterValues = [value]
        self.prior = prior
        self.selectedParameter = target
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

        if target is None:
            raise ConfigurationError('No parameter set for transition model "GaussianRandomWalk"')

    def __str__(self):
        return 'Gaussian random walk'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior(ndarray): Parameter distribution from current time step
            t(int): integer time step

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """
        axisToTransform = self.study.observationModel.parameterNames.index(self.selectedParameter)
        normedSigma = self.hyperParameterValues[0]/self.latticeConstant[axisToTransform]
        
        if normedSigma > 0.:
            newPrior = gaussian_filter1d(posterior, normedSigma, axis=axisToTransform)
        else:
            newPrior = posterior.copy()
        
        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class AlphaStableRandomWalk(TransitionModel):
    """
    Parameter changes follow alpha-stable distribution. This model assumes that parameter changes are distributed
    according to the symmetric alpha-stable distribution. For each parameter, two hyper-parameters can be set: the
    width of the distribution (c) and the shape (alpha).

    Args:
        name1(str): custom name of the hyper-parameter c
        value1(float, list, tuple, ndarray): width(s) of the distribution (c >= 0).
        name2(str): custom name of the hyper-parameter alpha
        value2(float, list, tuple, ndarray): shape(s) of the distribution (0 < alpha <= 2).
        target(str): parameter name of the observation model to apply transition model to
        prior: list of two hyper-prior distributions, where each may be passed as a(lambda) function, as a SymPy random
            variable, or directly as a Numpy array with probability values for each hyper-parameter value
    """
    def __init__(self, name1='c', value1=None, name2='alpha', value2=None, target=None, prior=(None, None)):
        if isinstance(value1, (list, tuple)):
            value1 = np.array(value1)
        if isinstance(value2, (list, tuple)):
            value2 = np.array(value2)

        self.study = None
        self.latticeConstant = None
        self.hyperParameterNames = [name1, name2]
        self.hyperParameterValues = [value1, value2]
        self.prior = prior
        self.selectedParameter = target
        self.kernel = None
        self.kernelParameters = None
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

        if target is None:
            raise ConfigurationError('No parameter set for transition model "AlphaStableRandomWalk"')

    def __str__(self):
        return 'Alpha-stable random walk'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior(ndarray): Parameter distribution from current time step
            t(int): integer time step

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """

        # if hyper-parameter values have changed, a new convolution kernel needs to be created
        if not self.kernelParameters == self.hyperParameterValues:
            normedC = []
            for lc in self.latticeConstant:
                normedC.append(self.hyperParameterValues[0] / lc)
            alpha = [self.hyperParameterValues[1]] * len(normedC)

            axisToTransform = self.study.observationModel.parameterNames.index(self.selectedParameter)
            selectedC = normedC[axisToTransform]
            normedC = [0.]*len(normedC)
            normedC[axisToTransform] = selectedC

            self.kernel = self.createKernel(normedC[0], alpha[0], 0)
            for i, (a, c) in enumerate(zip(alpha[1:], normedC[1:])):
                self.kernel *= self.createKernel(c, a, i+1)

            self.kernel = self.kernel.T
            self.kernelParameters = deepcopy(self.hyperParameterValues)

        newPrior = self.convolve(posterior)
        newPrior /= np.sum(newPrior)
        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)

    def createKernel(self, c, alpha, axis):
        """
        Create alpha-stable distribution on a grid as a kernel for convolution.

        Args:
            c(float): Scale parameter.
            alpha(float): Tail parameter (alpha = 1: Cauchy, alpha = 2: Gauss)
            axis(int): Axis along which the distribution is defined, for 2D-Kernels

        Returns:
            ndarray: kernel
        """
        gs = self.study.gridSize
        if len(gs) == 2:
            if axis == 1:
                l1 = gs[1]
                l2 = gs[0]
            elif axis == 0:
                l1 = gs[0]
                l2 = gs[1]
            else:
                raise ConfigurationError('Transformation axis must either be 0 or 1.')
        elif len(gs) == 1:
            l1 = gs[0]
            l2 = 0
            axis = 0
        else:
            raise ConfigurationError('Parameter grid must either be 1- or 2-dimensional.')

        kernel_fft = np.exp(-np.abs(c*np.linspace(0, np.pi, int(3*l1/2+1)))**alpha)
        kernel = np.fft.irfft(kernel_fft)
        kernel = np.roll(kernel, int(3*l1/2-1))

        if len(gs) == 2:
            kernel = np.array([kernel]*(3*l2))

        if axis == 1:
            return kernel.T
        elif axis == 0:
            return kernel

    def convolve(self, distribution):
        """
        Convolves distribution with alpha-stable kernel.

        Args:
            distribution(ndarray): Discrete probability distribution to convolve.

        Returns:
            ndarray: convolution
        """
        gs = np.array(self.study.gridSize)
        padded_distribution = np.zeros(3*np.array(gs))
        if len(gs) == 2:
            padded_distribution[gs[0]:2*gs[0], gs[1]:2*gs[1]] = distribution
        elif len(gs) == 1:
            padded_distribution[gs[0]:2*gs[0]] = distribution

        padded_convolution = fftconvolve(padded_distribution, self.kernel, mode='same')
        if len(gs) == 2:
            convolution = padded_convolution[gs[0]:2*gs[0], gs[1]:2*gs[1]]
        elif len(gs) == 1:
            convolution = padded_convolution[gs[0]:2*gs[0]]

        return convolution


class ChangePoint(TransitionModel):
    """
    Abrupt parameter change at a specified time step. Parameter values are allowed to change only at a single point in
    time, right after a specified time step (Hyper-parameter tChange). Note that a uniform parameter distribution is
    used at this time step to achieve this "reset" of parameter values.

    Args:
        name(str): custom name of the hyper-parameter tChange
        value(int, list, tuple, ndarray): Integer value(s) of the time step of the change point
        prior: hyper-prior distribution that may be passed as a(lambda) function, as a SymPy random variable, or
            directly as a Numpy array with probability values for each hyper-parameter value
    """
    def __init__(self, name='tChange', value=None, prior=None):
        if isinstance(value, (list, tuple)):
            value = np.array(value)

        self.study = None
        self.latticeConstant = None
        self.hyperParameterNames = [name]
        self.hyperParameterValues = [value]
        self.prior = prior
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Change-point'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior(ndarray): Parameter distribution from current time step
            t(int): integer time step

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """
        if t == self.hyperParameterValues[0]:
            # check if custom prior is used by observation model
            if hasattr(self.study.observationModel.prior, '__call__'):
                prior = self.study.observationModel.prior(*self.study.grid)
            elif isinstance(self.study.observationModel.prior, np.ndarray):
                prior = deepcopy(self.study.observationModel.prior)
            else:
                prior = np.ones(self.study.gridSize)  # flat prior

            # normalize prior (necessary in case an improper prior is used)
            prior /= np.sum(prior)
            prior *= np.prod(self.study.latticeConstant)
            return prior
        else:
            return posterior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class Independent(TransitionModel):
    """
    Observations are treated as independent. This transition model restores the prior distribution for the parameters
    at each time step, effectively assuming independent observations.

    Note:
        Mostly used with an instance of OnlineStudy.
    """
    def __init__(self):
        self.study = None
        self.latticeConstant = None
        self.hyperParameterNames = []
        self.hyperParameterValues = []
        self.prior = None
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Independent observations model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior(ndarray): Parameter distribution from current time step
            t(int): integer time step

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """
        # check if custom prior is used by observation model
        if hasattr(self.study.observationModel.prior, '__call__'):
            prior = self.study.observationModel.prior(*self.study.grid)
        elif isinstance(self.study.observationModel.prior, np.ndarray):
            prior = deepcopy(self.study.observationModel.prior)
        else:
            prior = np.ones(self.study.gridSize)  # flat prior

        # normalize prior (necessary in case an improper prior is used)
        prior /= np.sum(prior)
        return prior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class RegimeSwitch(TransitionModel):
    """
    Small probability for a parameter jump in each time step. In case the number of change-points in a given data set
    is unknown, the regime-switching model may help to identify potential abrupt changes in parameter values. At each
    time step, all parameter values within the set boundaries are assigned a minimal probability density of being
    realized in the next time step, effectively allowing abrupt parameter changes at every time step.

    Args:
        name(str): custom name of the hyper-parameter log10pMin
        value(float, list, tuple, ndarray): Minimal probability density (log10 value) that is assigned to every
            parameter value
        prior: hyper-prior distribution that may be passed as a(lambda) function, as a SymPy random variable, or
            directly as a Numpy array with probability values for each hyper-parameter value
    """
    def __init__(self, name='log10pMin', value=None, prior=None):
        if isinstance(value, (list, tuple)):
            value = np.array(value)

        self.study = None
        self.latticeConstant = None
        self.hyperParameterNames = [name]
        self.hyperParameterValues = [value]
        self.prior = prior
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Regime-switching model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Parameters:
            posterior(ndarray): Parameter distribution from current time step
            t(int): integer time step

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """
        newPrior = posterior.copy()
        limit = (10.**self.hyperParameterValues[0])*np.prod(self.latticeConstant)  # convert prob. density to prob.
        newPrior[newPrior < limit] = limit

        # transformation above violates proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class NotEqual(TransitionModel):
    """
    Unlikely parameter values are preferred in the next time step. Assumes an "inverse" parameter distribution at each
    new time step. The new prior is derived by substracting the posterior probability values from their maximal value
    and subsequently re-normalizing. To assure that no parameter value is set to zero probability, one may specify a
    minimal probability for all parameter values. This transition model is mostly used in instances of OnlineStudy to
    detect time step when parameter distributions change significantly.

    Args:
        name(str): custom name of the hyper-parameter log10pMin
        value(float, list, tuple, ndarray): Log10-value of the minimal probability that is set to all possible
            parameter values of the inverted parameter distribution
        prior: hyper-prior distribution that may be passed as a(lambda) function, as a SymPy random variable, or
            directly as a Numpy array with probability values for each hyper-parameter value

    Note:
        Mostly used with an instance of OnlineStudy.
    """
    def __init__(self, name='log10pMin', value=None, prior=None):
        if isinstance(value, (list, tuple)):
            value = np.array(value)

        self.study = None
        self.latticeConstant = None
        self.hyperParameterNames = [name]
        self.hyperParameterValues = [value]
        self.prior = prior
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Not-Equal model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Parameters:
            posterior(ndarray): Parameter distribution from current time step
            t(int): integer time step

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """
        newPrior = posterior.copy()
        limit = (10**self.hyperParameterValues[0])*np.prod(self.latticeConstant)  # convert prob. density to prob.

        newPrior = np.amax(newPrior) - newPrior
        newPrior /= np.sum(newPrior)
        newPrior[newPrior < limit] = limit

        # transformation above violates proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class Deterministic(TransitionModel):
    """
    Deterministic parameter variations. Given a function with time as the first argument and further  keyword-arguments
    as hyper-parameters, plus the name of a parameter of the observation model that is supposed to follow this function
    over time, this transition model shifts the parameter distribution accordingly. Note that these models are entirely
    deterministic, as the hyper-parameter values are entered by the user. However, the hyper-parameter distributions can
    be inferred using a Hyper-study or can be optimized using the 'optimize' method of the Study class.

    Args:
        function(function): A function that takes the time as its first argument and further takes keyword-arguments
            that correspond to the hyper-parameters of the transition model which the function defines.
        target(str): The observation model parameter that is manipulated according to the function defined above.
        prior: List of hyper-prior distributions (one for each hyper-parameter), where each may be passed as a(lambda)
            function, as a SymPy random variable, or directly as a Numpy array with probability values for each
            hyper-parameter value

    Example:
    ::
        def quadratic(t, a=0, b=0):
            return a*(t**2) + b*t

        S = bl.Study()
        ...
        S.setObservationModel(bl.om.WhiteNoise('std', bl.oint(0, 3, 1000)))
        S.setTransitionModel(bl.tm.Deterministic(quadratic, target='signal'))
    """
    def __init__(self, function=None, target=None, prior=None):
        self.study = None
        self.latticeConstant = None
        self.function = function
        self.selectedParameter = target
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

        if target is None:
            raise ConfigurationError('No parameter set for transition model "Deterministic"')

        # create ordered dictionary of hyper-parameters from keyword-arguments of function
        argspec = getargspec(self.function)

        # only keyword arguments are allowed
        if not len(argspec.args) == len(argspec.defaults)+1:
            raise ConfigurationError('Function to define deterministic transition model can only contain one '
                                     'non-keyword argument (time; first argument) and keyword-arguments '
                                     '(hyper-parameters) with default values.')

        # define hyper-parameters of transition model
        self.hyperParameterNames = []
        self.hyperParameterValues = []
        for arg, default in zip(argspec.args[1:], argspec.defaults):
            if isinstance(default, (list, tuple)):
                default = np.array(default)
            self.hyperParameterNames.append(arg)
            self.hyperParameterValues.append(default)

        if prior is None:
            # provide as many "None"-priors as there are hyper-parameters
            self.prior = [None]*len(argspec.defaults)
        else:
            # if list of priors is supplied, check length
            if isinstance(prior, Iterable):
                if not len(prior) == len(argspec.defaults):
                    raise ConfigurationError('{} priors are defined for transition model "{}", but model contains {}'
                                             'hyper-parameters.'
                                             .format(len(prior), self.function.__name__, len(argspec.defaults)))
            # if single prior is defined, pack it in a list
            else:
                self.prior = [prior]

    def __str__(self):
        return 'Deterministic model ({})'.format(self.function.__name__)

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior(ndarray): Parameter distribution from current time step
            t(int, float): time stamp (integer time index by default)

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """
        # determine grid axis along which to shift the distribution
        axisToTransform = self.study.observationModel.parameterNames.index(self.selectedParameter)

        # compute offset to shift parameter grid
        params = {name: value for (name, value) in zip(self.hyperParameterNames, self.hyperParameterValues)}
        ftp1 = self.function(t + 1 - self.tOffset, **params)
        ft = self.function(t-self.tOffset, **params)
        d = ftp1-ft

        # normalize offset with respect to lattice constant of parameter grid
        d /= self.latticeConstant[axisToTransform]

        # build list for all axes of parameter grid (setting only the selected axis to a non-zero value)
        dAll = [0] * len(self.latticeConstant)
        dAll[axisToTransform] = d

        # shift interpolated version of distribution
        newPrior = shift(posterior, dAll, order=3, mode='nearest')

        # transformation above may violate proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        # determine grid axis along which to shift the distribution
        axisToTransform = self.study.observationModel.parameterNames.index(self.selectedParameter)

        # compute offset to shift parameter grid
        params = {name: value for (name, value) in zip(self.hyperParameterNames, self.hyperParameterValues)}
        ftm1 = self.function(t - 1 - self.tOffset, **params)
        ft = self.function(t - self.tOffset, **params)
        d = ftm1 - ft

        # normalize offset with respect to lattice constant of parameter grid
        d /= self.latticeConstant[axisToTransform]

        # build list for all axes of parameter grid (setting only the selected axis to a non-zero value)
        dAll = [0] * len(self.latticeConstant)
        dAll[axisToTransform] = d

        # shift interpolated version of distribution
        newPrior = shift(posterior, dAll, order=3, mode='nearest')

        # transformation above may violate proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior


class CombinedTransitionModel(TransitionModel):
    """
    Different models act at the same time. This class allows to combine different transition models to be
    able to explore more complex parameter dynamics. All sub-models are passed to this class as arguments on
    initialization. Note that a different order of the sub-models can result in different parameter dynamics.

    Args:
        *args: Sequence of transition models
    """
    def __init__(self, *args):
        self.study = None
        self.latticeConstant = None
        self.models = args
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

        # check if any sub-model is a break-point and raise error if so
        if np.any([str(arg) == 'Break-point' for arg in args]):
            raise ConfigurationError('The "BreakPoint" transition model can only be used with the '
                                     '"SerialTransitionModel" class.')

    def __str__(self):
        return 'Combined transition model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior(ndarray): Parameter distribution from current time step
            t(int): integer time step

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """
        newPrior = posterior.copy()

        for m in self.models:
            m.latticeConstant = self.latticeConstant  # latticeConstant needs to be propagated to sub-models
            m.study = self.study  # study needs to be propagated to sub-models
            m.tOffset = self.tOffset
            newPrior = m.computeForwardPrior(newPrior, t)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        newPrior = posterior.copy()

        for m in self.models:
            m.latticeConstant = self.latticeConstant
            m.study = self.study
            m.tOffset = self.tOffset
            newPrior = m.computeBackwardPrior(newPrior, t)

        return newPrior


class SerialTransitionModel(TransitionModel):
    """
    Different models act at different time steps. To model fundamental changes in parameter dynamics, different
    transition models can be serially coupled. Depending on the time step, a corresponding sub-model is chosen to
    compute the new prior distribution from the posterior distribution. If a break-point lies in between two transition
    models, the parameter values do not change abruptly at the time step of the break-point, whereas a change-point not
    only changes the transition model, but also allows the parameters to change (the parameter distribution is re-set to
    the prior distribution).

    Args:
        *args: Sequence of transition models and break-points/change-points (for n models, n-1
            break-points/change-points have to be provided)

    Example:
    ::
        T = bl.tm.SerialTransitionModel(bl.tm.Static(),
                                        bl.tm.BreakPoint('t_1', 50),
                                        bl.tm.RegimeSwitch('log10pMin', -7),
                                        bl.tm.BreakPoint('t_2', 100),
                                        bl.tm.GaussianRandomWalk('sigma', 0.2, target='x'))

    In this example, parameters are assumed to be constant until 't_1' (time step 50), followed by a regime-switching-
    process until 't_2' (time step 100). Finally, we assume Gaussian parameter fluctuations for parameter 'x' until the
    last time step. Note that models and time steps do not necessarily have to be passed in an alternating way.
    """
    def __init__(self, *args):
        self.study = None
        self.latticeConstant = None

        # determine time steps of structural breaks and other sub-models
        self.hyperParameterNames = []
        self.hyperParameterValues = []
        self.prior = []
        self.models = []
        self.changePointMask = []
        for arg in args:
            if str(arg) == 'Break-point':
                self.hyperParameterNames.append(arg.name)
                self.prior.append(arg.prior)

                # exclude 'all' case, conversion to list is needed to avoid future warning about element-wise comparison
                if isinstance(arg.value, str) and arg.value == 'all':  # 'all' is passed without type change
                    self.hyperParameterValues.append(arg.value)
                elif isinstance(arg.value, Iterable):  # convert list/tuple in numpy array
                    self.hyperParameterValues.append(np.array(arg.value))
                else:  # single values are passed without type change
                    self.hyperParameterValues.append(arg.value)
                self.changePointMask.append(0)
            elif str(arg) == 'Change-point':
                name = arg.hyperParameterNames[0]
                value = arg.hyperParameterValues[0]
                self.hyperParameterNames.append(name)
                self.prior.append(arg.prior)

                # exclude 'all' case, conversion to list is needed to avoid future warning about element-wise comparison
                if isinstance(value, str) and value == 'all':  # 'all' is passed without type change
                    self.hyperParameterValues.append(value)
                elif isinstance(value, Iterable):  # convert list/tuple in numpy array
                    self.hyperParameterValues.append(np.array(value))
                else:  # single values are passed without type change
                    self.hyperParameterValues.append(value)
                self.changePointMask.append(1)
            else:  # sub-model
                self.models.append(arg)

        self.changePointMask = np.array(self.changePointMask).astype(bool)

        # check: break times have to be passed in monotonically increasing order
        # since multiple values can be passed for one break-point at init, we check first values only
        firstValues = []
        for v in self.hyperParameterValues:
            if isinstance(v, str) and v == 'all':
                firstValues.append(v)
            elif isinstance(v, Iterable):
                firstValues.append(v[0])
            else:
                firstValues.append(v)

        if not all(x < y if not ((isinstance(x, str) and x == 'all') or (isinstance(y, str) and y == 'all')) else True
                   for x, y in zip(firstValues, firstValues[1:])):
            raise ConfigurationError('Time steps for structural breaks and/or change-points have to be passed in '
                                     'monotonically increasing order.')

        # check: n models require n-1 break times
        if not (len(self.models)-1 == len(self.hyperParameterValues)):
            raise ConfigurationError('Wrong number of structural breaks/change-points and models. For n models, n-1 '
                                     'structural breaks/change-points are required.')

    def __str__(self):
        return 'Serial transition model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior(ndarray): Parameter distribution from current time step
            t(int): integer time step

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """
        # the index of the model to choose at time t is given by the number of break times <= t
        modelIndex = np.sum(np.array(self.hyperParameterValues) <= t)

        self.models[modelIndex].latticeConstant = self.latticeConstant  # latticeConstant needs to be propagated
        self.models[modelIndex].study = self.study  # study needs to be propagated
        self.models[modelIndex].tOffset = self.hyperParameterValues[modelIndex-1] if modelIndex > 0 else 0
        newPrior = self.models[modelIndex].computeForwardPrior(posterior, t)
        newPrior = self._forwardChangePointCheck(newPrior, t)
        return newPrior

    def computeBackwardPrior(self, posterior, t):
        # the index of the model to choose at time t is given by the number of break times <= t
        modelIndex = np.sum(np.array(self.hyperParameterValues) <= t-1)

        self.models[modelIndex].latticeConstant = self.latticeConstant  # latticeConstant needs to be propagated
        self.models[modelIndex].study = self.study  # study needs to be propagated
        self.models[modelIndex].tOffset = self.hyperParameterValues[modelIndex-1] if modelIndex > 0 else 0
        newPrior = self.models[modelIndex].computeBackwardPrior(posterior, t)
        newPrior = self._backwardChangePointCheck(newPrior, t)
        return newPrior

    def _forwardChangePointCheck(self, posterior, t):
        """
        This function checks if a change-point is set to the current time step and replaces the posterior with the prior
        distribution, just like the change-point transition model. This allows to use change-points in serial transition
        models.

        Args:
            posterior(ndarray): Parameter distribution from current time step
            t(int): integer time step

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """
        if t in np.array(self.hyperParameterValues)[self.changePointMask]:
            # check if custom prior is used by observation model
            if hasattr(self.study.observationModel.prior, '__call__'):
                prior = self.study.observationModel.prior(*self.study.grid)
            elif isinstance(self.study.observationModel.prior, np.ndarray):
                prior = deepcopy(self.study.observationModel.prior)
            else:
                prior = np.ones(self.study.gridSize)  # flat prior

            # normalize prior (necessary in case an improper prior is used)
            prior /= np.sum(prior)
            prior *= np.prod(self.study.latticeConstant)
            return prior
        else:
            return posterior

    def _backwardChangePointCheck(self, posterior, t):
        return self._forwardChangePointCheck(posterior, t - 1)


class BreakPoint(TransitionModel):
    """
    Break-point. This class can only be used to specify break-point within a SerialTransitionModel instance.

    Args:
        name(str): custom name of the hyper-parameter tBreak
        value(int, list, tuple, ndarray): Value(s) of the time step(s) of the break point
        prior: hyper-prior distribution that may be passed as a(lambda) function, as a SymPy random variable, or
            directly as a Numpy array with probability values for each hyper-parameter value
    """
    def __init__(self, name='tBreak', value=None, prior=None):
        if isinstance(value, (list, tuple)):
            value = np.array(value)

        self.name = name
        self.value = value
        self.prior = prior

    def __str__(self):
        return 'Break-point'


class BivariateRandomWalk(TransitionModel):
    """
    Correlated Gaussian parameter fluctuations. This model assumes that parameter changes follow a bivariate Gaussian
    distribution.
    """
    def __init__(self, name1='sigma1', value1=None,
                 name2='sigma2', value2=None,
                 name3='rho', value3=None,
                 prior=(None, None, None)):

        if isinstance(value1, (list, tuple)):
            value1 = np.array(value1)
        if isinstance(value2, (list, tuple)):
            value2 = np.array(value2)
        if isinstance(value3, (list, tuple)):
            value2 = np.array(value3)

        self.study = None
        self.latticeConstant = None
        self.hyperParameterNames = [name1, name2, name3]
        self.hyperParameterValues = [value1, value2, value3]
        self.prior = prior
        self.kernel = None
        self.kernelParameters = None
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Bivariate random walk'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior(ndarray): Parameter distribution from current time step
            t(int): integer time step

        Returns:
            ndarray: Prior parameter distribution for subsequent time step
        """

        # if hyper-parameter values have changed, a new convolution kernel needs to be created
        if not self.kernelParameters == self.hyperParameterValues:
            normedSigma1 = self.hyperParameterValues[0] / self.latticeConstant[0]
            normedSigma2 = self.hyperParameterValues[1] / self.latticeConstant[1]

            self.kernel = self.createKernel(normedSigma1, normedSigma2, self.hyperParameterValues[2])
            self.kernelParameters = deepcopy(self.hyperParameterValues)

        newPrior = convolve2d(posterior, self.kernel, mode='same')
        newPrior /= np.sum(newPrior)
        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)

    @staticmethod
    def createKernel(sigma1, sigma2, rho):
        rv = multivariate_normal(cov=[[sigma1 ** 2., rho * sigma1 * sigma2],
                                      [rho * sigma1 * sigma2, sigma2 ** 2.]])

        x = np.arange(-3 * np.ceil(sigma1), 3 * np.ceil(sigma1) + 1)
        y = np.arange(-3 * np.ceil(sigma2), 3 * np.ceil(sigma2) + 1)

        xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

        kernel = rv.pdf(np.array([xv, yv]).T).T
        kernel /= np.sum(kernel)
        return kernel




# ---------------------------------------------------------------------------
# CIR-based transition model (Ref by ChatGPT 5; Only for 1 target, like DD)
# ---------------------------------------------------------------------------

class CIRTransition(TransitionModel):
    """
    Parameter changes follow the stationary CIR diffusion:
        dX_t = kappa (theta - X_t) dt + sigma sqrt(X_t) dW_t   with X_t >= 0.

    One-step transition density is known in closed form via a scaled
    noncentral chi-square; we discretize it on the target grid and update
    the prior by direct quadrature (matrix multiply) along the selected axis.

    Hyper-parameters (names are user-definable via name1..name4):
      - kappa  (>0): mean-reversion speed
      - theta  (>0): long-run mean (stationary mean)
      - sigma  (>0): volatility-of-volatility
      - dt     (>0): time increment between observations (default 1.0)

    Notes
    -----
    * Only the 'target' parameter axis is transformed; other axes are left
      unchanged (delta kernel), matching the pattern of other TMs.
    * Target grid should be non-negative (CIR support).
    * Forward update uses K[j,i] ≈ p(x_{t+1}=grid[j] | x_t=grid[i]) * lattice.
    * Backward update uses the time-reversed kernel B = D_pi K^T D_pi^{-1},
      where pi is the discretized stationary density of the CIR on the grid.
    """

    def __init__(self,
                 name1='kappa', value1=1.0,
                 name2='theta', value2=1.0,
                 name3='sigma', value3=0.5,
                 name4='dt',    value4=1.0,
                 target=None, prior=(None, None, None, None)):

        # Accept array-valued inputs (for HyperStudy/Optimize) like other TMs
        if isinstance(value1, (list, tuple)): value1 = np.array(value1, dtype=float)
        if isinstance(value2, (list, tuple)): value2 = np.array(value2, dtype=float)
        if isinstance(value3, (list, tuple)): value3 = np.array(value3, dtype=float)
        if isinstance(value4, (list, tuple)): value4 = np.array(value4, dtype=float)

        self.study = None
        self.latticeConstant = None
        self.hyperParameterNames  = [name1, name2, name3, name4]
        self.hyperParameterValues = [value1, value2, value3, value4]
        self.prior = prior
        self.selectedParameter = target
        self.tOffset = 0

        # Caches
        self._kernel_matrix = None     # forward matrix K (N,N)
        self._kernel_params = None     # [kappa, theta, sigma, dt]
        self._cached_axis    = None
        self._cached_grid    = None
        self._cached_lattice = None

        self._backward_matrix = None   # backward matrix B (N,N)
        self._pi              = None   # stationary weights on grid
        self._pi_params       = None   # [kappa, theta, sigma]
        self._pi_grid         = None

        if target is None:
            raise ConfigurationError('No parameter set for transition model "CIRTransition"')

    def __str__(self):
        return 'CIR transition'

    # --------------------------- CIR math helpers ---------------------------

    @staticmethod
    def _cir_pdf(x_next, x_cur, kappa, theta, sigma, dt):
        """
        Vectorized CIR one-step transition pdf at x_next given x_cur.
        Uses the noncentral chi-square mapping:
            phi = exp(-kappa*dt)
            nu  = 4*kappa*theta/sigma^2
            c   = sigma^2*(1-phi)/(4*kappa)
            lam = 4*kappa*phi*x_cur / (sigma^2*(1-phi))
            X_{t+dt} = c * Y,  Y ~ ncx2(nu, lam)
            f_X(x | x_cur) = (1/c) * f_ncx2(x/c; df=nu, nc=lam)
        Shapes:
            x_next: (N,1), x_cur: (1,N) -> output (N,N)
        """
        # numeric guards
        x_next = np.maximum(x_next, 0.0)
        x_cur  = np.maximum(x_cur,  0.0)

        phi = np.exp(-kappa * dt)
        one_minus_phi = np.maximum(1.0 - phi, 1e-14)

        nu  = 4.0 * kappa * theta / (sigma * sigma)
        c   = (sigma * sigma) * one_minus_phi / (4.0 * kappa)
        lam = (4.0 * kappa * phi * x_cur) / ((sigma * sigma) * one_minus_phi)

        z = x_next / c
        dens = ncx2.pdf(z, df=nu, nc=lam) / c
        return dens

    @staticmethod
    def _cir_stationary_unnormalized(x, kappa, theta, sigma):
        """
        Unnormalized stationary density of CIR (Gamma with mean theta):
            shape a = 2*kappa*theta/sigma^2, scale b = sigma^2/(2*kappa).
        We return an unnormalized value and discretely normalize later.
        """
        x = np.maximum(x, 0.0)
        a = 2.0 * kappa * theta / (sigma * sigma)     # shape
        b = (sigma * sigma) / (2.0 * kappa)           # scale
        # unnormalized gamma pdf ~ x^{a-1} exp(-x/b)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            g = np.power(x, np.maximum(a - 1.0, 0.0)) * np.exp(-x / max(b, 1e-300))
        g[~np.isfinite(g)] = 0.0
        return g

    # ------------------------ Kernel / matrix builders ----------------------

    def _resolve_axis_grid_lattice(self):
        """
        Return (axis, grid_1d, lattice) for the selected parameter.
        Works for both Study and HyperStudy instances.
        """
        # Which axis does the TM act on?
        axis = self.study.observationModel.parameterNames.index(self.selectedParameter)
    
        # Try several places where bayesloop stores the 1D grids
        grids = None
        # 1) Newer/typical: Study/HyperStudy keeps the grids as .marginalGrid (list of 1D arrays)
        if hasattr(self.study, 'marginalGrid') and self.study.marginalGrid:
            grids = self.study.marginalGrid
        # 2) Some versions keep them on the observation model
        elif hasattr(self.study, 'observationModel') and hasattr(self.study.observationModel, 'parameterValues'):
            grids = self.study.observationModel.parameterValues
        # 3) Fallback (older code paths you might have): parameterGrids
        elif hasattr(self.study, 'parameterGrids'):
            grids = self.study.parameterGrids
    
        if grids is None:
            raise ConfigurationError(
                'CIRTransition could not find parameter grids on the Study/HyperStudy. '
                'Expected one of: study.marginalGrid, observationModel.parameterValues, study.parameterGrids.'
            )
    
        grid = np.asarray(grids[axis], dtype=float)
    
        # Lattice constants are already injected by bayesloop when the TM is attached
        if self.latticeConstant is None:
            raise ConfigurationError('latticeConstant not set on transition model.')
        lattice = float(self.latticeConstant[axis])
    
        # (Optional) Guard for CIR support
        if grid.min() < 0:
            raise ConfigurationError(
                f'CIRTransition requires a non-negative grid for "{self.selectedParameter}". Got min={grid.min():.3g}.'
            )
        return axis, grid, lattice

    def _maybe_rebuild_forward_matrix(self):
        """Build (or reuse) K[j,i] ≈ p(x_next=grid[j] | x_cur=grid[i]) * lattice."""
        kappa, theta, sigma, dt = [float(v) for v in self.hyperParameterValues]
        axis, grid, lattice = self._resolve_axis_grid_lattice()
        new_params = (kappa, theta, sigma, dt)
        need = (
            self._kernel_matrix is None or
            # self._kernel_params != [kappa, theta, sigma, dt] or
            any( not np.isclose(a, b, rtol=1e-12, atol=1e-15) for a, b in zip(self._kernel_params, new_params)) or
            self._cached_axis    != axis or
            not np.isclose(self._cached_lattice, lattice, rtol=1e-12, atol=1e-15) or
            self._cached_grid is None or
            self._cached_grid.shape != grid.shape or
            not np.allclose(self._cached_grid, grid)
        )
        if not need:
            return

        N = grid.size
        x_next = grid[:, None]   # (N,1)
        x_cur  = grid[None, :]   # (1,N)
        pdf = self._cir_pdf(x_next, x_cur, kappa, theta, sigma, dt)  # (N,N)

        K = pdf * lattice  # Riemann factor: columns approximate integral weights

        # Column-normalize for numerical stability (preserve mass)
        col_sums = K.sum(axis=0, keepdims=True)
        bad = (col_sums <= 0)
        if np.any(bad):
            # If a column collapsed numerically, fall back to a delta at nearest cell
            for i in np.where(bad[0])[0]:
                K[:, i] = 0.0
                K[min(i, N-1), i] = 1.0
        else:
            K /= col_sums

        self._kernel_matrix = K
        self._kernel_params = [kappa, theta, sigma, dt]
        self._cached_axis    = axis
        self._cached_grid    = grid.copy()
        self._cached_lattice = lattice

        # Invalidate backward cache (depends on K and pi)
        self._backward_matrix = None

    def _maybe_rebuild_backward_matrix(self):
        """
        Build (or reuse) the time-reversed kernel:
            B = D_pi K^T D_pi^{-1},
        where pi is the discretized stationary density on the target grid.
        """
        self._maybe_rebuild_forward_matrix()
        K = self._kernel_matrix
        kappa, theta, sigma, dt = [float(v) for v in self.hyperParameterValues]
        axis, grid, lattice = self._cached_axis, self._cached_grid, self._cached_lattice
        new_params_B = (kappa, theta, sigma, dt)
        
        # (Re)compute stationary weights pi on the grid if needed
        need_pi = (
            self._pi is None or
            any( not np.isclose(a, b, rtol=1e-12, atol=1e-15) for a, b in zip(self._pi_params, new_params_B)) or
            # self._pi_params != [kappa, theta, sigma] or
            self._pi_grid is None or
            not np.array_equal(self._pi_grid, grid)
        )
        if need_pi:
            pi = self._cir_stationary_unnormalized(grid, kappa, theta, sigma)
            Z = (pi * lattice).sum()
            if Z > 0:
                pi = pi / Z
            else:
                # degenerate safeguard: concentrate mass near theta
                j = np.argmin(np.abs(grid - theta))
                pi = np.zeros_like(grid)
                pi[j] = 1.0
            self._pi = pi
            self._pi_params = [kappa, theta, sigma]
            self._pi_grid = grid.copy()

        pi = self._pi
        # Form B = D_pi K^T D_pi^{-1} without building dense diagonals explicitly
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_pi = np.where(pi > 0, 1.0 / pi, 0.0)
        B = (K.T * pi[np.newaxis, :]) * inv_pi[:, np.newaxis]

        # Column-normalize (numerical guard)
        col_sums = B.sum(axis=0, keepdims=True)
        bad = (col_sums <= 0)
        if np.any(bad):
            for j in np.where(bad[0])[0]:
                B[:, j] = 0.0
                B[j, j] = 1.0
        else:
            B /= col_sums

        self._backward_matrix = B

    # ------------------------------ main API ------------------------------- #

    def computeForwardPrior(self, posterior, t):
        """
        Prior at t+1 from posterior at t via CIR transition along the target axis:
            prior_axis = K @ posterior_axis,
        applied to the last axis (after moving/reshaping like other TMs).
        """
        self._maybe_rebuild_forward_matrix()
        K = self._kernel_matrix
        axis = self._cached_axis

        arr = np.asarray(posterior, dtype=float)
        # move target axis to end
        if axis != arr.ndim - 1:
            arr = np.moveaxis(arr, axis, -1)
        leading = arr.shape[:-1]
        N = arr.shape[-1]
        arr2 = arr.reshape(-1, N)  # (M, N)

        prior2 = arr2 @ K.T        # (M, N)
        prior = prior2.reshape(*leading, N)
        if axis != prior.ndim - 1:
            prior = np.moveaxis(prior, -1, axis)

        s = prior.sum()
        if s > 0.0:
            prior /= s
        return prior

    def computeBackwardPrior(self, posterior, t):
        """
        Prior at t-1 from posterior at t using the time-reversed CIR kernel:
            prior_axis = B @ posterior_axis,
        where B = D_pi K^T D_pi^{-1} ensures detailed balance with the
        CIR stationary density pi on the discretized grid.
        """
        self._maybe_rebuild_backward_matrix()
        B = self._backward_matrix
        axis = self._cached_axis

        arr = np.asarray(posterior, dtype=float)
        if axis != arr.ndim - 1:
            arr = np.moveaxis(arr, axis, -1)
        leading = arr.shape[:-1]
        N = arr.shape[-1]
        arr2 = arr.reshape(-1, N)

        prior2 = arr2 @ B.T
        prior = prior2.reshape(*leading, N)
        if axis != prior.ndim - 1:
            prior = np.moveaxis(prior, -1, axis)

        s = prior.sum()
        if s > 0.0:
            prior /= s
        return prior

