import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial as fact
import seaborn as sns
from tqdm.notebook import trange
import torch as pt

from .errors import log_perc_error

class Smoother:
    """ Core state smoother class
    
    Attributes
    ----------
    actuals: PyTorch Tensor
        A post-processed copy of data_frame, includes derivative dimensions
        Shape: Observations, Columns, Dims
    coefs: dict
        Contains learned coefficients for smoothing, values in pt.Tensor form.
    columns: Pandas Index
        Copy of original data_frame columns.
    dims: int
        Number of derivative dimensions to smooth by. More will get a better
        smoothing fit, but will make the fit more subject to data variance.
    error_func: function
        The function to use to calculate the observation-level error. Must be
        PyTorch Tensor compatible.
    horizon: float
        How many of observations to smooth out by. Larger numbers mean a weaker
        fit, but a smoother result. Can be decimal valued.
    index: Pandas Index
        Copy of original data_frame index.
    learn_seed: bool
        Set to True to enable updating of the seed data. Not recommend, as it
        adds significant learning tine, and makes learning unstable.
    loss_rcd: list
        Record of loss values at each iteration.
    state: PyTorch Tensor
        A copy of the final smoothed data.
        Shape: Observations, Columns, Dims
    verbose: bool
        Set to True to allow for verbose expression of class functions, including
        TQDM progress bars.
        
    Hidden Attributes
    -----------------
    alpha: float
        Gradient update smoothing factor. Higher values updates gradient more per epoc.
    beta: float
        Gradient^2 update smoothing factor. Higher values updates gradient^2 more per epoc.
    data: PyTorch Tensor
        A Tensor copy of the original data_frame.
        Shape: Observations, Columns
    enforce_zero:
        Set to True enforces the first derivative dimension to never go below a zero value.
    flow_clamp: float
        Constrains the flow coefficient as to not cause inf/nan values in the sigmoid.
    grad: dict
        Contains current smoothed gradient in pt.Tensor form. Keys match self.coefs.
    grad_sq: dict
        Contains current smoothed gradient^2 in pt.Tensor form. Keys match self.coefs.
    incrementor: PyTorch Tensor
        Tensor for predicting one time step from any state.
        Shape: Dims, Dims
    learn_clamp: float
        Minimum value for learn step denominator, prevents div0 errors.
    seed: PyTorch Tensor
        A Tensor copy of the seed data.
        Shape: Columns
    """
    
    def __init__(self, data_frame, dims, horizon, error_func=log_perc_error,
                 seed_data=None, coef_targets=0, learn_seed=False,
                 alpha=0.2, beta=0.1, learn_clamp=1e-16, flow_clamp=32,
                 enforce_zero=False, verbose=False, tqdm_leave=True,
                ):
        """ Initializer for Smoother class
        
        Parameters
        ----------
        data_frame: Pandas DataFrame
            Contains the data to be smoothed.
            Shape: Observations, Columns
        dims: int (restriction: >= 1)
            Number of derivative dimensions to smooth by. More will get a better
            smoothing fit, but will make the fit more subject to data variance.
        horizon: float (restriction: > 0)
            How many of observations to smooth out by. Larger numbers mean a weaker
            fit, but a smoother result. Can be decimal valued.
        error_func: function (optional, default: log_perc_error)
            The function to use to calculate the observation-level error. Must be
            PyTorch Tensor compatible.
        seed_data: Pandas DataFrame (optional, default: takes from first row of data_frame)
            Initial values to seed the smoothing algorithm with. If not provided will take
            from the first row of data_frame, and fill derivative dimensions with 0's.
            Shape: Dims, Columns
        coef_targets: float (optional, default: 0)
            Seed value for the learned part of the flow coefficent.
        learn_seed: bool (optional, default: False)
            Set to True to enable updating of the seed data. Not recommend, as it
            adds significant learning tine, and makes learning unstable.
        alpha: float (optional, default: 0.2, restriction: > 0, < 1)
            Gradient update smoothing factor. Higher values updates gradient more per epoc.
        beta: float (optional, default: 0.1, restriction: > 0, < 1)
            Gradient^2 update smoothing factor.Higher values updates gradient^2 more per epoc.
        learn_clamp: float (optional, default: 1e-16, restriction: > 0)
            Minimum value for learn step denominator, prevents div0 errors.
        flow_clamp: float (optional, default: 32, restriction: > 0)
            Constrains the flow coefficient as to not cause inf/nan values in the sigmoid.
        enforce_zero: bool (optional, default False)
            Set to True enforces the first derivative dimension to never go below a zero value.
        verbose: bool (optional, default: False)
            Set to True to allow for verbose expression of class functions, including
            TQDM progress bars.
        tqdm_leave: bool (optional, default: True)
            Set to True to have the TQDM progress bar stay when complete. Only relevant
            if verbose is also set to True.
        """
        
        self.dims = dims
        self.horizon = horizon
        self.error_func = error_func
        self.learn_seed = learn_seed
        self._enforce_zero = enforce_zero
        self.verbose = verbose
        self.tqdm_leave = tqdm_leave
        self.loss_rcd = []
        
        self._alpha = alpha
        self._beta = beta
        self._flow_clamp = flow_clamp
        self._learn_clamp = learn_clamp
        self._grad = {}
        self._grad_sq = {}
        
        self._init_frame(data_frame, seed_data, self.dims)
        self._init_incrementor(self.dims)
        self._init_coefs(self.dims, self.columns.size, coef_targets, learn_seed)
        
    def _init_frame(self, data_frame, seed_data, dims):
        """ Initializes the provided data
        
        Will take the provided data_frame and convert it to a PyTorch Tensor.
        
        Will do the same with seed_data, however if no seed_data is available,
        the first row of data_frame will be removed from data_frame and be used
        as seed_data, with zero-fills for the derivative dimensions. If seed_data
        is provided but with only one dimension of data, missing dimensions
        will be zero-filled.
        
        All values are directly assigned as class attributes.
        
        Parameters
        ----------
        data_frame: Pandas DataFrame
            Contains the data to be smoothed.
            Shape: Observations, Columns
        seed_data: Pandas DataFrame
            Initial values to seed the smoothing algorithm with. If not provided will take
            from the first row of data_frame, and fill derivative dimensions with 0's.
            Shape: Dims, Columns
        dims: int (restriction: >= 1)
            Number of derivative dimensions to smooth by. More will get a better
            smoothing fit, but will make the fit more subject to data variance.
            
        Returns
        -------
        None
        
        Attribute Assignments
        ---------------------
        columns: Pandas Index
            Copy of original data_frame columns.
        data: PyTorch Tensor
            A Tensor copy of the original data_frame.
            Shape: Observations, Columns
        index: Pandas Index
            Copy of original data_frame index.
        seed: PyTorch Tensor
            A Tensor copy of the seed data.
            Shape: Columns
        """
        
        if seed_data is None:
            if self.verbose: print('No seed data provided, using first row of data_frame.')
            seed_data = data_frame.iloc[0,:]
            data_frame = data_frame.iloc[1:,:]
            
        self.index = data_frame.index
        self.columns = data_frame.columns
        self._data = pt.from_numpy(data_frame.values).type(pt.Tensor)
        
        if len(seed_data.shape) == 1:
            if self.verbose: print('Seed data one dimensional, generating {} 0-value derivative dimensions.'.format(dims-1))
            self._seed = [pt.from_numpy(seed_data[self.columns].values).type(pt.Tensor)]
            for d in range(1, dims): self._seed.append(pt.zeros(self.columns.size))
            self._seed = pt.stack(self._seed, dim=0)
        else:
            self._seed = pt.from_numpy(seed_data[self.columns].values).type(pt.Tensor)
            
    def _init_incrementor(self, dims):
        """ Constructs the incrementor
        
        The incrementor is used to predict off of the current state. The current
        state can be matrix multiplied against the incrementor to increment
        by one time step. Multiple applications of the incrementor can be made to
        increment additional days.
        
        Parameters
        ----------
        dims: int (restriction: >= 1)
            Number of derivative dimensions to smooth by. More will get a better
            smoothing fit, but will make the fit more subject to data variance.
            
        Returns
        -------
        None
        
        Attribute Assignments
        ---------------------
        incrementor: PyTorch Tensor
            Tensor for predicting one time step from any state.
            Shape: Dims, Dims
        """
        
        self._incrementor = pt.from_numpy(np.stack([
            np.pad(1 / fact(np.arange(dims-dim)), (dim,0), 'constant', constant_values=0)
            for dim in range(dims)
        ])).type(pt.Tensor)
        
    def _init_coefs(self, dims, col_size, coef_targets, learn_seed):
        """ Initializes the learning coefficents
        
        Flow is the learned component of the flow value, which determines how much
        a current observation influences the smoothing.
        
        Seed is only present if learn_seed is True, it is the seed values that the
        smoothing process is started from.
        
        Parameters
        ----------
        dims: int (restriction: >= 1)
            Number of derivative dimensions to smooth by. More will get a better
            smoothing fit, but will make the fit more subject to data variance.
        col_size: int (restriction: >= 1)
            Number of columns in the data.
        coef_targets: float (optional, default: 0)
            Seed value for the learned part of the flow coefficent.
        learn_seed: bool (optional, default: False)
            Set to True to enable updating of the seed data. Not recommend, as it
            adds significant learning tine, and makes learning unstable.
            
        Returns
        -------
        None
        
        Attribute Assignments
        ---------------------
        coefs: dict
            Contains learned coefficients for smoothing, values in pt.Tensor form.
        """
        
        self.coefs = {
            'flow': pt.empty(dims, col_size).fill_(coef_targets),
        }
        if learn_seed: self.coefs['seed'] = self._seed
            
    def learn(self, epocs, learn_rate=1e-3):
        """ Starts the learning process
        
        This method is the master controller for training the model, performing
        the multiple iterations that optimize the coefficients.
        
        Parameters
        ----------
        epocs: int (restriction: >= 1)
            Number of epocs to learn from.
        learn_rate: float (optional, default: 1e-3, restriction: > 0)
            Multiplier on the final learning steps made for gradient update.
            Smaller values mean slower, but more reliable learning. Larger
            values mean faster, but more unstable learning. It is *strongly*
            recommended to never go over 1e-1 with this.
        
        Returns
        -------
        None
        
        Attribute Assignments
        ---------------------
        actuals: PyTorch Tensor
            A post-processed copy of data_frame, includes derivative dimensions
            Shape: Observations, Columns, Dims
        state: PyTorch Tensor
            A copy of the final smoothed data.
            Shape: Observations, Columns, Dims
        """
        
        t = trange(epocs, leave=self.tqdm_leave) if self.verbose else range(epocs)
        for epoc in t:
            self.actuals, self.state = self._epoc(learn_rate, t)
        
    def _epoc(self, learn_rate, t):
        """ Runs a single epoc of training
        
        Parameters
        ----------
        learn_rate: float (restriction: > 0)
            Multiplier on the final learning steps made for gradient update.
            Smaller values mean slower, but more reliable learning. Larger
            values mean faster, but more unstable learning. It is *strongly*
            recommended to never go over 1e-1 with this.
        t: TQDM iterator
            Used for updating the postfix of the TQDM iterator if verbose is True.
            
        Returns
        -------
        actuals: PyTorch Tensor
            A post-processed copy of data_frame, includes derivative dimensions
            Shape: Observations, Columns, Dims
        state: PyTorch Tensor
            A copy of the final smoothed data.
            Shape: Observations, Columns, Dims
        
        Attribute Assignments
        ---------------------
        None
        """
        
        for coef in self.coefs.values(): coef.requires_grad = True
        flow = self._init_flow(self.coefs['flow'], self.horizon, self._flow_clamp)

        actuals = [self.coefs['seed'] if self.learn_seed else self._seed]
        state = [self.coefs['seed'] if self.learn_seed else self._seed]
        error = []
        for actual in self._data:
            new_actual = self._form_actual(actual, self.dims, actuals[-1])
            new_state, new_error = self._smooth_data(new_actual, self._incrementor, state[-1], flow, self.error_func, self._enforce_zero)
            actuals.append(new_actual)
            state.append(new_state)
            error.append(new_error)

        actuals = pt.stack(actuals[1:], dim=0).transpose(1,2)
        state = pt.stack(state[1:], dim=0).transpose(1,2)
        error = pt.stack(error, dim=0)
        loss = error.mean(dim=0).mean()
        self._update_loss_rcd(loss, t)

        gradients = self._calc_grad(loss, self.coefs)
        self._update_coefs(gradients, learn_rate)
        
        return actuals, state
    
    @staticmethod
    def _init_flow(target, horizon, clamp):
        """ Initializes the flow smoothing values
        
        Flow dictates how much of the new state information overwrites the predicted state information.
        Flow's formula is organised around the idea that at each state update there is a % of
        the actual states over a horizon window that will be present in the current observation.
        
        The 'proper' formula for flow is actually $F = 1 - (1 - T)^{1 / H}$, where T = target and H = horizon.
        A simplified version is used here, with the one-minus's removed, as gradient descent doesn't
        need to maintain logical integrity with the original meaning, and it computes faster.
        
        Parameters
        ----------
        target: PyTorch Tensor
            Controls the % of flow over the horizon period
            Shape = Dims, Columns
        horizon: float
            How many of observations to smooth out by. Larger numbers mean a weaker
            fit, but a smoother result.
        clamp: float (restriction: > 0)
            Constrains the flow coefficient as to not cause inf/nan values in the sigmoid.
            
        Returns
        -------
        flow: PyTorch Tensor
            Contains the flow values that will be used in smoothing.
            Shape: Dims, Columns
        
        Attribute Assignments
        ---------------------
        None
        """
        
        flow = pt.sigmoid(target.clamp(-clamp, clamp))**(1/horizon)
        return flow
                
    @staticmethod
    def _form_actual(actual, dims, prior_actual):
        """ Forms derivative dimensions for unsmoothed data
        
        Constructs additional derivatives based on differencing the prior derivative's
        change between the current values and prior values.
        
        Parameters
        ----------
        actual: PyTorch Tensor
            A row of unsmoothed data from the data class attribute
            Shape: Columns
        dims: int (restriction: >= 1)
            Number of derivative dimensions to smooth by. More will get a better
            smoothing fit, but will make the fit more subject to data variance.
        prior_actual: PyTorch Tensor
            A copy of the previous unsmoothed record, with derivative dimensions.
            Shape: Dims, Columns
            
        Returns
        -------
        dimmed_actual: PyTorch Tensor
            Unsmoothed data with derivative dimensions calculated.
            Shape: Dims, Columns
            
        Attribute Assignments
        ---------------------
        None
        """
        
        dimmed_actual = [actual]
        for d in range(1, dims): dimmed_actual.append(dimmed_actual[-1] - prior_actual[d-1,:])
        dimmed_actual = pt.stack(dimmed_actual, dim=0)
        return dimmed_actual
    
    @staticmethod
    def _smooth_data(actual, incrementor, prior_state, flow, error_func, enforce_zero):
        """ Performs smoothing overation
        
        Performs a prediction from the prior state, which is used in calculating both
        the current smoothed values, and the learning error.
        
        Parameters
        ----------
        actual: PyTorch Tensor
            Unsmoothed data with derivative dimensions.
            Shape: Dims, Columns
        incrementor: PyTorch Tensor
            Tensor for predicting one time step from any state.
            Shape: Dims, Dims
        prior_state: PyTorch Tensor
            Unsmoothed previous timestep data with derivative dimensions.
            Shape: Dims, Columns
        flow: PyTorch Tensor
            Contains the flow values that will be used in smoothing.
            Shape: Dims, Columns
        error_func: function
            The function to use to calculate the observation-level error. Must be
            PyTorch Tensor compatible.
        enforce_zero: bool
            Set to True enforces the first derivative dimension to never go below a zero value.
            
        Returns
        -------
        new_state: PyTorch Tensor
            The current smoothed state
            Shape: Dims, Columns
        new_error: PyTorch Tensor
            The prediction error between the incremented prior state and the actual
            unsmoothed values.
            Shape: Dims, Columns
            
        Attribute Assignments
        ---------------------
        None
        """
        
        predict = incrementor @ prior_state
        new_state = ((1-flow) * actual) + (flow * predict)
        if enforce_zero: new_state[0,:] = new_state[0,:].clamp(0, np.inf)
        new_error = error_func(predict[0,:], actual[0,:])
        return new_state, new_error
    
    @staticmethod
    def _calc_grad(loss, coefs):
        """ Calculates gradient of loss with respect to coefs
        
        Parses through active coefficients, including them in PyTorch's AutoGrad
        calculation, and reforms them in the original coefs dictionary form.
        
        Parameters
        ----------
        loss: PyTorch Tensor
            Loss value to calculate gradient with
            Shape: Scalar
        coefs: dict
            Contains learned coefficients for smoothing, values in pt.Tensor form.
            
        Returns
        -------
        gradients: dict
            Contains the gradients of loss, with respect to coefs, within dict values.
            
        Attribute Assignments
        ---------------------
        None
        """
        
        gradients = np.array([(key, coef) for key, coef in coefs.items()])
        gradients[:,1] = pt.autograd.grad(loss, gradients[:,1])
        gradients = {key:grad for key, grad in gradients}
        return gradients
    
    def _update_loss_rcd(self, loss, t):
        """ Updates the loss record
        
        Will take the current loss value and convert it to a Numpy value, storing
        it in the loss_rcd attribute.
        
        If verbose is True, will append the current value to the TQDM postfix.
        
        Paramaters
        ----------
        loss: PyTorch Tensor
            Loss value to calculate gradient with
            Shape: Scalar
        t: TQDM iterator
            Used for updating the postfix of the TQDM iterator if verbose is True.
            
        Returns
        -------
        None
        
        Attribute Assignments
        ---------------------
        loss_rcd: list (update)
            Adds the current loss values to the loss_rcd list.
        """
        
        self.loss_rcd.append(loss.detach().cpu().numpy())
        if self.verbose: t.set_postfix({'loss': self.loss_rcd[-1]})
    
    def _update_coefs(self, gradients, learn_rate):
        """ Updates the coefficients
        
        Performis a modified form of ADAM gradient descent to update the
        coefficients.
        
        Parameters
        ----------
        gradients: dict
            Contains the gradients of loss, with respect to coefs, within dict values.
        learn_rate: float (restriction: > 0)
            Multiplier on the final learning steps made for gradient update.
            Smaller values mean slower, but more reliable learning. Larger
            values mean faster, but more unstable learning. It is *strongly*
            recommended to never go over 1e-1 with this.
            
        Returns
        -------
        None
        
        Attribute Assignments
        ---------------------
        grad: dict (update)
            Updates the gradient values based on the current and prior gradients
        grad_sq: dict (update)
            Updates the gradient^2 values based on the current and prior gradient^2's
        coefs: dict (update)
            Updates the coefficients based on their gradients.
        """
        
        with pt.no_grad():
            for key, gradient in gradients.items():
                self._grad[key] = (self._alpha * gradient) + ((1-self._alpha) * self._grad.get(key, gradient))
                self._grad_sq[key] = (self._beta * (gradient**2)) + ((1-self._beta) * self._grad_sq.get(key, gradient**2))
                learn_step = self._grad[key] / self._grad_sq[key].sqrt().clamp(self._learn_clamp, np.inf)
                self.coefs[key] = self.coefs[key] - (learn_step * learn_rate)
                
    def plot_fit(self, col, dim=0, figsize=(14,10)):
        """ Plots the smoothed state fit
        
        Plots the actual state data against the smoothed data. Good for fit diagnostics.
        
        Parameters
        ----------
        col: varies
            Column name for data to plot, from original data_frame.
        dim: int (optional, default 0)
            Dimension of data to plot.
        fig_size: tuple (optional, default: (14,10))
            Size of resulting plot
            
        Returns
        -------
        None, does render a plot to the view however
        
        Attribute Assignments
        ---------------------
        None
        """
        
        idx = self.columns.get_loc(col)
        plt.figure(figsize=figsize)
        sns.lineplot(x=self.index, y=self.actuals[:,idx,dim].detach().cpu().numpy())
        sns.lineplot(x=self.index, y=self.state[:,idx,dim].detach().cpu().numpy())
        plt.xlim(self.index[0], self.index[-1])
        plt.tight_layout()