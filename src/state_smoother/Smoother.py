from __future__ import annotations

from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Index, Series
from scipy.special import factorial as fact
import seaborn as sns
from tqdm.notebook import trange
import torch as pt

from .errors import sq_err


class Smoother:
    """Fit smoothed state trajectories for tabular time-series data.

    These instance attributes are established during ``__init__``.
    """

    dims: int
    horizon: float
    error_func: Callable[[pt.Tensor, pt.Tensor], pt.Tensor]
    learn_seed: bool
    verbose: bool
    tqdm_leave: bool
    loss_rcd: list[np.ndarray]
    actuals: pt.Tensor | None
    state: pt.Tensor | None
    columns: Index
    index: Index
    coefs: dict[str, pt.Tensor]
    _alpha: float
    _beta: float
    _flow_clamp: float
    _learn_clamp: float
    _grad: dict[str, pt.Tensor]
    _grad_sq: dict[str, pt.Tensor]
    _data: pt.Tensor
    _seed: pt.Tensor
    _incrementor: pt.Tensor

    def __init__(
        self,
        data_frame: DataFrame,
        dims: int,
        horizon: float,
        error_func: Callable[[pt.Tensor, pt.Tensor], pt.Tensor] = sq_err,
        seed_data: DataFrame | Series | None = None,
        coef_targets: float = 0,
        learn_seed: bool = False,
        alpha: float = 0.2,
        beta: float = 0.1,
        learn_clamp: float = 1e-16,
        flow_clamp: float = 32,
        verbose: bool = False,
        tqdm_leave: bool = True,
    ) -> None:
        """Initialize the smoother and seed internal tensors."""

        self.dims = dims
        self.horizon = horizon
        self.error_func = error_func
        self.learn_seed = learn_seed
        self.verbose = verbose
        self.tqdm_leave = tqdm_leave
        self.loss_rcd = []
        self.actuals = None
        self.state = None
        
        self._alpha = alpha
        self._beta = beta
        self._flow_clamp = flow_clamp
        self._learn_clamp = learn_clamp
        self._grad = {}
        self._grad_sq = {}
        
        self._init_frame(data_frame, seed_data, self.dims)
        self._init_incrementor(self.dims)
        self._init_coefs(self.dims, self.columns.size, coef_targets, learn_seed)

    def _init_frame(
        self,
        data_frame: DataFrame,
        seed_data: DataFrame | Series | None,
        dims: int,
    ) -> None:
        """Convert the input frame and seed data into tensors."""

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

    def _init_incrementor(self, dims: int) -> None:
        """Build the state transition incrementor."""

        self._incrementor = pt.from_numpy(np.stack([
            np.pad(1 / fact(np.arange(dims-dim)), (dim,0), 'constant', constant_values=0)
            for dim in range(dims)
        ])).type(pt.Tensor)

    def _init_coefs(
        self,
        dims: int,
        col_size: int,
        coef_targets: float,
        learn_seed: bool,
    ) -> None:
        """Initialize trainable coefficient tensors."""

        self.coefs = {
            'flow': pt.empty(dims, col_size).fill_(coef_targets),
        }
        if learn_seed: self.coefs['seed'] = self._seed

    def learn(self, epocs: int, learn_rate: float = 1e-3) -> None:
        """Run gradient-based fitting for the configured number of epochs."""

        t = trange(epocs, leave=self.tqdm_leave) if self.verbose else range(epocs)
        for _epoc in t:
            self.actuals, self.state = self._epoc(learn_rate, t)

    def _epoc(self, learn_rate: float, t: Any) -> tuple[pt.Tensor, pt.Tensor]:
        """Run one optimization epoch and return actual/state tensors."""

        for coef in self.coefs.values(): coef.requires_grad = True
        flow = self._init_flow(self.coefs['flow'], self.horizon, self._flow_clamp)

        actuals = [self.coefs['seed'] if self.learn_seed else self._seed]
        state = [self.coefs['seed'] if self.learn_seed else self._seed]
        error = []
        for actual in self._data:
            new_actual = self._form_actual(actual, self.dims, actuals[-1])
            new_state, new_error = self._smooth_data(new_actual, self._incrementor, state[-1], flow, self.error_func)
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
    def _init_flow(target: pt.Tensor, horizon: float, clamp: float) -> pt.Tensor:
        """Convert learned flow targets into bounded smoothing weights."""

        flow = pt.sigmoid(target.clamp(-clamp, clamp))**(1/(horizon**2))
        return flow
                
    @staticmethod
    def _form_actual(actual: pt.Tensor, dims: int, prior_actual: pt.Tensor) -> pt.Tensor:
        """Add derivative dimensions for the current observation."""

        dimmed_actual = [actual]
        for d in range(1, dims): dimmed_actual.append(dimmed_actual[-1] - prior_actual[d-1,:])
        dimmed_actual = pt.stack(dimmed_actual, dim=0)
        return dimmed_actual
    
    @staticmethod
    def _smooth_data(
        actual: pt.Tensor,
        incrementor: pt.Tensor,
        prior_state: pt.Tensor,
        flow: pt.Tensor,
        error_func: Callable[[pt.Tensor, pt.Tensor], pt.Tensor],
    ) -> tuple[pt.Tensor, pt.Tensor]:
        """Predict the next state, blend it with the observation, and score error."""

        predict = incrementor @ prior_state
        new_state = ((1-flow) * actual) + (flow * predict)
        new_error = error_func(predict[0,:], actual[0,:])
        return new_state, new_error
    
    @staticmethod
    def _calc_grad(loss: pt.Tensor, coefs: dict[str, pt.Tensor]) -> dict[str, pt.Tensor]:
        """Compute gradients for the active coefficient tensors."""

        keys = list(coefs.keys())
        coef_values = [coefs[key] for key in keys]
        grad_values = pt.autograd.grad(loss, coef_values)

        gradients = {}
        for key, gradient in zip(keys, grad_values):
            gradients[key] = pt.nan_to_num(gradient, nan=0.0)
        return gradients
    
    def _update_loss_rcd(self, loss: pt.Tensor, t: Any) -> None:
        """Append the current loss and update the progress display when enabled."""

        self.loss_rcd.append(loss.detach().cpu().numpy())
        if self.verbose: t.set_postfix({'loss': self.loss_rcd[-1]})
    
    def _update_coefs(self, gradients: dict[str, pt.Tensor], learn_rate: float) -> None:
        """Apply the smoothed gradient update to each trainable coefficient."""

        with pt.no_grad():
            for key, gradient in gradients.items():
                self._grad[key] = (self._alpha * gradient) + ((1-self._alpha) * self._grad.get(key, gradient))
                self._grad_sq[key] = (self._beta * (gradient**2)) + ((1-self._beta) * self._grad_sq.get(key, gradient**2))
                learn_step = self._grad[key] / self._grad_sq[key].sqrt().clamp(min=self._learn_clamp)
                self.coefs[key] = self.coefs[key] - (learn_step * learn_rate)
                
    def plot_fit(
        self,
        col: Any,
        dim: int = 0,
        figsize: tuple[int, int] = (14, 10),
        ax: Any | None = None,
    ) -> None:
        """Plot the observed and smoothed series for one column/dimension."""

        if ax is None:
            fig = plt.figure(figsize=figsize)
            fig.add_axes(plt.axes())
            ax = fig.axes[0]
            
        idx = self.columns.get_loc(col)
        sns.lineplot(x=self.index, y=self.actuals[:,idx,dim].detach().cpu().numpy(), ax=ax)
        sns.lineplot(x=self.index, y=self.state[:,idx,dim].detach().cpu().numpy(), ax=ax)
        ax.set_xlim(self.index[0], self.index[-1])
