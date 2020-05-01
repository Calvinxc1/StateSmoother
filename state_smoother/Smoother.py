import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial as fact
import seaborn as sns
from tqdm.notebook import trange
import torch as pt

from .errors import log_perc_error

class Smoother:
    def __init__(self, data_frame, dims, horizon, error_func=log_perc_error,
                 seed_data=None, seed_targets=0, learn_seed=False,
                 alpha=0.2, beta=0.1, clamp=1e-16, verbose=False,
                ):
        self.dims = dims
        self.horizon = horizon
        self.error_func = error_func
        self.learn_seed = learn_seed
        self.verbose = verbose
        self.loss_rcd = []
        
        self._alpha = alpha
        self._beta = beta
        self._clamp = clamp
        self._grad = {}
        self._grad_sq = {}
        
        self._init_frame(data_frame, seed_data, self.dims)
        self._init_incrementor(self.dims)
        self._init_coefs(self.dims, self.columns.size, seed_targets, learn_seed)
        
    def _init_frame(self, data_frame, seed_data, dims):
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
            self._seed = pt.from_numpy(seed_data[self.columns]).type(pt.Tensor)
            
    def _init_incrementor(self, dims):
        self._incrementor = pt.from_numpy(np.stack([
            np.pad(1 / fact(np.arange(dims-dim)), (dim,0), 'constant', constant_values=0)
            for dim in range(dims)
        ])).type(pt.Tensor)
        
    def _init_coefs(self, dims, col_size, seed_targets, learn_seed):
        self.coefs = {
            'flow': pt.empty(dims, col_size).fill_(seed_targets),
        }
        if learn_seed: self.coefs['seed'] = self._seed
            
    def learn(self, epocs, learn_rate=1e-3):
        t = trange(epocs) if self.verbose else range(epocs)
        for epoc in t:
            self.actuals, self.state = self._epoc(learn_rate, t)
        
    def _epoc(self, learn_rate, t):
        for coef in self.coefs.values(): coef.requires_grad = True
        flow = self._init_flow(self.coefs['flow'], self.horizon)

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
    def _init_flow(target, horizon):
        flow = pt.sigmoid(target.clamp(-32,32))**(1/horizon)
        return flow
                
    @staticmethod
    def _form_actual(actual, dims, prior_actual):
        actual = [actual]
        for d in range(1, dims): actual.append(actual[-1] - prior_actual[d-1,:])
        actual = pt.stack(actual, dim=0)
        return actual
    
    @staticmethod
    def _smooth_data(actual, incrementor, prior_state, flow, error_func):
        predict = incrementor @ prior_state
        new_state = ((1-flow) * actual) + (flow * predict)
        new_error = error_func(predict[0,:], actual[0,:])
        return new_state, new_error
    
    @staticmethod
    def _calc_grad(loss, coefs):
        gradients = np.array([(key, coef) for key, coef in coefs.items()])
        gradients[:,1] = pt.autograd.grad(loss, gradients[:,1])
        gradients = {key:grad for key, grad in gradients}
        return gradients
    
    def _update_loss_rcd(self, loss, t):
        self.loss_rcd.append(loss.detach().cpu().numpy())
        if self.verbose: t.set_postfix({'loss': self.loss_rcd[-1]})
    
    def _update_coefs(self, gradients, learn_rate):
        with pt.no_grad():
            for key, gradient in gradients.items():
                self._grad[key] = (self._alpha * gradient) + ((1-self._alpha) * self._grad.get(key, gradient))
                self._grad_sq[key] = (self._beta * (gradient**2)) + ((1-self._beta) * self._grad_sq.get(key, gradient**2))
                learn_step = self._grad[key] / self._grad_sq[key].sqrt().clamp(self._clamp, np.inf)
                self.coefs[key] = self.coefs[key] - (learn_step * learn_rate)
                
    def plot_fit(self, col, figsize=(14,10)):
        idx = self.columns.get_loc(col)
        plt.figure(figsize=figsize)
        sns.lineplot(x=self.index, y=self.actuals[:,idx,0].detach().cpu().numpy())
        sns.lineplot(x=self.index, y=self.state[:,idx,0].detach().cpu().numpy())
        plt.xlim(self.index[0], self.index[-1])
        plt.tight_layout()