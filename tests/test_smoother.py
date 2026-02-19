from pathlib import Path

import numpy as np
import pandas as pd
import torch as pt

from state_smoother import Smoother
from state_smoother.errors import log_perc_error, sq_err


def _load_market_data():
    return pd.read_csv(
        Path(__file__).parent / "data" / "eve_market_data.csv",
        parse_dates=["record_date"],
        index_col="record_date",
    )


def _small_frame():
    return pd.DataFrame(
        {
            "alpha": [10.0, 11.0, 13.0, 12.0, 14.0, 15.0],
            "beta": [2.0, 3.0, 2.0, 4.0, 3.0, 5.0],
        },
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )


def test_learn_produces_expected_shapes_and_loss_history():
    data = _small_frame()
    model = Smoother(data, dims=3, horizon=4)

    model.learn(epocs=3, learn_rate=1e-2)

    expected_shape = (data.shape[0] - 1, data.shape[1], 3)
    assert model.actuals.shape == expected_shape
    assert model.state.shape == expected_shape
    assert len(model.loss_rcd) == 3


def test_default_seed_uses_first_row_and_zero_fills_derivatives():
    data = _small_frame()
    model = Smoother(data, dims=3, horizon=3)

    assert model._seed.shape == (3, data.shape[1])
    expected_first_dim = pt.from_numpy(data.iloc[0].to_numpy()).type(pt.Tensor)
    assert pt.allclose(model._seed[0], expected_first_dim)
    assert pt.allclose(model._seed[1], pt.zeros(data.shape[1]))
    assert pt.allclose(model._seed[2], pt.zeros(data.shape[1]))
    assert model._data.shape[0] == data.shape[0] - 1
    assert model.index[0] == data.index[1]


def test_seed_data_keeps_all_observations_and_uses_provided_dims():
    data = _small_frame()
    seed_data = data.iloc[[0, 1]]
    model = Smoother(data, dims=2, horizon=3, seed_data=seed_data)

    assert model._data.shape[0] == data.shape[0]
    assert model._seed.shape == (2, data.shape[1])
    expected_seed = pt.from_numpy(seed_data.to_numpy()).type(pt.Tensor)
    assert pt.allclose(model._seed, expected_seed)


def test_learn_seed_adds_seed_to_coefficients():
    model = Smoother(_small_frame(), dims=2, horizon=2, learn_seed=True)
    assert "seed" in model.coefs


def test_init_flow_applies_clamp_before_sigmoid_and_horizon():
    target = pt.tensor([[-100.0, 0.0, 100.0]])
    flow = Smoother._init_flow(target=target, horizon=2, clamp=2)
    expected = pt.sigmoid(target.clamp(-2, 2)) ** (1 / (2**2))
    assert pt.allclose(flow, expected)


def test_form_actual_builds_derivative_dimensions():
    actual = pt.tensor([10.0, 7.0])
    prior_actual = pt.tensor([[8.0, 5.0], [1.0, 2.0], [0.5, 0.5]])
    dimmed = Smoother._form_actual(actual=actual, dims=3, prior_actual=prior_actual)

    expected = pt.tensor(
        [
            [10.0, 7.0],
            [2.0, 2.0],
            [1.0, 0.0],
        ]
    )
    assert pt.allclose(dimmed, expected)


def test_smooth_data_calculates_prediction_state_and_error():
    actual = pt.tensor([[4.0], [1.0]])
    incrementor = pt.tensor([[1.0, 1.0], [0.0, 1.0]])
    prior_state = pt.tensor([[2.0], [1.0]])
    flow = pt.tensor([[0.5], [0.25]])

    new_state, new_error = Smoother._smooth_data(
        actual=actual,
        incrementor=incrementor,
        prior_state=prior_state,
        flow=flow,
        error_func=sq_err,
    )

    predict = incrementor @ prior_state
    expected_state = ((1 - flow) * actual) + (flow * predict)
    expected_error = sq_err(predict[0, :], actual[0, :])
    assert pt.allclose(new_state, expected_state)
    assert pt.allclose(new_error, expected_error)


def test_error_functions_return_expected_values():
    assert pt.allclose(
        sq_err(pt.tensor([2.0, 3.0]), pt.tensor([1.0, 5.0])),
        pt.tensor([1.0, 4.0]),
    )
    assert pt.allclose(
        log_perc_error(pt.tensor([2.0, 4.0]), pt.tensor([1.0, 2.0])),
        pt.tensor([0.0, 0.0]),
    )


def test_market_data_smoke_learns_and_records_finite_loss():
    market_data = _load_market_data()
    model = Smoother(market_data, dims=2, horizon=14)

    model.learn(epocs=2, learn_rate=1e-2)

    losses = np.array(model.loss_rcd)
    assert len(losses) == 2
    assert np.isfinite(losses).all()
