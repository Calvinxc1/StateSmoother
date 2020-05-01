import pandas as pd

from state_smoother import Smoother

dims = 3
horizon = 14

market_data = pd.read_csv(
    './_test/data/eve_market_data.csv',
    parse_dates=['record_date'], index_col='record_date'
)

def test_run():
    exp = Smoother(market_data, dims=dims, horizon=horizon)
    exp.learn(100, learn_rate=1e-2)
    
    assert exp.state.shape[0] == market_data.shape[0] - 1
    assert exp.state.shape[1] == market_data.shape[1]
    assert exp.state.shape[2] == dims
    
    assert exp.actuals.shape[0] == market_data.shape[0] - 1
    assert exp.actuals.shape[1] == market_data.shape[1]
    assert exp.actuals.shape[2] == dims