import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import ray.rllib.agents.ppo as ppo
import tensortrade.env.default as default
from gym.spaces import Discrete
from ray import tune
from ray.tune.registry import register_env
from symfit import parameters, variables, sin, cos, Fit
from tensortrade.env.default.actions import TensorTradeActionScheme
from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.env.default.rewards import RiskAdjustedReturns, PBR, SimpleProfit
from tensortrade.env.generic import Renderer
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.orders import proportion_order
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio

from tensortrade.data.cdd import CryptoDataDownload
from ta import add_all_ta_features
from ta.utils import dropna
import yfinance as yf

# from tensortrade.oms.instruments import USD, BTC, EUR 

# Instrument
USD = Instrument('USD', 8, 'U.S. Dollar')
EUR = Instrument('EUR', 8, 'Euro')

### download data
# cdd = CryptoDataDownload()
# data = cdd.fetch("Bitstamp", "USD", "BTC", "1h")
# print(data)

data = yf.download("EURUSD=X", start="2020-12-20", end="2021-01-31", interval='15m')
data.insert(0, "Date", "Any")
data.insert(1, "Unix", "Any")
data['Date'] = data.index
data = data.reset_index()
del data['Datetime']
del data['Adj Close']
# data = dropna(data)
data = add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume")
data = data.dropna(1) 
# reset time zone 
data['Date'] = data['Date'].dt.tz_localize(None)
print(data)

class BSH(TensorTradeActionScheme):

    registered_name = "bsh"

    def __init__(self, cash: 'Wallet', asset: 'Wallet'):
        super().__init__()
        self.cash = cash
        self.asset = asset

        self.listeners = []
        self.action = 0

    @property
    def action_space(self):
        return Discrete(2)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: 'Portfolio'):
        order = None

        if abs(action - self.action) > 0:
            src = self.cash if self.action == 0 else self.asset
            tgt = self.asset if self.action == 0 else self.cash
            order = proportion_order(portfolio, src, tgt, 1.0)
            self.action = action

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self.action = 0

class PBR(TensorTradeRewardScheme):

    registered_name = "pbr"

    def __init__(self, price: 'Stream'):
        super().__init__()
        self.position = -1

        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")

        reward = (r * position).fillna(0).rename("reward")

        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, action: int):
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: 'Portfolio'):
        return self.feed.next()["reward"]

    def reset(self):
        self.position = -1
        self.feed.reset()

def create_env(config, save_path='./agents/charts/', is_eval=False):

    # Load data
    k = -3000
    w = -200
    if is_eval:
        k = -200
        w = None
    y = data['Close'][k:w].to_numpy()
    
    features = []
    for c in data.columns[5:]:
        s = Stream.source(list(data[c][k:w]), dtype="float").rename(data[c].name)
        features += [s]
        
    cp = Stream.source(y, dtype="float").rename("EUR-USD")

    coinbase = Exchange("coinbase", service=execute_order, options=ExchangeOptions(commission=0.00005))(
        cp
    )
    
    feature_add = [
        cp,
        cp.ewm(span=10).mean().rename("fast"),
        cp.ewm(span=50).mean().rename("medium"),
        cp.ewm(span=100).mean().rename("slow"),
        cp.log().diff().fillna(0).rename("lr")
    ]
    features = features + feature_add

    feed = DataFeed(features)

    feed.compile()
    
    cash = Wallet(coinbase, 10000 * EUR)
    asset = Wallet(coinbase, 10000 * USD)

    portfolio = Portfolio(EUR, [
        cash,
        asset
    ])

    reward_scheme = PBR(price=cp)
    # reward_scheme = SimpleProfit(window_size=500)

    action_scheme = BSH(
        cash=cash,
        asset=asset
    ).attach(reward_scheme)

    # renderer_feed = DataFeed([
    #     Stream.source(y, dtype="float").rename("price"),
    #     Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    # ])

    renderer_feed = DataFeed([
        Stream.source(list(data["Date"][k:w])).rename("date"),
        Stream.source(list(data["Open"][k:w]), dtype="float").rename("open"),
        Stream.source(list(data["High"][k:w]), dtype="float").rename("high"),
        Stream.source(list(data["Low"][k:w]), dtype="float").rename("low"),
        Stream.source(list(data["Close"][k:w]), dtype="float").rename("close"), 
        Stream.source(list(data["Volume"][k:w]), dtype="float").rename("volume") 
    ])

    environment = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme, #"managed-risk",
        reward_scheme=reward_scheme,
        renderer_feed=renderer_feed,
        renderer=default.renderers.PlotlyTradingChart(display=False, save_format='html', path=save_path),
        # renderer=PositionChangeChart(),
        window_size=config["window_size"],
        max_allowed_loss=0.6
    )
    return environment


def main():

    Trainer_config = {
    "env": "TradingEnv",
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 40,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": True,
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": True,
    },
    "env_config": {
        "window_size": 25
    },
    "log_level": "DEBUG",
    "framework": "torch",
    "ignore_worker_failures": True,
    "num_workers": 1,
    "num_gpus": 0,
    "clip_rewards": True,
    "lr": 8e-6,
    "gamma": 0,
    "observation_filter": "MeanStdFilter",
    "lambda": 0.72,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "eager_tracing": True
    }

    register_env("TradingEnv", create_env)

    analysis = tune.run(
        "PPO",
        stop={
          "training_iteration": 10
        },
        config=Trainer_config,
        checkpoint_at_end=True,
        local_dir="./results_PPO"
    )
    
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial(metric="episode_reward_mean", mode="max"),
        metric="episode_reward_mean"
    )
    checkpoint_path = checkpoints[0][0]

    # Restore agent
    agent = ppo.PPOTrainer(
        env="TradingEnv",
        config=Trainer_config
    )
    agent.restore(checkpoint_path)

    # Test on sine curve
    env = create_env({
        "window_size": 25
    }, save_path='./agents/charts/')

    episode_reward = 0
    done = False
    obs = env.reset()
    state = agent.get_policy().model.get_initial_state()

    while not done:
        action, state, logit = agent.compute_action(observation=obs, prev_action=1.0, prev_reward = 0.0, state = state)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    print("episode_reward: ", episode_reward)

    env.render()    

    # Eval
    agent.restore(checkpoint_path)

    # Test on sine curve
    env = create_env({
        "window_size": 25
    }, save_path='./agents/charts_eval/', is_eval=True)

    episode_reward = 0
    done = False
    obs = env.reset()
    state = agent.get_policy().model.get_initial_state()

    while not done:
        action, state, logit = agent.compute_action(observation=obs, prev_action=1.0, prev_reward = 0.0, state = state)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    print("episode_reward: ", episode_reward)

    env.render()    
    
if __name__ == "__main__":
    main()
