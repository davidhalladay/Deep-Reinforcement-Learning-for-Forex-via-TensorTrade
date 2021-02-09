import pandas as pd
import tensortrade.env.default as default
from datetime import datetime

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.rewards import RiskAdjustedReturns, PBR, SimpleProfit
from tensortrade.env.default.actions import BSH

from tensortrade.agents import DQNAgent
from ray import tune
from ray.tune.registry import register_env
import ray
import ray.rllib.agents.ppo as ppo
# from stable_baselines.common.policies import MlpLnLstmPolicy
# from stable_baselines import PPO2, A2C
import os
from ta import add_all_ta_features
from ta.utils import dropna
import numpy as np
import yfinance as yf

cdd = CryptoDataDownload()
# feature engineer
data = cdd.fetch("Bitstamp", "USD", "BTC", "minute")
# data = yf.download("EURUSD=X", start="2021-01-01", end="2021-01-31", interval='15m')
data = dropna(data)
# print(data)
data = add_all_ta_features(
    data, open="open", high="high", low="low", close="close", volume="volume")
data = data.dropna(1) 
print(data)
# exit()
def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100*(1 - (1 + rs) ** -1)


def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal


def create_env(envs_config=None):
    features = []
    for c in data.columns[5:]:
        s = Stream.source(list(data[c][-100:]), dtype="float").rename(data[c].name)
        features += [s]

    cp = Stream.select(features, lambda s: s.name == "close")
    
    features = [cp.log().diff().fillna(0).rename("lr")] + features[1:]

    feed = DataFeed(features)
    feed.compile()

    bitstamp = Exchange("bitstamp", service=execute_order)(
        Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    )

    portfolio = Portfolio(USD, [
        Wallet(bitstamp, 10000 * USD),
        Wallet(bitstamp, 10 * BTC)
    ])


    renderer_feed = DataFeed([
        Stream.source(list(data["date"])).rename("date"),
        Stream.source(list(data["open"]), dtype="float").rename("open"),
        Stream.source(list(data["high"]), dtype="float").rename("high"),
        Stream.source(list(data["low"]), dtype="float").rename("low"),
        Stream.source(list(data["close"]), dtype="float").rename("close"), 
        Stream.source(list(data["volume"]), dtype="float").rename("volume") 
    ])

    # reward_scheme = RiskAdjustedReturns(
    #     return_algorithm='sortino', 
    #     risk_free_rate=0.025, 
    #     target_returns=0.1, 
    #     window_size=200
    # )
    reward_scheme = SimpleProfit(window_size=200) #PBR(price=cp)

    # action_scheme = BSH(
    #     cash=portfolio.wallets[0],
    #     asset=portfolio.wallets[1]
    # ).attach(reward_scheme)

    env = default.create(
        portfolio=portfolio,
        action_scheme="managed-risk",
        reward_scheme=reward_scheme, #"risk-adjusted",
        feed=feed,
        renderer_feed=renderer_feed,
        renderer=default.renderers.PlotlyTradingChart(display=False, save_format='html', path='./agents/charts/'),
        window_size=40
    )
    return env

Trainer_config = {
    "env": "TradingEnv",
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "use_lstm": True,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 60,
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
ray.init()

print("Run training! PPO analysis.")
analysis = tune.run(
    ppo.PPOTrainer,
    stop={
      "episode_reward_mean": 5000
    },
    config=Trainer_config,
    checkpoint_at_end=True,
    local_dir="./results"
)

print("Done.")
print("Save checkpoint.")

# Get checkpoint
checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean", mode="max"), 
    metric="episode_reward_mean"
)
checkpoint_path = checkpoints[0][0]

print("Restore agent from checkpoint.")

# Restore agent

agent = ppo.PPOTrainer(
    env="TradingEnv",
    config=Trainer_config
)

# agent.train()
# checkpoint_pat[h = "/home/davidfan/VLL/FX/tensortrade/2_PPO-lstm-BTCUSD/results/PPO_2021-02-04_00-50-12/PPO_TradingEnv_e1b5f_00000_0_2021-02-04_00-50-12/checkpoint_2/checkpoint-2"
agent.restore(checkpoint_path)

# # Instantiate the environment
env = create_env()

# # Run until episode ends
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
