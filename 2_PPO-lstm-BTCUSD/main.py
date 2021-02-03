import pandas as pd
import tensortrade.env.default as default
from datetime import datetime

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent
from ray import tune
from ray.tune.registry import register_env
import ray
import ray.rllib.agents.ppo as ppo
# from stable_baselines.common.policies import MlpLnLstmPolicy
# from stable_baselines import PPO2, A2C
import os


cdd = CryptoDataDownload()

data = cdd.fetch("Bitstamp", "USD", "BTC", "1h")

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


def create_env(config=None):
    features = []
    for c in data.columns[1:]:
        s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
        features += [s]

    cp = Stream.select(features, lambda s: s.name == "close")

    features = [
        cp.log().diff().rename("lr"),
        rsi(cp, period=20).rename("rsi"),
        macd(cp, fast=10, slow=50, signal=5).rename("macd")
    ]

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


    env = default.create(
        portfolio=portfolio,
        action_scheme="managed-risk",
        reward_scheme="risk-adjusted",
        feed=feed,
        renderer_feed=renderer_feed,
        renderer=default.renderers.PlotlyTradingChart(display=False, save_format='html', path='./agents/charts/'),
        window_size=40
    )
    return env


##############################################
#　PPO ray
env = create_env()
agent = DQNAgent(env)

agent.train(n_steps=15000, n_episodes=1000, save_path="./agents/", save_every=100)

# register_env("TradingEnv", create_env)[]
# ray.init()
# print("Run training! PPO analysis.")
# analysis = tune.run(
#     "PPO",
#     stop={
#       "episode_reward_mean": 5000
#     },
#     config={
#         "env": "TradingEnv",
#         "env_config": {
#             "window_size": 25
#         },
#         "log_level": "DEBUG",
#         "framework": "torch",
#         "ignore_worker_failures": True,
#         "num_workers": 1,
#         "num_gpus": 0,
#         "clip_rewards": True,
#         "lr": 8e-6,
#         "gamma": 0,
#         "observation_filter": "MeanStdFilter",
#         "lambda": 0.72,
#         "vf_loss_coeff": 0.5,
#         "entropy_coeff": 0.01
#     },
#     checkpoint_at_end=True,
#     local_dir="./results"
# )

# print("Done.")
# print("Save checkpoint.")

# # Get checkpoint
# checkpoints = analysis.get_trial_checkpoints_paths(
#     trial=analysis.get_best_trial("episode_reward_mean"),
#     metric="episode_reward_mean"
# )
# checkpoint_path = checkpoints[0][0]

# print("Restore agent from checkpoint.")

# Restore agent
# agent = ppo.PPOTrainer(
#     env="TradingEnv",
#     config={
#         "env_config": {
#             "window_size": 25
#         },
#         "framework": "torch",
#         "log_level": "DEBUG",
#         "ignore_worker_failures": True,
#         "num_workers": 4,
#         "num_gpus": 2,
#         "clip_rewards": True,
#         "lr": 8e-6,
#         "gamma": 0,
#         "observation_filter": "MeanStdFilter",
#         "lambda": 0.72,
#         "vf_loss_coeff": 0.5,
#         "entropy_coeff": 0.01
#     }
# )

# agent.train()
# # agent.restore(checkpoint_path)

# # # Instantiate the environment
# # env = create_env({
# #     "window_size": 25
# # })

# # # Run until episode ends
# episode_reward = 0
# done = False
# obs = env.reset()

# while not done:
#     action = agent.compute_action(obs)
#     obs, reward, done, info = env.step(action)
#     episode_reward += reward

# date_string = "_".join(str(datetime.utcnow()).split())

# os.mkdir(f"charts/{date_string}")

# fig = env.render()
# # fig.savefig(f"charts/{date_string}/test_sine_curve.png")

# # env.render()

########################################
