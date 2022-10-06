{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Reinforcement Learning for Trading - Deep Q-learning & the stock market"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To train a trading agent, we need to create a market environment that provides price and other information, offers trading-related actions, and keeps track of the portfolio to reward the agent accordingly."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## How to Design an OpenAI trading environment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The OpenAI Gym allows for the design, registration, and utilization of environments that adhere to its architecture, as described in its [documentation](https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym). The [trading_env.py](trading_env.py) file implements an example that illustrates how to create a class that implements the requisite `step()` and `reset()` methods.\n",
    "\n",
    "The trading environment consists of three classes that interact to facilitate the agent's activities:\n",
    " 1. The `DataSource` class loads a time series, generates a few features, and provides the latest observation to the agent at each time step. \n",
    " 2. `TradingSimulator` tracks the positions, trades and cost, and the performance. It also implements and records the results of a buy-and-hold benchmark strategy. \n",
    " 3. `TradingEnvironment` itself orchestrates the process. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The book chapter explains these elements in more detail."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## A basic trading game"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To train the agent, we need to set up a simple game with a limited set of options, a relatively low-dimensional state, and other parameters that can be easily modified and extended.\n",
    "\n",
    "More specifically, the environment samples a stock price time series for a single ticker using a random start date to simulate a trading period that, by default, contains 252 days, or 1 year. The state contains the (scaled) price and volume, as well as some technical indicators like the percentile ranks of price and volume, a relative strength index (RSI), as well as 5- and 21-day returns. The agent can choose from three actions:\n",
    "\n",
    "- **Buy**: Invest capital for a long position in the stock\n",
    "- **Flat**: Hold cash only\n",
    "- **Sell short**: Take a short position equal to the amount of capital\n",
    "\n",
    "The environment accounts for trading cost, which is set to 10bps by default. It also deducts a 1bps time cost per period. It tracks the net asset value (NAV) of the agent's portfolio and compares it against the market portfolio (which trades frictionless to raise the bar for the agent)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We use the same DDQN agent and neural network architecture that successfully learned to navigate the Lunar Lander environment. We let exploration continue for 500,000 time steps (~2,000 1yr trading periods) with linear decay of ε to 0.1 and exponential decay at a factor of 0.9999 thereafter."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Imports & Settings"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:10:21.897529Z",
     "start_time": "2021-11-16T04:10:21.892603Z"
    }
   },
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:10:23.441506Z",
     "start_time": "2021-11-16T04:10:21.898569Z"
    }
   },
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "from pathlib import Path\n",
    "from time import time\n",
    "from collections import deque\n",
    "from random import sample\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.ticker import FuncFormatter\n",
    "import seaborn as sns\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras import Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.regularizers import l2\n",
    "\n",
    "import gym\n",
    "from gym.envs.registration import register"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Settings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:10:24.669692Z",
     "start_time": "2021-11-16T04:10:24.664510Z"
    }
   },
   "outputs": [],
   "source": [
    "np.random.seed(42)\n",
    "tf.random.set_seed(42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:10:24.791466Z",
     "start_time": "2021-11-16T04:10:24.786681Z"
    }
   },
   "outputs": [],
   "source": [
    "sns.set_style('whitegrid')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:10:25.000425Z",
     "start_time": "2021-11-16T04:10:24.926689Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Using GPU\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2021-11-15 22:10:24.956597: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
      "2021-11-15 22:10:24.960919: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n",
      "2021-11-15 22:10:24.961256: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n"
     ]
    }
   ],
   "source": [
    "gpu_devices = tf.config.experimental.list_physical_devices('GPU')\n",
    "if gpu_devices:\n",
    "    print('Using GPU')\n",
    "    tf.config.experimental.set_memory_growth(gpu_devices[0], True)\n",
    "else:\n",
    "    print('Using CPU')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:10:25.033871Z",
     "start_time": "2021-11-16T04:10:25.031449Z"
    }
   },
   "outputs": [],
   "source": [
    "results_path = Path('results', 'trading_bot')\n",
    "if not results_path.exists():\n",
    "    results_path.mkdir(parents=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Helper functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:10:30.326477Z",
     "start_time": "2021-11-16T04:10:30.324003Z"
    }
   },
   "outputs": [],
   "source": [
    "def format_time(t):\n",
    "    m_, s = divmod(t, 60)\n",
    "    h, m = divmod(m_, 60)\n",
    "    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Set up Gym Environment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Before using the custom environment, just like with the Lunar Lander environment, we need to register it:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:10:31.011029Z",
     "start_time": "2021-11-16T04:10:31.004868Z"
    }
   },
   "outputs": [],
   "source": [
    "trading_days = 252"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:10:31.167371Z",
     "start_time": "2021-11-16T04:10:31.162398Z"
    }
   },
   "outputs": [],
   "source": [
    "register(\n",
    "    id='trading-v0',\n",
    "    entry_point='trading_env:TradingEnvironment',\n",
    "    max_episode_steps=trading_days\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Initialize Trading Environment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can instantiate the environment by using the desired trading costs and ticker:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:10:33.062989Z",
     "start_time": "2021-11-16T04:10:33.061026Z"
    }
   },
   "outputs": [],
   "source": [
    "trading_cost_bps = 1e-3\n",
    "time_cost_bps = 1e-4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:10:33.253407Z",
     "start_time": "2021-11-16T04:10:33.239317Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Trading costs: 0.10% | Time costs: 0.01%'"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f'Trading costs: {trading_cost_bps:.2%} | Time costs: {time_cost_bps:.2%}'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-11-16T04:14:48.346878Z",
     "start_time": "2021-11-16T04:14:46.084768Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:trading_env:loading data for AAPL...\n",
      "INFO:trading_env:got data for AAPL...\n",
      "INFO:trading_env:None\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "MultiIndex: 9367 entries, (Timestamp('1981-01-30 00:00:00'), 'AAPL') to (Timestamp('2018-03-27 00:00:00'), 'AAPL')\n",
      "Data columns (total 10 columns):\n",
      " #   Column   Non-Null Count  Dtype  \n",
      "---  ------   --------------  -----  \n",
      " 0   returns  9367 non-null   float64\n",
      " 1   ret_2    9367 non-null   float64\n",
      " 2   ret_5    9367 non-null   float64\n",
      " 3   ret_10   9367 non-null   float64\n",
      " 4   ret_21   9367 non-null   float64\n",
      " 5   rsi      9367 non-null   float64\n",
      " 6   macd     9367 non-null   float64\n",
      " 7   atr      9367 non-null   float64\n",
      " 8   stoch    9367 non-null   float64\n",
      " 9   ultosc   9367 non-null   float64\n",
      "dtypes: float64(10)\n",
      "memory usage: 1.5+ MB\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[42]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trading_environment = gym.make('trading-v0', \n",
    "                               ticker='AAPL',\n",
    "                               trading_days=trading_days,\n",
    "                               trading_cost_bps=trading_cost_bps,\n",
    "                               time_cost_bps=time_cost_bps)\n",
    "trading_environment.seed(42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Get Environment Params"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.548145Z",
     "start_time": "2021-02-25T06:20:32.545830Z"
    }
   },
   "outputs": [],
   "source": [
    "state_dim = trading_environment.observation_space.shape[0]\n",
    "num_actions = trading_environment.action_space.n\n",
    "max_episode_steps = trading_environment.spec.max_episode_steps"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Define Trading Agent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.563692Z",
     "start_time": "2021-02-25T06:20:32.549782Z"
    }
   },
   "outputs": [],
   "source": [
    "class DDQNAgent:\n",
    "    def __init__(self, state_dim,\n",
    "                 num_actions,\n",
    "                 learning_rate,\n",
    "                 gamma,\n",
    "                 epsilon_start,\n",
    "                 epsilon_end,\n",
    "                 epsilon_decay_steps,\n",
    "                 epsilon_exponential_decay,\n",
    "                 replay_capacity,\n",
    "                 architecture,\n",
    "                 l2_reg,\n",
    "                 tau,\n",
    "                 batch_size):\n",
    "\n",
    "        self.state_dim = state_dim\n",
    "        self.num_actions = num_actions\n",
    "        self.experience = deque([], maxlen=replay_capacity)\n",
    "        self.learning_rate = learning_rate\n",
    "        self.gamma = gamma\n",
    "        self.architecture = architecture\n",
    "        self.l2_reg = l2_reg\n",
    "\n",
    "        self.online_network = self.build_model()\n",
    "        self.target_network = self.build_model(trainable=False)\n",
    "        self.update_target()\n",
    "\n",
    "        self.epsilon = epsilon_start\n",
    "        self.epsilon_decay_steps = epsilon_decay_steps\n",
    "        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps\n",
    "        self.epsilon_exponential_decay = epsilon_exponential_decay\n",
    "        self.epsilon_history = []\n",
    "\n",
    "        self.total_steps = self.train_steps = 0\n",
    "        self.episodes = self.episode_length = self.train_episodes = 0\n",
    "        self.steps_per_episode = []\n",
    "        self.episode_reward = 0\n",
    "        self.rewards_history = []\n",
    "\n",
    "        self.batch_size = batch_size\n",
    "        self.tau = tau\n",
    "        self.losses = []\n",
    "        self.idx = tf.range(batch_size)\n",
    "        self.train = True\n",
    "\n",
    "    def build_model(self, trainable=True):\n",
    "        layers = []\n",
    "        n = len(self.architecture)\n",
    "        for i, units in enumerate(self.architecture, 1):\n",
    "            layers.append(Dense(units=units,\n",
    "                                input_dim=self.state_dim if i == 1 else None,\n",
    "                                activation='relu',\n",
    "                                kernel_regularizer=l2(self.l2_reg),\n",
    "                                name=f'Dense_{i}',\n",
    "                                trainable=trainable))\n",
    "        layers.append(Dropout(.1))\n",
    "        layers.append(Dense(units=self.num_actions,\n",
    "                            trainable=trainable,\n",
    "                            name='Output'))\n",
    "        model = Sequential(layers)\n",
    "        model.compile(loss='mean_squared_error',\n",
    "                      optimizer=Adam(lr=self.learning_rate))\n",
    "        return model\n",
    "\n",
    "    def update_target(self):\n",
    "        self.target_network.set_weights(self.online_network.get_weights())\n",
    "\n",
    "    def epsilon_greedy_policy(self, state):\n",
    "        self.total_steps += 1\n",
    "        if np.random.rand() <= self.epsilon:\n",
    "            return np.random.choice(self.num_actions)\n",
    "        q = self.online_network.predict(state)\n",
    "        return np.argmax(q, axis=1).squeeze()\n",
    "\n",
    "    def memorize_transition(self, s, a, r, s_prime, not_done):\n",
    "        if not_done:\n",
    "            self.episode_reward += r\n",
    "            self.episode_length += 1\n",
    "        else:\n",
    "            if self.train:\n",
    "                if self.episodes < self.epsilon_decay_steps:\n",
    "                    self.epsilon -= self.epsilon_decay\n",
    "                else:\n",
    "                    self.epsilon *= self.epsilon_exponential_decay\n",
    "\n",
    "            self.episodes += 1\n",
    "            self.rewards_history.append(self.episode_reward)\n",
    "            self.steps_per_episode.append(self.episode_length)\n",
    "            self.episode_reward, self.episode_length = 0, 0\n",
    "\n",
    "        self.experience.append((s, a, r, s_prime, not_done))\n",
    "\n",
    "    def experience_replay(self):\n",
    "        if self.batch_size > len(self.experience):\n",
    "            return\n",
    "        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))\n",
    "        states, actions, rewards, next_states, not_done = minibatch\n",
    "\n",
    "        next_q_values = self.online_network.predict_on_batch(next_states)\n",
    "        best_actions = tf.argmax(next_q_values, axis=1)\n",
    "\n",
    "        next_q_values_target = self.target_network.predict_on_batch(next_states)\n",
    "        target_q_values = tf.gather_nd(next_q_values_target,\n",
    "                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))\n",
    "\n",
    "        targets = rewards + not_done * self.gamma * target_q_values\n",
    "\n",
    "        q_values = self.online_network.predict_on_batch(states)\n",
    "        q_values[[self.idx, actions]] = targets\n",
    "\n",
    "        loss = self.online_network.train_on_batch(x=states, y=q_values)\n",
    "        self.losses.append(loss)\n",
    "\n",
    "        if self.total_steps % self.tau == 0:\n",
    "            self.update_target()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Define hyperparameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.575368Z",
     "start_time": "2021-02-25T06:20:32.565067Z"
    }
   },
   "outputs": [],
   "source": [
    "gamma = .99,  # discount factor\n",
    "tau = 100  # target network update frequency"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### NN Architecture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.584925Z",
     "start_time": "2021-02-25T06:20:32.576469Z"
    }
   },
   "outputs": [],
   "source": [
    "architecture = (256, 256)  # units per layer\n",
    "learning_rate = 0.0001  # learning rate\n",
    "l2_reg = 1e-6  # L2 regularization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Experience Replay"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.593134Z",
     "start_time": "2021-02-25T06:20:32.586645Z"
    }
   },
   "outputs": [],
   "source": [
    "replay_capacity = int(1e6)\n",
    "batch_size = 4096"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### $\\epsilon$-greedy Policy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.603464Z",
     "start_time": "2021-02-25T06:20:32.594606Z"
    }
   },
   "outputs": [],
   "source": [
    "epsilon_start = 1.0\n",
    "epsilon_end = .01\n",
    "epsilon_decay_steps = 250\n",
    "epsilon_exponential_decay = .99"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Create DDQN Agent"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We will use [TensorFlow](https://www.tensorflow.org/) to create our Double Deep Q-Network ."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.613239Z",
     "start_time": "2021-02-25T06:20:32.604766Z"
    }
   },
   "outputs": [],
   "source": [
    "tf.keras.backend.clear_session()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.720879Z",
     "start_time": "2021-02-25T06:20:32.614703Z"
    }
   },
   "outputs": [],
   "source": [
    "ddqn = DDQNAgent(state_dim=state_dim,\n",
    "                 num_actions=num_actions,\n",
    "                 learning_rate=learning_rate,\n",
    "                 gamma=gamma,\n",
    "                 epsilon_start=epsilon_start,\n",
    "                 epsilon_end=epsilon_end,\n",
    "                 epsilon_decay_steps=epsilon_decay_steps,\n",
    "                 epsilon_exponential_decay=epsilon_exponential_decay,\n",
    "                 replay_capacity=replay_capacity,\n",
    "                 architecture=architecture,\n",
    "                 l2_reg=l2_reg,\n",
    "                 tau=tau,\n",
    "                 batch_size=batch_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.725896Z",
     "start_time": "2021-02-25T06:20:32.722143Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "Dense_1 (Dense)              (None, 256)               2816      \n",
      "_________________________________________________________________\n",
      "Dense_2 (Dense)              (None, 256)               65792     \n",
      "_________________________________________________________________\n",
      "dropout (Dropout)            (None, 256)               0         \n",
      "_________________________________________________________________\n",
      "Output (Dense)               (None, 3)                 771       \n",
      "=================================================================\n",
      "Total params: 69,379\n",
      "Trainable params: 69,379\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "ddqn.online_network.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Run Experiment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Set parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.733088Z",
     "start_time": "2021-02-25T06:20:32.727071Z"
    }
   },
   "outputs": [],
   "source": [
    "total_steps = 0\n",
    "max_episodes = 1000"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Initialize variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.741126Z",
     "start_time": "2021-02-25T06:20:32.734309Z"
    }
   },
   "outputs": [],
   "source": [
    "episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-02-25T06:20:32.752721Z",
     "start_time": "2021-02-25T06:20:32.742471Z"
    }
   },
   "outputs": [],
   "source": [
    "def track_results(episode, nav_ma_100, nav_ma_10,\n",
    "                  market_nav_100, market_nav_10,\n",
    "                  win_ratio, total, epsilon):\n",
    "    time_ma = np.mean([episode_time[-100:]])\n",
    "    T = np.sum(episode_time)\n",
    "    \n",
    "    template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '\n",
    "    template += 'Market: {:>6.1%} ({:>6.1%}) | '\n",
    "    template += 'Wins: {:>5.1%} | eps: {:>6.3f}'\n",
    "    print(template.format(episode, format_time(total), \n",
    "                          nav_ma_100-1, nav_ma_10-1, \n",
    "                          market_nav_100-1, market_nav_10-1, \n",
    "                          win_ratio, epsilon))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Train Agent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "ExecuteTime": {
     "start_time": "2021-02-25T06:20:28.016Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  10 | 00:00:01 | Agent: -39.1% (-39.1%) | Market:   4.6% (  4.6%) | Wins: 20.0% | eps:  0.960\n",
      "  20 | 00:00:51 | Agent: -34.0% (-28.9%) | Market:  23.2% ( 41.8%) | Wins: 20.0% | eps:  0.921\n",
      "  30 | 00:03:02 | Agent: -27.7% (-15.2%) | Market:  20.6% ( 15.4%) | Wins: 16.7% | eps:  0.881\n",
      "  40 | 00:05:11 | Agent: -22.8% ( -8.2%) | Market:  21.2% ( 23.0%) | Wins: 20.0% | eps:  0.842\n",
      "  50 | 00:07:25 | Agent: -21.8% (-17.5%) | Market:  20.2% ( 16.4%) | Wins: 20.0% | eps:  0.802\n",
      "  60 | 00:09:57 | Agent: -22.5% (-26.4%) | Market:  24.5% ( 45.6%) | Wins: 21.7% | eps:  0.762\n",
      "  70 | 00:12:38 | Agent: -20.4% ( -7.5%) | Market:  29.4% ( 59.3%) | Wins: 21.4% | eps:  0.723\n",
      "  80 | 00:15:26 | Agent: -20.9% (-24.7%) | Market:  27.6% ( 14.7%) | Wins: 22.5% | eps:  0.683\n",
      "  90 | 00:18:24 | Agent: -21.1% (-22.2%) | Market:  24.3% ( -2.1%) | Wins: 23.3% | eps:  0.644\n",
      " 100 | 00:21:15 | Agent: -19.5% ( -5.3%) | Market:  24.0% ( 20.9%) | Wins: 23.0% | eps:  0.604\n",
      " 110 | 00:24:16 | Agent: -16.0% ( -4.4%) | Market:  26.0% ( 25.0%) | Wins: 24.0% | eps:  0.564\n",
      " 120 | 00:27:18 | Agent: -10.4% ( 27.0%) | Market:  29.9% ( 80.6%) | Wins: 25.0% | eps:  0.525\n",
      " 130 | 00:30:25 | Agent:  -9.5% ( -5.8%) | Market:  36.4% ( 80.3%) | Wins: 26.0% | eps:  0.485\n",
      " 140 | 00:33:52 | Agent:  -7.8% (  8.6%) | Market:  35.4% ( 13.4%) | Wins: 26.0% | eps:  0.446\n",
      " 150 | 00:37:15 | Agent:  -5.5% (  6.0%) | Market:  37.9% ( 41.0%) | Wins: 28.0% | eps:  0.406\n",
      " 160 | 00:40:44 | Agent:  -4.4% (-15.8%) | Market:  34.9% ( 15.7%) | Wins: 28.0% | eps:  0.366\n",
      " 170 | 00:44:21 | Agent:  -4.7% (-10.8%) | Market:  32.7% ( 37.8%) | Wins: 28.0% | eps:  0.327\n",
      " 180 | 00:47:52 | Agent:  -2.6% ( -2.9%) | Market:  37.6% ( 63.3%) | Wins: 28.0% | eps:  0.287\n",
      " 190 | 00:51:28 | Agent:   0.8% ( 11.2%) | Market:  43.9% ( 61.3%) | Wins: 28.0% | eps:  0.248\n",
      " 200 | 00:55:07 | Agent:  -0.2% (-14.5%) | Market:  43.9% ( 20.4%) | Wins: 28.0% | eps:  0.208\n",
      " 210 | 00:57:39 | Agent:   1.7% ( 14.5%) | Market:  45.1% ( 37.2%) | Wins: 28.0% | eps:  0.168\n",
      " 220 | 01:00:23 | Agent:   0.6% ( 15.3%) | Market:  41.2% ( 41.3%) | Wins: 30.0% | eps:  0.129\n",
      " 230 | 01:03:12 | Agent:   1.4% (  2.2%) | Market:  35.5% ( 24.2%) | Wins: 33.0% | eps:  0.089\n",
      " 240 | 01:06:03 | Agent:   1.5% (  9.4%) | Market:  32.7% (-14.7%) | Wins: 35.0% | eps:  0.050\n",
      " 250 | 01:08:51 | Agent:   0.3% ( -6.2%) | Market:  31.5% ( 28.6%) | Wins: 32.0% | eps:  0.010\n",
      " 260 | 01:11:38 | Agent:   3.6% ( 17.9%) | Market:  33.1% ( 31.6%) | Wins: 34.0% | eps:  0.009\n",
      " 270 | 01:14:24 | Agent:   6.4% ( 17.4%) | Market:  31.5% ( 21.8%) | Wins: 36.0% | eps:  0.008\n",
      " 280 | 01:17:19 | Agent:   8.6% ( 18.6%) | Market:  33.6% ( 84.3%) | Wins: 35.0% | eps:  0.007\n",
      " 290 | 01:20:21 | Agent:   6.9% ( -5.2%) | Market:  28.4% (  9.6%) | Wins: 35.0% | eps:  0.007\n",
      " 300 | 01:23:18 | Agent:  10.1% ( 16.6%) | Market:  31.0% ( 46.1%) | Wins: 36.0% | eps:  0.006\n",
      " 310 | 01:26:13 | Agent:   8.1% ( -4.9%) | Market:  31.0% ( 36.7%) | Wins: 36.0% | eps:  0.005\n",
      " 320 | 01:29:12 | Agent:   7.3% (  7.4%) | Market:  30.9% ( 40.5%) | Wins: 34.0% | eps:  0.005\n",
      " 330 | 01:32:03 | Agent:  10.9% ( 37.6%) | Market:  32.3% ( 38.6%) | Wins: 34.0% | eps:  0.004\n",
      " 340 | 01:34:48 | Agent:  15.3% ( 54.2%) | Market:  41.8% ( 79.7%) | Wins: 33.0% | eps:  0.004\n",
      " 350 | 01:38:07 | Agent:  18.1% ( 21.0%) | Market:  42.4% ( 35.0%) | Wins: 38.0% | eps:  0.004\n",
      " 360 | 01:41:09 | Agent:  20.5% ( 41.9%) | Market:  39.9% (  6.4%) | Wins: 39.0% | eps:  0.003\n",
      " 370 | 01:44:11 | Agent:  21.6% ( 28.7%) | Market:  40.8% ( 30.6%) | Wins: 39.0% | eps:  0.003\n",
      " 380 | 01:47:17 | Agent:  19.6% ( -1.4%) | Market:  39.3% ( 69.3%) | Wins: 40.0% | eps:  0.003\n",
      " 390 | 01:50:20 | Agent:  22.5% ( 23.7%) | Market:  42.6% ( 42.7%) | Wins: 41.0% | eps:  0.002\n",
      " 400 | 01:53:19 | Agent:  22.2% ( 13.9%) | Market:  40.2% ( 22.7%) | Wins: 44.0% | eps:  0.002\n",
      " 410 | 01:57:03 | Agent:  24.5% ( 18.5%) | Market:  42.3% ( 57.8%) | Wins: 44.0% | eps:  0.002\n",
      " 420 | 02:00:58 | Agent:  28.4% ( 45.9%) | Market:  38.6% (  2.7%) | Wins: 48.0% | eps:  0.002\n",
      " 430 | 02:04:21 | Agent:  25.8% ( 11.1%) | Market:  36.7% ( 19.6%) | Wins: 48.0% | eps:  0.002\n",
      " 440 | 02:07:56 | Agent:  25.8% ( 54.4%) | Market:  30.4% ( 17.3%) | Wins: 49.0% | eps:  0.001\n",
      " 450 | 02:11:14 | Agent:  27.4% ( 37.6%) | Market:  31.8% ( 48.6%) | Wins: 46.0% | eps:  0.001\n",
      " 460 | 02:14:15 | Agent:  26.1% ( 28.9%) | Market:  38.0% ( 68.8%) | Wins: 44.0% | eps:  0.001\n",
      " 470 | 02:17:15 | Agent:  27.0% ( 37.6%) | Market:  37.1% ( 21.5%) | Wins: 45.0% | eps:  0.001\n",
      " 480 | 02:20:20 | Agent:  34.1% ( 69.4%) | Market:  35.9% ( 56.9%) | Wins: 48.0% | eps:  0.001\n",
      " 490 | 02:23:17 | Agent:  34.8% ( 30.6%) | Market:  31.6% ( -0.1%) | Wins: 51.0% | eps:  0.001\n",
      " 500 | 02:26:40 | Agent:  36.1% ( 27.3%) | Market:  32.3% ( 29.9%) | Wins: 49.0% | eps:  0.001\n",
      " 510 | 02:30:11 | Agent:  35.4% ( 11.0%) | Market:  30.2% ( 36.8%) | Wins: 49.0% | eps:  0.001\n",
      " 520 | 02:33:49 | Agent:  31.6% (  7.5%) | Market:  32.8% ( 29.0%) | Wins: 46.0% | eps:  0.001\n",
      " 530 | 02:37:32 | Agent:  34.1% ( 36.8%) | Market:  33.1% ( 22.4%) | Wins: 46.0% | eps:  0.001\n",
      " 540 | 02:41:29 | Agent:  35.4% ( 67.0%) | Market:  33.1% ( 17.4%) | Wins: 50.0% | eps:  0.001\n",
      " 550 | 02:45:14 | Agent:  35.4% ( 38.0%) | Market:  33.2% ( 49.0%) | Wins: 51.0% | eps:  0.000\n",
      " 560 | 02:48:50 | Agent:  36.0% ( 35.0%) | Market:  29.9% ( 35.9%) | Wins: 54.0% | eps:  0.000\n",
      " 570 | 02:52:27 | Agent:  36.1% ( 38.4%) | Market:  31.1% ( 34.2%) | Wins: 53.0% | eps:  0.000\n",
      " 580 | 02:56:06 | Agent:  33.4% ( 41.9%) | Market:  33.4% ( 79.9%) | Wins: 50.0% | eps:  0.000\n",
      " 590 | 02:59:44 | Agent:  31.5% ( 12.6%) | Market:  37.0% ( 35.8%) | Wins: 46.0% | eps:  0.000\n",
      " 600 | 03:03:31 | Agent:  31.7% ( 29.3%) | Market:  39.2% ( 51.3%) | Wins: 45.0% | eps:  0.000\n",
      " 610 | 03:07:22 | Agent:  35.1% ( 44.4%) | Market:  38.5% ( 29.6%) | Wins: 50.0% | eps:  0.000\n",
      " 620 | 03:11:15 | Agent:  38.5% ( 42.1%) | Market:  39.6% ( 40.6%) | Wins: 52.0% | eps:  0.000\n",
      " 630 | 03:15:03 | Agent:  36.5% ( 16.7%) | Market:  40.7% ( 33.5%) | Wins: 51.0% | eps:  0.000\n",
      " 640 | 03:18:45 | Agent:  31.8% ( 20.2%) | Market:  44.5% ( 54.8%) | Wins: 44.0% | eps:  0.000\n",
      " 650 | 03:22:24 | Agent:  32.7% ( 46.9%) | Market:  40.2% (  6.4%) | Wins: 47.0% | eps:  0.000\n",
      " 660 | 03:26:06 | Agent:  32.1% ( 28.6%) | Market:  38.9% ( 22.9%) | Wins: 46.0% | eps:  0.000\n",
      " 670 | 03:29:49 | Agent:  29.5% ( 12.3%) | Market:  37.5% ( 20.2%) | Wins: 46.0% | eps:  0.000\n",
      " 680 | 03:33:33 | Agent:  28.3% ( 29.7%) | Market:  31.2% ( 16.4%) | Wins: 50.0% | eps:  0.000\n",
      " 690 | 03:37:19 | Agent:  32.8% ( 57.9%) | Market:  30.5% ( 29.1%) | Wins: 53.0% | eps:  0.000\n",
      " 700 | 03:41:07 | Agent:  32.2% ( 23.2%) | Market:  27.8% ( 24.7%) | Wins: 53.0% | eps:  0.000\n",
      " 710 | 03:44:55 | Agent:  32.2% ( 44.6%) | Market:  25.4% (  5.0%) | Wins: 51.0% | eps:  0.000\n",
      " 720 | 03:48:46 | Agent:  33.6% ( 55.8%) | Market:  25.5% ( 41.7%) | Wins: 49.0% | eps:  0.000\n",
      " 730 | 03:52:39 | Agent:  32.4% (  4.3%) | Market:  26.5% ( 44.2%) | Wins: 51.0% | eps:  0.000\n",
      " 740 | 03:56:31 | Agent:  32.2% ( 18.9%) | Market:  25.7% ( 46.5%) | Wins: 54.0% | eps:  0.000\n",
      " 750 | 04:00:26 | Agent:  28.4% (  8.5%) | Market:  28.4% ( 33.3%) | Wins: 51.0% | eps:  0.000\n",
      " 760 | 04:04:22 | Agent:  26.1% (  5.5%) | Market:  26.9% (  8.3%) | Wins: 52.0% | eps:  0.000\n",
      " 770 | 04:08:19 | Agent:  25.8% (  9.4%) | Market:  28.1% ( 31.9%) | Wins: 51.0% | eps:  0.000\n",
      " 780 | 04:12:19 | Agent:  27.2% ( 44.1%) | Market:  28.0% ( 15.5%) | Wins: 52.0% | eps:  0.000\n",
      " 790 | 04:16:20 | Agent:  24.1% ( 26.4%) | Market:  29.0% ( 38.5%) | Wins: 52.0% | eps:  0.000\n",
      " 800 | 04:20:23 | Agent:  23.8% ( 20.3%) | Market:  29.3% ( 28.1%) | Wins: 53.0% | eps:  0.000\n",
      " 810 | 04:24:27 | Agent:  27.1% ( 77.7%) | Market:  35.6% ( 67.6%) | Wins: 51.0% | eps:  0.000\n",
      " 820 | 04:28:34 | Agent:  24.5% ( 30.3%) | Market:  34.7% ( 33.0%) | Wins: 51.0% | eps:  0.000\n",
      " 830 | 04:32:43 | Agent:  26.3% ( 21.6%) | Market:  30.3% (  0.7%) | Wins: 52.0% | eps:  0.000\n",
      " 840 | 04:36:53 | Agent:  30.3% ( 59.3%) | Market:  26.7% (  9.7%) | Wins: 54.0% | eps:  0.000\n",
      " 850 | 04:41:05 | Agent:  33.6% ( 41.2%) | Market:  26.2% ( 28.5%) | Wins: 54.0% | eps:  0.000\n",
      " 860 | 04:45:20 | Agent:  38.5% ( 55.0%) | Market:  27.7% ( 23.3%) | Wins: 52.0% | eps:  0.000\n",
      " 870 | 04:49:36 | Agent:  42.9% ( 53.4%) | Market:  24.0% ( -4.8%) | Wins: 56.0% | eps:  0.000\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " 880 | 04:53:54 | Agent:  48.2% ( 96.8%) | Market:  23.8% ( 13.1%) | Wins: 56.0% | eps:  0.000\n",
      " 890 | 04:58:14 | Agent:  50.0% ( 44.1%) | Market:  22.2% ( 22.3%) | Wins: 56.0% | eps:  0.000\n",
      " 900 | 05:02:35 | Agent:  52.9% ( 49.5%) | Market:  22.0% ( 26.5%) | Wins: 58.0% | eps:  0.000\n",
      " 910 | 05:06:58 | Agent:  52.3% ( 72.1%) | Market:  15.5% (  2.9%) | Wins: 60.0% | eps:  0.000\n",
      " 920 | 05:11:23 | Agent:  53.9% ( 45.7%) | Market:  16.9% ( 46.4%) | Wins: 60.0% | eps:  0.000\n",
      " 930 | 05:15:49 | Agent:  60.2% ( 85.2%) | Market:  15.2% (-16.4%) | Wins: 60.0% | eps:  0.000\n",
      " 940 | 05:20:18 | Agent:  56.8% ( 25.5%) | Market:  15.8% ( 15.7%) | Wins: 60.0% | eps:  0.000\n",
      " 950 | 05:24:51 | Agent:  58.4% ( 56.6%) | Market:  17.2% ( 43.3%) | Wins: 61.0% | eps:  0.000\n",
      " 960 | 05:29:24 | Agent:  57.4% ( 45.0%) | Market:  21.4% ( 65.3%) | Wins: 59.0% | eps:  0.000\n",
      " 970 | 05:33:58 | Agent:  54.2% ( 21.4%) | Market:  22.2% (  2.4%) | Wins: 59.0% | eps:  0.000\n",
      " 980 | 05:38:33 | Agent:  49.7% ( 52.2%) | Market:  22.9% ( 20.5%) | Wins: 57.0% | eps:  0.000\n",
      " 990 | 05:43:11 | Agent:  47.6% ( 22.9%) | Market:  19.9% ( -7.9%) | Wins: 57.0% | eps:  0.000\n",
      "1000 | 05:47:51 | Agent:  46.8% ( 41.5%) | Market:  17.6% (  3.5%) | Wins: 57.0% | eps:  0.000\n"
     ]
    }
   ],
   "source": [
    "start = time()\n",
    "results = []\n",
    "for episode in range(1, max_episodes + 1):\n",
    "    this_state = trading_environment.reset()\n",
    "    for episode_step in range(max_episode_steps):\n",
    "        action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))\n",
    "        next_state, reward, done, _ = trading_environment.step(action)\n",
    "    \n",
    "        ddqn.memorize_transition(this_state, \n",
    "                                 action, \n",
    "                                 reward, \n",
    "                                 next_state, \n",
    "                                 0.0 if done else 1.0)\n",
    "        if ddqn.train:\n",
    "            ddqn.experience_replay()\n",
    "        if done:\n",
    "            break\n",
    "        this_state = next_state\n",
    "\n",
    "    # get DataFrame with seqence of actions, returns and nav values\n",
    "    result = trading_environment.env.simulator.result()\n",
    "    \n",
    "    # get results of last step\n",
    "    final = result.iloc[-1]\n",
    "\n",
    "    # apply return (net of cost) of last action to last starting nav \n",
    "    nav = final.nav * (1 + final.strategy_return)\n",
    "    navs.append(nav)\n",
    "\n",
    "    # market nav \n",
    "    market_nav = final.market_nav\n",
    "    market_navs.append(market_nav)\n",
    "\n",
    "    # track difference between agent an market NAV results\n",
    "    diff = nav - market_nav\n",
    "    diffs.append(diff)\n",
    "    \n",
    "    if episode % 10 == 0:\n",
    "        track_results(episode, \n",
    "                      # show mov. average results for 100 (10) periods\n",
    "                      np.mean(navs[-100:]), \n",
    "                      np.mean(navs[-10:]), \n",
    "                      np.mean(market_navs[-100:]), \n",
    "                      np.mean(market_navs[-10:]), \n",
    "                      # share of agent wins, defined as higher ending nav\n",
    "                      np.sum([s > 0 for s in diffs[-100:]])/min(len(diffs), 100), \n",
    "                      time() - start, ddqn.epsilon)\n",
    "    if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):\n",
    "        print(result.tail())\n",
    "        break\n",
    "\n",
    "trading_environment.close()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Store Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "ExecuteTime": {
     "start_time": "2021-02-25T06:20:28.020Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 1000 entries, 1 to 1000\n",
      "Data columns (total 4 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Agent              1000 non-null   float64\n",
      " 1   Market             1000 non-null   float64\n",
      " 2   Difference         1000 non-null   float64\n",
      " 3   Strategy Wins (%)  901 non-null    float64\n",
      "dtypes: float64(4)\n",
      "memory usage: 39.1 KB\n"
     ]
    }
   ],
   "source": [
    "results = pd.DataFrame({'Episode': list(range(1, episode+1)),\n",
    "                        'Agent': navs,\n",
    "                        'Market': market_navs,\n",
    "                        'Difference': diffs}).set_index('Episode')\n",
    "\n",
    "results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()\n",
    "results.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "ExecuteTime": {
     "start_time": "2021-02-25T06:20:28.023Z"
    }
   },
   "outputs": [],
   "source": [
    "results.to_csv(results_path / 'results.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "ExecuteTime": {
     "start_time": "2021-02-25T06:20:28.025Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "with sns.axes_style('white'):\n",
    "    sns.distplot(results.Difference)\n",
    "    sns.despine()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Evaluate Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "ExecuteTime": {
     "start_time": "2021-02-25T06:20:28.029Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 1000 entries, 1 to 1000\n",
      "Data columns (total 4 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Agent              1000 non-null   float64\n",
      " 1   Market             1000 non-null   float64\n",
      " 2   Difference         1000 non-null   float64\n",
      " 3   Strategy Wins (%)  901 non-null    float64\n",
      "dtypes: float64(4)\n",
      "memory usage: 39.1 KB\n"
     ]
    }
   ],
   "source": [
    "results.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following diagram shows the rolling average of agent and market returns over 100 periods on the left, and the share of the last 100 periods the agent outperformed the market on the right. It uses AAPL stock data with some 9,000 daily price and volume observations, corresponding to ~35 years of data. \n",
    "\n",
    "It shows how the agent's performance improves significantly while exploring at a higher rate over the first ~600 periods (that is, years) and approaches a level where it outperforms the market around 40 percent of the time, despite transaction costs. In an increasing number of instances, it beats the market over half the time out of 100 periods:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "ExecuteTime": {
     "start_time": "2021-02-25T06:20:28.031Z"
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA+gAAAEYCAYAAADPrtzUAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAACqdElEQVR4nOzdd3hT1RvA8W/SdO+9C7RAKbtQKMjee8hQHKg4AJHtwoWKAxWFnyJDHIiIA5CWPQQZyix7lVXoXnSvpG3G749L04ZuKLSF83mePklu7jj33jQ37z3nvEem0+l0CIIgCIIgCIIgCIJQq+S1XQBBEARBEARBEARBEESALgiCIAiCIAiCIAh1ggjQBUEQBEEQBEEQBKEOEAG6IAiCIAiCIAiCINQBIkAXBEEQBEEQBEEQhDpABOiCIAiCIAiCIAiCUAeIAF0Q7oGjR4/SvXv32i7GffHVV1/x888/37ftzZ07lyVLlty37dU38+fP5/fff6/tYgiCINQLKpWKyZMn0759e6ZPn17bxblv/vjjDz755JPaLsYdWb58Oe+8805tF6PO+uWXX/jyyy9ruxjCXRABuvBAGD9+PB06dKCgoKC2i1Il/v7+tG3blsDAQLp168b8+fPRaDRVWrZ3794cOnToHpewatLS0ggNDWXcuHGAdGPC39+fqVOnGsx36dIl/P39GT9+/F1vc968ebzyyit3vZ7y5OXlERgYyEsvvXTPtnEvvfDCCyxfvrze/C8IgnD3auMa6O/vT1RUVIXzJCYm8uqrrxIcHEzbtm0ZM2YMe/furfI2NmzYwBNPPHG3Ra3Qjh07SElJ4ejRo3zzzTf3dFt1RUFBAcuWLePFF18EIDs7mxdeeIGgoCBeffVVg98j7777Ln///XeV1x0bG4u/vz+PPvqowfS0tDRatmxJ796977r8kydPvqc3F3Q6HX369GHw4MH3bBv30uOPP86mTZtITU2t7aIId0gE6EK9Fxsby/Hjx5HJZOzZs6e2i1NlGzdu5NSpU/z6669s27aNv/76675sV61W19i6NmzYQI8ePTAzM9NPc3Bw4NSpU6Snp+unhYSE0LBhwxrb7r20c+dOTExMOHjwIMnJyTW+/po8/mVxcXHB19eXf/75555uRxCEuqGuXgMzMjJ48sknMTExYcuWLRw5coTnnnuOV199lR07dtR28QDQaDTEx8fTsGFDFApFtZe/19/n98qePXvw9fXF1dUVkGrTAwICOHToEHFxcfqA/NSpU9y8eZN+/fpVext5eXlcuXJF/3rLli14enrWzA7cY2FhYaSlpRETE8PZs2drfP33+nNjampK9+7dCQ0NvafbEe4dEaAL9V5oaCht2rTh0UcfLfVlNGfOHD788EMmTpxIYGAgY8eOJTo6Wv++v78/v//+O/3796dDhw58+OGH6HQ6ABYvXsxrr72mn7fornDRF+tff/3FoEGDCAwMpE+fPvzxxx93VP4GDRrQrl07wsPD9dP27t3LiBEjCAoKYty4cVy6dAmA119/nfj4eCZPnkxgYCDff/99mc3pS9ayL168mOnTp/Paa6/Rrl07QkJCGD9+PP/73/8YN24cgYGBPP/886SlpQGQn5/Pa6+9RnBwMEFBQYwePZqUlJQyy37gwAE6dOhgMM3Y2Jg+ffqwbds2QPoBtH37doYNG2Yw38mTJxk9ejTt27dn9OjRnDx5EoCtW7cyatQog3l//vlnJk+eDEjndNGiRUBxV4KffvqJzp0707VrV4MbHenp6UyePJl27doxevRoFi1aVGltTEhICOPGjcPf35/NmzcDcPr0abp06WJQq/D333/r90mr1bJixQr69u1LcHAwM2bMICMjAyj+3Kxbt46ePXvy7LPPAjB9+nS6dOlC+/bteeqpp7h69WqVyx0REcGECRPo2LEjAwYM0B/rIh07dmT//v0V7qcgCA+Giq6Bd/NdUtH186mnngJgxIgRBAYGlvoOAul728LCgk8++QRnZ2fMzMwYOnQokydP5vPPP0en05W6roLUGmDdunVERETw/vvvc/r0aQIDAwkKCtKXa+7cuUyYMIHAwECefvpp4uLiqrxP77//Pi+99BJt27blqaeeYunSpWzfvp3AwEDWrVuHVqtl6dKl9OrVi86dO/PGG2+QnZ0NlP19vmHDBsaNG8enn35KUFAQffr04eTJk/ob2J07dyYkJERfhn379jFy5EjatWtHjx49WLx4sf69ovWHhITQs2dPgoODWbZsmf59jUbD8uXL6du3L4GBgYwaNYqEhIRK9/t2t1+7Y2NjCQ4OxsTEhKCgIGJiYtBoNMyfP/+Om5KPGDHCYL9DQ0MZOXKkwTwRERGMHz+eoKAghgwZor/BVNk1t+Tvs8qOmUql4s0336RDhw4M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\n",
      "text/plain": [
       "<Figure size 1008x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)\n",
    "\n",
    "df1 = (results[['Agent', 'Market']]\n",
    "       .sub(1)\n",
    "       .rolling(100)\n",
    "       .mean())\n",
    "df1.plot(ax=axes[0],\n",
    "         title='Annual Returns (Moving Average)',\n",
    "         lw=1)\n",
    "\n",
    "df2 = results['Strategy Wins (%)'].div(100).rolling(50).mean()\n",
    "df2.plot(ax=axes[1],\n",
    "         title='Agent Outperformance (%, Moving Average)')\n",
    "\n",
    "for ax in axes:\n",
    "    ax.yaxis.set_major_formatter(\n",
    "        FuncFormatter(lambda y, _: '{:.0%}'.format(y)))\n",
    "    ax.xaxis.set_major_formatter(\n",
    "        FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))\n",
    "axes[1].axhline(.5, ls='--', c='k', lw=1)\n",
    "\n",
    "sns.despine()\n",
    "fig.tight_layout()\n",
    "fig.savefig(results_path / 'performance', dpi=300)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This relatively simple agent uses **no information beyond the latest market data and the reward signal** compared to the machine learning models we covered elsewhere in this book. Nonetheless, it learns to make a profit and achieve performance similar to that of the market (after training on <1,000 years' worth of data, which takes only a fraction of the time on a GPU).\n",
    "\n",
    "Keep in mind that using a single stock also increases the **risk of overfitting** to the data — by a lot. You can test your trained agent on new data using the saved model (see the notebook for Lunar Lander).\n",
    "\n",
    "In summary, we have demonstrated the **mechanics of setting up an RL trading environment** and experimented with a basic agent that uses a small number of technical indicators. You should **try to extend both the environment and the agent** - for example, by allowing it to choose from several assets, size the positions, and manage risks.\n",
    "\n",
    "Reinforcement learning is often considered **one of the most promising approaches to algorithmic trading** because it most accurately models the task an investor is facing. However, our dramatically simplified examples illustrate that creating a **realistic environment poses a considerable challenge**. Moreover, deep reinforcement learning that has achieved impressive breakthroughs in other domains may face greater obstacles given the noisy nature of financial data, which makes it even harder to learn a value function based on delayed rewards.\n",
    "\n",
    "Nonetheless, the substantial interest in this subject makes it likely that institutional investors are working on larger-scale experiments that may yield tangible results. An interesting complementary approach beyond the scope of this book is **Inverse Reinforcement Learning**, which aims to identify the reward function of an agent(for example, a human trader) given its observed behavior; see [Arora and Doshi (2019)](https://www.semanticscholar.org/paper/A-Survey-of-Inverse-Reinforcement-Learning%3A-Methods-Arora-Doshi/9d4d8509f6da094a7c31e063f307e0e8592db27f) for a survey and [Roa-Vicens et al. (2019)](https://deepai.org/publication/towards-inverse-reinforcement-learning-for-limit-order-book-dynamics) for an application on trading in the limit-order book context.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": true,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {
    "height": "calc(100% - 180px)",
    "left": "10px",
    "top": "150px",
    "width": "230.906px"
   },
   "toc_section_display": true,
   "toc_window_display": true
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}