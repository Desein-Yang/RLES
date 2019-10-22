# Evolution Strategy (REINFORCE)

本Repository为OpenAI 2017年论文《Evolution Strategy as a scalable alternative of reinforcement learning》的复现。

#### Environment
Python 3.7.2
OpenAI Gym:Mujoco Atari

#### Framework
1.  **preprocess** (from **raw pixel+reward signal** to 84x84x4 **tensor)**

2.  **agent**： create agent
3.  **policy**： create class policy，construct a network with different updater to optimize policy
4.  **other**：parallel, callback, logging and load from hdf5,etc.

![Uml class](.\uml.png)

#### Dependency
- keras
- tensorflow/theano
- pickle
- multiprocessing
- gym

#### Reference
[OpenAi的Gym开放测试环境](http://gym.openai.com/)
[OpenAI Gym Documention](http://gym.openai.com/docs/)

#### Github Source code
[Open AI source code](https://github.com/openai/evolution-strategies-starter)
[keras TRPO network](https://github.com/joschu/modular_rl)
[Evolutionary Strategy python ](https://github.com/alirezamika/evostra)
