# Residual Reinforcement Learning

Implementation of [Residual Reinforcement Learning for Robot
Control](https://arxiv.org/abs/1812.03201) (JSRL) with [Stable
Baselines3](https://github.com/DLR-RM/stable-baselines3).

## Installation

```bash
pip install residual-rl
```

## Usage

See `examples/train_rrl.py` for examples on how to train TD3 + RRL on the
`PointMaze-v3` environment.

## References

```bibtex
@misc{https://doi.org/10.48550/arxiv.1812.03201,
  doi = {10.48550/ARXIV.1812.03201},
  url = {https://arxiv.org/abs/1812.03201},
  author = {Johannink,  Tobias and Bahl,  Shikhar and Nair,  Ashvin and Luo,  Jianlan and Kumar,  Avinash and Loskyll,  Matthias and Ojea,  Juan Aparicio and Solowjow,  Eugen and Levine,  Sergey},
  keywords = {Robotics (cs.RO),  Machine Learning (cs.LG),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Residual Reinforcement Learning for Robot Control},
  publisher = {arXiv},
  year = {2018},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
@software{gymnasium_robotics2023github,
  author = {Rodrigo de Lazcano and Kallinteris Andreas and Jun Jet Tai and Seungjae Ryan Lee and Jordan Terry},
  title = {Gymnasium Robotics},
  url = {http://github.com/Farama-Foundation/Gymnasium-Robotics},
  version = {1.2.0},
  year = {2023},
}
```
