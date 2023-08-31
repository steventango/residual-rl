from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from typing import Union, Optional, Tuple, Dict
import numpy as np


def get_rrl_policy(ExplorationPolicy: BasePolicy):
    class RRLPolicy(ExplorationPolicy):
        def __init__(self, *args, control_policy: BasePolicy = None, alpha: float = 0.5, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.control_policy = control_policy

            # Used for logging
            self.residual_actions = None
            self.control_actions = None
            self.actions = None
            self.alpha = alpha

        def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            """
            Get the policy action from an observation (and optional hidden state).
            Includes sugar-coating to handle different observations (e.g. normalizing images).

            :param observation: the input observation
            :param state: The last hidden states (can be None, used in recurrent policies)
            :param episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            :param deterministic: Whether or not to return deterministic actions.
            :return: the model's action and the next hidden state
                (used in recurrent policies)
            """
            residual_actions, residual_state = super().predict(observation, state, episode_start, deterministic)
            control_actions, _ = self.control_policy.predict(observation, state, episode_start, deterministic)
            self.residual_actions = residual_actions
            self.control_actions = control_actions
            self.actions = self.alpha * residual_actions + (1 - self.alpha) * control_actions
            return self.actions, residual_state

    return RRLPolicy


def get_rrl_algorithm(Algorithm: BaseAlgorithm):
    class RRLAlgorithm(Algorithm):
        def __init__(self, policy, *args, **kwargs):
            if isinstance(policy, str):
                policy = self._get_policy_from_name(policy)
            else:
                policy = policy
            policy = get_rrl_policy(policy)
            super().__init__(policy, *args, **kwargs)

    return RRLAlgorithm
