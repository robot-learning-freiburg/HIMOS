import time
import warnings
from typing import Optional, Tuple

import numpy as np

#from src.highlevel_policy.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper

class VecMonitor(VecEnvWrapper):
    """
    A vectorized monitor wrapper for *vectorized* Gym environments,
    it is used to record the episode reward, length, time and other data.

    Some environments like `openai/procgen <https://github.com/openai/procgen>`_
    or `gym3 <https://github.com/openai/gym3>`_ directly initialize the
    vectorized environments, without giving us a chance to use the ``Monitor``
    wrapper. So this class simply does the job of the ``Monitor`` wrapper on
    a vectorized level.

    :param venv: The vectorized environment
    :param filename: the location to save a log file, can be None for no log
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    def __init__(
        self,
        venv: VecEnv,
        filename: Optional[str] = None,
        info_keywords: Tuple[str, ...] = (),
    ):
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor, ResultsWriter

        # This check is not valid for special `VecEnv`
        # like the ones created by Procgen, that does follow completely
        # the `VecEnv` interface
        try:
            is_wrapped_with_monitor = venv.env_is_wrapped(Monitor)[0]
        except AttributeError:
            is_wrapped_with_monitor = False

        if is_wrapped_with_monitor:
            warnings.warn(
                "The environment is already wrapped with a `Monitor` wrapper"
                "but you are wrapping it with a `VecMonitor` wrapper, the `Monitor` statistics will be"
                "overwritten by the `VecMonitor` ones.",
                UserWarning,
            )

        VecEnvWrapper.__init__(self, venv)
        self.episode_returns = None
        self.episode_lengths = None
        self.episode_count = 0
        self.t_start = time.time()

        env_id = None
        if hasattr(venv, "spec") and venv.spec is not None:
            env_id = venv.spec.id

        if filename:
            self.results_writer = ResultsWriter(
                filename, header={"t_start": self.t_start, "env_id": env_id}, extra_keys=info_keywords
            )
        else:
            self.results_writer = None
        self.info_keywords = info_keywords

    def reset(self,indices=None) -> VecEnvObs:

        obs = self.venv.reset(indices)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

      
        return obs
    
    #added here to avoid errors in base_vec_env in SB3's common lib.
    def step(self, actions: np.ndarray,indices=None) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        if indices is None:
            self.step_async(actions)
            return self.step_wait()
        else:
            self.step_async(actions,indices)
            return self.step_wait(indices)

    def step_wait(self,indices=None) -> VecEnvStepReturn:

        obs, rewards, dones, infos = self.venv.step_wait(indices)
           
        
        new_infos = list(infos[:])
      
        for i,indice in enumerate(indices):
            self.episode_returns[indice] += rewards[indice]
            self.episode_lengths[indice] += 1
            if dones[indice]:
                info = infos[indice].copy()
                episode_return = self.episode_returns[indice]
                episode_length = self.episode_lengths[indice]
                episode_info = {"r": episode_return, "l": episode_length, "t": round(time.time() - self.t_start, 6)}
                
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[indice] = 0
                self.episode_lengths[indice] = 0
                if self.results_writer:
                   
                    self.results_writer.write_row(episode_info)
                
                
                new_infos[indice] = info
                #need this line there tbecause otherwise, the returned value would be simply overwritten by the next iteration within the 
                #low level 8 steps execution process. Therefore, it needs to be written info self.info buffer which is held in subproc_vec_env_HRL.py
                #print("SET info due to DONE")
                self.infos = np.array(self.infos)
                self.infos[indice] = info
                self.infos = tuple(self.infos)  
                

        return obs, rewards, dones, new_infos

    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return self.venv.close()
