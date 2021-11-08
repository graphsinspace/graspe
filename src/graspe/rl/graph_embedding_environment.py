from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


class PyEnvironment(object):

  def reset(self):
    """Return initial_time_step."""
    self._current_time_step = self._reset()
    return self._current_time_step

  def step(self, action):
    """Apply action and return new time_step."""
    if self._current_time_step is None:
        return self.reset()
    self._current_time_step = self._step(action)
    return self._current_time_step

  def current_time_step(self):
    return self._current_time_step

  def time_step_spec(self):
    """Return time_step_spec."""

  @abc.abstractmethod
  def observation_spec(self):
    """Return observation_spec."""

  @abc.abstractmethod
  def action_spec(self):
    """Return action_spec."""

  @abc.abstractmethod
  def _reset(self):
    """Return initial_time_step."""

  @abc.abstractmethod
  def _step(self, action):
    """Apply action and return new time_step."""
