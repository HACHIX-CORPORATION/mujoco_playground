# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.locomotion.hunter import hunter_constants
from mujoco_playground._src.locomotion.hunter.base import get_assets
# from mujoco_playground.experimental.sim2sim.keyboard_reader import Gamepad

_HERE = epath.Path(__file__).parent
_ONNX_DIR = epath.Path("/home/sandbox/Work/mujoco_playground/learning/notebooks/onnx/")


class OnnxController:
  """ONNX controller for the Hunter humanoid."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      ctrl_dt: float,
      n_substeps: int,
      action_scale: float = 0.5,
      vel_scale_x: float = 1.0,
      vel_scale_y: float = 1.0,
      vel_scale_rot: float = 1.0,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)

    self._counter = 0
    self._n_substeps = n_substeps
    self._ctrl_dt = ctrl_dt

    self._phase = np.array([0.0, np.pi])
    self._gait_freq = 1.5
    self._phase_dt = 2 * np.pi * self._gait_freq * ctrl_dt

    # self._joystick = Gamepad(
    #     vel_scale_x=vel_scale_x,
    #     vel_scale_y=vel_scale_y,
    #     vel_scale_rot=vel_scale_rot
    # )

  def get_obs(self, model, data) -> np.ndarray:
    linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    gravity = data.sensor("upvector").data
    joint_angles = data.qpos[7:]
    joint_velocities = data.qvel[6:]
    # command = self._joystick.get_command()
    # ph = self._phase if np.linalg.norm(command) >= 0.01 else np.ones(2) * np.pi
    # phase = np.concatenate([np.cos(ph), np.sin(ph)])
    # joint_angles[:2] *= 0.0
    # joint_velocities[:2] *= 0.0
    linvel_scale = 2.0
    joint_velocities_scale = 0.05
    obs = np.hstack([
        linvel * linvel_scale,
        gyro,
        gravity,
        joint_angles - self._default_angles,
        joint_velocities * joint_velocities_scale,
        self._last_action,
        # command,
        # phase,
    ])
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    # if self._counter % self._n_substeps == 0:
    #   obs = self.get_obs(model, data)
    #   onnx_input = {"obs": obs.reshape(1, -1)}
    #   onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
    #   self._last_action = onnx_pred.copy()
    #   data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
    #   phase_tp1 = self._phase + self._phase_dt
    #   self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
    data.ctrl[:] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    data.ctrl[0] = 0.2 * np.sin(
      2 * np.pi * 0.5 * self._counter * self._ctrl_dt)

def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
      hunter_constants.HUNTER_FIXED_TERRAIN_XML.as_posix(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 0)

  ctrl_dt = 0.02
  sim_dt = 0.002
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  policy = OnnxController(
      policy_path=(_ONNX_DIR / "stand_202511180751.onnx").as_posix(),
      default_angles=np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
      ctrl_dt=ctrl_dt,
      n_substeps=n_substeps,
      action_scale=0.5,
      vel_scale_x=1.0,
      vel_scale_y=0.8,
      vel_scale_rot=1.0,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
