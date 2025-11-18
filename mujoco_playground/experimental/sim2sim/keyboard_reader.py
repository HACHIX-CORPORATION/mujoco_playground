# Copyright 2025 DeepMind Technologies Limited
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
# pylint: disable=line-too-long
"""Keyboard-driven Gamepad class using the `keyboard` library instead of pynput.
"""
import threading
import time

import numpy as np
import keyboard as kb


def _interpolate(value, old_max, new_scale, deadzone=0.01):
  ret = value * new_scale / old_max
  if abs(ret) < deadzone:
    return 0.0
  return ret


class Gamepad:
  """Gamepad class that reads keyboard keys (w/a/s/d/q/e/arrow keys/space)."""

  def __init__(
      self,
      vendor_id=0x046D,
      product_id=0xC219,
      vel_scale_x=0.4,
      vel_scale_y=0.4,
      vel_scale_rot=1.0,
  ):
    self._vendor_id = vendor_id
    self._product_id = product_id
    self._vel_scale_x = vel_scale_x
    self._vel_scale_y = vel_scale_y
    self._vel_scale_rot = vel_scale_rot

    self.vx = 0.0
    self.vy = 0.0
    self.wz = 0.0
    self.is_running = True

    self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
    self.read_thread.start()

  def read_loop(self):
    # Poll keyboard state at fixed frequency using the `keyboard` library.
    update_hz = 30.0
    try:
      while self.is_running:
        self._update_from_keys()
        time.sleep(1.0 / update_hz)
    except Exception:
      # keep loop robust to unexpected keyboard library errors
      pass

  def _update_from_keys(self):
    # Query relevant keys directly.
    # Note: keyboard.is_pressed may require elevated privileges on some platforms.
    vx = 0.0
    vy = 0.0
    wz = 0.0

    # Forward / backward
    if kb.is_pressed("w") or kb.is_pressed("up"):
      vx += self._vel_scale_x
    if kb.is_pressed("s") or kb.is_pressed("down"):
      vx -= self._vel_scale_x

    # Left / right (lateral)
    if kb.is_pressed("d") or kb.is_pressed("right"):
      vy += self._vel_scale_y
    if kb.is_pressed("a") or kb.is_pressed("left"):
      vy -= self._vel_scale_y

    # Rotation
    if kb.is_pressed("e"):
      wz += self._vel_scale_rot
    if kb.is_pressed("q"):
      wz -= self._vel_scale_rot

    # Emergency stop (space) clears motions
    if kb.is_pressed("space"):
      vx = 0.0
      vy = 0.0
      wz = 0.0

    # Apply deadzone via the same interpolate helper (input assumed full-scale ±1).
    self.vx = _interpolate(vx, 1.0, 1.0)  # vx already scaled
    self.vy = _interpolate(vy, 1.0, 1.0)
    self.wz = _interpolate(wz, 1.0, 1.0)

  def update_command(self, data):
    # kept for compatibility but not used for keyboard input
    # Accept a tuple/list (vx, vy, wz) to directly set velocities if needed.
    try:
      vx, vy, wz = data
      self.vx = float(vx)
      self.vy = float(vy)
      self.wz = float(wz)
    except Exception:
      pass

  def get_command(self):
    return np.array([self.vx, self.vy, self.wz])

  def stop(self):
    self.is_running = False


if __name__ == "__main__":
  gamepad = Gamepad()
  try:
    while True:
      print(gamepad.get_command())
      time.sleep(0.1)
  except KeyboardInterrupt:
    gamepad.stop()
