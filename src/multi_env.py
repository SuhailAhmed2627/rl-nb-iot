import random
from typing import Literal
import numpy as np
import random
from typing import Tuple
from typing import List
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Device():
   id: int
   ce_group: int
   status: Literal['idle', 'collision', 'success']

   def __init__(self, id: int, ce_group: int):
      self.id = id
      self.status = 'idle'
      self.ce_group = ce_group

   def __str__(self):
      return f'{self.id}'

   def create_sequence(self, transmitting_freq: int, n_repe: int):
      # create a sequence of n_repe frequencies
      n_sequences = np.ndarray((n_repe, 4), dtype=int)

      for n in range(n_repe):
         sequence = []
         sequence.append(transmitting_freq)
         sequence.append(transmitting_freq + random.randint(-4, 4))
         sequence.append(transmitting_freq + random.randint(-4, 4))
         sequence.append(transmitting_freq + random.randint(-4, 4))
         n_sequences[n] = sequence

      return n_sequences

   def initiate_connection(self, f_lb: int, f_ub: int, n_repe: int):
      # pick a random frequency between f_lb and f_ub - 4
      transmitting_freq = random.randint(f_lb, f_ub - 4)
      sequence = self.create_sequence(transmitting_freq, n_repe)
      return sequence


TTIS = np.load('TTI-Final.npy')
MAX_FREQ = 128
LEN_SEQ = 4

class MultiNBIoT(gym.Env):
   devices: List[Device]
   tti: int # The current TTI
   obs: int # The number of devices that have successfully transmitted
   id_counter: int # The id of the next device to be created 

   def __init__(self):
      self.action_space = spaces.MultiDiscrete([4, 6, 3, 4, 6, 3, 4, 6, 3]) 
      self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int8)
      self.id_counter = 0

   def _get_obs(self):
      return np.array([self.obs])
   
   def _get_new_devices(self, ce_group: int = 0):
      new_devices = []
      for i in range(int(TTIS[self.tti])):
         device = Device(self.id_counter, ce_group)
         new_devices.append(device)
         self.id_counter += 1
      return new_devices / 3
   
   def _perform_rach(self, preamble_freq: int, n_repe: int, devices:List[Device]):
      f_lb, f_ub = self._frequency_bounds(preamble_freq)

      device_wise_sequences: List[Tuple[List[int], int]] = []
      for device in devices:
         sequence = device.initiate_connection(f_lb, f_ub, n_repe)
         device_wise_sequences.append((sequence, device.id))

      curr_devices_ids = [device.id for device in devices]
      overall_successful_device_ids = []

      for i in range(0, n_repe):
         time_sequence: List[List[Tuple[int, int]]] = []

         for s in overall_successful_device_ids:
            for d in device_wise_sequences:
               if d[1] == s:
                  del d
                  break

         for t in range(0, LEN_SEQ):
            time_sequence_t: List[Tuple[int, int]] = []
            for d in range(0, len(device_wise_sequences)):
               sequence, device_id = device_wise_sequences[d]
               sequence_i = sequence[i]
               sequence_t = sequence_i[t]
               time_sequence_t.append((sequence_t, device_id))
            time_sequence.append(time_sequence_t)
         
         successful_device_ids = curr_devices_ids.copy()

         for n in range(0, LEN_SEQ):
            curr_time_seq = time_sequence[n]
            curr_time_seq.sort(key=lambda x: x[0])
            for i in range(0, len(curr_time_seq) - 1):
               freq_i, device_id_i = curr_time_seq[i]
               freq_j, device_id_j = curr_time_seq[i + 1]
               if freq_i == freq_j:
                  if device_id_i in successful_device_ids:
                     successful_device_ids.remove(device_id_i)
                  if device_id_j in successful_device_ids:
                     successful_device_ids.remove(device_id_j)

         for device_id in successful_device_ids:
            curr_devices_ids.remove(device_id)
            overall_successful_device_ids.append(device_id)

      return overall_successful_device_ids


   def _action_to_params(self, action):
      f_prea_vals = [12, 24, 36, 48] # Preamble Frequency
      n_repe_vals = [1, 2, 4, 8, 16, 32] # Number of repetitions
      n_rach_vals = [1, 2, 4] # Number of RACH occasions
      return f_prea_vals[action[0]], n_repe_vals[action[1]], n_rach_vals[action[2]], f_prea_vals[action[3]], n_repe_vals[action[4]], n_rach_vals[action[5]], f_prea_vals[action[6]], n_repe_vals[action[7]], n_rach_vals[action[8]]

   def _frequency_bounds(self, preamble_freq):
      f_lb = random.randint(0, MAX_FREQ - preamble_freq)
      f_ub = f_lb + preamble_freq
      return f_lb, f_ub

   def reset(self, seed=None):
      super().reset(seed=seed)
      self.devices = []
      self.tti = 0
      obs = np.array([0])
      
      # only return the observation, no additional info
      return obs, "Some info"

   def step(self, action: int):
      # action is an array of size 3 containing the preamble frequency, number of repeats, and number of RACH occasions
      action = self._action_to_params(action)
      preamble_freq = action[0]
      n_repe = action[1]
      n_rach = action[2]

      print(f'Preamble freq: {preamble_freq}, n_repe: {n_repe}, n_rach: {n_rach}')
      print(f'TTI: {self.tti}')

      total_no_of_successes = 0
      for _ in range(3):
         new_devices = self._get_new_devices()
         print(f'New devices: {[device.id for device in new_devices]}')
         self.devices.extend(new_devices)

         no_of_successes = 0

         for _ in range(0, n_rach):
            successful_device_ids = self._perform_rach(preamble_freq, n_repe, self.devices)
            no_of_successes += len(successful_device_ids)
            self.devices = [device for device in self.devices if device.id not in successful_device_ids]
         
         total_no_of_successes += no_of_successes

      self.obs = np.array(total_no_of_successes)
      self.tti += 1

      return self._get_obs(), total_no_of_successes/10, self.tti == len(TTIS), self.tti == len(TTIS), {}