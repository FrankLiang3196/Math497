# Author: Jonathan Siegel
#
# Contains an implementation of the l1 prox which calculates the maximum across
# all of the convolutions on the same grid.

import torch
import math

class l1_prox_resnet:

  def __init__(self, lam, maximum_factor, mode = 'normal'):
    self.lam = lam
    self.maximum_factor = maximum_factor
    self.mode = mode
    self.maxima = {}
    return

  def apply(self, p, backward_step):
    p.data.copy_(torch.clamp(p - self.lam * backward_step, min=0) + torch.clamp(p + self.lam * backward_step, max=0))

  def get_zero_params(self, p):
    return torch.zeros_like(p.data)

  def get_running_av(self, p):
    with torch.no_grad():
      if len(p.shape) == 4 and self.mode == 'kernel':
        norms = torch.norm(p, p=1, dim=[2,3])
        return (1.0 / (p.shape[2] * p.shape[3])) * norms[:,:,None,None] * torch.ones_like(p.data)
      return torch.abs(p.data)

  def calculate_backward_v(self, running_av): 
    maximum = self.maxima[running_av.shape]
    if maximum > 0 and len(running_av.shape) == 4 and self.mode == 'kernel':
      return math.sqrt(running_av.shape[2] * running_av.shape[3]) * self.maximum_factor / (1.0 + (self.maximum_factor - 1.0)*(running_av / maximum))
    elif maximum > 0 and len(running_av.shape) == 4 or len(running_av.shape) == 2:
      return self.maximum_factor / (1.0 + (self.maximum_factor - 1.0)*(running_av / maximum))
    else:
      return torch.zeros_like(running_av)

  def register_running_av(self, running_av):
    if not running_av.shape in self.maxima:
      self.maxima[running_av.shape] = torch.max(running_av)
    if self.maxima[running_av.shape] < torch.max(running_av):
      self.maxima[running_av.shape] = torch.max(running_av) 
    return

  def reset(self):
    # Reset the maximum calculation at the beginning of each iteration.
    self.maxima.clear()
