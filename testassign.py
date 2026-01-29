import numpy as np
import torch
cost = np.array([[4., 1., 3], [2, 0, 5], [3, 2, 2]])
from scipy.optimize import linear_sum_assignment
gx = np.tanh(cost,cost)
tgx = torch.tanh(torch.from_numpy(cost))
a= 1
