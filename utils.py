#utils　一応真似して書いたけど未使用

import numpy as np

def one_hot_encode(value, max_value):
    encoding = np.zeros(max_value)
    if value < max_value:
        encoding[value] = 1
    return encoding

def encode_direction(direction):
    mapping = {'up': 0, 'down': 1, 'idle': 2}
    one_hot = np.zeros(3)
    if direction in mapping:
        one_hot[mapping[direction]] = 1
    return one_hot
