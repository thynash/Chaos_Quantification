from logistic import logistic_map
from henon import henon_map
from coupled_logistic import coupled_logistic_map

x = logistic_map(3.9, 0.2, 1000)
print("Logistic OK:", x.shape)

h = henon_map(1.4, 0.3, 0.1, 0.1, 1000)
print("Henon OK:", h.shape)

c = coupled_logistic_map(3.9, 0.05, 0.2, 0.21, 1000)
print("Coupled OK:", c.shape)

