# %%
import numpy as np
from nelder_mead import bbox

v_f1 = np.load("vertices_1.npy")

v_f1[-1]
print(v_f1[-1][0])
print(bbox(v_f1[-1][0], 1))


v_f2 = np.load("vertices_2.npy")

v_f2[-1]
print(v_f2[-1][0])
print(bbox(v_f2[-1][0], 2))

v_f3 = np.load("vertices_3.npy")

v_f3[-1]
print(v_f3[-1][0])
print(bbox(v_f3[-1][0], 3))
