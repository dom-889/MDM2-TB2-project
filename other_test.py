import numpy as np

y = np.trunc(np.linspace(0, 10, 9))

print(y)

n = [i for i in range(100)]

for i in y:
    print(n[int(i)])