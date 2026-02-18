import numpy as np

x = np.linspace(12, 134, 10)
y = np.linspace(10, 123, 10)
print(f"{x}\n{y}")
for a, b in zip(x, y):
    print(a, b)