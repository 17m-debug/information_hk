import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

df = pd.read_excel('data/data2.xls', header=None)
data_np = df.to_numpy()
# print(data_np)

print(f"下为对data2的求解")
# H(X)
H_x = 0
p_x = np.zeros((1, 5))
for i in range(1, 4):
    for j in range(1, 5):
        p_x[0, j-1] += data_np[i-1, j-1]
# print(p_x)

for i in range(1, 5):
    H_x += -p_x[0, i-1] * np.log2(p_x[0, i-1])
# print(H_x)
print(f'H(X)=' + str(H_x))

# H(Y)
H_y = 0
p_y = np.zeros((1, 4))
for i in range(1, 4):
    for j in range(1, 5):
        p_y[0, i-1] += data_np[i-1, j-1]
# print(p_y)

for i in range(1, 4):
    H_y += -p_y[0, i-1] * np.log2(p_y[0, i-1])
# print(H_y)
print(f'H(Y)=' + str(H_y))

# H(X,Y)
H_xy = 0
for i in range(1, 4):
    for j in range(1, 5):
        # print(i,j)
        if data_np[i-1, j-1] == 0:
            H_xy += 0
        else:
            H_xy += -data_np[i - 1, j - 1] * np.log2(data_np[i - 1, j - 1])
print(f'H(X,Y)=' + str(H_xy))
# print(data_np[3, 3])

# H(X|Y)
H_x_y = 0
for i in range(1, 4):
    for j in range(1, 5):
        if data_np[i-1, j-1] == 0:
            H_x_y += 0
        else:
            H_x_y += -data_np[i - 1, j - 1] * np.log2(data_np[i - 1, j - 1] / p_y[0, i-1])
print(f'H(X|Y)=' + str(H_x_y))
# print(data_np[3, 3])

# H(Y|X)
H_y_x = 0
for i in range(1, 4):
    for j in range(1, 5):
        if data_np[i-1, j-1] == 0:
            H_y_x += 0
        else:
            H_y_x += -data_np[i - 1, j - 1] * np.log2(data_np[i - 1, j - 1] / p_x[0, j-1])
print(f'H(X|Y)=' + str(H_y_x))
# print(data_np[3, 3])

# I(X;Y)
I_xy = 0
for i in range(1, 4):
    for j in range(1, 5):
        if data_np[i-1, j-1] == 0:
            I_xy += 0
        else:
            I_xy += data_np[i - 1, j - 1] * np.log2(data_np[i - 1, j - 1] / (p_y[0, i-1] * p_x[0, j-1]))
print(f'I(X;Y)=' + str(I_xy))
print(f'I(X;Y)=' + str(H_x - H_x_y))
# print(data_np[3, 3])