import csv
import numpy as np
import pandas as pd

# with open('1018data.csv') as f:
#     f_csv = csv.reader(f)
#     headers = next(f_csv)
#     for row in f_csv:
#         print(row)

data = pd.read_csv('1018data.csv')
# print(data)
headers = data.columns
# print(headers)
index = np.array(data['Index'])
x = np.array(data['X'])
y = np.array(data['Y'])
z = np.array(data['Z'])
# print(x)
x = x[:10800]
y = y[:10800]
z = z[:10800]
# print(x.size)
x_new = x.reshape(270, 40)
y_new = y.reshape(270, 40)
z_new = z.reshape(270, 40)
index_new = index[:270]
PCI = [50, 60, 50, 50, 60, 70, 60, 80, 40, 50, 50, 60, 70, 60, 60, 60, 40, 10, 60, 50, 60, 70, 50, 70, 80, 30, 60,
       60, 60, 40, 60, 60, 50, 20, 50, 60, 40, 50, 60, 60, 40, 60, 50, 50, 40, 50, 30, 70, 80, 60, 40, 60, 60, 60,
       60, 40, 50, 60, 70, 70, 60, 40, 60, 50, 40, 60, 40, 60, 50, 50, 40, 50, 60, 50, 70, 40, 50, 60, 60, 60, 60,
       40, 60, 60, 60, 60, 40, 40, 40, 60, 50, 50, 50, 70, 60, 40, 60, 40, 60, 60, 50, 60, 10, 50, 70, 50, 50, 60,
       50, 70, 50, 40, 60, 40, 60, 60, 40, 60, 70, 60, 70, 60, 70, 60, 40, 50, 60, 50, 60, 40, 50, 40, 60, 50, 50,
       70, 40, 50, 50, 60, 40, 60, 80, 60, 50, 40, 60, 50, 50, 60, 60, 50, 50, 40, 60, 60, 50, 60, 50, 40, 60, 40,
       50, 60, 50, 50, 40, 70, 60, 40, 40, 50, 40, 60, 50, 60, 70, 60, 40, 70, 60, 50, 60, 80, 50, 70, 40, 30, 50,
       50, 40, 40, 60, 60, 50, 50, 50, 40, 50, 50, 60, 80, 60, 60, 50, 40, 50, 50, 40, 70, 70, 60, 60, 50, 50, 70,
       40, 30, 90, 40, 60, 60, 60, 40, 60, 50, 50, 60, 70, 60, 50, 60, 40, 50, 60, 50, 60, 50, 50, 60, 60, 60, 60,
       50, 50, 50, 50, 40, 40, 60, 70, 70, 50, 50, 40, 50, 50, 70, 50, 50, 60, 50, 60, 50, 40, 60, 60, 50, 70, 50]
PCI = np.array(PCI)
x_out = []
for i, row in enumerate(x_new):
    row_new = ",".join(map(lambda p: str(p), row))
    x_out.append(row_new)
x_out = np.array(x_out)

y_out = []
for i, row in enumerate(y_new):
    row_new = ",".join(map(lambda p: str(p), row))
    y_out.append(row_new)
y_out = np.array(y_out)

z_out = []
for i, row in enumerate(z_new):
    row_new = ",".join(map(lambda p: str(p), row))
    z_out.append(row_new)
z_out = np.array(z_out)
# print(PCI)
# print(x_out.size)

data_out = pd.DataFrame({'ID': index_new, 'x': x_out, 'y': y_out, 'z': z_out, 'PCI': PCI})
data_out.to_csv('ARI_PCI.csv', index=False, header=True)
