import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma, factorial, cython_special, jv

'''
Question 1A
'''


def stirling(n):
    return math.sqrt(2 * math.pi * n) * (n / math.e) ** n


nf = 1
print('n', '\t', 'Stirling', '\t\t', 'Factorial')
for x in range(1, 11):
    nf *= x
    print(x, "\t", stirling(x), "\t\t", nf)

x = np.linspace(0, 5.5, 2251)
y_stirling = []
for i in (x):
    y_stirling.append(stirling(i))

plt.plot(x, y_stirling, 'b', alpha=0.6, label='stirling(x)')
k = np.arange(1, 7)

plt.plot(k, factorial(k), 'k*', alpha=0.6,

         label='(x)!, x = 1, 2, ...')

plt.xlim(0, 5.5)

plt.ylim(-1, 25)

plt.grid()

plt.xlabel('x')

plt.legend(loc='lower right')

plt.show()

'''
Question 1B
'''

x = np.linspace(-3.5, 5.5, 2251)

y = gamma(x)

plt.plot(x, y, 'b', alpha=0.6, label='gamma(x)')

k = np.arange(1, 7)

plt.plot(k, factorial(k - 1), 'k*', alpha=0.6,

         label='(x-1)!, x = 1, 2, ...')

plt.xlim(-3.5, 5.5)

plt.ylim(-10, 25)

plt.grid()

plt.xlabel('x')

plt.legend(loc='lower right')

plt.show()

'''
Question 1C
'''

x = np.linspace(1, 20, 20)
y_stirling = []
for i in (x):
    y_stirling.append(stirling(i))
df = pd.DataFrame(columns=['Value', 'Gamma', 'Stirlings', 'Difference %'])
df['Value'] = x
df['Gamma'] = gamma(x + 1)
df['Stirlings'] = y_stirling
df['Difference %'] = ((df['Stirlings'] - df['Gamma']) / df['Gamma']) * 100

'''
Question 3
'''
x = np.linspace(1, 100, 100)
bessel1 = jv(1, x)
bessel5 = jv(5, x)

approx1 = (np.sqrt(2 / (np.pi * x))) * np.cos(x - (np.pi * (2 + 1) / 4))
approx5 = (np.sqrt(2 / (np.pi * x))) * np.cos(x - (np.pi * (10 + 1) / 4))

plt.subplot(2, 2, 1)
graph1 = plt.plot(x, approx1, 'k*', alpha=0.6, label='approx1')
graph2 = plt.plot(x, bessel1, 'b', alpha=0.6, label='bessel1')
plt.xlabel('x')
plt.ylabel('value')
plt.legend(loc='lower right')

plt.subplot(2, 2, 2)
graph5 = plt.plot(x, approx5, 'k*', alpha=0.6, label='approx5')
graph3 = plt.plot(x, bessel5, 'b', alpha=0.6, label='bessel5')
plt.xlabel('x')
plt.ylabel('value')
plt.legend(loc='lower right')

plt.subplot(2, 2, 3)
diff1 = (np.abs((bessel1 - approx1) / bessel1) * 100)
diff2 = (np.abs((bessel5 - approx5) / bessel5) * 100)
graph4 = plt.plot(x, diff1)
graph6 = plt.plot(x, diff2)
plt.xlim([0, 10])


df_bessel1 = pd.DataFrame(columns=['Bessel 1', 'Approximation 1', 'Difference %'])

def approx_1(n):
    return (np.sqrt(2 / (np.pi * x))) * np.cos(x - (np.pi * (2 + 1) / 4))

df_bessel1['Bessel 1'] = bessel1
df_bessel1['Approximation 1'] = approx_1(x)
df_bessel1['Difference %'] = (np.abs((bessel1 - approx_1(x)) / bessel1) * 100)

df_b1_05 = df_bessel1[df_bessel1['Difference %'] <= 0.5]
df_b1_1 = df_bessel1[df_bessel1['Difference %'] <= 1]

df_bessel5 = df_bessel1 = pd.DataFrame(columns=['Bessel 5', 'Approximation 5', 'Difference %'])

def approx_5(n):
    return (np.sqrt(2 / (np.pi * x))) * np.cos(x - (np.pi * (10 + 1) / 4))

df_bessel5['Bessel 5'] = bessel5
df_bessel5['Approximation 5'] = approx_5(x)
df_bessel5['Difference %'] =(np.abs((bessel1 - approx_5(x)) / bessel1) * 100)

df_b5_05 = df_bessel5[df_bessel5['Difference %'] <= 0.5]
df_b5_1 = df_bessel5[df_bessel5['Difference %'] <= 1]