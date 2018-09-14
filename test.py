#question1
import numpy as np
matx_10 = np.random.rand(10,10)
print(matx_10)
from numpy import linalg as la
eig_m = la.eigvals(matx_10)
print(eig_m)
import pandas as pd
df_m = pd.DataFrame(matx_10, columns=["column " + str(i) for i in range(matx_10.shape[0])])
print(df_m)
df_m.to_csv('qz1.csv', encoding='utf-8')
U, s, V = np.linalg.svd(matx_10, full_matrices=True)
print(U)
print(s)
print(V)

#question2
#list1
import math
from numpy import inf
l = []
for i in range(1,100):
    a = (((-1) ** (i + 1)) / i) * ((6 / 7) ** i)
    l.append(a)
print(sum(l))

#ndarray1
import math
from numpy import inf
import numpy as np
l = []
for i in range(1,100):
    a = (((-1) ** (i + 1)) / i) * ((6 / 7) ** i)
    l.append(a)
    arr=np.array(l)
print(sum(arr))

#list2
from numpy import inf
l = []
for i in range(1,100000):
    a = (1-1/i) **(2*i)
    l.append(a)
print(l[99998])
#ndarray2
import numpy as np
from numpy import inf
l = []
for i in range(1,100000):
    a = (1-1/i) **(2*i)
    l.append(a)
    arr = np.array(l)
print(arr[99998])
#sympy
import sympy
from sympy import *
n = Symbol('n')
s = (1-(1/n))**(2*n)
print(limit(s, n, oo))
i = sympy.symbols('i',integer = True)
a = float(sympy.summation((((-1)**(i+1))/i)*((6.0/7.0)**i),(i,1,sympy.oo)))
print(a)

