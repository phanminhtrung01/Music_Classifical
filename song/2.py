import numpy as np

a = input()

a = a.split(',')
for i in range(len(a)):
    a[i] = a[i].strip(" ").replace("'", '')
a = set(a)
print(len(a))
print(a)
