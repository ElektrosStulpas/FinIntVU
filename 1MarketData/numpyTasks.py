import enum
import numpy as np

#1 suma 2 vektoriu
a = np.random.randint(low=1, high=10, size=10)
b = np.random.randint(low=1, high=10, size=10)
c = 0
for i, _ in enumerate(a):
    c += a[i] + b[i]
print(c)

np.sum(np.add(a, b))
np.sum(a+b)

#2 anuliavimas teigiamu elementu
a = np.random.randint(low=-10, high=10, size=10)
for i, val in enumerate(a):
    if val > 0:
        a[i] = 0
print(a)

ar = a.clip(max=0)
print(ar)

#3 ismetimas > 6
a = np.random.randint(low=1, high=10, size=10)
b = []
for i, val in enumerate(a):
    if val <= 6:
        b = np.append(b, val)
b = b.astype(int)
print(a)
print(b)

b = a[a <= 6]


#MATLAB translate part
#4 vienodu salia esanciu elementu radimas
a = np.random.randint(low=1, high=5, size=10)
print(a)
for i, val in enumerate(a[:-1]):
    if a[i] == a[i+1]:
        print(f"Next to each other indexes: {i} and {i+1}. Element values at these indexes: {a[i]}")

a[np.asarray(np.where(np.ediff1d(a) == 0)) + 1] #gives back elements since we take indices from np.where. asarray is not needed if we don't want to increment all indices to get the second same value element

#5 radimas a > b
a = np.random.randn(10)
b = np.random.randn(10)
print(a)
print(b)
for i in range(len(a)):
    if a[i] > b[i]:
        print(f"a array value was bigger than b array value at index {i}. \na value was {a[i]} and b value was {b[i]}")

list(zip(a[np.where(a > b)], b[np.where(a > b)]))

#6 elementu perstumimas vektoriuje pakartojant paskutini
a = np.random.randint(low=1, high=10, size=10)
print(a)
for i in range(1, len(a)):
    a[i-1] = a[i]
print(a)

print(np.concatenate([np.roll(a, -1)[:-1], [a[-1]]])) #arrays have to be the same dimensions to be concatenated

#7 elementu eiles tvarkos sukeitimas
a = np.random.randint(low=1, high=10, size=10)
print(a)
for i in range(int(len(a)/2)):
    b = a[i]
    a[i] = a[-(i+1)]
    a[-(i+1)] = b
print(a)

print(np.flip(a))

#8 kas antro elemento uznulinimas
a = np.random.randint(low=1, high=10, size=10)
print(a)
for i in range(0, len(a)-1, 2):
    a[i+1] = 0
print(a)

a[1::2] = 0

#9 rasti matricos eiluciu vidurkius, stulpeliu vidurkius
a = np.random.randn(10, 20)
for i in range(len(a[:, 0])):
    mean = 0
    print(a[i])
    for j, val in enumerate(a[i]):
        mean += val
    print(mean/len(a[i]))

a.mean(0) #axis 0 is columns
a.mean(1) #axis 1 is rows

#10 gauti diagonalinius elementus
a = np.random.randint(10, size=(10, 10))
print(a)
for i, _ in enumerate(a[0]):
    print(a[i, i])

a[np.nonzero(a*np.eye(10,10))]