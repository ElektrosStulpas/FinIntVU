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
np.where(a > 0) # throws out the zeros alltogether

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
        
#TODO write a single liner for this

#5 radimas a > b
a = np.random.randn(10)
b = np.random.randn(10)
print(a)
print(b)
for i in range(len(a)):
    if a[i] > b[i]:
        print(f"a array value was bigger than b array value at index {i}. \na value was {a[i]} and b value was {b[i]}")

list(zip(a[np.where(a > b)], b[np.where(a > b)]))

#6 elementu perstumimas vektoriuje pakartojant paskutni
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

