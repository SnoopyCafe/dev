__author__ = 'admin'
a = 1
for i in range(1000000):
    a = a + 1e-6
print(a - 1000000000)