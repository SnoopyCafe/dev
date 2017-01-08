import numpy as np


int_array = np.array([2,3,4])
float_array = np.array([1.2,3.5,5.3])
print(int_array)
print (int_array.dtype)

print (np.zeros([2,4]))  #zeros array

print (np.ones([3,3]))  # Ones array

#range of numbers
print (np.arange(1,10,1))

even_numbers = np.array([2,4,6,8,10])
odd_numbers = np.array([1,3,5,7,9])

add_even_odd = even_numbers + odd_numbers
print ("add even + odd arrays ", add_even_odd)
print ("sum even ", even_numbers.sum())
print ("even mean ", even_numbers.mean())
print ("max even ", even_numbers.max())

# Matrix multiplication - Dot product
m1 = np.array([[1,1],[2,3]])
m2 = np.array([[2,0],[1,6]])
print ("dot product \n", m1.dot(m2))

# Matrix Addition
print (m1 + m2,"\n")

# Matrix Subtraction
print (m1 - m2)