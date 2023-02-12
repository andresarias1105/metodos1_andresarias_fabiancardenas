import numpy as np

def factorial(n):
   fact=(1) 
   for i in range(n):    
     fact*=(n-i)
   return fact

n=20

nums=np.zeros(n,dtype="object")

for i in range(1,n+1):
     nums[i-1]=(factorial(i))
     
print(nums)
