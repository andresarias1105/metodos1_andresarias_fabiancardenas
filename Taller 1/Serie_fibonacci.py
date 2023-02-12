import numpy as np
import matplotlib.pyplot as plt

n=20

nums=np.zeros(n+1)
nums[0]=0
nums[1]=1


for i in range(2,n+1):
    
    nums[i]=nums[i-1]+nums[i-2]
    
G_ratio=np.zeros(n-1)

for i in range(0,n-1):
    
    G_ratio[i]=nums[i+2]/nums[i+1]
    

    
    
    
    
    
k=np.arange(0,n+1,1)


fig, (ax, ax1) = plt.subplots(1,2)
ax.plot(k,nums,label="Serie Fibonacci")
ax.legend()

ax1.plot(k[0:19],G_ratio,label="Estimacion usando la serie")
ax1.axhline(y=(1+np.sqrt(5))/2,color="r",ls="--",label="Numero Aureo")
ax1.legend()


plt.show()

