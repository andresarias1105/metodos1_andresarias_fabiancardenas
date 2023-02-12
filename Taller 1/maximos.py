import urllib.request
import numpy as np
import matplotlib.pyplot as plt
url = 'https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/EstrellaEspectro.txt'
filename = 'Data/DatosMaximo.txt'
urllib.request.urlretrieve(url, filename)
# El proceso de descarga funciona en Windows,Linux y Mac

data = np.loadtxt(filename)
plt.plot(data[:,0],data[:,1])

x=(data[:,0])
y=data[:,1]

maximos=([])
for i in range(1,len(x)-1):
    
    if y[i-1]<y[i]>y[i+1]:
        maximos.append((x[i],y[i]))
    
maximos=np.array(maximos)
plt.plot(x,y,color="b")
plt.scatter(maximos[:,0],maximos[:,1],color="r",s=10)

plt.show()