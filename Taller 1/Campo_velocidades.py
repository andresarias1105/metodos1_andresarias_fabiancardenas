import numpy as np
import matplotlib.pyplot as plt
"""Definir una discretizaci´on en los ejes x e y, donde la regi´on es: A ∈ [−4, 4] con 25
puntos en cada eje."""

n=25
x=np.linspace(-4,4,n)
y=np.linspace(-4,4,n)
X,Y=np.meshgrid(x,y)




"""Definir la funcion potencial del flujo"""

V=2
R=2
def flujo_de_potencial(X,Y):
    
    
        
    
      potencial=V*X*(1-(R**2/(X**2+Y**2)))
    
      return potencial

"""Calcule y guarde adecuadamente el campo de velocidades usando la definici´on de
derivada parcial central"""

h=0.001
def derivada_Central_x(f,x,y,h):
    d=0.
    if h!=0:
        d=(f(x+h,y)-f(x-h,y))/(2*h)
    return d

def derivada_Central_y(f,x,y,h):
    d=0.
    if h!=0:
        d=(f(x,y+h)-f(x,y-h))/(2*h)
    return d


Vx= derivada_Central_x(flujo_de_potencial, X, Y, h)


Vy=-1*derivada_Central_y(flujo_de_potencial, X, Y, h)

for i in range(n):
    for j in range(n):
        if np.sqrt(X[i,j]**2+Y[i,j]**2)<=2:
            Vx[i,j]=0
            Vy[i,j]=0
fig=plt.figure(figsize=(5,5))

ax=fig.add_subplot(111)
circle = plt.Circle((0,0),2, fill=False)
ax.add_patch(circle)
for i in range(n):
    for j in range(n):
     ax.quiver(x[i],y[j],Vx[i,j],Vy[i,j])


