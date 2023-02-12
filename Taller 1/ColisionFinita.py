import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm
from time import sleep

class Particle:
    def __init__(self,r0,v0,a0,t,f,m=2,radius=2,id=0,k=100):
        
        self.dt=t[1]-t[0]
        self.r=r0
        self.v=v0
        self.a=a0
        self.R=np.zeros((len(t),len(r0)))
        self.V=np.zeros_like(self.R)
        self.A=np.zeros_like(self.R)
        self.radius=radius
        self.m=m
        self.f=f
        self.F=np.zeros_like(self.R)
        self.k=k
        
    def Evolution(self,i,p1,p2):
        
       self.SetForce(i)
       self.f=self.f
       
       self.SetAccel(i)
       self.a=self.f/self.m
       
       self.SetVelocity(i)
       self.v=self.v+self.a*self.dt
       
        
       self.SetPosition(i)
       self.r=self.r+self.v*self.dt
        
      
       
      
        
      
        
        
    
    def SetPosition(self,i):
        self.R[i]=self.r
    
    def GetPosition(self,scale=1):
        return self.R[::scale]
    
    def SetVelocity(self,i):
        self.V[i]=self.v
    
    def GetVelocity(self,scale=1):
        return self.V[::scale]
    
    def SetAccel(self,i):
        self.A[i]=self.a
    
    def GetAccel(self,scale=1):
        return self.A[::scale]
    
    def SetForce(self,i):
        self.F[i]=self.f
    
    def GetForce(self,scale=1):
        return self.F[::scale]
    
    def CheckLimits(self,Limits):
        if self.r[0] + self.radius > Limits[0][1] or self.r[0] - self.radius < Limits[0][0]:
           self.v[0] = -self.v[0]
           
           
        if self.r[1] + self.radius > Limits[1][1] or self.r[1] - self.radius < Limits[1][0]:
           self.v[1] = -self.v[1]   
    
    def Force(self,i,p1,p2):
        if np.linalg.norm([p1.r-p2.r])<p1.radius+p2.radius:
         self.f=1*((np.linalg.norm([p1.r-p2.r]))**3)*((p1.r-p2.r)/np.linalg.norm([p1.r-p2.r]))        
        else:
         self.f=np.zeros(2)

    
        

def RunSimulation(t):
    r01=np.array([-15.,1.])
    v01=np.array([10.,0.])
    a01=np.array([0.,0.])
    
    
    r02=np.array([0.,-1.5])
    v02=np.array([0.,0.])
    a02=np.array([0.,0.])
    
    f=np.array([0.,0.]) 
    Limits = np.array([[-20.,20.],[-20.,20.]])
    
    p1=Particle(r01,v01,a01,t,f)
    p2=Particle(r02,v02,a02,t,f)
    
    for it in tqdm(range(len(t)),desc='Running simulation', unit=' Steps'):
        sleep(0.0001)
        
        p1.Force(it, p1, p2)
        p2.Force(it, p2, p1)
      
        
        p1.Evolution(it,p1,p2)
        p1.CheckLimits(Limits)
        
        
        p2.Evolution(it,p2,p1)
        p2.CheckLimits(Limits)
     
  

    return p1,p2


dt = 0.1
tmax = 10
t = np.arange(0,tmax,dt)
Particles = RunSimulation(t)[0]
particle1= RunSimulation(t)[1]


fig=plt.figure(figsize=(5,5))
ax=fig.add_subplot(111)
def init():
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)

def Update(i):
    ax.clear()
    init()
    ax.set_title(r'$ t=%.2f \ s$' %(t[i]))
    
    x=Particles.GetPosition()[i,0]
    y=Particles.GetPosition()[i,1]
    """
    fx=Particles.GetForce()[i,0]
    fy=Particles.GetForce()[i,1]
    """
    
    
    vx=Particles.GetVelocity()[i,0]
    vy=Particles.GetVelocity()[i,1]
    
    circle = plt.Circle((x,y),Particles.radius, fill=True)
    ax.add_patch(circle)
    ax.arrow( x,y,vx,vy,head_width=0.5,color='r' )
    """
    ax.arrow( x,y,fx,fy,head_width=0.5,color='g' )
    """
    
    x0=particle1.GetPosition()[i,0]
    y0=particle1.GetPosition()[i,1]
    
    vx0=particle1.GetVelocity()[i,0]
    vy0=particle1.GetVelocity()[i,1]
    """
    fx0=particle1.GetForce()[i,0]
    fy0=particle1.GetForce()[i,1]
    """
    circle1 = plt.Circle((x0,y0),particle1.radius, fill=True)
    ax.add_patch(circle)
    ax.add_patch(circle1)
    
    ax.arrow( x0,y0,vx0,vy0,head_width=0.5,color='b' )
    """
    ax.arrow( x0,y0,fx0,fy0,head_width=0.5,color='y' )
    """
    

Animation = anim.FuncAnimation(fig,Update,frames=len(t),init_func=init)

plt.show()



        