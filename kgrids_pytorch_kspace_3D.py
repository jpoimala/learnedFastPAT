
import numpy as np
import math 
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

class kspaceKGrids(object):
    def __init__(self, dx,dy,dz, dt,c, NtFactor):
        
        self._dx=dx
        self._dy=dy
        self._dz=dz
        self._dt=dt
        self._c=c
        self._NtFactor=NtFactor

      
    @property
    def dx(self):        
         return self._dx
    
    @property
    def dy(self):        
         return self._dy
  
    @property
    def dz(self):        
         return self._dz        
  

    @property
    def dt(self):        
         return self._dt
     
    @property
    def c(self):        
         return self._c   

    @property
    def NtFactor(self):        
         return self._NtFactor  
    

   
        
    def inverse(self,pT):
        
        
        pT=np.concatenate((pT[::-1,:,:],pT[1::,:, :]),0)
        
        
        size=pT.shape
        
        Nt=size[0]
        Ny=size[1]
        Nz=size[2]
        
        # compute kgrids
        kgridBack = kgrid(math.ceil(Nt/self.NtFactor), self.dx, Ny, self.dy, Nz, self.dz, self.c)
        
        kgridBackC = kgrid(Nt, self.dx, Ny, self.dy, Nz, self.dz,  self.dx/self.dt)
     
        
        '''ALL Constant'''
        c=kgridBackC.c
        w=c*kgridBackC.kx
        w_new = kgridBack.c*kgridBack.k
        
        sf=np.square(w/c) - np.square(kgridBackC.ky)-np.square(kgridBackC.kz)
        sf=c*c*np.sqrt( sf.astype(np.complex)  )



        sf=np.divide(sf , 2*w)

        idx1  = np.where( (w == 0) & (kgridBackC.ky==0) & (kgridBackC.kz==0) )
        sf[idx1]=c/2 
        
        idx2  = np.where(np.abs(w)< c*np.sqrt(np.square(kgridBackC.ky)+np.square(kgridBackC.kz)) )
        sf[idx2]=0 
        
    
        idx  = np.where(np.isnan(sf))
        sf[idx]=0   
        
        
        
        ky=kgridBackC.ky_vec
        kyI=kgridBack.ky
        
        kz=kgridBackC.kz_vec
        kzI=kgridBack.kz
        
        
        w=c*kgridBackC.kx_vec
        wI=w_new
        
        
    
        
        return sf, ky, w, kz, kyI, wI, kzI
    
    
   
    
         
 
 
class kgrid(object):
    def __init__(self, Nx, dx, Ny, dy, Nz, dz,  c):

        [k, kx, ky, kz,  kx_vec, ky_vec, kz_vec ] = makeKgrid(Nx, dx, Ny, dy, Nz, dz)    

        self._kx=kx
        self._ky=ky
        self._kz=kz
        self._k=k
        
        self._kx_vec=kx_vec
        self._ky_vec=ky_vec
        self._kz_vec=kz_vec
        

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.c  = c

    @property
    def kx(self):        
         return self._kx
     
        
    @property
    def ky(self):        
         return self._ky
     
    @property
    def kz(self):        
         return self._kz      
     
    @property
    def k(self):        
         return self._k
     
        
    @property
    def kx_vec(self):        
         return self._kx_vec
     
    @property
    def ky_vec(self):        
         return self._ky_vec
     
    @property
    def kz_vec(self):        
         return self._kz_vec    
     
        
     

       
def makeKgrid(Nx, dx, Ny, dy, Nz, dz):
    
    kx_vec=makeDim(Nx, dx)
    ky_vec=makeDim(Ny, dy)
    kz_vec=makeDim(Nz, dz)
    
    
    kx = np.tile(kx_vec.reshape(-1,1)[:, np.newaxis], (1, Ny,  Nz))
    

    ky=np.tile(ky_vec[:, np.newaxis], (Nx, 1, Nz))
    
   
    kz=np.tile(kz_vec, (Nx, Ny, 1))
    
    k=np.zeros([Nx, Ny, Nz])
    
    
    k=(kx_vec**2).reshape(-1,1)[:, np.newaxis]+k
    
    
    k=(ky_vec**2).reshape(-1,1)+k
    

    
    k=(kz_vec**2)+k
    
    
    
    k=np.sqrt(k)
                   
    return k, kx, ky, kz, kx_vec, ky_vec, kz_vec 

       
def makeDim(Nx, dx):
    
    
    if  (Nx % 2) == 0:
    
    
        nx = np.arange(-Nx/2, Nx/2)/Nx
    
    else:
        
        
        nx = np.arange(-(Nx-1)/2, (Nx)/2)/Nx
        
    # force middle value to be zero in case 1/Nx is a recurring number and the series doesn't give exactly zero
    nx[math.floor(Nx/2)] = 0
            
    # define the wavenumber vector components
    kx_vec = (2*np.pi/dx)*nx
    
    return kx_vec

