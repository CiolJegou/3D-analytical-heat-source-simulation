# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 08:38:02 2022

@author: ljegou
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import integrate
from pyevtk.hl import gridToVTK  
import bz2
import _pickle as cPickle

import os
import shutil
from os import listdir
from os.path import isfile, join

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from subprocess import Popen, PIPE, STDOUT
import time

import matplotlib as mpl
# Default properties 
mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams['figure.figsize'] = (10.0, 6.0)    
mpl.rcParams['axes.unicode_minus'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams["axes.grid"] =True
# Define size
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 20, 25, 30
plt.rc('font', size=SMALL_SIZE)           # controls default text sizes
plt.rc('axes', titlesize=35)              # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

#This class defines the geometry of the surface
class geometry():
    
    #All dimensions in m
    def __init__(self, L1, L2, h0):
        self.L1 = L1     #Length of melt pool after deepest point (m)
        self.L2 = L2     #Length of melt pool before deepest point (m)
        self.h0 = h0     #highest height of melt pool (m)
        
        self.w0 = L1*2   #widest sidth of melt pool (m)
    
    #Heights along the x axis in y = 0
    def h_y0(self, x):
        return self.h0*(1 -(x+self.L2)**2/(self.L1+self.L2)**2)
    
    #Heights along the y axis in x = 0
    def h_x0(self, y):
        return self.h0-4*self.h0*y**2/self.w0**2
    
    #width depending on x
    def w(self, x):
        w_tmp = np.where(x>0,self.w0**2/4-x**2,(self.w0/2)**2)
        return np.sqrt(w_tmp)*2        

    #height on all domain
    def h_all(self, x,y):
        xy = x**2 + y**2
        cond = np.where((xy>(self.w0/2)**2) & (x>0),0,np.where((xy<(self.w0/2)**2) | (x>=-self.L2), 1,2))
        #Height in zone 0 (previous layer)
        h_0 = 0
        #Height in zone 1 (melt pool)
        wx = np.where(self.w(x)==0,1,self.w(x))
        h_1 = np.where(cond==1,self.h_y0(x) * (1 - 4*y**2/wx**2)*cond,0)
        #Height in zone 2 (new layer)
        h_2 = np.where(cond==2,self.h_x0(y),0)
        return h_0 + h_1 + h_2

    #Show the geometry
    def show(self, X=None, Y= None):
        if X is None:
            x = np.linspace(-2*self.L2,self.L1,100)
        else:
            x = X
        if Y is None:
            y = np.linspace(-self.L1,self.L1,100)
        else: 
            y = Y
        X,Y = np.meshgrid(x,y)
        z = self.h_all(X,Y)
        
        my_col = cm.jet(z/np.amax(z))
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X axis (mm)', labelpad = 20)
        ax.set_ylabel('Y axis (mm)', labelpad = 20)
        ax.set_zlabel('Z axis (mm)', labelpad = 20)
        # Plot a 3D surface
        ax.plot_surface(X*1e3, Y*1e3, z*1e3, facecolors = my_col)
        plt.show()
        
#This class defines the properties of the laser
class laser():
    
    def __init__(self, rb, beta, Pn):
        self.rb = rb                                   #Beam radius (mm)
        self.beta = beta                               #Laser absorbtivity
        self.Pn = Pn                                   #Average value of laser power (W)

        self.sigma = np.sqrt(2)/2*rb                   #Effective radius for gaussian distribution        
        self.I0 = (beta*Pn)/(np.pi*self.sigma**2*
                  (1 - np.exp(-rb**2/self.sigma**2)))  #Intensity scale factor (W.m-2)

    #Laser intensity 
    def I(self, r):
        return self.I0*np.exp(-r**2/self.sigma**2)
    
    #Show the laser profile
    def show(self):
        r = np.linspace(-self.rb,self.rb)
        plt.figure(figsize=(6,6))
        plt.title('Gaussian heat source')
        plt.plot(r*1e3, self.I(r)/1e6)
        plt.xlabel(r'Laser radius $r$ (mm)')
        plt.ylabel(r'Laser intensity I $(W.mm^{-2})$')

#This classe defines the discretization of the space for the visualisation        
class space():
    
    def __init__(self, x, y, z, dx, geom):
        self.dx = dx         #Voxel dimension (m)
        self.x = x           #X-axis length (m)
        self.y = y           #Y-axis length (m)
        self.z = z           #Z-axis length (m)
        
        self.geom = geom
        self.L1 = geom.L1    #See geometry class
        self.L2 = geom.L2    #See geometry class    
        self.h0 = geom.h0    #See geometry class
        self.w0 = geom.w0    #See geometry class
        
        self.check_bounds()
        self.check()
        #Beginning of x linspace
        self.x0 = -self.L2-2/3*(self.x-(self.L1+self.L2))
        self.x1 = +self.L1-1/3*(self.x-(self.L1+self.L2))
        #Beginning of z linspace
        self.z0 = -(self.z-self.h0)
        self.z1 = self.h0
        #Beginning of y linspace
        self.y0 = -(self.y/2)
        self.y1 = self.y/2
        
        self.get_space()
    
    #Check that the space dimension are greater than the geometry dimensions
    def check_bounds(self):
        if self.L1+self.L2 >self.x:
            print('Geometrical dimensions out of the space (x_axis)')
        if self.h0 > self.z:
            print('Geometrical dimensions out of the space (z_axis)')
        if self.w0 > self.y:
            print('Geometrical dimensions out of the space (y_axis)')
    
    #Check if the space is correctly discretisized (same dx and an integer number of points in every directions)
    def check(self):
        if not (self.x/self.dx).is_integer:
            print("NOT INTEGER : X dimension can not be divided by dx")
        if not (self.y/self.dx).is_integer:
            print("NOT INTEGER : Y dimension can not be divided by dx")
        if not (self.z/self.dx).is_integer:
            print("NOT INTEGER : Z dimension can not be divided by dx")
            
    #Discretization of the working space
    def get_space(self):
        self.X = np.linspace(self.x0,self.x1,int(self.x/self.dx))
        self.Y = np.linspace(self.y0,self.y1,int(self.y/self.dx))
        self.Z = np.linspace(self.z0,self.z1,int(self.z/self.dx))
    
    #Overlay the working space over the geometry
    def show(self):
        #sx = np.linspace(-2*self.L2,self.L1,100)
        #sy = np.linspace(-self.L1,self.L1,100)
        msx, msy = np.meshgrid(self.X, self.Y)
        sh = self.geom.h_all(msx, msy)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        mx, my, mz = np.meshgrid(self.X, self.Y, self.Z)
        ax.scatter(mx, my, mz, alpha = 0.2, edgecolors = 'black')
        my_col = cm.jet(sh/np.amax(sh))
        ax.plot_surface(msx, msy, sh, facecolors = my_col)
        ax.set_xlabel('X (mm)', labelpad = 20)
        ax.set_ylabel('Y (mm)', labelpad = 20)
        ax.set_zlabel('Z (mm)', labelpad = 20)
        
#This class it the analytical thermal modeling
class temperature():

    #constants = [K, Cp, rho, V, tol_z]
    def __init__(self, laser, geom, space, constants):
        self.laser = laser
        self.geom = geom
        self.space = space
        
        self.K = constants[0]           #Thermal conductivity (W.m.K-1)
        self.Cp = constants[1]          #Specific heat capacity (700 J.kg-1.K)
        self.rho  = constants[2]        #Density of the materiel (kg.m^-3)
        self.V = constants[3]           #Velocity of the laser (m.s-1)
        self.tol = constants[4]         #How close the geometry needs to fit the discretization space
        self.alpha = self.K/(self.rho*self.Cp)
        self.rb = self.laser.rb         #laser radius (m)
    
    #Main loop to do everything
    def main(self):
        print('Step 1: Computation of the temperature (self.T)')
        self.get_T()
        print('Step 2: Get upside view and area (self.camera)')
        self.get_camera()
        self.area([1673])
        #self.show_camera()
        print('Step 3: Get density (self.T_rho)')
        print('You can get help with (self.help_rho / self.show_rho')
        self.T_rho = self.to_rho(self.T)
        print("All done, don't forget to SAVE")
        print('T.to_VTK(name) ; T.save(name)')

    def T_point(self, ksi, eta, x, y, z):
        R = np.sqrt((x-0-ksi)**2 + (y-0-eta)**2 + (z-0-self.geom.h_all(ksi,eta))**2)
        A = self.laser.I0/ (2*np.pi*self.rho*self.Cp*self.alpha)
        B1 = np.exp(-(self.V*(R + (x-0-ksi))/(2*self.alpha)))
        B2 = np.exp(-(ksi**2 + eta**2)/self.laser.sigma**2)/R
        return A*B1*B2

    def T_steady_all(self, x, y, z):
        R = np.sqrt((x-0)**2 + (y-0)**2 + (z-0)**2)
        if R == 0:
            a = 0
        else:
            a = integrate.dblquad(self.T_point,-self.rb,self.rb, 0, lambda x:self.rb*(1-x**2), args=(x,y,z))[0]
        return a
    
    #Get the temperature considering irregular geometry
    def get_T(self):
        
        T = np.zeros((len(self.space.X), len(self.space.Y), len(self.space.Z)))
        c = 0
        a = 10

        for i,xi in enumerate(self.space.X):
            for j,yj in enumerate(self.space.Y):
                for k,zk in enumerate(self.space.Z):
                    c += 1
                    if zk > self.geom.h_all(xi,yj) + self.tol:
                        #If out of the domain, négative temperature
                        m = -1
                    else:
                        m = self.T_steady_all(xi,yj,zk)
                        
                    #if m > 2500:
                    #    m = 2500
                    
                    T[i,j,k] = m
                    if c*100/(len(self.space.X)*len(self.space.Y)*len(self.space.Z))> a:
                        print('avancement {}%'.format(a))
                        a +=10
                    
        self.T = T
    #Get the density from the temperature
    def to_rho(self, T):
        rho_liq = lambda x : 7981.4164 - 0.8*x+676.08
        rho_sol = lambda x : 7961.5946 - 0.5*(x-298)
        #If the temperature is below 0 '<=>' out of the space, the density is 0
        T_space = np.where(T>0,1,0)
        T_rho = np.where(T<1823, rho_sol(T), rho_liq(T))*T_space
        T_rho = np.where((T_rho<6600) & (T_rho>1), 6600, np.where(T_rho>7980, 7980, T_rho))
        return T_rho
    
    #show the relation between T and rho
    def show_rho(self):
        Tx = np.linspace(300, 2200, 1900)
        y = np.linspace(6000, 8000, 2000)
        x = np.array([1823 for i in range(len(y))])
        rhox = self.to_rho(Tx)
        plt.plot(Tx, rhox)
        plt.plot(x,y,'--r', linewidth = 3, label = 'T = 1823 K')
        plt.legend()
        plt.axis([300, 2200, 6600, 8000])
        plt.ylabel(r"Density $\rho$ (kg.m$^{-3}$")
        plt.xlabel("Temperature (K)")

    #Briefly explain how the relation between rho and T is found
    def help_rho(self):
        self.show_rho()
        mats = ['Fer', 'Carbone', 'Cuivre', 'Chrome', 'Nickel', 'Manganese', 'Molybdène', 'Sillicium', 'Soufre']
        #DENSITIES
        rho_fe = 7874
        rho_c = 2000
        rho_cu = 8960
        rho_cr = 7190
        rho_ni = 8908
        rho_mn = 7260
        rho_mo = 10280
        rho_si = 2650
        rho_s = 2070
        rhos = [rho_fe, rho_c, rho_cu, rho_cr, rho_ni, rho_mn, rho_mo, rho_si, rho_s]
        #WHEIGHT FRACTIONS
        w_c = 0.015
        w_cu = 0
        w_cr = 17.3
        w_ni = 12
        w_mn = 1.61
        w_mo = 2.23
        w_si = 0.53
        w_s = 0.01
        w_fe = 100 - (w_c+w_cu+w_cr+w_ni+w_mn+w_mo+w_si+w_s)
        ws = [w_fe, w_c, w_cu, w_cr, w_ni, w_mn, w_mo, w_si, w_s]
        #MOLECULAR MASS
        M_fe = 55.85
        M_c = 12.011
        M_cu = 63.55
        M_cr = 52
        M_ni = 58.69
        M_mn = 54.94
        M_mo = 95.95
        M_si = 28.09
        M_s = 32.065
        Ms = [M_fe, M_c, M_cu, M_cr, M_ni, M_mn, M_mo, M_si, M_s]
        #M_tot = M_fe*w_fe + M_c*w_c + M_cu*w_cu + M_cr*w_cr + M_ni*w_ni + M_mn*w_mn + M_mo*w_mo + M_si*w_si + M_s*w_s
        #MOLE FRACTIONS
        #x_i = M_i*w_i/M_tot

        #LIQUID WEIGHT FRACTION
        lw_c = 0.012
        lw_cu = 0
        lw_cr = 18.1
        lw_ni = 12.8
        lw_mn = 0.23
        lw_mo = 2.31
        lw_si = 0.54
        lw_s = 0.01
        lw_fe = 100 - (lw_c+lw_cu+lw_cr+lw_ni+lw_mn+lw_mo+lw_si+lw_s)
        lws = [lw_fe, lw_c, lw_cu, lw_cr, lw_ni, lw_mn, lw_mo, lw_si, lw_s]
        #LIQUID MOLE FRACTIONS
        #lx_i = M_i*lw_i/M_tot
        print("Composition massique de l'acier inox 316L:\n")
        print('Material\t|\tMass fraction (sol)\t|\tMass fraction (liq)\t|\tDensiy (kg.m-3)\t|\tMol mass (u)\n')
        for i,mat in enumerate(mats):
            print('{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\n'.format(mat, ws[i], lws[i], rhos[i], Ms[i]))
        print('\\\\\\\\\\\\\\\\\\\\\\\\\n')
        print('rho_ss = sum(rho_i*mole_frac_i)\n')
        print('mole_frac_i = Mi*wi/Mtot\n')
        print('Source : Fukuyama, H., Higashi, H., & Yamano, H. (2019).\nThermophysical Properties of Molten Stainless Steel Containing 5 mass % B4C.\n')
        print('Source : Mills, K. C., Yuchu, S. U., Zushu, L. I., & Brooks, R. F. (2004). Equations for the calculation of the thermo-physical properties of stainless steel.')
    def T_point_simple(self, ksi, eta, x, y, z):
        R = np.sqrt((x-0-ksi)**2 + (y-0-eta)**2 + (z-self.geom.h0)**2)
        A = self.laser.I0/ (2*np.pi*self.rho*self.Cp*self.alpha)
        B1 = np.exp(-(self.V*(R + (x-0-ksi))/(2*self.alpha)))
        B2 = np.exp(-(ksi**2 + eta**2)/self.laser.sigma**2)/R
        return A*B1*B2
    
    def T_steady_all_simple(self, x, y, z):
        R = np.sqrt((x-0)**2 + (y-0)**2 + (z-0)**2)
        if R == 0:
            a = 0
        else:
            a = integrate.dblquad(self.T_point_simple,-self.rb,self.rb, 0, lambda x:self.rb*(1-x**2), args=(x,y,z))[0]
        return a

    #Get the temperature without considering the geometry
    def get_T_simple(self):
        T_simple = np.zeros((len(self.space.X), len(self.space.Y), len(self.space.Z)))
        c = 0
        a = 10

        for i,xi in enumerate(self.space.X):
            for j,yj in enumerate(self.space.Y):
                for k,zk in enumerate(self.space.Z):
                    m = self.T_steady_all_simple(xi,yj,zk)
                    if m > 2500:
                        m = 2500
                    
                    T_simple[i,j,k] = m
                    if c> a/10*int(len(self.space.X)*len(self.space.Y)*len(self.space.Z)):
                        print('avancement {}%'.format(a*10))
                        a +=1
                    c +=1
        self.T_simple = T_simple
        
    #Export file to VTK for paraview
    def to_VTK(self, name):
        a = np.arange(0, len(self.space.X)+1)
        b = np.arange(0, len(self.space.Y)+1)
        c = np.arange(0, len(self.space.Z)+1)
        
        gridToVTK("./"+name, a, b, c, cellData = {name: self.T})
    
    #Get an upside view of the meltpool
    def get_camera(self):
        E = np.stack([self.T[:,:,i] for i in range(len(self.T[0,0,:]))])
        Ar = np.argmin(abs(E - 3000), 0)
        self.camera = np.array([[self.T[i,j,Ar[i,j]] for j in range(len(Ar[0,:]))] for i in range(len(Ar[:,0]))])
        
    def show_camera(self):
        fig, ax = plt.subplots(1,1, figsize = (6,8))
        im0 = ax.imshow(np.transpose(self.camera), vmin = 0, vmax = 2000, cmap = 'jet')
        ax.set_title('Camera view')
        ax.axis('off')
        cbar_ax = fig.add_axes([1, 0.21, 0.05, 0.58])
        fig.colorbar(im0, cax=cbar_ax, label ='Temperature (K)')
        
        fontprops = fm.FontProperties(size=30)
        scalebar = AnchoredSizeBar(ax.transData,20, '2 mm', 'lower right', pad=0.1,color='black',frameon=False,size_vertical=1,fontproperties=fontprops)
        ax.add_artist(scalebar)
    #Get the area of multiple isotherms
    def area(self, T = [1673]):
        for ti in T:
            ar = np.count_nonzero(np.array(self.camera) > ti)
            print('Aire à {} K : {} px = {:.2f} mm²'.format(ti, ar, ar*(self.space.dx*1e3)**2))
    
    #Save the temperature
    def save(self, name):
        if 'pkl' not in name:
            nameT = name + '_T.pkl'
            namerho = name + '_rho.pkl'
        else:
            nameT = name
            namerho = name
        with bz2.BZ2File(nameT, 'w') as f:
            cPickle.dump(self.T, f)
        with bz2.BZ2File(namerho, 'w') as f:
            cPickle.dump(self.T_rho, f)
    
    #Load the temperature
    def load(self, name):
        if 'pkl' in name:
            data = bz2.BZ2File(name, 'rb')
        else:
            data = bz2.BZ2File(name + '.pkl', 'rb')
        datar = cPickle.load(data)
        return datar