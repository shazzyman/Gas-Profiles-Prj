#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt
from colossus.utils import constants
from colossus.cosmology import cosmology
from scipy.optimize import curve_fit
from tabulate import tabulate

gasprofs100_filename = 'groups_gasprofs_tng100_099.hdf5'
gasprofs100_vals = h5py.File(gasprofs100_filename, 'r', libver='earliest', swmr=True)

#FOR TEST: Extract data for A-2390 from Vik paper
A2390_x, A2390_y =  np.loadtxt('plot-data.csv', skiprows = 1, unpack=True, delimiter = ',', usecols=(0,1))

mycosmo = cosmology.setCosmology('myCosmo', params = cosmology.cosmologies['planck18'], Om0 = 0.3, Ode0=0.7, H0 = 72, sigma8 = 0.9)
print(mycosmo)

gamma=3

mp = constants.M_PROTON

G = constants.G

G_CGS = constants.G_CGS

def get_rho_norm(r, n0, rc, rs, a, B, epsilon, n02, rc2, B2):
    npne = ((n0*(10**-3))**2)*(((r/rc)**(-a))/((1+(r**2)/(rc**2))**(3*B + (-a/2))))* \
        (1/((1+((r**gamma)/(rs**gamma)))**(epsilon/gamma))) \
            + (((n02*(10**-1))**2)/ ((1 + (r**2)/(rc2**2))**(3*B2)))
            
    rho_g = 1.624 * mp * (npne)**(1/2)
    
    #rho_c_ill = mycosmo.rho_c(0.2302) * (constants.MSUN)*(.677**2)/(constants.KPC**3)
    
    H_col = mycosmo.Hz(0.2302)
    
    h_col = H_col * (10 ** (-2))
    
    #rho_c_meeting = 2.775*(10**2) * constants.MSUN * (h_col**2)/(constants.KPC**3)
    
    rho_c_col = mycosmo.rho_c(0.2302) * (constants.MSUN)*((h_col)**2)/(constants.KPC**3)
    
    rho_c_Vik = (3*((H_col)**2))/(8*np.pi*G_CGS)  * ((10**5)/constants.MPC)**2
    
    rho_norm_col = rho_g/rho_c_col
    
    rho_norm_Vik = rho_g/rho_c_Vik
    
    return rho_g, rho_norm_col, rho_norm_Vik

n0, rc, rs, a, B, epsilon, n02, rc2, B2 = np.loadtxt('Vikhlinin_tab2.csv', skiprows = False, unpack=True, delimiter = ',', usecols=(0,1,2,3,4,5,6,7,8))

radii_all100 = np.array(gasprofs100_vals['profile_bins'])
bin_centers100 = (radii_all100[:, :-1] + radii_all100[:, 1:]) / 2
first_bin_center100 = (radii_all100[:, 0]/2)
bin_centers100 = np.insert(bin_centers100, 0, first_bin_center100, axis=1)
median_radii100 = np.median(bin_centers100, axis=0)
R_Crit500_100 = np.array(gasprofs100_vals['catgrp_Group_R_Crit500'])
median_R_Crit500_100 = np.median(R_Crit500_100, axis=0)
m_radii_norm_100 = median_radii100/median_R_Crit500_100


first_bin_center100 = [(radii_all100[:, 0]/2), (radii_all100[:, 0]/4), (radii_all100[:, 0]/8),(radii_all100[:, 0]/16), (radii_all100[:, 0]/32)]
bin_centers100 = np.insert(bin_centers100, 0, first_bin_center100, axis=1)
median_radii_Vik_kpc = np.median(bin_centers100, axis=0)
median_radii_Vik_mpc = median_radii_Vik_kpc * (constants.KPC) / (constants.MPC)

Vik_bins = np.linspace(78, 23400, 100)

m_radii_norm_A2390 = Vik_bins/1416

rho_g, rho_norm_col, rho_norm_Vik = get_rho_norm(Vik_bins, n0[9], rc[9], rs[9], a[9], B[9], epsilon[9], n02[9], rc2[9], B2[9])



#PLOTTING
#====================================================================================

plt.rcParams.update({'font.size': 16})
fig2 = plt.figure(figsize=(10,8), dpi=200)
ax = plt.subplot(111)
plt.xscale("log")

plt.semilogy(m_radii_norm_A2390, rho_norm_col, color="yellow", lw=3, alpha=1, label='rho c collosus')
#plt.semilogy(m_radii_norm_A2390, rho_norm_Vik, color="limegreen", lw=3, alpha=1, label='rho c Vik')

plt.scatter(A2390_x, A2390_y, marker='o',color="red", lw=3, alpha=1, label='Viklinin')

plt.xlabel('r/R500c', fontsize=20)
plt.ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
plt.title("Density from Vihklinin Paper and Formula Results  vs. radius", fontsize=30)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0, box.width, box.height * 0.9])
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, fancybox=True, shadow=True, fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

    
