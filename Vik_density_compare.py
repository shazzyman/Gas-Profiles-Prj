#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt
from colossus.utils import constants
from colossus.cosmology import cosmology
from scipy.interpolate import interp1d
#SIMULATION DEFINITIONS
#====================================================================================

gasprofs100_v5_filename = 'groups_gasprofs_v5_tng100_099.hdf5'
gasprofs100_v5_vals = h5py.File(gasprofs100_v5_filename, 'r', libver='earliest', swmr=True)

gasprofs300_v5_filename = 'groups_gasprofs_v5_tng300_099.hdf5'
gasprofs300_v5_vals = h5py.File(gasprofs300_v5_filename, 'r', libver='earliest', swmr=True)


rho_crit = constants.RHO_CRIT_0_KPC3

#CLUSTER DICTIONARY (CLUSTER NAME, Z, R500)


cluster_dict = {'A133': ('A133', 0.0569, 1007),
                'A262': ('A262', 0.0162 , 650),
                'A383': ('A383', 0.1883, 944),
                'A478': ('A478', 0.0881, 1337),
                'A907': ('A907', 0.1603, 1096),
                'A1413':('A1413', 0.1429, 1299),
                'A1795':('A1795', 0.0622, 1235),
                'A1991':('A1991', 0.0592, 732),
                'A2029':('A2029', 0.0779, 1362),
                'A2390':('A2390', 0.2302, 1416),
                'RX':('RX J1159+5531', 0.0810, 700),
                'MKW 4':('MKW 4', 0.0199, 634),
                'USGC S152':('USGC S152', 0.0153, None)
        }

clusters = ['A2390', 'A133']

Vik_Cosmo = cosmology.setCosmology('Vik_Cosmo', params = cosmology.cosmologies['planck18'], Om0 = 0.3, Ode0=0.7, H0 = 72, sigma8 = 0.9)
print(Vik_Cosmo)

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
    
    H_col = Vik_Cosmo.Hz(z)
    
    h_col = H_col * (10 ** (-2))
    
    rho_c_col = Vik_Cosmo.rho_c(z) * (constants.MSUN)*((h_col)**2)/(constants.KPC**3)
    
    rho_norm_col = rho_g/rho_c_col #* 100
    
    
    return rho_g, rho_norm_col

def get_rho_norm_fit(r, n0, rc, rs, a, B, epsilon, n02, rc2, B2):
    npne = ((n0*(10**-3))**2)*(((r/rc)**(-a))/((1+(r**2)/(rc**2))**(3*B + (-a/2))))* \
        (1/((1+((r**gamma)/(rs**gamma)))**(epsilon/gamma))) \
            + (((n02*(10**-1))**2)/ ((1 + (r**2)/(rc2**2))**(3*B2)))

            
    rho_g = 1.624 * mp * (npne)**(1/2)
    
    #rho_c_ill = mycosmo.rho_c(0.2302) * (constants.MSUN)*(.677**2)/(constants.KPC**3)
    
    H_col = Vik_Cosmo.Hz(z)
    
    h_col = H_col * (10 ** (-2))
    
    rho_c_col = Vik_Cosmo.rho_c(z) * (constants.MSUN)*((h_col)**2)/(constants.KPC**3)
    
    rho_norm_col = rho_g/rho_c_col #* 100
    
    
    return rho_norm_col

radii_all100 = np.array(gasprofs100_v5_vals['profile_bins'])
R_Crit500_100 = np.array(gasprofs100_v5_vals['catgrp_Group_R_Crit500'])

radii_all100_norm = radii_all100/R_Crit500_100[:,None]
              
rho_vals100 = np.array(gasprofs100_v5_vals['profile_gas_rho_3d'])
rho_vals100_norm = rho_vals100/rho_crit

radii_all300 = np.array(gasprofs300_v5_vals['profile_bins'])
R_Crit500_300 = np.array(gasprofs300_v5_vals['catgrp_Group_R_Crit500'])

radii_all300_norm = radii_all300/R_Crit500_300[:,None]

rho_vals300 = np.array(gasprofs300_v5_vals['profile_gas_rho_3d'])
rho_vals300_norm = rho_vals300/rho_crit

#EXTRACT CLUSTER OBSERVATION DATA + PLUG AND CHUG
#====================================================================================

Vik_bins = np.linspace(78, 23400, 500) #Minimum and maximum radial bins in both  TNG 100 and 300

n0, rc, rs, a, B, epsilon, n02, rc2, B2 = np.loadtxt('Vikhlinin_tab2.csv', skiprows = False, unpack=True, delimiter = ',', usecols=(0,1,2,3,4,5,6,7,8))
                                                                 
z=cluster_dict['A133'][1]
rho_g_A133, rho_norm_Vik_A133 = get_rho_norm(Vik_bins, n0[0], rc[0], rs[0], a[0], B[0], epsilon[0], n02[0], rc2[0], B2[0])

z=cluster_dict['A262'][1]
rho_g_A262, rho_norm_Vik_A262 = get_rho_norm(Vik_bins, n0[1], rc[1], rs[1], a[1], B[1], epsilon[1], n02[1], rc2[1], B2[1])

z=cluster_dict['A383'][1]
rho_g_A383, rho_norm_Vik_A383 = get_rho_norm(Vik_bins, n0[2], rc[2], rs[2], a[2], B[2], epsilon[2], n02[2], rc2[2], B2[2])

z=cluster_dict['A478'][1]
rho_g_A478, rho_norm_Vik_A478 = get_rho_norm(Vik_bins, n0[3], rc[3], rs[3], a[3], B[3], epsilon[3], n02[3], rc2[3], B2[3])

z=cluster_dict['A907'][1]
rho_g_A907, rho_norm_Vik_A907 = get_rho_norm(Vik_bins, n0[4], rc[4], rs[4], a[4], B[4], epsilon[4], n02[4], rc2[4], B2[4])

z=cluster_dict['A1413'][1]
rho_g_A1413, rho_norm_Vik_A1413 = get_rho_norm(Vik_bins, n0[5], rc[5], rs[5], a[5], B[5], epsilon[5], n02[5], rc2[5], B2[5])

z=cluster_dict['A1795'][1]
rho_g_A1795, rho_norm_Vik_A1795 = get_rho_norm(Vik_bins, n0[6], rc[6], rs[6], a[6], B[6], epsilon[6], n02[6], rc2[6], B2[6])

z=cluster_dict['A1991'][1]
rho_g_A1991, rho_norm_Vik_A1991 = get_rho_norm(Vik_bins, n0[7], rc[7], rs[7], a[7], B[7], epsilon[7], n02[7], rc2[7], B2[7])

z=cluster_dict['A2029'][1]
rho_g_A2029, rho_norm_Vik_A2029 = get_rho_norm(Vik_bins, n0[8], rc[8], rs[8], a[8], B[8], epsilon[8], n02[8], rc2[8], B2[8])

z=cluster_dict['A2390'][1]
rho_g_A2390, rho_norm_Vik_A2390 = get_rho_norm(Vik_bins, n0[9], rc[9], rs[9], a[9], B[9], epsilon[9], n02[9], rc2[9], B2[9])

z=cluster_dict['RX'][1]
rho_g_RX, rho_norm_Vik_RX = get_rho_norm(Vik_bins, n0[10], rc[10], rs[10], a[10], B[10], epsilon[10], n02[10], rc2[10], B2[10])

z=cluster_dict['MKW 4'][1]
rho_g_MKW, rho_norm_Vik_MKW = get_rho_norm(Vik_bins, n0[11], rc[11], rs[11], a[11], B[11], epsilon[11], n02[11], rc2[11], B2[11])
#emis_profile_USGC = get_rho_norm(Vik_bins, n0[12], rc[12], rs[12], a[12], B[12], epsilon[12], n02[12], rc2[12], B2[12])


m_radii_norm_A133 = Vik_bins/cluster_dict['A133'][2]
m_radii_norm_A262 = Vik_bins/cluster_dict['A262'][2]
m_radii_norm_A383 = Vik_bins/cluster_dict['A383'][2]
m_radii_norm_A478 = Vik_bins/cluster_dict['A478'][2]
m_radii_norm_A907 = Vik_bins/cluster_dict['A907'][2]
m_radii_norm_A1413 = Vik_bins/cluster_dict['A1413'][2]
m_radii_norm_A1795 = Vik_bins/cluster_dict['A1795'][2]
m_radii_norm_A1991 = Vik_bins/cluster_dict['A1991'][2]
m_radii_norm_A2029 = Vik_bins/cluster_dict['A2029'][2]
m_radii_norm_A2390 = Vik_bins/cluster_dict['A2390'][2]
m_radii_norm_RX = Vik_bins/cluster_dict['RX'][2]
m_radii_norm_MKW = Vik_bins/cluster_dict['MKW 4'][2]

#Median Density Calculations
#====================================================================================
my_bins100 = [np.min(radii_all100_norm[:,0]), np.max(radii_all100_norm[:,-1])]
my_bins100 = np.logspace(np.log10(my_bins100[0]), np.log10(my_bins100[1]), 50)

my_bins100_centers = (my_bins100[:-1] + my_bins100[1:]) / 2.0
              
bins_4_clus100 = [[] for _ in range(100)]
bins_4_clus100_centers = [[] for _ in range(100)]

for cluster in range(len(radii_all100_norm)):
    mask = (my_bins100 >= radii_all100_norm[cluster][0]) & (my_bins100 <= radii_all100_norm[cluster][-1])
    bins_4_clus100[cluster].extend(my_bins100[mask])

    for i in range(len(bins_4_clus100[cluster])-1):
        midpoints = (bins_4_clus100[cluster][i] + bins_4_clus100[cluster][i+1]) / 2
        bins_4_clus100_centers[cluster].append(midpoints)

rho_vals_interp100 = np.empty((len(radii_all100_norm),len(my_bins100_centers)))
rho_vals_interp100[:]=np.nan

for cluster in range(len(radii_all100_norm)):
    rho_vals_interp_func = interp1d(radii_all100_norm[cluster], rho_vals100_norm[cluster], bounds_error=False, fill_value=np.nan)
    for j in range(len(my_bins100)-1):
        if bins_4_clus100[cluster][0] == my_bins100[j]: 
            interp_values = rho_vals_interp_func(bins_4_clus100_centers[cluster])
            rho_vals_interp100[cluster, j:j+len(interp_values)] = interp_values
            
for j in range(len(my_bins100_centers)):
    if np.sum(np.isnan(rho_vals_interp100[:,j])) > 95:
        rho_vals_interp100[:,j] = np.nan

#Add line that will only include rho interp if more than 5 clusters have a value
median_rho_100 = np.nanmedian(rho_vals_interp100, axis=0)


radii_all300 = np.array(gasprofs300_v5_vals['profile_bins'])
R_Crit500_300 = np.array(gasprofs300_v5_vals['catgrp_Group_R_Crit500'])

radii_all300_norm = radii_all300/R_Crit500_300[:,None]

my_bins300 = [np.min(radii_all300_norm[:,0]), np.max(radii_all300_norm[:,-1])]
my_bins300 = np.logspace(np.log10(my_bins300[0]), np.log10(my_bins300[1]), 50)

my_bins300_centers = (my_bins300[:-1] + my_bins300[1:]) / 2.0
              
rho_vals300 = np.array(gasprofs300_v5_vals['profile_gas_rho_3d'])
rho_vals300_norm = rho_vals300/rho_crit

 
bins_4_clus300 = [[] for _ in range(100)]
bins_4_clus300_centers = [[] for _ in range(100)]

for cluster in range(len(radii_all300_norm)):
    mask = (my_bins300 >= radii_all300_norm[cluster][0]) & (my_bins300 <= radii_all300_norm[cluster][-1])
    bins_4_clus300[cluster].extend(my_bins300[mask])

    for i in range(len(bins_4_clus300[cluster])-1):
        midpoints = (bins_4_clus300[cluster][i] + bins_4_clus300[cluster][i+1]) / 2
        bins_4_clus300_centers[cluster].append(midpoints)

rho_vals_interp300 = np.empty((len(radii_all300_norm),len(my_bins300_centers)))
rho_vals_interp300[:]=np.nan

for cluster in range(len(radii_all300_norm)):
    rho_vals_interp_func = interp1d(radii_all300_norm[cluster], rho_vals300_norm[cluster], bounds_error=False, fill_value=np.nan)
    for j in range(len(my_bins300)-1):
        if bins_4_clus300[cluster][0] == my_bins300[j]: 
            interp_values = rho_vals_interp_func(bins_4_clus300_centers[cluster])
            
            rho_vals_interp300[cluster, j:j+len(interp_values)] = interp_values
            
for j in range(len(my_bins300_centers)):
    if np.sum(np.isnan(rho_vals_interp300[:,j])) > 95:
        rho_vals_interp300[:,j] = np.nan
        
median_rho_300 = np.nanmedian(rho_vals_interp300, axis=0)

#PLOTTING
#====================================================================================

plt.rcParams.update({'font.size': 16})
fig2 = plt.figure(figsize=(10,8), dpi=200)
ax = plt.subplot(111)
plt.xscale("log")



# for i in range(0, 99):
#     radii_all100 = np.array(gasprofs100_v5_vals['profile_bins'])
#     bin_centers100 = (radii_all100[:, :-1] + radii_all100[:, 1:]) / 2
#     first_bin_center100 = (radii_all100[:, 0]/2)
#     bin_centers100 = np.insert(bin_centers100, 0, first_bin_center100, axis=1)
#     R_500c = np.array(gasprofs100_v5_vals['catgrp_Group_R_Crit500'])
#     r_normalized_per = bin_centers100[i]/R_500c[i]
#     rho_vals_per = np.array(gasprofs100_v5_vals['profile_gas_rho_3d'][i])/rho_crit
#     plt.xscale("log")
#     plt.semilogy(r_normalized_per, rho_vals_per, color="yellow", lw=.5, alpha=0.25)            

# for i in range(0, 99):
#     radii_all300 = np.array(gasprofs300_v5_vals['profile_bins'])
#     bin_centers300 = (radii_all300[:, :-1] + radii_all300[:, 1:]) / 2
#     first_bin_center300 = (radii_all300[:, 0]/2)
#     bin_centers300 = np.insert(bin_centers300, 0, first_bin_center300, axis=1)
#     R_500c = np.array(gasprofs300_v5_vals['catgrp_Group_R_Crit500'])
#     r_normalized_per = bin_centers300[i]/R_500c[i]
#     rho_vals_per = np.array(gasprofs300_v5_vals['profile_gas_rho_3d'][i])/rho_crit
#     plt.xscale("log")
#     plt.semilogy(r_normalized_per, rho_vals_per, color="purple", lw=.5, alpha=0.25)            

plt.semilogy(m_radii_norm_A133, rho_norm_Vik_A133, color="red", lw=1, alpha=1, label='A133')
plt.semilogy(m_radii_norm_A262, rho_norm_Vik_A262, color="indianred", lw=1, alpha=1, label='A262')
plt.semilogy(m_radii_norm_A383, rho_norm_Vik_A383, color="tomato", lw=1, alpha=1, label='A383')
plt.semilogy(m_radii_norm_A478, rho_norm_Vik_A478, color="mistyrose", lw=1, alpha=1, label='A478')
plt.semilogy(m_radii_norm_A907, rho_norm_Vik_A907, color="sienna", lw=1, alpha=1, label='A907')
plt.semilogy(m_radii_norm_A1413, rho_norm_Vik_A1413, color="peru", lw=1, alpha=1, label='A1413')
plt.semilogy(m_radii_norm_A1795, rho_norm_Vik_A1795, color="darkgoldenrod", lw=1, alpha=1, label='A1795')
plt.semilogy(m_radii_norm_A1991, rho_norm_Vik_A1991, color="khaki", lw=1, alpha=1, label='A1991')
plt.semilogy(m_radii_norm_A2029, rho_norm_Vik_A2029, color="olive", lw=1, alpha=1, label='A2029')
plt.semilogy(m_radii_norm_A2390, rho_norm_Vik_A2390, color="limegreen", lw=1, alpha=1, label='A2390')
plt.semilogy(m_radii_norm_RX, rho_norm_Vik_RX, color="blue", lw=1, alpha=1, label='RX J1159+5531')
plt.semilogy(m_radii_norm_MKW, rho_norm_Vik_MKW, color="navy", lw=1, alpha=1, label='MKW 4')


plt.semilogy(my_bins100_centers, median_rho_100, color="#1F77B4", lw=5, alpha=1, label='TNG100')
plt.semilogy(my_bins300_centers, median_rho_300, color="#FF7F0E", lw=5, alpha=1, label='TNG300')


sixteenth_percentile= np.nanpercentile(rho_vals_interp100, 16, axis=0)
plt.semilogy(my_bins100_centers, sixteenth_percentile, color="#1F77B4", lw=3, alpha=0.2)

eightfour_percentile= np.nanpercentile(rho_vals_interp100, 84, axis=0)
plt.semilogy(my_bins100_centers, eightfour_percentile, color="#1F77B4", lw=3, alpha=0.2)

plt.fill_between(my_bins100_centers, sixteenth_percentile, eightfour_percentile, color="#1F77B4", alpha=0.2)



sixteenth_percentile= np.nanpercentile(rho_vals_interp300, 16, axis=0)
plt.semilogy(my_bins300_centers, sixteenth_percentile, color="#FF7F0E", lw=3, alpha=0.2)

eightfour_percentile= np.nanpercentile(rho_vals_interp300, 84, axis=0)
plt.semilogy(my_bins300_centers, eightfour_percentile, color="#FF7F0E", lw=3, alpha=0.2)

plt.fill_between(my_bins300_centers, sixteenth_percentile, eightfour_percentile, color="#FF7F0E", alpha=0.2)

plt.xlabel('r/R500c', fontsize=20)
plt.ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
plt.title("Density of TNG simulations and cluster observations vs. radius", fontsize=30)


box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0, box.width, box.height * 0.9])
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, fancybox=True, shadow=True, fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()




# ax1.semilogy(my_bins100_centers, median_rho_100, color="#1F77B4", lw=8, alpha=1, label='TNG100')
# ax1.semilogy(my_bins300_centers, median_rho_300, color="#FF7F0E", lw=8, alpha=1, label='TNG300')

# ax1.semilogy(my_bins100_centers_red_2x, curve_fit_result100_p1, color="red", lw=3, alpha=1)
# ax1.semilogy(my_bins300_centers_red_2x, curve_fit_result300_p1, color="black", lw=3, alpha=1)


# for i in range(0, 99):
#     radii_all100 = np.array(gasprofs100_v5_vals['profile_bins'])
#     bin_centers100 = (radii_all100[:, :-1] + radii_all100[:, 1:]) / 2
#     first_bin_center100 = (radii_all100[:, 0]/2)
#     bin_centers100 = np.insert(bin_centers100, 0, first_bin_center100, axis=1)
#     R_500c = np.array(gasprofs100_v5_vals['catgrp_Group_R_Crit500'])
#     r_normalized_per = bin_centers100[i]/R_500c[i]
#     rho_vals_per = np.array(gasprofs100_v5_vals['profile_gas_rho_3d'][i])/rho_crit
#     plt.xscale("log")
#     ax1.semilogy(r_normalized_per[2:], rho_vals_per[2:], color="yellow", lw=.5, alpha=0.25)            

# for i in range(0, 99):
#     radii_all300 = np.array(gasprofs300_v5_vals['profile_bins'])
#     bin_centers300 = (radii_all300[:, :-1] + radii_all300[:, 1:]) / 2
#     first_bin_center300 = (radii_all300[:, 0]/2)
#     bin_centers300 = np.insert(bin_centers300, 0, first_bin_center300, axis=1)
#     R_500c = np.array(gasprofs300_v5_vals['catgrp_Group_R_Crit500'])
#     r_normalized_per = bin_centers300[i]/R_500c[i]
#     rho_vals_per = np.array(gasprofs300_v5_vals['profile_gas_rho_3d'][i])/rho_crit
#     plt.xscale("log")
#     ax1.semilogy(r_normalized_per[2:], rho_vals_per[2:], color="purple", lw=.5, alpha=0.25)            

# sixteenth_percentile= np.nanpercentile(rho_vals_interp100, 16, axis=0)
# ax1.semilogy(my_bins100_centers, sixteenth_percentile, color="#1F77B4", lw=3, alpha=0.2)

# eightfour_percentile= np.nanpercentile(rho_vals_interp100, 84, axis=0)
# ax1.semilogy(my_bins100_centers, eightfour_percentile, color="#1F77B4", lw=3, alpha=0.2)

# ax1.fill_between(my_bins100_centers, sixteenth_percentile, eightfour_percentile, color="#1F77B4", alpha=0.2)



# sixteenth_percentile= np.nanpercentile(rho_vals_interp300, 16, axis=0)
# ax1.semilogy(my_bins300_centers, sixteenth_percentile, color="#FF7F0E", lw=3, alpha=0.2)

# eightfour_percentile= np.nanpercentile(rho_vals_interp300, 84, axis=0)
# ax1.semilogy(my_bins300_centers, eightfour_percentile, color="#FF7F0E", lw=3, alpha=0.2)

# ax1.fill_between(my_bins300_centers, sixteenth_percentile, eightfour_percentile, color="#FF7F0E", alpha=0.2)


# ax1.set_xlabel('r/R500c', fontsize=20)
# ax1.set_ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
# ax1.set_title("TNG Simulations Fit vs. radius", fontsize=20)


# ax1.set_xscale("log")



