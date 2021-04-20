from __future__ import print_function
import pytempura
from cosmo_cleaner import cosmology
import numpy as np
import camb
"""
a=cosmology.CMB_Lensing(lens_beam = 1.4,lens_noiseT = 1.,lens_noiseP = 1.4,lens_tellmin = 50,lens_tellmax = 3000,lens_pellmin = 50,
             lens_pellmax = 5000,lens_kmin = 30,lens_kmax = 2000, lens_f_sky=0.4, estimators=('TT','TE','EE','EB','TB','MV'), 
                NlTT=None, NlEE=None, NlBB=None)

print(a.noise_k)
np.savetxt("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/nlkkswn15.txt",a.noise_k)
"""

pars = np.array( ['ombh2', 'omch2', 'H0',  'As', 'ns', 'tau', 'mnu'])
centers = np.array([0.02222,  0.1197,  69,  2.196e-9, 0.9655, 0.06, 0.06])
defaultCosmology = {'omch2': 0.1197
                    ,'ombh2': 0.02222
                    ,'H0': 69
                    ,'ns': 0.9655
                    ,'As': 2.2e-9
                    ,'mnu': 0.06
                    ,'w0': -1.0
                    ,'tau':0.06
                    ,'nnu':3.046
                    ,'wa': 0.,'bias':2,'A_l':1.2}


nz = 1000
kmax = 10
zmin = 0.

pars = camb.CAMBparams()
pars.set_dark_energy(w=defaultCosmology['w0'],wa = defaultCosmology['wa'], dark_energy_model = 'ppf')
pars.set_cosmology(H0=defaultCosmology['H0'], cosmomc_theta = None,ombh2=defaultCosmology['ombh2'], 
               omch2=defaultCosmology['omch2'], mnu=defaultCosmology['mnu'], tau = defaultCosmology['tau'],
               nnu = defaultCosmology['nnu'], num_massive_neutrinos = 3)
#pars.NonLinear = model.NonLinear_both
pars.InitPower.set_params(ns=defaultCosmology['ns'],As=defaultCosmology['As'])

results = camb.get_results(pars)
parameters = np.array( ['ombh2', 'omch2', 'H0',  'As', 'ns', 'tau', 'mnu'])
centers = np.array([0.02222,  0.1197,  69,  2.196e-9, 0.9655, 0.06, 0.06])
left_steps = np.array([0.0008, 0.003,   0.2,   0.1e-9, 0.010, 0.02, 0.0])
right_steps = np.array([0.0008, 0.003,   0.2,   0.1e-9, 0.010, 0.02, 0.02])
delta=[]
for i in range(len(left_steps)):
    delta.append((left_steps[i],right_steps[i]))

#lensing for S2
"""
a=cosmology.CMB_Lensing(lens_kmax=2000, lens_tellmax=2500, lens_pellmax=2500,lens_f_sky=0.44)
fisher=a.compute_fisher_from_camb(results,defaultCosmology,parameters,delta)

np.savetxt("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/fisher_lensing_allisons.txt",fisher)

a=cosmology.CMB_Lensing(lens_tellmin = 50,lens_pellmin = 50,lens_pellmax = 4000,lens_kmax=3000,lens_beam=1.4, lens_noiseT=8.0, lens_noiseP=11.3)
fisher=a.compute_fisher_from_camb(results,defaultCosmology,parameters,delta)

np.savetxt("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/fisherS3.txt",fisher)
"""


#fisher lensing S4

defaultCosmology = {'omch2': 0.1197
                    ,'ombh2': 0.02222
                    ,'H0': 69
                    ,'ns': 0.9655
                    ,'As': 2.2e-9
                    ,'mnu': 0.06
                    ,'w0': -1.0
                    ,'tau':0.06
                    ,'nnu':3.046
                    ,'wa': 0.,'bias':2,'A_l':1.2,'N_eff':3.046,'omega_e':0.007,'omega_k':0}


nz = 1000
kmax = 10
zmin = 0.

pars = camb.CAMBparams()
pars.set_dark_energy(w=defaultCosmology['w0'],wa = defaultCosmology['wa'], dark_energy_model = 'ppf')
pars.set_cosmology(H0=defaultCosmology['H0'], cosmomc_theta = None,ombh2=defaultCosmology['ombh2'], 
               omch2=defaultCosmology['omch2'],omk=defaultCosmology['omega_k'], mnu=defaultCosmology['mnu'], tau = defaultCosmology['tau'],
               nnu = defaultCosmology['nnu'], num_massive_neutrinos = 3)
#pars.NonLinear = model.NonLinear_both
pars.InitPower.set_params(ns=defaultCosmology['ns'],As=defaultCosmology['As'])

results = camb.get_results(pars)
"""
parameters = np.array( ['ombh2', 'omch2', 'H0',  'As', 'ns', 'tau', 'mnu'])
centers = np.array([0.02222,  0.1197,  69,  2.196e-9, 0.9655, 0.06, 0.06])
left_steps = np.array([0.0008, 0.003,   2,   0.1e-9, 0.010, 0.02, 0.0])
right_steps = np.array([0.0008, 0.003,   2,   0.1e-9, 0.010, 0.02, 0.02])
delta=[]
for i in range(len(left_steps)):
    delta.append((left_steps[i],right_steps[i]))

a=cosmology.CMB_Lensing(lens_beam=3.0, lens_noiseT=1.0, lens_noiseP=1.4)
fisher=a.compute_fisher_from_camb(results,defaultCosmology,parameters,delta)

np.savetxt("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/fisherS4.txt",fisher)
"""

#include omega_k now
parameters = np.array( ['ombh2', 'omch2', 'H0',  'As', 'ns', 'tau', 'mnu','omega_k'])
centers = np.array([0.02222,  0.1197,  69,  2.196e-9, 0.9655, 0.06, 0.06,0])
left_steps = np.array([0.0008, 0.003,   2,   0.1e-9, 0.010, 0.02, 0.0,0.01])
right_steps = np.array([0.0008, 0.003,   2,   0.1e-9, 0.010, 0.02, 0.02,0.01])
delta=[]
for i in range(len(left_steps)):
    delta.append((left_steps[i],right_steps[i]))

a=cosmology.CMB_Lensing(lens_beam=3.0, lens_noiseT=1.0, lens_noiseP=1.4)
fisher=a.compute_fisher_from_camb(results,defaultCosmology,parameters,delta)

np.savetxt("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/fisherS4_omegak.txt",fisher)

"""
#include w_0 now
parameters = np.array( ['ombh2', 'omch2', 'H0',  'As', 'ns', 'tau', 'mnu','w0'])
centers = np.array([0.02222,  0.1197,  69,  2.196e-9, 0.9655, 0.06, 0.06,-1.0])
left_steps = np.array([0.0008, 0.003,   2,   0.1e-9, 0.010, 0.02, 0.0,0.3])
right_steps = np.array([0.0008, 0.003,   2,   0.1e-9, 0.010, 0.02, 0.02,0.3])
delta=[]
for i in range(len(left_steps)):
    delta.append((left_steps[i],right_steps[i]))

a=cosmology.CMB_Lensing(lens_beam=3.0, lens_noiseT=1.0, lens_noiseP=1.4)
fisher=a.compute_fisher_from_camb(results,defaultCosmology,parameters,delta)

np.savetxt("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/fisherS4_w0.txt",fisher)


#include w_0  and w_a now
parameters = np.array( ['ombh2', 'omch2', 'H0',  'As', 'ns', 'tau', 'mnu','w0','wa'])
centers = np.array([0.02222,  0.1197,  69,  2.196e-9, 0.9655, 0.06, 0.06,-1.0,0])
left_steps = np.array([0.0008, 0.003,   2,   0.1e-9, 0.010, 0.02, 0.0,0.3,0.6])
right_steps = np.array([0.0008, 0.003,   2,   0.1e-9, 0.010, 0.02, 0.02,0.3,0.6])
delta=[]
for i in range(len(left_steps)):
    delta.append((left_steps[i],right_steps[i]))

a=cosmology.CMB_Lensing(lens_beam=3.0, lens_noiseT=1.0, lens_noiseP=1.4)
fisher=a.compute_fisher_from_camb(results,defaultCosmology,parameters,delta)

np.savetxt("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/fisherS4_w0_wa.txt",fisher)
"""

#include w_0  and w_a  and omega_k
parameters = np.array( ['ombh2', 'omch2', 'H0',  'As', 'ns', 'tau', 'mnu','w0','wa','omega_k'])
centers = np.array([0.02222,  0.1197,  69,  2.196e-9, 0.9655, 0.06, 0.06,-1.0,0,0])
left_steps = np.array([0.0008, 0.003,   2,   0.1e-9, 0.010, 0.02, 0.0,0.3,0.6,0.01])
right_steps = np.array([0.0008, 0.003,   2,   0.1e-9, 0.010, 0.02, 0.02,0.3,0.6,0.01])
delta=[]
for i in range(len(left_steps)):
    delta.append((left_steps[i],right_steps[i]))

a=cosmology.CMB_Lensing(lens_beam=3.0, lens_noiseT=1.0, lens_noiseP=1.4)
fisher=a.compute_fisher_from_camb(results,defaultCosmology,parameters,delta)

np.savetxt("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/fisherS4_w0_wa_omegak.txt",fisher)