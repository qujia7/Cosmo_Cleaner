
import numpy as np
from orphics import maps, cosmology,io,stats
from scipy import optimize
import cosmo_cleaner
import matplotlib.pyplot as plt
from cosmo_cleaner import fisher as fisher,plot
from cosmo_cleaner import cosmology,triangle_plot
import camb
import warnings
warnings.filterwarnings('ignore')


defaultCosmology = {'omch2': 0.1198
                    ,'ombh2': 0.02225
                    ,'H0': 67.3
                    ,'ns': 0.9645
                    ,'As': 2.2e-9
                    ,'mnu': 0.06
                    ,'w0': -1.
                    ,'tau':0.06
                    ,'nnu':3.046
                    ,'wa': 0.0,'bias':1,'A_e':1.,'A_l':1.,'omega_k':0,'mctheta': 0.01040909}
nz = 1000
kmax = 10
zmin = 0.
ells=np.arange(3000)

pars = camb.CAMBparams()
pars.set_dark_energy(w=defaultCosmology['w0'],wa = defaultCosmology['wa'], dark_energy_model = 'ppf')
pars.set_cosmology( cosmomc_theta = defaultCosmology['mctheta'],ombh2=defaultCosmology['ombh2'], 
               omch2=defaultCosmology['omch2'], mnu=defaultCosmology['mnu'], tau = defaultCosmology['tau'],
               nnu = defaultCosmology['nnu'], num_massive_neutrinos = 3)
pars.InitPower.set_params(ns=defaultCosmology['ns'],As=defaultCosmology['As'])

results = camb.get_results(pars)

#include w_0 now
parameters = np.array( ['ombh2', 'omch2','mctheta',  'As', 'ns', 'tau', 'mnu'])
centers = np.array([0.02222,  0.1197, 0.01040909, 2.196e-9, 0.9655, 0.06, 0.06,-1.0])
left_steps = np.array([0.0008, 0.003, 0.00000001,  0.1e-9, 0.010, 0.02, 0.0,0.3])
right_steps = np.array([0.0008, 0.003, 0.00000001,   0.1e-9, 0.010, 0.02, 0.02,0.3])
delta=[]
for i in range(len(left_steps)):
    delta.append((left_steps[i],right_steps[i]))


lensing=cosmology.CMB_Lensing(lens_beam=3.0, lens_noiseT=1.0, lens_noiseP=1.4)
fisher_lensing=lensing.compute_fisher_from_camb(results,defaultCosmology,parameters,delta,use_theta=True)

np.save("./data/neutrino_lensing_theta.npy",fisher_lensing)

Allison_S4 = [cosmology.CMB_Primary(l_min=50, l_max=3000,
                theta_fwhm=(1.0,), sigma_T=(1.0,), 
                sigma_P=(1.4,), f_sky=0.4),
              cosmology.CMB_Primary(l_min=3000, l_max=5000,
                theta_fwhm=(3.0,), sigma_T=(1e100,),
                sigma_P=(1.4,), f_sky=0.4),
                cosmology.CMB_Primary(l_min=2, l_max=50, f_sky=0.44)]


fisher_Allison_S4=np.sum( [obs.compute_fisher_from_camb(results,defaultCosmology,parameters,delta,use_theta=True) for obs in Allison_S4], axis=0 )
np.save("./data/neutrino_primary_theta.npy",fisher_Allison_S4)

BAO=[cosmology.get_DESI_lowz(),cosmology.get_DESI_highz()]

BAO_fisher=np.sum(obs.compute_fisher_from_camb(pars,defaultCosmology,parameters,delta,use_theta=True) for obs in BAO)

np.save("./data/neutrino_bao_theta.npy",BAO_fisher)
BAO_h=cosmology.get_DESI_highz().compute_fisher_from_camb(pars,defaultCosmology,parameters,delta,use_theta=True)
np.save("./data/neutrino_bao_theta_h.npy",BAO_h)

BAO_ly=cosmology.get_DESI_ly().compute_fisher_from_camb(pars,defaultCosmology,parameters,delta,use_theta=True)
np.save("./data/neutrino_bao_theta_ly.npy",BAO_ly)