from orphics import cosmology
from cosmo_cleaner import fisher
import numpy as np
from orphics import maps, cosmology,io,stats
from scipy import optimize
import cosmo_cleaner
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
#from IPython.display import set_matplotlib_formats set_matplotlib_formats('retina')
csfont = {'fontname':'Latin Modern Roman'}
font = font_manager.FontProperties(family='Latin Modern Roman',style='normal', size = 8)

def get_corrcoef(clgg,clcibcib,clkk,clkg,clcibk,clcibg):
    rho={}
    rho_gk=np.sqrt(clkg**2/(clgg*clkk))
    rho_gcib=np.sqrt(clcibg**2/(clcibcib*clgg))
    rho_kcib=np.sqrt(clcibk**2/(clcibcib*clkk))
    rho['gk']=rho_gk
    rho['gcib']=rho_gcib
    rho['kcib']=rho_kcib
    return rho

def get_corr(cross,f1,f2):
    return np.sqrt(cross**2/(f1*f2))

def get_der(spectra,pars,delta,cleaned=False):
    """derivative function used for fisher"""
    der_spectra_alpha = np.ones((len(list(spectra.items())[0][1]), len(spectra), len(pars)))
    for i in range(len(pars)):
        print(f"Taking field derivatives wrt {pars[i]}")
        der=cosmology.derivative_parameter(ells,1,0.5,defaultCosmology,pars[i],delta[i])
        der_spectra_alpha[:, 0, i] = der[1][:cut] #kg
        der_spectra_alpha[:, 1, i] = der[0][:cut] #gg
        if cleaned==True:
            der_spectra_alpha[:, 2, i] = der[-1][:cut]
        else:
            der_spectra_alpha[:, 2, i] = der[2][:cut]
    return der_spectra_alpha









import camb
defaultCosmology = {'omch2': 0.120
                    ,'ombh2': 0.0224
                    ,'H0': 67.4
                    ,'ns': 0.965
                    ,'As': 2.1e-9
                    ,'mnu': 0.06
                    ,'w0': -1.0
                    ,'tau':0.06
                    ,'nnu':3.046
                    ,'wa': 0.,'bias':1,'A_e':1.,'A_l':1.,'N_eff':3.046,'omega_e':0.007,'omega_k':0}

nz = 1000
kmax = 10
zmin = 0.
ells=np.arange(2000)
for i in range(16):
    defaultCosmology[f'lsst_bias{i}']=1
pars = camb.CAMBparams()
pars.set_dark_energy(w=defaultCosmology['w0'],wa = defaultCosmology['wa'], dark_energy_model = 'ppf')
pars.set_cosmology(H0=defaultCosmology['H0'], cosmomc_theta = None,ombh2=defaultCosmology['ombh2'], 
               omch2=defaultCosmology['omch2'], mnu=defaultCosmology['mnu'], tau = defaultCosmology['tau'],
               nnu = defaultCosmology['nnu'], num_massive_neutrinos = 3)
#pars.NonLinear = model.NonLinear_both
pars.InitPower.set_params(ns=defaultCosmology['ns'],As=defaultCosmology['As'])

results = camb.get_results(pars)
from cosmo_cleaner import cosmology
from cosmo_cleaner import triangle_plot
default=cosmology.cosmology(nz,kmax,zmin,ells,defaultCosmology,pars,results)


clgg,clkg,clkk,clcibcib,clcibk,cl_clean=default.get_cibspectra(0.1,2)

pars=['bias','A_e','A_l']
delta=[(0.01,0.01),(0.01,0.01),(0.01,0.01)]

cut=300
"""
spectra = {'kg': clkg[:cut], 'gg' :clgg[:cut], 'kk': clkk[:cut]}
der_spectra_alpha=get_der(spectra,pars,delta,cleaned=False)
a=fisher.Fisher(1,len(pars),np.arange(len(clkg[:cut])),spectra,der_spectra_alpha)
a.get_fisher()
np.save(f"/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/cibcleaning/fisherlmax{cut}.npy",a.fisher)
"""
pars=['bias','A_e','A_l']
delta=[(0.01,0.01),(0.01,0.01),(0.01,0.01)]

base_error_nonoise=[]
fishermatrix_nonoise=[]

cut=202
spectra = {'kg': clkg[:cut], 'gg' :clgg[:cut], 'kk': clkk[:cut]}
der_spectra_alpha=get_der(spectra,pars,delta,cleaned=False)
a=fisher.Fisher(1,len(pars),np.arange(len(clkg[:cut])),spectra,der_spectra_alpha)
base_error_nonoise.append(a.get_fisher()[0])
fishermatrix_nonoise.append(a.fisher)

np.save(f"/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/cibcleaning/fisherlmaxnocleancut{cut}.npy",a.fisher)


"""
spectra = {'kg': clkg[:cut], 'gg' :clgg[:cut], 'kk': cl_clean[:cut]}
der_spectra_alpha=get_der(spectra,pars,delta,cleaned=True)
a=fisher.Fisher(1,len(pars),np.arange(len(clkg[:cut])),spectra,der_spectra_alpha)
a.get_fisher()
np.save(f"/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/cibcleaning/fisherlmax{cut}_clean.npy",a.fisher)
"""
