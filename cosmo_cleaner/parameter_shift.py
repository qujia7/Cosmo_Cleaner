import numpy as np
from orphics import maps, cosmology,io,stats
import matplotlib.pyplot as plt
from scipy import optimize
import cosmo_cleaner
from cosmo_cleaner import fisher as fisher,plot
from orphics import stats,io,mpi,maps
import camb
from cosmo_cleaner import cosmology
from cosmo_cleaner import triangle_plot
import camb
from camb import model, initialpower

def get_der(spectra,pars,delta,lensingzmin,cleaned=False):
    """derivative function used for fisher"""
    der_spectra_alpha = np.ones((len(list(spectra.items())[0][1]), len(spectra), len(pars)))
    for i in range(len(pars)):
        print(f"Taking field derivatives wrt {pars[i]}")
        der=cosmology.derivative_lensing(ells,1,0.5,defaultCosmology,pars[i],delta[i],nz=1000,kmax=10,zmin=0,zlensing=lensingzmin,idealised=False)

        der_spectra_alpha[:, 0, i] = der[:cut] #kk
    return der_spectra_alpha


defaultCosmology = {'omch2': 0.1198
                    ,'ombh2': 0.02225
                    ,'H0': 67.3
                    ,'ns': 0.9645
                    ,'As': 2.2e-9
                    ,'mnu': 0.06
                    ,'w0': -1.0
                    ,'tau':0.06
                    ,'nnu':3.046
                    ,'wa': 0.0,'bias':1,'A_e':1.,'A_l':1.,'omega_k':0}
for i in range(17):
    defaultCosmology[f'lsst_bias{i}']=1

nz = 1000
kmax = 10
zmin = 0.
ells=np.arange(2000)

pars = camb.CAMBparams()
pars.set_cosmology(H0=defaultCosmology['H0'], cosmomc_theta = None,ombh2=defaultCosmology['ombh2'], 
               omch2=defaultCosmology['omch2'], mnu=defaultCosmology['mnu'], tau = defaultCosmology['tau'],
               nnu = defaultCosmology['nnu'], num_massive_neutrinos = 3)
#pars.NonLinear = model.NonLinear_both
pars.InitPower.set_params(ns=defaultCosmology['ns'],As=defaultCosmology['As'])
results = camb.get_results(pars)
default=cosmology.cosmology(nz,kmax,zmin,ells,defaultCosmology,pars,results)

clgg,clkg,clkk,clcibcib,clcibk,cl_clean=default.get_cibspectra(0.1,2)

parameters = np.array( ['ombh2', 'omch2', 'H0',  'As', 'ns', 'tau','mnu'])
centers = np.array([0.02225,  0.1198,  67.3,  2.2e-9, 0.9645, 0.06, 0.06])
left_steps = np.array([0.0008, 0.003,   0.5,   0.1e-9, 0.010, 0.02, 0.0])
right_steps = np.array([0.0008, 0.003,   0.5,   0.1e-9, 0.010, 0.02, 0.02])
delta=[]
for i in range(len(left_steps)):
    delta.append((left_steps[i],right_steps[i]))


zmin=[0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,1]
print(zmin)
clkks=[]
clkkdes=[]
for i in range(len(zmin)):
    clkks.append(default.get_clkk(zmin=zmin[i]))


lensing=cosmology.CMB_Lensing(lens_beam=3.0, lens_noiseT=0.0, lens_noiseP=0.0,lens_kmin = 0,lens_kmax = 600, lens_f_sky=0.4)
fisher_lensing=lensing.compute_fisher_from_camb(results,defaultCosmology,parameters,delta)

np.save("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/parshift_lensing_test.npy",fisher_lensing)
print("lensing")
#we use a cut of Lmax=600 and gradually change the clkk used. Threshold change for zmin.
cut=2000
print(cut)
comm,rank,my_tasks = mpi.distribute(len(zmin))

for task in my_tasks:
    spectra = {'kk': clkks[task][:cut]}
    spectra_noise = {'kk': clkks[task][:cut]}
    der_spectra_alpha=get_der(spectra,parameters,delta,zmin[task],cleaned=False)
    a=fisher.Fisher(1,len(parameters),np.arange(len(clkg[:cut])),spectra_noise,der_spectra_alpha)
    a.get_fisher()[0]
    fishermatrix=a.fisher

    np.save(f"/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/derivativen{zmin[task]}.npy",der_spectra_alpha)
    np.save(f"/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/parshiftsfishern{zmin[task]}.npy",fishermatrix)


"""
lensing=cosmology.CMB_Lensing(lens_beam=3.0, lens_noiseT=1.0, lens_noiseP=1.4)
fisher_lensing=lensing.compute_fisher_from_camb(results,defaultCosmology,parameters,delta)

np.save("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/parshifts4fisherlensing.npy",fisher_lensing)

Allison_S4 = [cosmology.CMB_Primary(l_min=50, l_max=3000,
                theta_fwhm=(1.0,), sigma_T=(1.0,), 
                sigma_P=(1.4,), f_sky=0.4),
              cosmology.CMB_Primary(l_min=3000, l_max=5000,
                theta_fwhm=(3.0,), sigma_T=(1e100,),
                sigma_P=(1.4,), f_sky=0.4),
                cosmology.CMB_Primary(l_min=2, l_max=50, f_sky=0.44)]


fisher_Allison_S4=np.sum( [obs.compute_fisher_from_camb(results,defaultCosmology,parameters,delta) for obs in Allison_S4], axis=0 )
np.save("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/parshifts4primary.npy",fisher_Allison_S4)
"""