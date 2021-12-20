import numpy as np
from orphics import maps, cosmology,io,stats
import matplotlib.pyplot as plt
from scipy import optimize
import cosmo_cleaner
from cosmo_cleaner import fisher as fisher,plot
import camb
from cosmo_cleaner import cosmology
from cosmo_cleaner import triangle_plot
from orphics import stats,io,mpi,maps
from camb import model, initialpower
from pixell import utils
def get_corr(cross,f1,f2):
    return np.sqrt(cross**2/(f1*f2))

def get_der(spectra,defaultCosmology,parameter,delta,nz=1000,kmax=10,zmin=0,gal_bin=1,cleaned=False):
    """derivative function used for fisher"""
    der_spectra_alpha = np.ones((len(list(spectra.items())[0][1]), len(spectra), len(parameter)))
    for i in range(len(parameter)):
        left,right=delta[i]
        cosmology_pars,pars,res=cosmology.get_cosmology_var(defaultCosmology,parameter[i],delta[i])
        high=cosmology.cosmology(nz,kmax,zmin,ells,cosmology_pars[0],pars[0],res[0])
        low=cosmology.cosmology(nz,kmax,zmin,ells,cosmology_pars[1],pars[1],res[1])
        der=(high.get_LSST_lensing_test(gal_bin)-low.get_LSST_lensing_test(gal_bin))/((right+left))
        der_spectra_alpha[:, 0, i] = der[1][:cut]
        der_spectra_alpha[:, 1, i] = der[0][:cut]
        if cleaned==True:
            der_spectra_alpha[:, 2, i] = der[-1][:cut]
        else:
            der_spectra_alpha[:, 2, i] = der[2][:cut]
    return der_spectra_alpha

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
for i in range(17):
    defaultCosmology[f'lsst_bias{i}']=1

nz = 1000
kmax = 10
zmin = 0.
ells=np.arange(5000)

pars = camb.CAMBparams()
pars.set_dark_energy(w=defaultCosmology['w0'],wa = defaultCosmology['wa'], dark_energy_model = 'ppf')
pars.set_cosmology( cosmomc_theta = defaultCosmology['mctheta'],ombh2=defaultCosmology['ombh2'], 
               omch2=defaultCosmology['omch2'], mnu=defaultCosmology['mnu'], tau = defaultCosmology['tau'],
               nnu = defaultCosmology['nnu'], num_massive_neutrinos = 3)
#pars.NonLinear = model.NonLinear_both
pars.InitPower.set_params(ns=defaultCosmology['ns'],As=defaultCosmology['As'])
results = camb.get_results(pars)
default=cosmology.cosmology(nz,kmax,zmin,ells,defaultCosmology,pars,results)


#prepare lsst noise

bin_centers=[]
for i in range(len(default.LSST_z)):
    bin_centers.append(np.mean(default.LSST_z[i]))
bin_centers=np.array(bin_centers)
A=40/np.sum(default.dn_dz_LSST(bin_centers))

n_i=A*default.dn_dz_LSST(bin_centers)

n=[]
for i in range(len(n_i)):
    n.append(np.ones(3000)*n_i[i])
n=np.array(n)


#lensing noise get N0
clkk=np.loadtxt("./data/clkk.txt")[:4000]
s4noise=np.loadtxt("./data/nlkks4.txt")[:4000]
f_sky=0.4
ells=np.arange(len(s4noise))
lensnoise=(s4noise+clkk)/np.sqrt(f_sky*ells)

clgg,clkg,clkk,clkk_n=default.get_LSST_lensing_test(1,n[0],lensnoise)


pars = np.array( ['ombh2', 'omch2','mctheta',  'As', 'ns', 'tau', 'mnu'])
centers = np.array([0.02222,  0.1197, 0.01040909, 2.196e-9, 0.9655, 0.06, 0.06,-1.0])
left_steps = np.array([0.0008, 0.003, 0.00000001,  0.1e-9, 0.010, 0.02, 0.0,0.3])
right_steps = np.array([0.0008, 0.003, 0.00000001,   0.1e-9, 0.010, 0.02, 0.02,0.3])
Npars = len(pars)


#take the derivatives of spectra over the parameters
comm,rank,my_tasks = mpi.distribute(Npars)
derivative=[]


deltas=[]
for i in range(len(left_steps)):
    deltas.append((left_steps[i],right_steps[i]))
cut=0.1*0.67
for task in my_tasks:
    der=cosmology.LSST_derivative_parameter(defaultCosmology,pars[task],deltas[task],kmax=kmax,ells=3000,cut=cut,use_theta=True)
    derivative.append(der)
derivative=utils.allgatherv(derivative,comm)
np.save(f"./parshift_data/derivatives{kmax}theta{cut}.npy",derivative)
derdict=[]

keys=['g0g0', 'g1g1', 'g2g2', 'g3g3', 'g4g4', 'g5g5', 'g6g6', 'g7g7', 'g8g8', 'g9g9', 'g10g10', 'g11g11', 'g12g12', 'g13g13', 'g14g14', 'g15g15', 'kg0', 'kg1', 'kg2', 'kg3', 'kg4', 'kg5', 'kg6', 'kg7', 'kg8', 'kg9', 'kg10', 'kg11', 'kg12', 'kg13', 'kg14', 'kg15', 'kk']
for i in range(len(derivative)):
    derdict.append(cosmology.arraytodict(keys,derivative[i]))
np.save(f"./parshift_data/derivativesdictall{kmax}limber0.5theta.npy",derdict)
