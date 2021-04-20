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
from pixell import utils # These are needed for MPI. Relevant functions can be copied over.
from camb import model, initialpower




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
ells=np.arange(3000)
for i in range(16):
    defaultCosmology[f'lsst_bias{i}']=1
pars = camb.CAMBparams()
pars.set_dark_energy(w=defaultCosmology['w0'],wa = defaultCosmology['wa'], dark_energy_model = 'ppf')
pars.set_cosmology(H0=defaultCosmology['H0'], cosmomc_theta = None,ombh2=defaultCosmology['ombh2'], 
               omch2=defaultCosmology['omch2'],omk=defaultCosmology['omega_k'], mnu=defaultCosmology['mnu'], tau = defaultCosmology['tau'],
               nnu = defaultCosmology['nnu'], num_massive_neutrinos = 1)
pars.InitPower.set_params(ns=defaultCosmology['ns'],As=defaultCosmology['As'])
pars.NonLinear = model.NonLinear_both
results = camb.get_results(pars)



default=cosmology.cosmology(nz,kmax,zmin,ells,defaultCosmology,pars,results)
fiducial_cls=default.get_lsst_lensing(lenszmin=0)

print(fiducial_cls)
from tqdm import tqdm
pars = ['ombh2', 'omch2', 'H0',  'As', 'ns', 'tau', 'mnu']
for i in range(16):
    pars.append(f'lsst_bias{i}')
Npars = len(pars)


#take the derivatives of spectra over the parameters
Npars=23
comm,rank,my_tasks = mpi.distribute(Npars)
derivative=[]
left_steps = [0.0008, 0.003,   0.2,   0.01e-9, 0.005, 0.02, 0.0]
right_steps = [0.0008, 0.003,   0.2,   0.01e-9, 0.005, 0.02, 0.02]
for i in range(16):
    left_steps.append(0.01)
    right_steps.append(0.01)
deltas=[]
for i in range(len(left_steps)):
    deltas.append((left_steps[i],right_steps[i]))

for task in my_tasks:
    der=cosmology.LSST_derivative_parameter(defaultCosmology,pars[task],deltas[task],kmax=kmax,ells=3000)
    derivative.append(der)
derivative=utils.allgatherv(derivative,comm)
np.save(f"/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/derivatives{kmax}.npy",derivative)
derdict=[]

keys=['g0g0', 'g1g1', 'g2g2', 'g3g3', 'g4g4', 'g5g5', 'g6g6', 'g7g7', 'g8g8', 'g9g9', 'g10g10', 'g11g11', 'g12g12', 'g13g13', 'g14g14', 'g15g15', 'kg0', 'kg1', 'kg2', 'kg3', 'kg4', 'kg5', 'kg6', 'kg7', 'kg8', 'kg9', 'kg10', 'kg11', 'kg12', 'kg13', 'kg14', 'kg15', 'kk']
for i in range(len(derivative)):
    derdict.append(cosmology.arraytodict(keys,derivative[i]))
np.save(f"/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/derivativesdict{kmax}limber0.5.npy",derdict)
"""
derdict=[]

for i in range(len(derivative1)):
    derdict.append(cosmology.arraytodict(derivative1[i]))
np.save("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/derivatives1.npy",derdict)
"""
"""
derdict=[]
for i in range(len(derivative1)):
    derdict.append(cosmology.arraytodict(derivative1[i]))
np.save("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/derivatives1.npy",derdict)
s4noise=np.loadtxt("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/data/nlkks4.txt")
a=cosmology.LSSTxlensing(fiducial_cls[0],fiducial_cls[0],derdict,pars,s4noise,0.4,l_min=0,l_max=2000,nbar=40)

fisher=a.get_fisher()
cov=np.linalg.inv(fisher)
mnu_error=(np.diag(cov)**0.5)[6]*1000
if rank==0:
    print(f"mnu sum error: {mnu_error}meV")
"""