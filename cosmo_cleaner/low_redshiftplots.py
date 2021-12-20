import numpy as np
from orphics import maps, cosmology,io,stats
import matplotlib.pyplot as plt
import argparse
from scipy import optimize
import cosmo_cleaner
from cosmo_cleaner import fisher as fisher,plot
import camb
from cosmo_cleaner import cosmology
from cosmo_cleaner import triangle_plot
from orphics import stats,io,mpi,maps
from camb import model, initialpower
import camb

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
def get_der(spectra,pars,delta,lmin,cleaned=False,lsst=False):
    """derivative function used for fisher"""
    der_spectra_alpha = np.ones((len(list(spectra.items())[0][1]), len(spectra), len(pars)))
    for i in range(len(pars)):
        print(f"Taking field derivatives wrt {pars[i]}")
        der=cosmology.derivative_parameter_CIB(ells,0.1,2,defaultCosmology,pars[i],delta[i],lsst=lsst)
        der_spectra_alpha[:, 0, i] = der[1][lmin:cut] #kg
        der_spectra_alpha[:, 1, i] = der[0][lmin:cut] #gg
        if cleaned==True:
            der_spectra_alpha[:, 2, i] = der[-1][lmin:cut]
        else:
            der_spectra_alpha[:, 2, i] = der[2][lmin:cut]
    return der_spectra_alpha

#nohup srun -n 1 python low_redshiftplots.py --experiment S4 --fname S4lmin100width01 --width 0.1 --corr&
parser = argparse.ArgumentParser()
parser.add_argument('--experiment',type=str,default='ACT',help='experiment name')
parser.add_argument('--fname',type=str,default=None,help='file name')
parser.add_argument("--corr", action='store_true',help='Use conservative correlation')
parser.add_argument("--lsst", action='store_true',help='Use gg lsst 1st bin')
parser.add_argument("--width",type=float, default=0.5,help="galaxy redshift width")

args = parser.parse_args()

if args.experiment=='ACT':
    act=cosmology.CMB_Lensing(lens_beam = 1.4,lens_noiseT = 15.,lens_noiseP = 15.3,lens_tellmin = 50,lens_tellmax = 3000,lens_pellmin = 50, lens_pellmax = 3000,lens_kmin = 30,lens_kmax = 3000, lens_f_sky=0.3, estimators=('TT','TE','EE','EB','TB','MV'), NlTT=None, NlEE=None, NlBB=None)
    n0=act.noise_rec
    lensnoise=n0[:3000]

elif args.experiment=='SO':
    so=cosmology.CMB_Lensing(lens_beam = 1.4,lens_noiseT = 8.,lens_noiseP = 11.3,lens_tellmin = 50,lens_tellmax = 3000,lens_pellmin = 50, lens_pellmax = 4000,lens_kmin = 30,lens_kmax = 3000, lens_f_sky=0.3, estimators=('TT','TE','EE','EB','TB','MV'), NlTT=None, NlEE=None, NlBB=None)
    n0=so.noise_rec
    lensnoise=n0[:3000]

elif args.experiment=='S4':
    s4=cosmology.CMB_Lensing(lens_beam = 1.4,lens_noiseT = 1.,lens_noiseP = 1.4,lens_tellmin = 50,lens_tellmax = 3000,lens_pellmin = 50, lens_pellmax = 5000,lens_kmin = 30,lens_kmax = 3000, lens_f_sky=0.4, estimators=('TT','TE','EE','EB','TB','MV'), NlTT=None, NlEE=None, NlBB=None)
    n0=s4.noise_rec
    lensnoise=n0[:3000]
np.savetxt(f"/global/u1/j/jia_qu/cosmo_cleaner/cosmo_cleaner/low_redshift_data/{args.fname}lensnoise.txt",lensnoise)
print("saved noise")
noise=np.zeros(3000)+(1/1.06)*(np.pi/60./180.)**2

defaultCosmology = {'omch2': 0.1198
                    ,'ombh2': 0.02225
                    ,'H0': 67.3
                    ,'ns': 0.9645
                    ,'As': 2.2e-9
                    ,'mnu': 0.06
                    ,'w0': -1.0
                    ,'tau':0.06
                    ,'nnu':3.046
                    ,'wa': 0.,'bias':2,'A_e':1.2,'A_l':1.2,'omega_k':0}
for i in range(17):
    defaultCosmology[f'lsst_bias{i}']=1

nz = 1000
kmax = 10
zmin = 0.
ells=np.arange(2000)

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

clgg,clkg,clkk,clcibcib,clcibk,cl_clean=default.get_cibspectra(0.1,args.width,lsst=args.lsst)

if args.corr:
    correlation=np.loadtxt("/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/low_redshift_data/cibcorr.txt")
    cl_clean=clkk*(1-correlation**2)
parameters=['bias','A_e','A_l']
delta=[(0.01,0.01),(0.01,0.01),(0.01,0.01)]

base_error=[]
fishermatrix=[]

lmin=100

for i in range(202,1000,100):
    print(i)
    cut=i
    spectra = {'kg': clkg[lmin:cut], 'gg' :clgg[lmin:cut], 'kk': clkk[lmin:cut]}
    spectra_noise = {'kg': clkg[lmin:cut], 'gg' :clgg[lmin:cut]+noise[lmin:cut], 'kk': clkk[lmin:cut]+lensnoise[lmin:cut]}
    der_spectra_alpha=get_der(spectra,parameters,delta,lmin,cleaned=False)
    a=fisher.Fisher(1,len(parameters),np.arange(lmin,cut),spectra_noise,der_spectra_alpha)
    base_error.append(a.get_fisher()[0])
    fishermatrix.append(a.fisher)

clean_error=[]
fishermatrix_clean=[]
for i in range(202,1000,100):
    print(i)
    cut=i
    spectra = {'kg': clkg[lmin:cut], 'gg' :clgg[lmin:cut], 'kk': cl_clean[lmin:cut]}
    spectra_noise = {'kg': clkg[lmin:cut], 'gg' :clgg[lmin:cut]+noise[lmin:cut], 'kk': cl_clean[lmin:cut]+lensnoise[lmin:cut]}
    der_spectra_alpha=get_der(spectra,parameters,delta,lmin,cleaned=True,lsst=args.lsst)
    a=fisher.Fisher(1,len(parameters),np.arange(lmin,cut),spectra_noise,der_spectra_alpha)
    clean_error.append(a.get_fisher()[0])
    fishermatrix_clean.append(a.fisher)

covlist=[]
covclean=[]
base_error=[]
cleaned_error=[]

for i in range(len(fishermatrix_clean)):
    cov=np.linalg.inv(fishermatrix[i])
    cov_clean=np.linalg.inv(fishermatrix_clean[i])
    covlist.append(cov)
    covclean.append(cov_clean)
    error_non_marginalizedbase = np.diag(fishermatrix[i])**-0.5 
    error_non_marginalized = np.diag(fishermatrix_clean[i])**-0.5 
    base_error.append(error_non_marginalizedbase[0])
    cleaned_error.append(error_non_marginalized[0])
cleaned_error=np.array(cleaned_error)
base_error=np.array(base_error)

if args.corr and args.lsst:
    np.save(f"/global/u1/j/jia_qu/cosmo_cleaner/cosmo_cleaner/low_redshift_data/{args.fname}cov_corr_lsst.npy",covlist)
    np.save(f"/global/u1/j/jia_qu/cosmo_cleaner/cosmo_cleaner/low_redshift_data/{args.fname}covclean_corr_lsst.npy",covclean)
elif args.lsst:
    np.save(f"/global/u1/j/jia_qu/cosmo_cleaner/cosmo_cleaner/low_redshift_data/{args.fname}cov_lsst.npy",covlist)
    np.save(f"/global/u1/j/jia_qu/cosmo_cleaner/cosmo_cleaner/low_redshift_data/{args.fname}covclean_lsst.npy",covclean)
elif args.corr:
    np.save(f"/global/u1/j/jia_qu/cosmo_cleaner/cosmo_cleaner/low_redshift_data/{args.fname}cov_corr.npy",covlist)
    np.save(f"/global/u1/j/jia_qu/cosmo_cleaner/cosmo_cleaner/low_redshift_data/{args.fname}covclean_corr.npy",covclean)
else:
    np.save(f"/global/u1/j/jia_qu/cosmo_cleaner/cosmo_cleaner/low_redshift_data/{args.fname}cov.npy",covlist)
    np.save(f"/global/u1/j/jia_qu/cosmo_cleaner/cosmo_cleaner/low_redshift_data/{args.fname}covclean.npy",covclean)   

"""
parameters=['bias','A_e','A_l']
fiducial=[2., 1.2,1.2]
fig, axes = triangle_plot.plot_triangle_base(parameters,fiducial,covlist[0] , one_sigma_only=True,color_1d='red',ellipse_kwargs1={'ls': '-', 'edgecolor': 'red'})
# now plot a slightly cleaned instrument
triangle_plot.plot_triangle_base(parameters,fiducial, covclean[0], one_sigma_only=True,
                                  f=fig, ax=axes,
                                  ellipse_kwargs1={'ls': '-', 'edgecolor': 'blue'},
                                  ellipse_kwargs2={'ls': '-', 'edgecolor': 'blue'},
                                  color_1d='blue'
                                 )

l1, = axes[0, -1].plot([],[],'-',color="red", label='No cleaning')
l2, = axes[0, -1].plot([],[],'-',color="blue", label='Cleaning')
axes[0, -1].legend()
plt.savefig(f"/global/homes/j/jia_qu/cosmo_cleaner/cosmo_cleaner/plots/{args.fname}constraints.png")
"""


