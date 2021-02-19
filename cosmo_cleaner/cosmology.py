import camb
import numpy as np

#finite difference
#insert the parameter you would like to vary
def get_cosmology_var(defaultCosmology,par,delta=0.005):
    """
    default cosmology: dict containing default cosmology values
    par: parameter you are finding derivative of, keeping others constant
    delta: value you change your default cosmological parameter by
    """
    delta=delta*defaultCosmology[par]
    defaultCosmology_h=defaultCosmology.copy()
    defaultCosmology_l=defaultCosmology.copy()
    defaultCosmology_h[par]+=delta
    defaultCosmology_l[par]-=delta
    Cosmology=[defaultCosmology_h,defaultCosmology_l]
    parameters=[]
    cambres=[]
    for i in range(2):
        pars = camb.CAMBparams()
        pars.set_dark_energy(w=Cosmology[i]['w0'],wa = Cosmology[i]['wa'], dark_energy_model = 'ppf')
        pars.set_cosmology(H0=Cosmology[i]['H0'], cosmomc_theta = None,ombh2=Cosmology[i]['ombh2'], 
                       omch2=Cosmology[i]['omch2'], mnu=Cosmology[i]['mnu'], tau = Cosmology[i]['tau'],
                       nnu = Cosmology[i]['nnu'], num_massive_neutrinos = 3)
        #pars.NonLinear = model.NonLinear_both
        pars.InitPower.set_params(ns=Cosmology[i]['ns'],As=Cosmology[i]['As'])

        results = camb.get_results(pars)
        cambres.append(results)
        parameters.append(pars)
    return (Cosmology,parameters,cambres)


class cosmology:

    def __init__(self,nz,kmax,zmin,ells,cosmology,pars,cambres):
        self.nz=nz
        self.kmax=kmax
        self.zmin=zmin
        self.ells=ells
        self.cosmology=cosmology
        self.pars=pars
        self.results=cambres
        
        self.chistar=self.results.conformal_time(0)- self.results.tau_maxvis
        self.chis=np.linspace(0,self.chistar,self.nz)
        self.zs=self.results.redshift_at_comoving_radial_distance(self.chis)
        self.dchis=(self.chis[2:]-self.chis[:-2])/2
        self.chis=self.chis[1:-1]
        self.zs = self.zs[1:-1]
        self.Hzs = np.array([self.results.hubble_parameter(z) for z in self.zs])
        self.pars.Transfer.accurate_massive_neutrinos = True  
        self.bias=cosmology['bias']  
        self.Alens=cosmology['A_l']
        
        self.PK = camb.get_matter_power_interpolator(self.pars, nonlinear=True, 
hubble_units=False, k_hunit=False, kmax=self.kmax,
var1='delta_nonu',var2='delta_nonu', zmax=self.zs[-1])
        
        

    def dndz_gauss(self,z,z0,sigma):
        ans = 1/np.sqrt(2*np.pi*sigma**2)* np.exp((-(z-z0)**2)/ (2.*sigma**2.))
        return self.bias*self.Alens*ans 
    
    def get_lensing_window(self):
        cSpeedKmPerSec = 299792.458
        lensingwindow = 1.5*(self.cosmology['omch2']+self.cosmology['ombh2']+self.pars.omnuh2)*100.*100.*(1.+self.zs)*self.chis*((self.chistar - self.chis)/self.chistar)/self.Hzs/cSpeedKmPerSec
        return self.Alens*lensingwindow

    def get_clgg(self,mean_z,width,lmax=2000):
        cSpeedKmPerSec = 299792.458
        galaxywindow=self.dndz_gauss(self.zs,mean_z,width)
        cl_autog=[]
        ellsgg=np.arange(lmax)
        wg = np.ones(self.chis.shape)

        precalcFactor= self.Hzs**2./self.chis/self.chis/cSpeedKmPerSec**2.
        for l in ellsgg:
            k=(l+0.5)/self.chis
            wg[:]=1
            wg[k<1e-4]=0
            wg[k>=self.kmax]=0
            pkin = self.PK.P(self.zs, k, grid=False)
            common = ((wg*pkin)*precalcFactor)[self.zs>=self.zmin]        
            estCl = np.dot(self.dchis[self.zs>=self.zmin], common*(galaxywindow*galaxywindow)[self.zs>=self.zmin])
            cl_autog.append(estCl)
        return np.array(cl_autog)
    
    def get_clkk(self,lmax=2000):
        cSpeedKmPerSec = 299792.458
        ells=np.arange(lmax)
        wk = np.ones(self.chis.shape)
        precalcFactor= self.Hzs**2./self.chis/self.chis/cSpeedKmPerSec**2.
        lensingwindow=self.get_lensing_window()

        clkk=[]
        for l in ells:
            k=(l+0.5)/self.chis
            wk[:]=1
            wk[k<1e-4]=0
            wk[k>=self.kmax]=0
            pkin = self.PK.P(self.zs, k, grid=False)
            common = ((wk*pkin)*precalcFactor)[self.zs>=self.zmin]        
            estCl = np.dot(self.dchis[self.zs>=self.zmin], common*(lensingwindow*lensingwindow)[self.zs>=self.zmin])
            clkk.append(estCl)
        return np.array(clkk)

    def get_clkg(self,mean_z,width,lmax=2000):
        cSpeedKmPerSec = 299792.458
        lensingwindow=self.get_lensing_window()
        galaxywindow=self.dndz_gauss(self.zs,mean_z,width)
        precalcFactor= self.Hzs**2./self.chis/self.chis/cSpeedKmPerSec**2.
        ellsgg=np.arange(lmax)
        cl_cross=[]
        w = np.ones(self.chis.shape)
        for l in ellsgg:
            k=(l+0.5)/self.chis
            w[:]=1
            w[k<1e-4]=0
            w[k>=self.kmax]=0
            pkin = self.PK.P(self.zs, k, grid=False)
            common = ((w*pkin)*precalcFactor)[self.zs>=self.zmin]        
            estCl = np.dot(self.dchis[self.zs>=self.zmin], common*(lensingwindow*galaxywindow)[self.zs>=self.zmin])
            cl_cross.append(estCl)
        return np.array(cl_cross)

    def get_spectra(self,mean_z,width):
        self.spectra_dict={'clgg':self.get_clgg(mean_z,width),
        'clkk':self.get_clkk(),'clkg':self.get_clkg(mean_z,width) }
        self.spectra=[self.get_clgg(mean_z,width),self.get_clkg(mean_z,width),self.get_clkk()]
        return np.array(self.spectra)



def derivative_parameter(ells,mean_z,width,defaultCosmology,parameter,delta=0.005,nz=1000,kmax=10,zmin=0):
    """

    """
    cosmology_pars,pars,res=get_cosmology_var(defaultCosmology,parameter,delta=delta)
    high=cosmology(nz,kmax,zmin,ells,cosmology_pars[0],pars[0],res[0])
    low=cosmology(nz,kmax,zmin,ells,cosmology_pars[1],pars[1],res[1])

    derivative=(high.get_spectra(mean_z,width)-low.get_spectra(mean_z,width))/(2*delta*defaultCosmology[parameter])
    return derivative
        


