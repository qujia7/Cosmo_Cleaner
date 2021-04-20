import camb
import numpy as np
import itertools
from astropy import units as u
from astropy.cosmology import Planck15
from camb import model, initialpower
from scipy.special import erf

#finite difference
#insert the parameter you would like to vary
def get_cosmology_var(defaultCosmology,par,delta):
    """
    default cosmology: dict containing default cosmology values
    par: parameter you are finding derivative of, keeping others constant
    delta: value you change your default cosmological parameter by
    """
    left,right=delta
    defaultCosmology_h=defaultCosmology.copy()
    defaultCosmology_l=defaultCosmology.copy()
    defaultCosmology_h[par]+=right
    defaultCosmology_l[par]-=left
    Cosmology_l=[defaultCosmology_h,defaultCosmology_l]
    parameters=[]
    cambres=[]
    for i in range(2):
        pars = camb.CAMBparams()
        pars.set_dark_energy(w=Cosmology_l[i]['w0'],wa = Cosmology_l[i]['wa'], dark_energy_model = 'ppf')
        pars.set_cosmology(H0=Cosmology_l[i]['H0'], cosmomc_theta = None,ombh2=Cosmology_l[i]['ombh2'], 
                       omch2=Cosmology_l[i]['omch2'],omk=Cosmology_l[i]['omega_k'], mnu=Cosmology_l[i]['mnu'], tau = Cosmology_l[i]['tau'],
                       nnu = Cosmology_l[i]['nnu'], num_massive_neutrinos = 1)
        pars.NonLinear = model.NonLinear_both
        pars.InitPower.set_params(ns=Cosmology_l[i]['ns'],As=Cosmology_l[i]['As'])

        results = camb.get_results(pars)
        cambres.append(results)
        parameters.append(pars)
    return (Cosmology_l,parameters,cambres)


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
        self.A_e=cosmology['A_e']  
        self.Alens=cosmology['A_l']
        
        self.PK = camb.get_matter_power_interpolator(self.pars, nonlinear=True, 
hubble_units=False, k_hunit=False, kmax=self.kmax, zmax=self.zs[-1])

        #LSST specification

        self.LSST_bins=np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.3, 2.6, 3, 3.5, 4, 7])
        self.lsst_bias=np.array([cosmology[f'lsst_bias{i}'] for i in range(len(self.LSST_bins)-1)])
        start=self.LSST_bins[:-1]
        end= self.LSST_bins[1:] 
        self.LSST_z=[]
        for i in range(len(start)):
            self.LSST_z.append(np.arange(start[i],end[i],0.001)) 

        

    def get_lensing_window(self,zmin=0):
        cSpeedKmPerSec = 299792.458
        lensingwindow = 1.5*(self.cosmology['omch2']+self.cosmology['ombh2']+self.pars.omnuh2)*100.*100.*(1.+self.zs)*self.chis*((self.chistar - self.chis)/self.chistar)/self.Hzs/cSpeedKmPerSec
        lensingwindow[self.zs<zmin]=0.
        lensingwindow[self.zs<0.5]*=self.A_e
        lensingwindow[self.zs>=0.5]*=self.Alens

        return lensingwindow
        

    def dn_dz_LSST(self,z,z0=0.3):
        return (1/(2*z0))*(z/z0)**2*np.exp(-z/z0)

    def lsst_window(self,i,B=1):
        zrange=self.zs
        sigma_z=0.05*(1+zrange)
        bias=B*(1+zrange)
        err_0=erf((zrange-self.LSST_bins[i])/(np.sqrt(2)*sigma_z))
        err_1=erf((zrange-self.LSST_bins[i+1])/(np.sqrt(2)*sigma_z))
        dn_idz=40*self.dn_dz_LSST(zrange)*0.5*(err_0-err_1)
        window=bias*dn_idz/np.trapz(dn_idz,zrange)
        return window

    def get_lsst_kappa(self,i,lmax=2000,zmin=0):
        #return lsstxlensing with the ith bin
        cSpeedKmPerSec = 299792.458
        lensingwindow=self.get_lensing_window(zmin=zmin)
        galaxywindow=self.lsst_window(i,self.lsst_bias[i])

        precalcFactor= self.Hzs**2./self.chis/self.chis/cSpeedKmPerSec**2.
        lbmax=self.kmax*self.results.comoving_radial_distance(np.mean(self.LSST_z[i]))
        ellsgg=np.arange(lmax)
        cl_cross=[]
        w = np.ones(self.chis.shape)
        for l in ellsgg:
            #f l<=lbmax:
            k=(l+0.5)/self.chis
            w[:]=1
            w[k<1e-4]=0
            w[k>=self.kmax]=0
            pkin = self.PK.P(self.zs, k, grid=False)
            common = ((w*pkin)*precalcFactor)[self.zs>=self.zmin]        
            estCl = np.dot(self.dchis[self.zs>=self.zmin], common*(lensingwindow*galaxywindow)[self.zs>=self.zmin])
            cl_cross.append(estCl)
            #else:
                #cl_cross.append(0)
        return np.array(cl_cross)

    def get_lsst_auto(self,i,j=None,lmax=2000):
        #int i: ith bin
        #int j:jth bin
        cSpeedKmPerSec = 299792.458
        galaxywindow=self.lsst_window(i,self.lsst_bias[i])
        if j is not None:
            galaxywindow2=self.lsst_window(j,self.lsst_bias[j])
        else:
            galaxywindow2=galaxywindow
            j=i

        cl_autog=[]
        ellsgg=np.arange(lmax)
        wg = np.ones(self.chis.shape)

        precalcFactor= self.Hzs**2./self.chis/self.chis/cSpeedKmPerSec**2.
        for l in ellsgg:
            #if l<=lbmax:
            k=(l+0.5)/self.chis
            wg[:]=1
            wg[k<1e-4]=0
            wg[k>=self.kmax]=0 
            pkin = self.PK.P(self.zs, k, grid=False)
            common = ((wg*pkin)*precalcFactor)[self.zs>=self.zmin]        
            estCl = np.dot(self.dchis[self.zs>=self.zmin], common*(galaxywindow*galaxywindow2)[self.zs>=self.zmin])
            cl_autog.append(estCl)
            #else:
                #cl_autog.append(0.)
        return np.array(cl_autog)

    def dndz_gauss(self,z,z0,sigma):
        ans = 1/np.sqrt(2*np.pi*sigma**2)* np.exp((-(z-z0)**2)/ (2.*sigma**2.))
        ans[self.zs<0.5]*=self.A_e
        ans[self.zs>=0.5]*=self.Alens
        return self.bias*ans

    def f_nu(self,f,beta):
        T=34
        k=1.38064852e-23
        h=6.62607015e-34
        fv=f.copy()
        fsmall=fv[fv<4955*10**9]
        a=(np.exp(h*fsmall/(k*T))-1)**(-1)*fsmall**(beta+3)
        fbig=fv[fv>=4955*10**9]
        b=((np.exp(h*fbig/(k*T))-1)**(-1))*fbig**(beta+3)*(fbig/(4955*10**9))**(-2)
        result=np.concatenate((a,b))
        return result

    def cib_window(self,nu,zs,b=3.6e-62,noise=None):
        z_c=2
        sigma_z=2
        beta=2
        nu=nu*10**9
        eta_0=1.5e18
        freq = nu * u.Hz
        equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
        conv=(1. * u.Jy/ u.sr).to(u.uK, equivalencies=equiv)  #convert from J/sr to uk
        cib_w=b*(self.chis**2/(1.+self.zs)**2)*np.exp(-(self.zs-z_c)**2/(2*sigma_z**2))*self.f_nu(nu*(1+self.zs),beta)*conv.value
        if noise is not None:
            cib_w+=noise
        cib_w[self.zs<0.5]*=self.A_e
        cib_w[self.zs>=0.5]*=self.Alens
        return cib_w

    def get_clcib(self,cib_window,lmax=3000):
        cSpeedKmPerSec = 299792.458

        cl_cib=[]
        cib_window=cib_window
        ells=np.arange(lmax)
        w = np.ones(self.chis.shape)
        precalcFactor= self.Hzs**2./self.chis/self.chis/cSpeedKmPerSec**2.

        for l in ells:
            k=(l)/self.chis
            w[:]=1
            w[k<1e-4]=0
            w[k>=self.kmax]=0
            pkin = self.PK.P(self.zs, k, grid=False)
            common = ((w*pkin)*precalcFactor)[self.zs>=self.zmin]        
            estCl = np.dot(self.dchis[self.zs>=self.zmin], common*(cib_window*cib_window)[self.zs>=self.zmin])
            cl_cib.append(estCl) 
        return np.array(cl_cib)

    def get_clcibg(self,galaxywindow,cib_window,lmax=3000):
        cSpeedKmPerSec = 299792.458

        galaxywindow=galaxywindow
        cl_cibg=[]
        ells=np.arange(lmax)
        wcg = np.ones(self.chis.shape)
        precalcFactor= self.Hzs**2./self.chis/self.chis/cSpeedKmPerSec**2.

        for l in ells:
            k=(l)/self.chis
            wcg[:]=1
            wcg[k<1e-4]=0
            wcg[k>=self.kmax]=0
            pkin = self.PK.P(self.zs, k, grid=False)
            common = ((wcg*pkin)*precalcFactor)[self.zs>=self.zmin]        
            estCl = np.dot(self.dchis[self.zs>=self.zmin], common*(cib_window*galaxywindow)[self.zs>=self.zmin])
            cl_cibg.append(estCl) 
        return np.array(cl_cibg)

    def get_clgg(self,galaxy_window,lmax=3000):
        cSpeedKmPerSec = 299792.458
        #galaxywindow=self.dndz_gauss(self.zs,mean_z,width)
        galaxywindow=galaxy_window
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
    
    def get_clkk(self,lmax=2000,lens_kmax=1000,zmin=0):
        cSpeedKmPerSec = 299792.458
        ells=np.arange(lmax)
        wk = np.ones(self.chis.shape)
        precalcFactor= self.Hzs**2./self.chis/self.chis/cSpeedKmPerSec**2.
        lensingwindow=self.get_lensing_window(zmin=zmin)

        clkk=[]
        for l in ells:
            k=(l+0.5)/self.chis
            wk[:]=1
            wk[k<1e-4]=0
            wk[k>=lens_kmax]=0
            pkin = self.PK.P(self.zs, k, grid=False)
            common = ((wk*pkin)*precalcFactor)[self.zs>=self.zmin]        
            estCl = np.dot(self.dchis[self.zs>=self.zmin], common*(lensingwindow*lensingwindow)[self.zs>=self.zmin])
            clkk.append(estCl)
        return np.array(clkk)

    def get_clkg(self,galaxy_window,lmax=3000):
        cSpeedKmPerSec = 299792.458
        lensingwindow=self.get_lensing_window()
        galaxywindow=galaxy_window
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

    def get_clcibk(self,cib_window,lmax=3000):
        cSpeedKmPerSec = 299792.458

        cl_cibk=[]
        wkc = np.ones(self.chis.shape)
        lensingwindow=self.get_lensing_window()
        ells=np.arange(lmax)
        precalcFactor= self.Hzs**2./self.chis/self.chis/cSpeedKmPerSec**2.

        for l in ells:
            k=(l)/self.chis
            wkc[:]=1
            wkc[k<1e-4]=0
            wkc[k>=self.kmax]=0
            pkin = self.PK.P(self.zs, k, grid=False)
            common = ((wkc*pkin)*precalcFactor)[self.zs>=self.zmin]      
            estCl = np.dot(self.dchis[self.zs>=self.zmin], common*(lensingwindow*cib_window)[self.zs>=self.zmin])
            cl_cibk.append(estCl)
        return np.array(cl_cibk)

    def get_cibspectra(self,mean_z,width,cib_noise=0.,galaxy_noise=0.):
        #prepare normal and cleaned fields for the cib
        cib_window=self.cib_window(545,self.zs,noise=cib_noise)
        galaxywindow=self.dndz_gauss(self.zs,mean_z,width)
        clgg=self.get_clgg(galaxywindow)
        #clcib=self.get_clcib(self.cib_window(545,self.zs,noise=0))
        clcib=self.get_clcib(cib_window)
        clkk=self.get_clkk()
        clkg=self.get_clkg(galaxywindow)
        clcibk=self.get_clcibk(cib_window)
        self.windows=[galaxywindow,self.get_lensing_window(),cib_window]
        #the clcibk should contain the noise
        self.spectra=[clgg,clkg,clkk,clcib,clcibk,clkk-(clcibk**2/clcib)]
        return np.array(self.spectra)
    
    def get_lsst_lensing(self,lenszmin=0,lmax=2000):
        fields={}
        field_list=[]
        for i in range(len(self.LSST_bins)-1):
            fields[f'g{i}g{i}']=self.get_lsst_auto(i,lmax=lmax)
            field_list.append(fields[f'g{i}g{i}'])
        for i in range(len(self.LSST_bins)-1):
            fields[f'kg{i}']=self.get_lsst_kappa(i,zmin=lenszmin,lmax=lmax)
            field_list.append(fields[f'kg{i}'])

        #include cross correlation between neighboring and next to neighboring bins
        """
        for i in range(len(self.LSST_bins)-3):
            fields[f'g{i}g{i+1}']=self.get_lsst_auto(i,i+1,lmax=lmax)
            fields[f'g{i}g{i+2}']=self.get_lsst_auto(i,i+2,lmax=lmax)
            field_list.append(fields[f'g{i}g{i+1}'])
            field_list.append(fields[f'g{i}g{i+2}'])
        fields[f'g{14}g{15}']=self.get_lsst_auto(14,15,lmax=lmax)
        field_list.append(fields[f'g{14}g{15}'])
        """


        fields['kk']=self.get_clkk(zmin=lenszmin,lmax=lmax)
        field_list.append(fields['kk'])
        #lets artificially set the low z part of kappa to zero
        field_list=np.array(field_list)
        return fields,field_list
     
    
    #for this problem we need three fields, g1, cmb and the external tracer
    def get_fields(self,idealised=False):
        if idealised==True:
            g1=self.bias*self.get_lensing_window()
            g1[self.zs>0.3]=0
            g2=self.get_lensing_window()
            g3=self.get_lensing_window()
            g3[self.zs<0.3]=0
        else:
            #g1=self.dndz_gauss(self.zs,0.1,0.25)
            g1=self.bias*self.get_lensing_window()
            g1[self.zs>0.3]=0
            g2=self.get_lensing_window()
            g3=self.dndz_gauss(self.zs,2,10)
            g3[self.zs<0.3]=0
        

        self.windows=[g1,g2,g3]
        self.fields=[self.get_clgg(g1),self.get_clkg(g1),self.get_clkk(),self.get_clgg(g3),self.get_clkg(g3),self.get_clkk()-(self.get_clkg(g3)**2/self.get_clgg(g3))]
        return np.array(self.fields)




def LSST_derivative_parameter(defaultCosmology,parameter,delta,lenszmin=0,nz=1000,kmax=10,zmin=0,ells=2000):
    """
    take derivative with respect to a given parameter.
    """
    left,right=delta
    cosmology_pars,pars,res=get_cosmology_var(defaultCosmology,parameter,delta)
    high=cosmology(nz,kmax,zmin,ells,cosmology_pars[0],pars[0],res[0])
    low=cosmology(nz,kmax,zmin,ells,cosmology_pars[1],pars[1],res[1])
    derivative=(high.get_lsst_lensing(lenszmin,lmax=ells)[1]-low.get_lsst_lensing(lenszmin,lmax=ells)[1])/((right+left))
    #put the above in a dictionary
    #return dicti
    return derivative

def arraytodict(keys,array):
    dicti={}
    for i in range(len(array)):
        dicti[keys[i]]=array[i]
     
    return dicti


def derivative_parameter(ells,mean_z,width,defaultCosmology,parameter,delta,nz=1000,kmax=10,zmin=0,idealised=False):
    """
    take derivative with respect to a given parameter.
    """
    left,right=delta
    cosmology_pars,pars,res=get_cosmology_var(defaultCosmology,parameter,delta)
    high=cosmology(nz,kmax,zmin,ells,cosmology_pars[0],pars[0],res[0])
    low=cosmology(nz,kmax,zmin,ells,cosmology_pars[1],pars[1],res[1])

    #derivative=(high.get_spectra(mean_z,width)-low.get_spectra(mean_z,width))/(2*delta*defaultCosmology[parameter])
    derivative=(high.get_fields(idealised=idealised)-low.get_fields(idealised=idealised))/((right+left))
    return derivative

def derivative_parameter_CIB(ells,mean_z,width,defaultCosmology,parameter,delta=0.005,nz=1000,kmax=10,zmin=0,idealised=False):
    """
    take derivative with respect to a given parameter.
    """
    cosmology_pars,pars,res=get_cosmology_var(defaultCosmology,parameter,delta=delta)
    high=cosmology(nz,kmax,zmin,ells,cosmology_pars[0],pars[0],res[0])
    low=cosmology(nz,kmax,zmin,ells,cosmology_pars[1],pars[1],res[1])

    #derivative=(high.get_spectra(mean_z,width)-low.get_spectra(mean_z,width))/(2*delta*defaultCosmology[parameter])
    derivative=(high.get_cibspectra(mean_z,width)-low.get_cibspectra(mean_z,width))/(2*delta*defaultCosmology[parameter])
    return derivative

def get_der(defaultCosmology,cut,spectra,pars,cleaned=False):
    """derivative function used for fisher"""
    ells=np.arange(len(list(spectra.items())[0][1]))
    der_spectra_alpha = np.ones((len(list(spectra.items())[0][1]), len(spectra), len(pars)))
    for i in range(len(pars)):
        print(f"Taking field derivatives wrt {pars[i]}")
        der=derivative_parameter(ells,1,0.5,defaultCosmology,pars[i],delta=0.00005)
        der_spectra_alpha[:, 0, i] = der[1][:cut]
        der_spectra_alpha[:, 1, i] = der[0][:cut]
        if cleaned==True:
            der_spectra_alpha[:, 2, i] = der[-1][:cut]
        else:
            der_spectra_alpha[:, 2, i] = der[2][:cut]
    return der_spectra_alpha


class CMB_Primary():
    """
    Modified from fish_chips repo
    Class for computing Fisher matrices from the CMB primary (TT/TE/EE).
    This experiment class requires some instrument parameters, and computes
    white noise for each multipole. The computation of the Fisher matrix
    follows equation 4 of arxiv:1402.4108.
    """

    def __init__(self, theta_fwhm=(10., 7., 5.),
                 sigma_T=(68.1, 42.6, 65.4),
                 sigma_P=(109.4, 81.3, 133.6),
                 f_sky=0.65, l_min=2, l_max=2500,
                 verbose=False):
        """
        Initialize the experiment with noise parameters.
        Uses the Planck bluebook parameters by default.
        Parameters
        ----------
            theta_fwhm (list of float): beam resolution in arcmin
            sigma_T (list of float): temperature resolution in muK
            sigma_P (list of float): polarization resolution in muK
            f_sky (float): sky fraction covered
            l_min (int): minimum ell for CMB power spectrum
            l_max (int): maximum ell for CMB power spectrum
        """

        # convert from arcmin to radians
        self.theta_fwhm = theta_fwhm * np.array([np.pi/60./180.])
        self.sigma_T = sigma_T * np.array([np.pi/60./180.])
        self.sigma_P = sigma_P * np.array([np.pi/60./180.])
        self.num_channels = len(theta_fwhm)
        self.f_sky = f_sky
        self.ells = np.arange(l_max+1)
        self.l_min = l_min
        self.l_max = l_max
        self.noise_T = np.zeros(self.l_max+1, 'float64')
        self.noise_P = np.zeros(self.l_max+1, 'float64')
        self.noise_TE = np.zeros(self.l_max+1, 'float64')

        for l in range(self.l_min, self.l_max+1):
            
            self.noise_T[l] = 0
            self.noise_P[l] = 0
            #inverse noise add the different channels
            for channel in range(self.num_channels):
                self.noise_T[l] += self.sigma_T[channel]**-2 *\
                    np.exp(
                        -l*(l+1)*self.theta_fwhm[channel]**2/8./np.log(2.))
                self.noise_P[l] += self.sigma_P[channel]**-2 *\
                    np.exp(
                        -l*(l+1)*self.theta_fwhm[channel]**2/8./np.log(2.))
            self.noise_T[l] = 1/self.noise_T[l]
            self.noise_P[l] = 1/self.noise_P[l]
            
        self.noise_T[self.ells < self.l_min] = 1e100
        self.noise_P[self.ells < self.l_min] = 1e100
        self.noise_T[self.ells > self.l_max] = 1e100
        self.noise_P[self.ells > self.l_max] = 1e100

    def compute_fisher_from_camb(self,cmbresults,defaultCosmology,parameters,delta):
        """
           cmbresults: camb object containing initial cosmology information
           defaultCosmology: dictionary containing original cosmology parameters and values
           parameters: list of strings containing name of experiments
           delta: list of tuples 
        """
        der_dict={}    
        fiducial={}
        results=cmbresults
        fid=results.get_cmb_power_spectra(lmax=self.l_max,spectra=['lensed_scalar'],CMB_unit='muK',raw_cl=True)['lensed_scalar'].transpose()
        fiducial['tt']=fid[0]
        fiducial['ee']=fid[1]
        fiducial['te']=fid[3]

        #create the cosmologies to take derivatives with
        for i in range(len(parameters)):
            left,right=delta[i]
            cosmology_pars,pars,res=get_cosmology_var(defaultCosmology,parameters[i],delta[i])
            cls_high=res[0].get_cmb_power_spectra(lmax=self.l_max,spectra=['lensed_scalar'],CMB_unit='muK',raw_cl=True)['lensed_scalar'].transpose()
            cls_low=res[1].get_cmb_power_spectra(lmax=self.l_max,spectra=['lensed_scalar'],CMB_unit='muK',raw_cl=True)['lensed_scalar'].transpose()

            der_dict[parameters[i]+'_tt']=(cls_high[0]-cls_low[0])/((right+left))
            der_dict[parameters[i]+'_ee']=(cls_high[1]-cls_low[1])/((right+left))
            der_dict[parameters[i]+'_te']=(cls_high[3]-cls_low[3])/((right+left))
        df=der_dict
        npar = len(parameters)
        self.fisher = np.zeros((npar, npar))

        self.fisher_ell = np.zeros(self.l_max)

        for i, j in itertools.combinations_with_replacement(range(npar), r=2):
            # following eq 4 of https://arxiv.org/pdf/1402.4108.pdf
            fisher_ij = 0.0
            # probably a more efficient way to do this exists
            for l in range(self.l_min, self.l_max):

                Cl = np.array([[fiducial['tt'][l] + self.noise_T[l], fiducial['te'][l] + self.noise_TE[l]],
                               [fiducial['te'][l] + self.noise_TE[l], fiducial['ee'][l] + self.noise_P[l]]])
                invCl = np.linalg.inv(Cl)

                dCl_i = np.array([[df[parameters[i]+'_tt'][l], df[parameters[i]+'_te'][l]],
                                  [df[parameters[i]+'_te'][l], df[parameters[i]+'_ee'][l]]])
                dCl_j = np.array([[df[parameters[j]+'_tt'][l], df[parameters[j]+'_te'][l]],
                                  [df[parameters[j]+'_te'][l], df[parameters[j]+'_ee'][l]]])

                inner_term = np.dot(np.dot(invCl, dCl_i), np.dot(invCl, dCl_j))
                fisher_contrib = (2*l+1)/2. * self.f_sky * np.trace(inner_term)
                fisher_ij += fisher_contrib

            # fisher is diagonal, so we get half of the matrix for free
            self.fisher[i, j] = fisher_ij
            self.fisher[j, i] = fisher_ij

        return self.fisher

    def compute_fisher_from_spectra(self, fid, df, pars):
        """
        Compute the Fisher matrix given fiducial and derivative dicts.
        This function is for generality, to enable easier interfacing with
        codes like CAMB. The input parameters must be in the units of the
        noise, muK^2.
        Parameters
        ----------
        fid (dictionary) : keys are '{parameter_XY}' with XY in {tt, te, ee}.
            These keys point to the actual power spectra.
        df (dictionary) :  keys are '{parameter_XY}' with XY in {tt, te, ee}.
            These keys point to numerically estimated derivatives generated
            from precomputed cosmologies.
        pars (list of strings) : the parameters being constrained in the
            Fisher analysis.
        """
        npar = len(pars)
        self.fisher = np.zeros((npar, npar))
        self.fisher_ell = np.zeros(self.l_max)

        for i, j in itertools.combinations_with_replacement(range(npar), r=2):
            # following eq 4 of https://arxiv.org/pdf/1402.4108.pdf
            fisher_ij = 0.0
            # probably a more efficient way to do this exists
            for l in range(self.l_min, self.l_max):

                Cl = np.array([[fid['tt'][l] + self.noise_T[l], fid['te'][l] + self.noise_TE[l]],
                               [fid['te'][l] + self.noise_TE[l], fid['ee'][l] + self.noise_P[l]]])
                invCl = np.linalg.inv(Cl)

                dCl_i = np.array([[df[pars[i]+'_tt'][l], df[pars[i]+'_te'][l]],
                                  [df[pars[i]+'_te'][l], df[pars[i]+'_ee'][l]]])
                dCl_j = np.array([[df[pars[j]+'_tt'][l], df[pars[j]+'_te'][l]],
                                  [df[pars[j]+'_te'][l], df[pars[j]+'_ee'][l]]])

                inner_term = np.dot(np.dot(invCl, dCl_i), np.dot(invCl, dCl_j))
                fisher_contrib = (2*l+1)/2. * self.f_sky * np.trace(inner_term)
                fisher_ij += fisher_contrib

            # fisher is diagonal, so we get half of the matrix for free
            self.fisher[i, j] = fisher_ij
            self.fisher[j, i] = fisher_ij

        return self.fisher



class BAO_Experiment():

    """
    Class for returning a BAO fisher matrix
    """

    def __init__(self, redshifts, errors):
        """Initialize BAO experiment with z and sigma_fk as a percentage of true sigma_fk
        """
        self.redshifts = np.array(redshifts)
        self.errors = np.array(errors)
    
    def compute_fisher_from_camb(self,pars,fiducial,parameters,delta):
        """
           camb oject
           cosmology: camb object containing initial cosmology information
           fiducial: dictionary containing original cosmology parameters and values
           parameters: list of strings containing name of experiments
        """
        der_dict={}
    
        #convert the %error into absolute error from fiducial f_k measurements
        results = camb.get_results(pars)
        matrix = results.get_BAO(self.redshifts, pars)
        rs_over_DV, H, DA, F_AP = matrix[:, 0], matrix[:, 1], matrix[:, 2], matrix[:, 3]

        self.errors=rs_over_DV*self.errors/100
        #create the cosmologies to take derivatives with
        for i in range(len(parameters)):
            left,right=delta[i]
            cosmology_pars,pars,res=get_cosmology_var(fiducial,parameters[i],delta[i])
            matrix_high=res[0].get_BAO(self.redshifts, pars[0])
            matrix_low=res[1].get_BAO(self.redshifts, pars[1])
            rs_over_DVhigh=matrix_high[:, 0]
            rs_over_DVlow=matrix_low[:, 0]
            der_dict[parameters[i]+'_dfdtheta']=(rs_over_DVhigh-rs_over_DVlow)/(left+right)       
        df=der_dict
        npar = len(parameters)
        self.fisher = np.zeros((npar, npar))

        for i, j in itertools.combinations_with_replacement(range(npar), r=2):
            fisher_ij = 0.0
            
            for z_ind, z in enumerate(self.redshifts):
                df_dtheta_i = df[parameters[i]+'_dfdtheta'][z_ind]
                df_dtheta_j = df[parameters[j]+'_dfdtheta'][z_ind]
                fisher_ij += np.sum( (df_dtheta_i*df_dtheta_j)/(self.errors[z_ind]**2) )

            # fisher is diagonal, so we get half of the matrix for free
            self.fisher[i, j] = fisher_ij
            self.fisher[j, i] = fisher_ij

        return self.fisher

class Prior():
    def __init__(self,parameter_name,error):
        self.parameter_name=parameter_name
        self.prior_error = error
    def get_fisher(self,parameters):
        npar = len(parameters)
        self.fisher = np.zeros((npar, npar))
        for index, parameter in enumerate(parameters):
            if self.parameter_name == parameter:
                fisher[index, index] = 1./self.prior_error**2
                return fisher

        return fisher
  

def get_PlanckPol_combine(other_exp_l_min=100):
    # planck from Allison + Madhavacheril
    
    TEB = CMB_Primary(theta_fwhm=[33,    23,  14,  10, 7, 5, 5], 
                           sigma_T = [145,  149,  137,65, 43,66,200],
                           sigma_P = [1e100,1e100,450,103,81,134,406],
                           f_sky = 0.2,
                           l_min = other_exp_l_min,
                           l_max = 2500)

    low_TEB = CMB_Primary(theta_fwhm=[33,    23,  14,  10, 7, 5, 5], 
                           sigma_T = [145,  149,  137,65, 43,66,200],
                           sigma_P = [1e100,1e100,450,103,81,134,406],
                           f_sky = 0.6,
                           l_min = 30,
                           l_max = other_exp_l_min)
    
    TT = CMB_Primary(theta_fwhm=[33,    23,  14,  10, 7, 5, 5], 
                           sigma_T = [145,   149,  137,   65,   43,   66,  200],
                           sigma_P = [1e100,1e100,1e100,1e100,1e100,1e100,1e100],
                           f_sky = 0.6,
                           l_min = 2,
                           l_max = 30)
    
    return [TT, low_TEB, TEB] # NOTE NO TAU PRIOR, MUST INCLUDE WITH OTHER


def get_S4(theta=1.5, error=1.0, f_sky=0.4):
    # S4 forecasting specs

    # just P for 100-300
    low_P = CMB_Primary(theta_fwhm=[theta], 
                           sigma_T = [1e100],
                           sigma_P = [1.4*error],
                           f_sky = f_sky,
                           l_min = 100,
                           l_max = 300)

    # 300-3000 both T+P
    low_ell = CMB_Primary(theta_fwhm=[theta], 
                           sigma_T = [error],
                           sigma_P = [1.4*error],
                           f_sky = f_sky,
                           l_min = 300,
                           l_max = 3000)
    
    # just P for 3000-5000
    high_ell = CMB_Primary(theta_fwhm=[theta], 
                           sigma_T = [1e100],
                           sigma_P = [1.4*error],
                           f_sky = f_sky,
                           l_min = 3000,
                           l_max = 5000)
    
    tau_prior = Prior('tau',0.01 )

    return [low_P, low_ell, high_ell, tau_prior] + get_PlanckPol_combine()


def get_BAO15():
    zs=np.array([0.106, 0.15, 0.32, 0.57])
    sigma_fk_error=np.array([4.83,3.87,2.35,1.33])
    bao=BAO_Experiment(zs,sigma_fk_error)
    return bao

def get_DESI_lowz():
    zs_low=np.array([0.05, 0.15, 0.25, 0.35, 0.45])
    sigma_fk_errorlow=np.array([4.33,1.66,1.07,0.91,1.56])
    lowz=BAO_Experiment(zs_low,sigma_fk_errorlow)
    return lowz

def get_DESI_highz():
    zs_high = np.array([0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85]) 
    sigma_fk_errorhigh=np.array([0.57,0.48,0.47,0.49,0.58,0.60,0.61,0.92,0.98,1.16,1.76,2.88,2.92])
    highz=BAO_Experiment(zs_high,sigma_fk_errorhigh)
    return highz


class CMB_Lensing():
    """Produce lensing kappa noise from pytempura. Useful to combine with CMB primary fisher forecast
    """
    
        # DEFAULTS ARE FOR PLANCK EXPERIMENT
    def __init__(self, 
             lens_beam = 7.0,lens_noiseT = 33.,lens_noiseP = 56.,
             lens_tellmin = 2,lens_tellmax = 3000,lens_pellmin = 2,
             lens_pellmax = 3000,lens_kmin = 80,lens_kmax = 3000, lens_f_sky=0.65, estimators=('TT','TE','EE','EB','TB','MV'), 
                NlTT=None, NlEE=None, NlBB=None):

        # get lensing noise
        # Initialize cosmology and Clkk. Later parts need dimensionless spectra.
        self.l_min = lens_tellmin
        self.l_max = min(max(lens_tellmax, lens_pellmax)+1000,5500) 
        self.k_min = lens_kmin
        self.k_max = lens_kmax
        self.f_sky = lens_f_sky
        
        from falafel import utils
        import pytempura
        # generate cosmology with orphics
        lmax = self.l_max
        ells = np.arange(0,lmax+1,1)
        Tcmb = 2.726
        
        # compute noise curves
        sT = lens_noiseT * (np.pi/60./180.)
        sP = lens_noiseP * (np.pi/60./180.)
        theta_FWHM = lens_beam * (np.pi/60./180.)
        muK = Tcmb*1.0e6
        # unitless white noise
        exp_term = np.exp(ells*(ells+1)*(theta_FWHM**2)/(8*np.log(2)))
        if NlTT is None:
            NlTT = sT**2 * exp_term #/ muK**2
        else:
            NlTT = NlTT[ells]
        if NlEE is None:
            NlEE = sP**2 * exp_term #/ muK**2
        else:
            NlEE = NlEE[ells]
        if NlBB is None:
            NlBB = sP**2 * exp_term #/ muK**2
        else:
            NlBB = NlBB[ells]
            
            
        NlTT[ells > lens_tellmax] = 1e100
        NlEE[ells > lens_pellmax] = 1e100
        NlBB[ells > lens_pellmax] = 1e100
            
        self.NlTT = NlTT
        self.NlEE = NlEE
        self.NlBB = NlBB
        noise={'TT':NlTT,'EE':NlEE,'BB':NlBB}

        est_norm_list=['TT','TE','EE','EB','TB','MV']

        ucls,tcls = utils.get_theory_dicts(nells=noise,lmax=lmax,grad=True)
        Als = pytempura.get_norms(est_norm_list,ucls,tcls,self.l_min,lmax,k_ellmax=lens_kmax)
        ls = np.arange(Als['MV'][0].size)
        self.noise_rec = np.interp(np.arange(self.l_max+1), ls, Als['MV'][0]* (ls*(ls+1.)/2.)**2.)
        self.noise_rec[np.arange(self.l_max+1) <= lens_kmin] = 1e100
        self.noise_rec[np.arange(self.l_max+1) >= lens_kmax] = 1e100

 
    def compute_fisher_from_camb(self,cmbresults,defaultCosmology,parameters,delta):
        """
           cosmology: camb object containing initial cosmology information
           fiducial: dictionary containing original cosmology parameters and values
           parameters: list of strings containing name of experiments
           delta: list of tuples
        """
        df={}
    
        #convert the %error into absolute error from fiducial f_k measurements
        results=cmbresults
        clkk=results.get_lens_potential_cls(lmax=self.k_max) #camb returns Dphi
        clkk=clkk[:,0]*np.pi*0.5
        self.noise_kappa=(clkk+self.noise_rec[:self.k_max+1])


        #set up derivative for clkk

        npar = len(parameters)
        self.fisher = np.zeros((npar, npar))
        for i,j in itertools.combinations_with_replacement( range(npar),r=2):
            fisher_ij = 0.0
            print(i)
            # following eq 4 of https://arxiv.org/pdf/1402.4108.pdf
            left_i,right_i=delta[i]
            left_j,right_j=delta[j]
            _,pars_i,res_i=get_cosmology_var(defaultCosmology,parameters[i],delta[i])
            cls_high_i=res_i[0].get_lens_potential_cls(lmax=self.k_max)
            cls_high_i=cls_high_i[:,0]*np.pi*0.5
            cls_low_i=res_i[1].get_lens_potential_cls(lmax=self.k_max)
            cls_low_i=cls_low_i[:,0]*np.pi*0.5
            df[parameters[i]+'_kk']=(cls_high_i-cls_low_i)/((right_i+left_i))
            _,pars_j,res_j=get_cosmology_var(defaultCosmology,parameters[j],delta[j])
            cls_high_j=res_j[0].get_lens_potential_cls(lmax=self.k_max)
            cls_high_j=cls_high_j[:,0]*np.pi*0.5
            cls_low_j=res_j[1].get_lens_potential_cls(lmax=self.k_max)
            cls_low_j=cls_low_j[:,0]*np.pi*0.5
            df[parameters[j]+'_kk']=(cls_high_j-cls_low_j)/((right_j+left_j))
            for l in range(self.k_min, self.k_max):
                Clkk_plus_Nlkk_sq = (self.noise_kappa[l])**2

                fisher_contrib = (2*l+1)/2. * self.f_sky * \
                    (df[parameters[i]+'_kk'][l] * 
                     df[parameters[j]+'_kk'][l])/Clkk_plus_Nlkk_sq
                fisher_ij += fisher_contrib
                
            # fisher is diagonal
            self.fisher[i,j] = fisher_ij
            self.fisher[j,i] = fisher_ij
        
        return self.fisher
    
    


import itertools
class LSSTxlensing():
    """
    Class for computing Fisher matrices from LSSTxlensing.

    """

    def __init__(self, fid,spectra,der_spectra,pars,lensingnoise,f_sky=0.65, l_min=10, l_max=1000,nbar=66):
        self.f_sky = f_sky
        self.spectra=spectra
        self.der_spectra=der_spectra
        self.fid=fid
        self.pars=pars
        self.Npars=len(self.pars)
        self.ells = np.arange(l_min,l_max)
        self.l_min = l_min
        self.l_max = l_max
        self.shotnoise=1/nbar
        #convert shotnoise from arcmin^2 to rad
        self.shotnoise*= (np.pi/60./180.)**2
        self.lensingnoise=lensingnoise


    def get_modes(self):
        deltaL=np.zeros(len(self.ells))
        deltaL[0]=self.ells[0]-0
        deltaL[1:]=self.ells[1:]-self.ells[:-1] 
        result = (2*self.ells*deltaL*self.f_sky)
        return 1/result

    def get_Cl(self,X,Y):
        """X:g,k
           Y:gi,g
        """
        kappa_noise=(self.lensingnoise[self.l_min:self.l_max]+self.fid[f'kk'][self.l_min:self.l_max])*np.sqrt(self.get_modes())
        if X+Y in [f'g{i}g{i}' for i in range(16)]:
            Cl=self.fid[f'{X+Y}'][self.l_min:self.l_max]+self.shotnoise
        elif Y+X in [f'g{i}g{i}' for i in range(16)]:
            Cl=self.fid[f'{Y+X}'][self.l_min:self.l_max]+self.shotnoise
        elif X+Y in [f'kg{i}' for i in range(16)]:
            Cl=self.fid[f'{X+Y}'][self.l_min:self.l_max]
        elif Y+X in [f'kg{i}' for i in range(16)]:
            Cl=self.fid[f'{Y+X}'][self.l_min:self.l_max]
        elif X+Y in ['kk']:
            Cl=self.fid[f'kk'][self.l_min:self.l_max]+kappa_noise
        else:
            #print("cross cls not present")
            Cl=np.ones(len(self.ells))*0
        return Cl

    def get_derCl(self,der,X,Y):
        """X:g,k
           Y:gi,g
        """
        if X+Y in [f'g{i}g{i}' for i in range(16)]:
            Cl=der[f'{X+Y}'][self.l_min:self.l_max]+self.shotnoise
        elif Y+X in [f'g{i}g{i}' for i in range(16)]:
            Cl=der[f'{Y+X}'][self.l_min:self.l_max]+self.shotnoise
        elif X+Y in [f'kg{i}' for i in range(16)]:
            Cl=der[f'{X+Y}'][self.l_min:self.l_max]
        elif Y+X in [f'kg{i}' for i in range(16)]:
            Cl=der[f'{Y+X}'][self.l_min:self.l_max]
        elif X+Y in ['kk']:
            Cl=der[f'kk'][self.l_min:self.l_max]
        else:
            #print("cross cls not present")
            Cl=np.ones(len(self.ells))*0
        return Cl

    def get_cov(self,X,Y,W,Z):
            return self.get_modes()*(self.get_Cl(X,W)*self.get_Cl(Y,Z)+self.get_Cl(X,Z)*self.get_Cl(Y,W))

    def get_covmat(self):
        self.covmat=np.zeros((len(self.ells),len(self.spectra),len(self.spectra)))
        self.cov_dict = {}
        for i in range(len(self.spectra)): 
            for j in range(len(self.spectra)):
                if i<16 or i>32:
                    X=list(self.spectra)[i][:2]
                    Y=list(self.spectra)[i][2:]
                elif i in range(16,32):
                    X=list(self.spectra)[i][0]
                    Y=list(self.spectra)[i][1:]
                elif i==32:
                    X,Y=list(self.spectra)[i]
                if j<16 or j>32:
                    W=list(self.spectra)[j][:2]
                    Z=list(self.spectra)[j][2:]
                elif j in range(16,32):
                    W=list(self.spectra)[j][0]
                    Z=list(self.spectra)[j][1:]
                elif i==32:
                    W,Z=list(self.spectra)[j]
                self.covmat[:,i,j]=self.get_cov(X,Y,W,Z)
                self.cov_dict[list(self.spectra)[i]+','+list(self.spectra)[j]]= self.get_cov(X,Y,W,Z)
        return self.covmat
    
    def get_covmatnew(self):
        """use this one"""
        lista = [f'g{i}' for i in range(16)]
        lista.append('k')
        allcombs = list(itertools.combinations_with_replacement(lista, 2))
        self.covmat=np.zeros((len(self.ells),len(allcombs),len(allcombs)))
        self.cov_dict = {}
        for cA in allcombs:
            for cB in allcombs:
                a, b = cA
                c, d = cB
                i, j = allcombs.index(cA), allcombs.index(cB)
                self.covmat[:,i,j]=self.get_cov(a, b, c, d)
                self.cov_dict[a+b+c+d]= self.covmat[:,i,j]
        return self.covmat
    
    def prepare_derivatives(self):
        lista = [f'g{i}' for i in range(16)]
        lista.append('k')
        allcombs = list(itertools.combinations_with_replacement(lista, 2))
        der_spectra_alpha = np.ones((len(self.ells), len(allcombs), self.Npars))
        for i in range(self.Npars):
            for cA in allcombs:
                    a, b = cA
                    ind= allcombs.index(cA)
                    der_spectra_alpha[:,ind,i]=self.get_derCl(self.der_spectra[i],a,b)
        return der_spectra_alpha
    
    
    def get_fisher(self):
        lista = [f'g{i}' for i in range(16)]
        lista.append('k')
        allcombs = list(itertools.combinations_with_replacement(lista, 2))     
        #derivatives check shape (ells,len(spectra),len(pars))
        derivatives=self.prepare_derivatives()
        assert(derivatives.shape[0] == len(self.ells)) 
        assert(derivatives.shape[1] == len(allcombs)) 
        assert(derivatives.shape[2] == self.Npars)
        fisher_per_mode = np.einsum('...ik, ...ij, ...jm -> ...km',derivatives, np.nan_to_num(np.linalg.inv(self.get_covmatnew())), derivatives)
        self.fisher_per_mode=fisher_per_mode
        print(fisher_per_mode)
        self.error_per_mode_non_marginalized = np.nan_to_num(np.diagonal(fisher_per_mode,axis1 = 1, axis2 = 2)**-0.5)
        
        self.fisher=np.sum(self.fisher_per_mode,axis=0)
        return self.fisher

import itertools
class LSSTxlensing():
    """
    Class for computing Fisher matrices from LSSTxlensing.

    """

    def __init__(self, fid,spectra,der_spectra,pars,lensingnoise,lbmax,n,f_sky=0.65, l_min=2, l_max=1000):
        self.f_sky = f_sky
        self.spectra=spectra
        self.der_spectra=der_spectra
        self.fid=fid
        self.pars=pars
        self.Npars=len(self.pars)
        self.ells = np.arange(l_min,l_max)
        self.l_min = l_min
        self.l_max = l_max
        self.shotnoise=1/n
        #convert shotnoise from arcmin^2 to rad^2
        self.shotnoise*= (np.pi/60./180.)**2
        self.noise_list=[]
        self.cross_noise=[]
        self.lbmax=lbmax
        for i in range(16):
            noise=self.shotnoise[i].copy()
            crossnoise=np.zeros(3000)
            noise[int(lbmax[i]):]=1e100
            #crossnoise[int(lbmax[i]):]=1e100
            self.noise_list.append(noise)
            self.cross_noise.append(crossnoise)
        self.lensingnoise=lensingnoise


    def get_modes(self):
        deltaL=np.zeros(len(self.ells))
        deltaL[0]=self.ells[0]-0
        deltaL[1:]=self.ells[1:]-self.ells[:-1] 
        result = (2*self.ells*deltaL*self.f_sky)
        return 1/result

    def get_Cl(self,X,Y):
        """X:g,k
           Y:gi,g
        """
        kappa_noise=(self.lensingnoise[self.l_min:self.l_max]+self.fid[f'kk'][self.l_min:self.l_max])*np.sqrt(self.get_modes())
        if X+Y in [f'g{i}g{i}' for i in range(16)]:
            j=int(X[1])
            Cl=self.fid[f'{X+Y}'][self.l_min:self.l_max]+self.noise_list[j][self.l_min:self.l_max]
        elif Y+X in [f'g{i}g{i}' for i in range(16)]:
            j=int(X[1])
            Cl=self.fid[f'{Y+X}'][self.l_min:self.l_max]+self.noise_list[j][self.l_min:self.l_max]
        elif X+Y in [f'kg{i}' for i in range(16)]:
            j=int(Y[1])
            Cl=self.fid[f'{X+Y}'][self.l_min:self.l_max]
            Cl=self.fid[f'{X+Y}'][self.l_min:self.l_max]+self.cross_noise[j][self.l_min:self.l_max]

        elif Y+X in [f'kg{i}' for i in range(16)]:
            j=int(X[1])
            Cl=self.fid[f'{Y+X}'][self.l_min:self.l_max]
            Cl=self.fid[f'{Y+X}'][self.l_min:self.l_max]+self.cross_noise[j][self.l_min:self.l_max]

        elif Y+X in [f'kg{i}' for i in range(16)]:
            j=int(X[1])
            Cl=self.fid[f'{Y+X}'][self.l_min:self.l_max]
            Cl=self.fid[f'{Y+X}'][self.l_min:self.l_max]+self.cross_noise[j][self.l_min:self.l_max]
        
        elif X+Y in ['kk']:
            Cl=self.fid[f'kk'][self.l_min:self.l_max]+kappa_noise[self.l_min:self.l_max]
        else:
            #print("cross cls not present")
            Cl=np.ones(len(self.ells))*0
        return Cl

    def get_derCl(self,der,X,Y):
        """X:g,k
           Y:gi,g
        """
        if X+Y in [f'g{i}g{i}' for i in range(16)]:
            j=int(X[1])
            Cl=der[f'{X+Y}']
            Cl[int(self.lbmax[j]):]=0.
            Cl=Cl[self.l_min:self.l_max]
            Cl=der[f'{X+Y}'][self.l_min:self.l_max]
        
        elif Y+X in [f'g{i}g{i}' for i in range(16)]:
            j=int(X[1])
            Cl=der[f'{Y+X}']
            Cl[int(self.lbmax[j]):]=0.
            Cl=Cl[self.l_min:self.l_max]
            
            #Cl=der[f'{Y+X}'][self.l_min:self.l_max]
        elif X+Y in [f'kg{i}' for i in range(16)]:
            j=int(Y[1])
            Cl=der[f'{X+Y}']
            Cl[int(self.lbmax[j]):]=0.
            Cl=Cl[self.l_min:self.l_max]           
            #Cl=der[f'{X+Y}'][self.l_min:self.l_max]
        elif Y+X in [f'kg{i}' for i in range(16)]:
            j=int(X[1])
            Cl=der[f'{Y+X}']
            Cl[int(self.lbmax[j]):]=0.
            Cl=Cl[self.l_min:self.l_max] 
            #Cl=der[f'{Y+X}'][self.l_min:self.l_max]

        elif X+Y in ['kk']:
            Cl=der[f'kk'][self.l_min:self.l_max]
        else:
            Cl=np.ones(len(self.ells))*0
        return Cl

    def get_cov(self,X,Y,W,Z):
            return self.get_modes()*(self.get_Cl(X,W)*self.get_Cl(Y,Z)+self.get_Cl(X,Z)*self.get_Cl(Y,W))

    
    def get_covmatnew(self):
        """use this one"""
        lista = [f'g{i}' for i in range(16)]
        lista.append('k')
        allcombs = list(itertools.combinations_with_replacement(lista, 2))
        self.covmat=np.zeros((len(self.ells),len(allcombs),len(allcombs)))
        self.cov_dict = {}
        for cA in allcombs:
            for cB in allcombs:
                a, b = cA
                c, d = cB
                i, j = allcombs.index(cA), allcombs.index(cB)
                self.covmat[:,i,j]=self.get_cov(a, b, c, d)
                self.cov_dict[a+b+c+d]= self.covmat[:,i,j]
        return self.covmat
    
    def prepare_derivatives(self):
        lista = [f'g{i}' for i in range(16)]
        lista.append('k')
        allcombs = list(itertools.combinations_with_replacement(lista, 2))
        der_spectra_alpha = np.ones((len(self.ells), len(allcombs), self.Npars))
        for i in range(self.Npars):
            for cA in allcombs:
                    a, b = cA
                    ind= allcombs.index(cA)
                    der_spectra_alpha[:,ind,i]=self.get_derCl(self.der_spectra[i],a,b)
        return der_spectra_alpha
    
    
    def get_fisher(self):
        lista = [f'g{i}' for i in range(16)]
        lista.append('k')
        allcombs = list(itertools.combinations_with_replacement(lista, 2))     
        #derivatives check shape (ells,len(spectra),len(pars))
        derivatives=self.prepare_derivatives()
        assert(derivatives.shape[0] == len(self.ells)) 
        assert(derivatives.shape[1] == len(allcombs)) 
        assert(derivatives.shape[2] == self.Npars)
        fisher_per_mode = np.einsum('...ik, ...ij, ...jm -> ...km',derivatives, np.nan_to_num(np.linalg.inv(self.get_covmatnew())), derivatives)
        self.fisher_per_mode=fisher_per_mode
        self.error_per_mode_non_marginalized = np.nan_to_num(np.diagonal(fisher_per_mode,axis1 = 1, axis2 = 2)**-0.5)
        self.fisher=np.sum(self.fisher_per_mode,axis=0)
        return self.fisher