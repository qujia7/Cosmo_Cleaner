import camb
import numpy as np
import itertools

#finite difference
#insert the parameter you would like to vary
def get_cosmology_var(defaultCosmology,par,delta=0.005):
    """
    default cosmology: dict containing default cosmology values
    par: parameter you are finding derivative of, keeping others constant
    delta: value you change your default cosmological parameter by
    """
    defaultCosmology_h=defaultCosmology.copy()
    defaultCosmology_l=defaultCosmology.copy()
    delta=delta*defaultCosmology[par]
    defaultCosmology_h[par]+=delta
    defaultCosmology_l[par]-=delta
    Cosmology_l=[defaultCosmology_h,defaultCosmology_l]
    parameters=[]
    cambres=[]
    for i in range(2):
        pars = camb.CAMBparams()
        pars.set_dark_energy(w=Cosmology_l[i]['w0'],wa = Cosmology_l[i]['wa'], dark_energy_model = 'ppf')
        pars.set_cosmology(H0=Cosmology_l[i]['H0'], cosmomc_theta = None,ombh2=Cosmology_l[i]['ombh2'], 
                       omch2=Cosmology_l[i]['omch2'], mnu=Cosmology_l[i]['mnu'], tau = Cosmology_l[i]['tau'],
                       nnu = Cosmology_l[i]['nnu'], num_massive_neutrinos = 3)
        #pars.NonLinear = model.NonLinear_both
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
        self.Alens=cosmology['A_l']
        
        self.PK = camb.get_matter_power_interpolator(self.pars, nonlinear=True, 
hubble_units=False, k_hunit=False, kmax=self.kmax,
var1='delta_nonu',var2='delta_nonu', zmax=self.zs[-1])

        #LSST specification

        self.LSST_bins=np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.3, 2.6, 3, 3.5, 4, 7])
        start=self.LSST_bins[:-1]
        end= self.LSST_bins[1:] 
        self.LSST_z=[]
        for i in range(len(start)):
            self.LSST_z.append(np.arange(start[i],end[i],0.001)) 

        

    def get_lensing_window(self):
        cSpeedKmPerSec = 299792.458
        lensingwindow = 1.5*(self.cosmology['omch2']+self.cosmology['ombh2']+self.pars.omnuh2)*100.*100.*(1.+self.zs)*self.chis*((self.chistar - self.chis)/self.chistar)/self.Hzs/cSpeedKmPerSec
        return self.Alens*lensingwindow

    def dn_dz_LSST(self,z,z0=0.3):
        return (1/(2*z0))*(z/z0)**2*np.exp(-z/z0)

    def lsst_window(self,zrange,dndz,B=1):
        bias=B*(1+zrange)
        window=bias*dndz/np.trapz(dndz,zrange)
        return np.interp(self.zs,zrange,window,right=0,left=0)

    def get_lsst_kappa(self,i,lmax=2000):
        #return lsstxlensing with the ith bin
        cSpeedKmPerSec = 299792.458
        lensingwindow=self.get_lensing_window()
        galaxywindow=self.lsst_window(self.LSST_z[i],self.dn_dz_LSST(self.LSST_z[i]))

        precalcFactor= self.Hzs**2./self.chis/self.chis/cSpeedKmPerSec**2.
        lbmax=self.kmax*self.results.comoving_radial_distance(np.mean(self.LSST_z[i]))
        ellsgg=np.arange(lmax)
        cl_cross=[]
        w = np.ones(self.chis.shape)
        for l in ellsgg:
            if l<lbmax:
                k=(l+0.5)/self.chis
                w[:]=1
                w[k<1e-4]=0
                w[k>=self.kmax]=0
                pkin = self.PK.P(self.zs, k, grid=False)
                common = ((w*pkin)*precalcFactor)[self.zs>=self.zmin]        
                estCl = np.dot(self.dchis[self.zs>=self.zmin], common*(lensingwindow*galaxywindow)[self.zs>=self.zmin])
                cl_cross.append(estCl)
            else:
                cl_cross.append(0)
        return np.array(cl_cross)

    def get_lsst_auto(self,i,j=None,lmax=2000):
        #int i: ith bin
        #int j:jth bin
        cSpeedKmPerSec = 299792.458
        #galaxywindow=self.dndz_gauss(self.zs,mean_z,width)
        galaxywindow=self.lsst_window(self.LSST_z[i],self.dn_dz_LSST(self.LSST_z[i]))
        if j is not None:
            galaxywindow2=self.lsst_window(self.LSST_z[j],self.dn_dz_LSST(self.LSST_z[j]))
        else:
            galaxywindow2=galaxywindow
            j=i

        cl_autog=[]
        lbmax=min(self.kmax*self.results.comoving_radial_distance(np.mean(self.LSST_z[i])),self.kmax*self.results.comoving_radial_distance(np.mean(self.LSST_z[j])))
        ellsgg=np.arange(lmax)
        wg = np.ones(self.chis.shape)

        precalcFactor= self.Hzs**2./self.chis/self.chis/cSpeedKmPerSec**2.
        for l in ellsgg:
            if l<lbmax:
                k=(l+0.5)/self.chis
                wg[:]=1
                wg[k<1e-4]=0
                wg[k>=self.kmax]=0
                pkin = self.PK.P(self.zs, k, grid=False)
                common = ((wg*pkin)*precalcFactor)[self.zs>=self.zmin]        
                estCl = np.dot(self.dchis[self.zs>=self.zmin], common*(galaxywindow*galaxywindow2)[self.zs>=self.zmin])
                cl_autog.append(estCl)
            else:
                cl_autog.append(0.)
        return np.array(cl_autog)

    def dndz_gauss(self,z,z0,sigma):
        ans = 1/np.sqrt(2*np.pi*sigma**2)* np.exp((-(z-z0)**2)/ (2.*sigma**2.))
        return self.bias*self.Alens*ans 


    def get_clgg(self,galaxy_window,lmax=2000):
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

    def get_clkg(self,galaxy_window,lmax=2000):
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

    def get_spectra(self,mean_z,width):
        self.spectra_dict={'clgg':self.get_clgg(mean_z,width),
        'clkk':self.get_clkk(),'clkg':self.get_clkg(mean_z,width) }
        self.spectra=[self.get_clgg(mean_z,width),self.get_clkg(mean_z,width),self.get_clkk()]
        return np.array(self.spectra)
    
    def get_lsst_lensing(self):
        fields={}
        field_list=[]
        for i in range(len(self.LSST_bins)-1):
            fields[f'g{i}g{i}']=self.get_lsst_auto(i)
            field_list.append(fields[f'g{i}g{i}'])
        for i in range(len(self.LSST_bins)-1):
            fields[f'kg{i}']=self.get_lsst_kappa(i)
            field_list.append(fields[f'kg{i}'])
    

            
        fields['kk']=self.get_clkk()
        field_list.append(fields['kk'])
        field_list=np.array(field_list)
        return fields,field_list
     
    
    #for this problem we need three fields, g1, cmb and the external tracer
    def get_fields(self):
        """
        g1=self.bias*self.get_lensing_window()
        g1[self.zs>0.3]=0
        g2=self.get_lensing_window()
        g3=self.get_lensing_window()
        g3[self.zs<0.3]=0
        """
        
        g1=self.dndz_gauss(self.zs,0.2,0.5)
        g1[self.zs>0.3]=0
        g2=self.get_lensing_window()
        g3=self.dndz_gauss(self.zs,4,1)
        g3[self.zs<0.3]=0
        

        self.windows=[g1,g2,g3]
        self.fields=[self.get_clgg(g1),self.get_clkg(g1),self.get_clkk(),self.get_clgg(g3),self.get_clkg(g3),self.get_clkk()-(self.get_clkg(g3)**2/self.get_clgg(g3))]
        return np.array(self.fields)




def LSST_derivative_parameter(ells,defaultCosmology,parameter,delta=0.005,nz=1000,kmax=10,zmin=0):
    """
    take derivative with respect to a given parameter.
    """
    cosmology_pars,pars,res=get_cosmology_var(defaultCosmology,parameter,delta=delta)
    high=cosmology(nz,kmax,zmin,ells,cosmology_pars[0],pars[0],res[0])
    low=cosmology(nz,kmax,zmin,ells,cosmology_pars[1],pars[1],res[1])
    #derivative=(high.get_spectra(mean_z,width)-low.get_spectra(mean_z,width))/(2*delta*defaultCosmology[parameter])
    derivative=(high.get_lsst_lensing()[1]-low.get_lsst_lensing()[1])/(2*delta*defaultCosmology[parameter])
    return derivative

def derivative_parameter(ells,mean_z,width,defaultCosmology,parameter,delta=0.005,nz=1000,kmax=10,zmin=0):
    """
    take derivative with respect to a given parameter.
    """
    cosmology_pars,pars,res=get_cosmology_var(defaultCosmology,parameter,delta=delta)
    high=cosmology(nz,kmax,zmin,ells,cosmology_pars[0],pars[0],res[0])
    low=cosmology(nz,kmax,zmin,ells,cosmology_pars[1],pars[1],res[1])

    #derivative=(high.get_spectra(mean_z,width)-low.get_spectra(mean_z,width))/(2*delta*defaultCosmology[parameter])
    derivative=(high.get_fields()-low.get_fields())/(2*delta*defaultCosmology[parameter])
    return derivative



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

    def compute_fisher_from_camb(self,cmbresults,defaultCosmology,parameters):
        """
           cmbresults: camb object containing initial cosmology information
           defaultCosmology: dictionary containing original cosmology parameters and values
           parameters: list of strings containing name of experiments
        """
        der_dict={}
        delta=0.0005
    
        fiducial={}
        results=cmbresults
        fid=results.get_cmb_power_spectra(lmax=self.l_max,spectra=['lensed_scalar'],CMB_unit='muK',raw_cl=True)['lensed_scalar'].transpose()
        fiducial['tt']=fid[0]
        fiducial['ee']=fid[1]
        fiducial['te']=fid[3]

        #create the cosmologies to take derivatives with
        for i in range(len(parameters)):
            cosmology_pars,pars,res=get_cosmology_var(defaultCosmology,parameters[i],delta)
            cls_high=res[0].get_cmb_power_spectra(lmax=self.l_max,spectra=['lensed_scalar'],CMB_unit='muK',raw_cl=True)['lensed_scalar'].transpose()
            cls_low=res[1].get_cmb_power_spectra(lmax=self.l_max,spectra=['lensed_scalar'],CMB_unit='muK',raw_cl=True)['lensed_scalar'].transpose()

            der_dict[parameters[i]+'_tt']=(cls_high[0]-cls_low[0])/(2*delta*defaultCosmology[parameters[i]])
            der_dict[parameters[i]+'_ee']=(cls_high[1]-cls_low[1])/(2*delta*defaultCosmology[parameters[i]])
            der_dict[parameters[i]+'_te']=(cls_high[3]-cls_low[3])/(2*delta*defaultCosmology[parameters[i]])
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
    
    def compute_fisher_from_camb(self,fiducial,parameters):
        """
           cosmology: camb object containing initial cosmology information
           fiducial: dictionary containing original cosmology parameters and values
           parameters: list of strings containing name of experiments
        """
        der_dict={}
        delta=0.0005
    
        #convert the %error into absolute error from fiducial f_k measurements
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=fiducial['H0'], ombh2=fiducial['ombh2'], omch2=fiducial['omch2'], mnu=fiducial['mnu'], omk=0, tau=fiducial['tau'])
        pars.InitPower.set_params(As=fiducial['As'], ns=fiducial['ns'], r=0)
        pars.set_for_lmax(2500, lens_potential_accuracy=0)
        results = camb.get_results(pars)
        matrix = results.get_BAO(self.redshifts, pars)
        rs_over_DV, H, DA, F_AP = matrix[:, 0], matrix[:, 1], matrix[:, 2], matrix[:, 3]
        print(self.errors)
        print(rs_over_DV)
        print(rs_over_DV*self.errors/100)
        self.errors=rs_over_DV*self.errors/100
        #create the cosmologies to take derivatives with
        for i in range(len(parameters)):
            cosmology_pars,pars,res=get_cosmology_var(fiducial,parameters[i],delta)
            matrix_high=res[0].get_BAO(self.redshifts, pars[0])
            matrix_low=res[1].get_BAO(self.redshifts, pars[1])
            rs_over_DVhigh=matrix_high[:, 0]
            rs_over_DVlow=matrix_low[:, 0]
            der_dict[parameters[i]+'_dfdtheta']=(rs_over_DVhigh-rs_over_DVlow)/(2*delta*fiducial[parameters[i]])       
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