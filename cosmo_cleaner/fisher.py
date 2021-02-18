


import healpy as hp
import numpy as np
import matplotlib.pyplot as plt


class Combiner():
    def __init__(self, ells, spectra_cross: dict = {}, spectra_auto: dict = {}):
        '''
        ells: bins in l space
        spectra_cross: cross of field fi with g, just use fi as the name 
        spectra_auto: fi, fj crosses
        '''
        self.nbins=len(ells)
        spectra_list=list(spectra_cross.keys())
        self.Nfields = len(spectra_list)
        self.spectra_cross = np.zeros((self.nbins, self.Nfields))
        self.spectra_auto = np.zeros((self.nbins, self.Nfields, self.Nfields))
        
        for i, name in enumerate(spectra_list):
            self.spectra_cross[:, i] = spectra_cross[name]
            for j, name2 in enumerate(spectra_list):
                try:
                    self.spectra_auto[:, i, j] = spectra_auto[name+name2]
                except:
                    self.spectra_auto[:,i,j]= spectra_auto[name2+name]
    
    def combine(self, a):
        cross = np.einsum('...j, ...j -> ...', a, self.spectra_cross)
        auto = np.einsum('...i, ...ij, ...j -> ...', a, self.spectra_auto, a)
        return cross, auto
    
class Forecaster():
    def __init__(self, fsky, ells, names : list = ['k', 'g'], spectra: dict ={}, use_as_data: list = ['kg', 'gg', 'kk']):
        '''
        fsky: fraction of sky
        ells: bins in l space, centers
        spectra:dict in which key corresponds to the name of C_ell, and values are C_ell array
        names: put field to be cleaned first, then the cross correlated one 
        use_as_data: spectra used in practice, e.g. kk, gg or kk, gg, kg 
        '''
        self.names = names
        self.k = names[0]
        self.g = names[1]
        self.Nfields = len(names)
        self.nbins = len(ells)
        self.Nmodes = self.get_Nmodes(ells, fsky) 
        self.fsky = fsky
        self.list_spectra = use_as_data #list(spectra.keys()) 
        self.Nspectra = len(self.list_spectra)
        #self.used = use_as_data
        self.spectra = spectra
        
    def get_Nmodes(self, ells, fsky):
        deltal = ells[1]-ells[0] #ells[1:]-ells[:-1] 
        result = (2*ells*deltal*fsky)
        return result
    
    def get_spectra_element(self, X, Y): 
        try:
            result=self.spectra[X+Y]
        except:
            result=self.spectra[Y+X]
        return result
    
    def get_gaussian_cov(self, X, Y, W, Z):
        result = 1/self.Nmodes*(self.get_spectra_element(X, W)*self.get_spectra_element(Y, Z)+self.get_spectra_element(X, Z)*self.get_spectra_element(Y, W))
        return result
    
    def get_cov(self, X, Y, W, Z):
        return self.get_gaussian_cov(X, Y, W, Z)
    
    def prepare_cov_for_error(self, separator = ','):
        self.cov = np.zeros((self.nbins, self.Nspectra, self.Nspectra))
        self.cov_dict = {}
        for a in self.list_spectra: 
            for b in self.list_spectra:
                X, Y = list(a)
                W, Z = list(b)
                i = self.list_spectra.index(a)
                j = self.list_spectra.index(b) 
                covariance = self.get_cov(X, Y, W, Z) 
                self.cov[:, i, j] = covariance 
                self.cov_dict[a+separator+b]= covariance
        self.invcov = np.linalg.inv(self.cov)
        return 0
    
    def make_fisher_and_err_bars(self, alphas, der_spectra_alpha):
        '''
        a: vector for the combination
        e.g., Ckk = a.T Ckiki a, Ckg = a.T Ckig 
        '''
        Npars = len(alphas)
        assert(der_spectra_alpha.shape[0] == self.nbins) 
        assert(der_spectra_alpha.shape[1] == self.Nspectra) 
        assert(der_spectra_alpha.shape[2] == Npars)
        
        fisher_per_mode = np.einsum('...ik, ...ij, ...jm -> ...km',der_spectra_alpha, np.nan_to_num(self.invcov), der_spectra_alpha)
        self.error_per_mode_non_marginalized = np.nan_to_num(np.diagonal(fisher_per_mode,axis1 = 1, axis2 = 2)**-0.5)
        fisher_integrated = np.sum(fisher_per_mode, axis = 0) 
        print(fisher_integrated)
        self.fisher = fisher_integrated 
        print(self.fisher)
        self.error_non_marginalized = np.nan_to_num(np.diag(self.fisher)**-0.5 )
        self.error_marginalized = np.nan_to_num(np.linalg.inv(self.fisher)**0.5 )
        return self.error_marginalized

#create forecaster class as a check of the above

#receive spectra as a dict

class Fisher():
    def __init__(self, fsky,Npars, ells, spectra, der_spectra):
        self.spectra=spectra
        self.fsky=fsky
        self.der_spectra=der_spectra
        self.Npars=Npars
        self.ells=ells
        #function to get the number of modes
        
    def get_modes(self):
        
        deltaL=np.zeros(len(self.ells))
        deltaL[0]=self.ells[0]-0
        deltaL[1:]=self.ells[1:]-self.ells[:-1] 
        result = (2*self.ells*deltaL*self.fsky)
        return 1/result

    def get_Cl(self,X,Y):
        """X:g,k,i
           Y:g,k,i
        """
        if X+Y in self.spectra:
            Cl=self.spectra[X+Y]
        elif Y+X in self.spectra:
            Cl=self.spectra[Y+X]
        else:
            #field not present assume its zero and do not return error
            Cl=np.zeros(len(self.ells))
        return Cl

    def get_cov(self,X,Y,W,Z):
        return self.get_modes()*(self.get_Cl(X,W)*self.get_Cl(Y,Z)+self.get_Cl(X,Z)*self.get_Cl(Y,W))

    def get_covmat(self):
        self.covmat=np.zeros((len(self.ells),len(self.spectra),len(self.spectra)))
        self.cov_dict = {}
        for i in range(len(self.spectra)): 
            for j in range(len(self.spectra)):
                X,Y=list(self.spectra)[i]
                W,Z=list(self.spectra)[j]
                self.covmat[:,i,j]=self.get_cov(X,Y,W,Z)
                self.cov_dict[list(spectra)[i]+','+list(spectra)[j]]= self.get_cov(X,Y,W,Z)
        return self.covmat
    
    def get_fisher(self):
        
        #derivatives check shape (ells,len(spectra),len(pars))
        assert(self.der_spectra.shape[0] == len(self.ells)) 
        assert(self.der_spectra.shape[1] == len(self.spectra)) 
        assert(self.der_spectra.shape[2] == self.Npars)
        fisher_per_mode = np.einsum('...ik, ...ij, ...jm -> ...km',self.der_spectra, np.nan_to_num(np.linalg.inv(self.get_covmat())), self.der_spectra)
        self.fisher_per_mode=fisher_per_mode
        self.error_per_mode_non_marginalized = np.nan_to_num(np.diagonal(fisher_per_mode,axis1 = 1, axis2 = 2)**-0.5)
        
        self.fisher=np.sum(self.fisher_per_mode,axis=0)
        self.error_non_marginalized = np.diag(self.fisher)**-0.5 
        self.error_marginalized = np.linalg.inv(self.fisher)**0.5
        
        return self.error_marginalized[0]
    
#example use case
"""
spectra = {'kg': clkg[:cut], 'gg' :clgg[:cut], 'kk': clkk[:cut]}
der_spectra_alpha = np.ones((len(clkg[:cut]), len(spectra), Npars))
#Derivatives with respect to b
der_spectra_alpha[:, 0, 0] = clgk0[:cut]
der_spectra_alpha[:, 1, 0] = 2*bias*clgg0[:cut]
der_spectra_alpha[:, 2, 0] = np.zeros(cut)

a=Fisher(1,1,ellrange,spectra,der_spectra_alpha)
a.get_fisher()
"""
    
def compare_cleaning(clgg,clcibcib,clkk,clkg,clcibk,clcibg,bias,cut=500,fsky=1,num_spectra=2):
    clgg0=clgg/4
    clgk0=clkg/2
    ellrange=np.arange(cut)
    
    #calculate the base error
    spectra = {'kg': clkg[:cut], 'gg' :clgg[:cut], 'kk': clkk[:cut]}
    spectra_used = ['kg', 'gg']
    pars = ['b']#, 's'
    Npars = len(pars)
    der_spectra_alpha = np.ones((len(clkg[:cut]), num_spectra, Npars))
    #Derivatives with respect to b
    der_spectra_alpha[:, 0, 0] = clgk0[:cut]
    der_spectra_alpha[:, 1, 0] = 2*bias*clgg0[:cut]
    
    F_kg_gg_only = Forecaster(fsky, ellrange, ['k', 'g'], spectra, use_as_data =spectra_used)

    F_kg_gg_only.prepare_cov_for_error()
    err_kg_gg_only = F_kg_gg_only.make_fisher_and_err_bars(pars, der_spectra_alpha) 
    print(f'Base error is {err_kg_gg_only}')   
    
    #incorporate cib
    ## Now combine convergence and CIB
    spectra_cross = {'k': clkg[:cut], 'i': clcibg[:cut]}
    spectra_auto = {'kk': clkk[:cut], 'ii': clcibcib[:cut], 'ki': clcibk[:cut]}
    C = Combiner(ellrange, spectra_cross, spectra_auto)
    a = np.ones((len(ellrange), 2)) 
    all_x=-clcibk[:cut]/clcibcib[:cut]
    a = np.ones((len(ellrange), 2)) 
    a[:, 1] = all_x
    cross, auto = C.combine(a = a)
    cross = clkg #do not change clkg, as ideally this should not be affected by cleaning, in this case they are the same
    spectra = {'kg': cross[:cut], 'gg' :clgg[:cut], 'kk': auto[:cut]}
    pars = ['b']#, 's']
    Npars = len(pars)
    der_spectra_alpha = np.ones((len(clkg[:cut]), 3, Npars))
    #Derivatives with respect to b
    der_spectra_alpha[:, 0, 0] = cross[:cut]/bias
    der_spectra_alpha[:, 1, 0] = 2*bias*clgg0[:cut]
    der_spectra_alpha[:, 2, 0] = clkk[:cut]*np.zeros(len(clkk[:cut]))
    F=Forecaster(fsky,ellrange,['k','g'],spectra)
    F.prepare_cov_for_error()
    result=F.make_fisher_and_err_bars(pars,der_spectra_alpha)
    print(f'{100*(err_kg_gg_only[0][0]-result[0][0])/err_kg_gg_only[0][0]} % improvement with cleaning')
    return (err_kg_gg_only[0][0],result[0][0])



def get_corrcoef(clgg,clcibcib,clkk,clkg,clcibk,clcibg):
    rho={}
    rho_gk=np.sqrt(clkg**2/(clgg*clkk))
    rho_gcib=np.sqrt(clcibg**2/(clcibcib*clgg))
    rho_kcib=np.sqrt(clcibk**2/(clcibcib*clkk))
    rho['gk']=rho_gk
    rho['gcib']=rho_gcib
    rho['kcib']=rho_kcib
    return rho