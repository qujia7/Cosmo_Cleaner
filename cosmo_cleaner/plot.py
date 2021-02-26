import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as stats

ALPHA_1=1.52
Plot_Fact = 2


def get_gaussian_plot(par,parameters,fiducial,covariance,covariance_clean):
    """
        par (string): parameter name
        parameters (dict): dictionary mapping parameters to the corresponding index in the covatiance matrix
        covariance (array): covariance matrix
        covariance_clean (array): covariance of cleaned array
    
    Return
    ------
        tuple, ellipse a, b, angle in degrees
    """

    cov=covariance
    cov_n=covariance_clean
    error=(np.diag(cov)**0.5)[parameters[par]]
    error_n=(np.diag(cov_n)**0.5)[parameters[par]]
    mu = fiducial[par]
    sigma=error
    sigma_n=error_n
    x = np.linspace(mu - 1.2*sigma, mu + 1.2*sigma, 100)
    plt.vlines(mu,min(stats.norm.pdf(x, mu, sigma)),max(stats.norm.pdf(x, mu, sigma)),linestyles='dashdot',color='k',label='mean')
    plt.vlines(mu+sigma,min(stats.norm.pdf(x, mu, sigma)),max(stats.norm.pdf(x, mu, sigma)),linestyles='dashdot',color='k',label='$+ 1\sigma$')
    plt.vlines(mu-sigma,min(stats.norm.pdf(x, mu, sigma)),max(stats.norm.pdf(x, mu, sigma)),linestyles='dashdot',color='k',label='$-1\sigma$')
    plt.plot(x, stats.norm.pdf(x, mu, sigma),'-.',color='k')
    plt.vlines(mu+sigma_n,min(stats.norm.pdf(x, mu, sigma_n)),max(stats.norm.pdf(x, mu, sigma_n)),color='blue',linestyles='solid',label='$+ 1\sigma cleaned$')
    plt.vlines(mu-sigma_n,min(stats.norm.pdf(x, mu, sigma_n)),max(stats.norm.pdf(x, mu, sigma_n)),color='blue',linestyles='solid',label='$-1\sigma cleaned$')
    plt.plot(x, stats.norm.pdf(x, mu, sigma_n),'-',color='blue')
    plt.title(par)
    plt.legend()
    plt.show()  


def get_ellipse_params(par1,par2,parameters,covariance):
    """
        par1 (string): parameter 1 name
        par2 (string): parameter 2 name
        parameters (dict): dictionary mapping parameters to the corresponding index in the covatiance matrix
        covariance (array): covariance matrix
    
    Return
    ------
        tuple, ellipse a, b, angle in degrees
    """
    cov=covariance
    i=parameters[par1]
    j=parameters[par2]
    sigma_x2=cov[i][i]
    sigma_xy=cov[i][j]
    sigma_y2=cov[j][j]
    a2 = (sigma_x2+sigma_y2)/2. + np.sqrt((sigma_x2 - sigma_y2)**2/4. +
                                          sigma_xy**2)
    b2 = (sigma_x2+sigma_y2)/2. - np.sqrt((sigma_x2 - sigma_y2)**2/4. +
                                          sigma_xy**2)
    angle = np.arctan(2.*sigma_xy/(sigma_x2-sigma_y2)) / 2.   
    return np.sqrt(a2), np.sqrt(b2), angle * 180.0 / np.pi,np.sqrt(sigma_x2), np.sqrt(sigma_y2), sigma_xy


def plot_ellipse(ax, par1, par2, parameters, fiducial, cov,cov_improved,
                 one_sigma_only=False,
                 scale1=1, scale2=1,
                 kwargs1={'ls': '--','edgecolor': 'blue'},
                 kwargs2={'ls': '-'},
                 default_kwargs={'lw': 1, 'facecolor': 'none',
                                 'edgecolor': 'black'}):
    a,b,theta,sigma_x,sigma_y,sigma_xy=get_ellipse_params(par1,par2,parameters,cov)
    a_n,b_n,theta_n,sigma_xn,sigma_yn,sigma_xyn=get_ellipse_params(par1,par2,parameters,cov_improved)
    fid1 = fiducial[par1] * scale1
    fid2 = fiducial[par2] * scale2
    # use defaults plotting arguments
    kwargs1_temp = default_kwargs.copy()
    kwargs1_temp.update(kwargs1)
    kwargs1 = kwargs1_temp
    kwargs2_temp = default_kwargs.copy()
    kwargs2_temp.update(kwargs2)
    kwargs2 = kwargs2_temp
    e1 = Ellipse(
        xy=(fid1, fid2),
        width=a_n * 2 * ALPHA_1, height=b_n * 2 * ALPHA_1,
        angle=theta_n, **kwargs1)  
    e2 = Ellipse(
        xy=(fid1, fid2),
        width=a * 2 * ALPHA_1, height=b * 2 * ALPHA_1,
        angle=theta, **kwargs2)  
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax.add_artist(e1)
    ax.add_artist(e2)
    ax.set_ylabel(par2)
    ax.set_xlabel(par1)

    ax.plot(fid1, fid2,'kx')

    ax.add_patch(e2)
    e2.set(clip_box=ax.bbox,label='Original Fields 1sigma error')
    ax.add_patch(e1)
    e1.set(clip_box=ax.bbox,label='Cleaned Fields 1sigma error')
    ax.set_xlim(fid1 - Plot_Fact * sigma_x,fid1 + Plot_Fact * sigma_x)
    ax.set_ylim(fid2 - Plot_Fact * sigma_y,fid2 + Plot_Fact * sigma_y)
    ax.legend()
    return e2