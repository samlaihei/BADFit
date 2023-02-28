###################
# Version History #
###################
# V1 - Created, NOT cleaned, check (1+z) factors
#	 - Use at your own risk

###################
# Import Packages #
###################
import numpy as np
from scipy import interpolate
import pandas as pd
import os
import glob
import scipy.constants as con
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import csv
from uncertainties import ufloat
from uncertainties.umath import *
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps

# Spectra Modules
from specutils import Spectrum1D
from specutils.manipulation import (extract_region, box_smooth, gaussian_smooth, trapezoid_smooth, median_smooth)


## Modules for dust extintction 
from dust_extinction.parameter_averages import F19
from dust_extinction.parameter_averages import G16


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

#############
# Functions #
#############

def write_to_file(filename, dataheader, datarow):
	if os.path.exists(filename):
		with open(filename, 'a', newline='') as file:
			writer = csv.writer(file, delimiter=',')
			writer.writerow(datarow)
	else:
		with open(filename, 'w', newline='') as file:
			writer = csv.writer(file, delimiter=',')
			writer.writerow(dataheader)
			writer.writerow(datarow)
	return

		
def sort_ybyx(x, y):
    x, y = np.array(x), np.array(y)
    temp_y = np.array([i for _, i in sorted(zip(x, y))])
    temp_x = np.array(sorted(x))
    return [temp_x, temp_y]

def calc_power(redshift, lam, flux, eflux):
	dl = cosmo.luminosity_distance(redshift).to(u.cm)
	freq = con.c/lam*10**10
	power = np.array(flux) * 4 * np.pi * dl.value**2 * np.array(lam)*(1+redshift)
	epower = np.array(eflux) * 4 * np.pi * dl.value**2 * np.array(lam)*(1+redshift)
	return [freq, power, epower]
	
def calc_flux(redshift, freq, power, epower):
	dl = cosmo.luminosity_distance(redshift).to(u.cm)
	lam = con.c/(freq)*10**10
	flux = np.array(power)/(4 * np.pi * dl.value**2 * np.array(lam)*(1+redshift))
	eflux = np.array(epower)/(4 * np.pi * dl.value**2 * np.array(lam)*(1+redshift))
	return [lam, flux, eflux]
	

def eval_PL(p, xx): # [Norm, slope]
    f_pl = p[0] * (np.array(xx) / 3000.0) ** p[1]
    return f_pl


###############
# Fauxtometry #
###############

def Fe_flux_mask(template_wave, template_flux, fwhm, threshold):
    xtest = np.arange(np.nanmin(template_wave), np.nanmax(template_wave), 0.01)
    pix_avg = (template_wave+np.roll(template_wave,1))[1:]/2.
    pix_dispersion = np.median((template_wave-np.roll(template_wave,1))[1:]/pix_avg*con.c/1000.)
    if fwhm < 900.0:
        sig_conv = np.sqrt(910.0 ** 2 - 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))
    else:
        sig_conv = np.sqrt(fwhm ** 2- 900.0 ** 2) / 2. / np.sqrt(2. * np.log(2.))  # in km/s
    # Get sigma in pixel space
    sig_pix = sig_conv / pix_dispersion # km/s dispersion
    khalfsz = np.round(4 * sig_pix + 1, 0)
    xx = np.arange(0, khalfsz * 2, 1) - khalfsz
    kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
    kernel = kernel / np.sum(kernel)

    flux_Fe_conv = np.convolve(template_flux, kernel, 'same')
    tck = interpolate.splrep(template_wave, flux_Fe_conv)
    yval = 1. * interpolate.splev(xtest, tck)
    
    #quantile_val = np.quantile(yval, threshold)
    quantile_val = threshold*np.nanmedian(yval)

    mod_yval = yval-quantile_val
    spline = UnivariateSpline(xtest, mod_yval, s=0)
    Fe_masks = []
    spline_roots = sorted(np.append(np.array(spline.roots()), [np.nanmin(xtest), np.nanmax(xtest)]))
    for index, root1 in enumerate(spline_roots[1:]):
        root0 = spline_roots[index]
        if np.nanmax(mod_yval[np.logical_and(xtest > root0, xtest < root1)]) > 0.:
            Fe_masks.append([root0, root1])
    return Fe_masks

def line_masks(line_wavs, line_width_vel):
    masks = []
    for line_wav in line_wavs:
        linewidth_ang = line_width_vel/con.c*1000.*line_wav
        masks.append([line_wav-linewidth_ang, line_wav+linewidth_ang])
    return masks

def make_mask(lam, all_masks):
    data_mask = np.array([True for i in lam])
    Lyman_break_wav = 1600
    data_mask[lam < Lyman_break_wav] = False
    #data_mask[np.logical_and(lam > 2800, lam < 4000)] = False # MODIFIED 
    for masks in all_masks:
        for lims in masks:
            data_mask[np.logical_and(lam > lims[0], lam < lims[1])] = False
    return data_mask

def make_fauxtometry(lam, mask, flux, eflux):
    temp_array = []
    faux_lam, faux_flux, faux_eflux = [], [], []
    for temp_lam, temp_bool in zip(lam, mask):
        if temp_bool:
            temp_array.append(temp_lam)
        else:
            if len(temp_array) > 0:
                temp_lams = lam[np.logical_and(lam > np.nanmin(temp_array), lam < np.nanmax(temp_array))]
                temp_flux = flux[np.logical_and(lam > np.nanmin(temp_array), lam < np.nanmax(temp_array))]
                temp_eflux = eflux[np.logical_and(lam > np.nanmin(temp_array), lam < np.nanmax(temp_array))]
                faux_lam.append(np.average(temp_lams))
                faux_flux.append(np.average(temp_flux, weights=1/temp_eflux**2))
                faux_eflux.append(np.sqrt(np.sum(temp_eflux**2))/len(temp_eflux))
            temp_array = []
    return [np.array(faux_lam), np.array(faux_flux), np.array(faux_eflux)]



##############
# Extinction #
##############
def MW_extinction(redshift, Rv, Ebv, lams, flux, eflux): # lams input in rest frame
	ext = F19(Rv=Rv)
	defined_range = np.array([0.115, 3.3])*10**4*u.angstrom
	lams_obs = lams*(1+redshift)*u.angstrom
	flux_obs = flux[np.logical_and(lams_obs>defined_range[0], lams_obs<defined_range[1])]
	eflux_obs = eflux[np.logical_and(lams_obs>defined_range[0], lams_obs<defined_range[1])]
	lams_obs_ext = lams_obs[np.logical_and(lams_obs>defined_range[0], lams_obs<defined_range[1])]
	flux_noext = flux_obs/ext.extinguish(lams_obs_ext, Ebv=Ebv)
	eflux_noext = eflux_obs/ext.extinguish(lams_obs_ext, Ebv=Ebv)
	
	flux[np.logical_and(lams_obs>defined_range[0], lams_obs<defined_range[1])] = flux_noext
	eflux[np.logical_and(lams_obs>defined_range[0], lams_obs<defined_range[1])] = eflux_noext
	
	return [lams, flux, eflux]
	
def AGN_extinction(RvA, fA, Av, lams, flux, eflux): # lams input in rest frame
	ext = G16(RvA=RvA, fA=fA)
	defined_range = np.array([0.1, 3.3])*10**4 #in angstrom
	flux_obs = flux[np.logical_and(lams>defined_range[0], lams<defined_range[1])]
	eflux_obs = eflux[np.logical_and(lams>defined_range[0], lams<defined_range[1])]
	lams_obs = lams[np.logical_and(lams>defined_range[0], lams<defined_range[1])]*u.angstrom
	flux_noext = flux_obs/ext.extinguish(lams_obs, Av=Av)
	eflux_noext = eflux_obs/ext.extinguish(lams_obs, Av=Av)
	
	flux[np.logical_and(lams>defined_range[0], lams<defined_range[1])] = flux_noext
	eflux[np.logical_and(lams>defined_range[0], lams<defined_range[1])] = eflux_noext
	
	return [lams, flux, eflux]
	
def GC10_MEC_extinction(Av, lams, flux, eflux): # lams input in rest frame
    ext_file = 'G10_MEC.csv'
    pdata = pd.read_csv(ext_file)
    wavs = pdata['Wavelength'].to_numpy() # in angstrom
    ext = pdata['Extinction'].to_numpy()
    f = interpolate.interp1d(wavs, ext, fill_value='extrapolate')
    
    defined_range = np.array([0.1, 3.3])*10**4 #in angstrom
    flux_obs = flux[np.logical_and(lams>defined_range[0], lams<defined_range[1])]
    eflux_obs = eflux[np.logical_and(lams>defined_range[0], lams<defined_range[1])]
    lams_obs = lams[np.logical_and(lams>defined_range[0], lams<defined_range[1])]*u.angstrom
    axav = f(lams_obs)
    flux_noext = flux_obs/np.power(10.0, -0.4 * axav * Av)
    eflux_noext = eflux_obs/np.power(10.0, -0.4 * axav * Av)

    flux[np.logical_and(lams>defined_range[0], lams<defined_range[1])] = flux_noext
    eflux[np.logical_and(lams>defined_range[0], lams<defined_range[1])] = eflux_noext

    return [lams, flux, eflux]


redshift = 4.692
##############
# Photometry #
##############
datafile = 'data/example_orig.csv' 

pdata = pd.read_csv(datafile)

PHOT_freq = pdata['Frequency'].to_numpy()# Hz
PHOT_efreq = pdata['e_Frequency'].to_numpy()
PHOT_power = pdata['Power'].to_numpy() # erg/s
PHOT_epower = pdata['e_Power'].to_numpy()

PHOT_freq_OG = np.copy(PHOT_freq)
PHOT_lam, PHOT_flux, PHOT_eflux = calc_flux(redshift, PHOT_freq, PHOT_power, PHOT_epower)
PHOT_lam_OG, PHOT_flux_OG, PHOT_eflux_OG = np.copy(PHOT_lam), np.copy(PHOT_flux), np.copy(PHOT_eflux)

############
# Spectrum #
############

J2157_file = 'data/example_spectra.csv'

pdata = pd.read_csv(J2157_file)
SPEC_lam = pdata['Wavelength'].to_numpy()
SPEC_flux = pdata['Flux'].to_numpy()
SPEC_eflux = pdata['eFlux'].to_numpy()

SPEC_1D = Spectrum1D(spectral_axis=SPEC_lam*u.angstrom, flux=SPEC_flux * u.Jy)
eSPEC_1D = Spectrum1D(spectral_axis=SPEC_lam*u.angstrom, flux=SPEC_eflux * u.Jy)
SPEC_1Dsmooth = median_smooth(SPEC_1D, width=9)
eSPEC_1Dsmooth = median_smooth(eSPEC_1D, width=9)

SPEC_freq, SPEC_power, SPEC_epower = calc_power(redshift, SPEC_1Dsmooth.spectral_axis.value, SPEC_1Dsmooth.flux.value, eSPEC_1Dsmooth.flux.value)

SPEC_freq_OG = np.copy(SPEC_freq)
SPEC_lam_OG = np.copy(SPEC_lam)


#########################
# Extinction Correction #
#########################

MW_Rv = 3.1
MW_Ebv = 0.013

AGN_RvA = 2.7
AGN_fA = 0.2
AGN_Ebv = 0.00
AGN_Av = AGN_Ebv * AGN_RvA

PHOT_lam, PHOT_flux, PHOT_eflux = MW_extinction(redshift, MW_Rv, MW_Ebv, PHOT_lam, PHOT_flux, PHOT_eflux)
SPEC_lam, SPEC_flux, SPEC_eflux = MW_extinction(redshift, MW_Rv, MW_Ebv, SPEC_lam, SPEC_flux, SPEC_eflux)

#PHOT_lam, PHOT_flux, PHOT_eflux = AGN_extinction(AGN_RvA, AGN_fA, AGN_Av, PHOT_lam, PHOT_flux, PHOT_eflux)
#SPEC_lam, SPEC_flux, SPEC_eflux = AGN_extinction(AGN_RvA, AGN_fA, AGN_Av, SPEC_lam, SPEC_flux, SPEC_eflux)

PHOT_lam, PHOT_flux, PHOT_eflux = GC10_MEC_extinction(AGN_Av, PHOT_lam, PHOT_flux, PHOT_eflux)
SPEC_lam, SPEC_flux, SPEC_eflux = GC10_MEC_extinction(AGN_Av, SPEC_lam, SPEC_flux, SPEC_eflux)

PHOT_freq, PHOT_power, PHOT_epower = calc_power(redshift, PHOT_lam, PHOT_flux, PHOT_eflux)
SPEC_freq, SPEC_power, SPEC_epower = calc_power(redshift, SPEC_lam, SPEC_flux, SPEC_eflux)



######################
# Fitting Thresholds #
######################
freq_low = 3.0E14
freq_high = 1.91E15

PHOT_freq = PHOT_freq[np.logical_and(PHOT_freq_OG > freq_low, PHOT_freq_OG < freq_high)]
PHOT_efreq = PHOT_efreq[np.logical_and(PHOT_freq_OG > freq_low, PHOT_freq_OG < freq_high)]
PHOT_power = PHOT_power[np.logical_and(PHOT_freq_OG > freq_low, PHOT_freq_OG < freq_high)]
PHOT_epower = PHOT_epower[np.logical_and(PHOT_freq_OG > freq_low, PHOT_freq_OG < freq_high)]
PHOT_lam, PHOT_flux, PHOT_eflux = calc_flux(redshift, PHOT_freq, PHOT_power, PHOT_epower)

SPEC_freq = SPEC_freq[np.logical_and(SPEC_freq_OG > freq_low, SPEC_freq_OG < freq_high)]
SPEC_power = SPEC_power[np.logical_and(SPEC_freq_OG > freq_low, SPEC_freq_OG < freq_high)]
SPEC_epower = SPEC_epower[np.logical_and(SPEC_freq_OG > freq_low, SPEC_freq_OG < freq_high)]
SPEC_lam, SPEC_flux, SPEC_eflux = calc_flux(redshift, SPEC_freq, SPEC_power, SPEC_epower)


###############
# Fauxtometry #
###############

# Iron Template Mask #
MgII_FWHM = 4500.
FWHM_factor = 3.
Fe_temp_threshold = 0.7 # 0.5 for VW01 is default, 0.7 for BV08

VW01 = np.genfromtxt('Fe_Templates/fe_uv_VW01.txt')
VW01[:,0], VW01[:,1] = 10**VW01[:,0], VW01[:,1]*10**15

BG92 = np.genfromtxt('Fe_Templates/fe_optical_BG92.txt')
BG92[:,0], BG92[:,1] = 10**BG92[:,0], BG92[:,1]*10**15

BV08 = np.genfromtxt('Fe_Templates/BruhweilerVerner2008/d11-m20-21-735.txt')
BV08[:,0], BV08[:,1] = BV08[:,0], BV08[:,1]*10**(-6) # also defined for optical

#Fe_masks = np.array(list(Fe_flux_mask(VW01[:,0], VW01[:,1], MgII_FWHM, Fe_temp_threshold)) + 
					#list(Fe_flux_mask(BG92[:,0], BG92[:,1], MgII_FWHM, Fe_temp_threshold)))
Fe_masks = np.array(Fe_flux_mask(BV08[:,0], BV08[:,1], MgII_FWHM, Fe_temp_threshold))


# Emission Lines Mask #
SDSS_linefile = 'SDSS_emission_lines.txt'
with open(SDSS_linefile,'r') as f:
	temp=[x.strip().split('\t') for x in f]
	line_wav, gal_weight, qso_weight, line_name = np.transpose(temp)
	line_wav = np.array([float(i) for i in line_wav])
	gal_weight = np.array([float(i) for i in gal_weight])
	qso_weight = np.array([float(i) for i in qso_weight])

linewidth_vel = FWHM_factor * MgII_FWHM #km/s
Emission_masks = line_masks(line_wav[qso_weight > 0], linewidth_vel)

all_masks = [Fe_masks, Emission_masks]
data_mask = make_mask(SPEC_lam, all_masks)

FAUX_lam, FAUX_flux, FAUX_eflux = make_fauxtometry(SPEC_lam, data_mask, SPEC_flux, SPEC_eflux)
FAUX_freq, FAUX_power, FAUX_epower = calc_power(redshift, FAUX_lam, FAUX_flux, FAUX_eflux)

#########################
# Photometry Correction #
#########################
Selsing_template = 'Templates/Selsing_Xshooter.txt'
Selsing_lam, Selsing_flux, Selsing_eflux = np.loadtxt(Selsing_template, unpack=True) 
lamS, (specS, e_specS) = [Selsing_lam, np.array([Selsing_flux, 0.01*Selsing_flux])*10**-16]
lamS = lamS[specS != 0]
e_specS = e_specS[specS != 0]
specS = specS[specS != 0]

Selsing_PL_params = np.array([3.75E-16, -1.7])
new_PL_params = np.array([3.75E-16, -1.1])

if False:
	specS = specS - eval_PL(Selsing_PL_params, lamS) # !! Modifies Selsing template
	specS = specS + eval_PL(new_PL_params, lamS) # !! Modifies Selsing template

filter_files = glob.glob('filters/*.txt')
filter_avglam = []
mod_PHOT_power = PHOT_power[np.logical_or(PHOT_lam < np.nanmin(SPEC_lam_OG), PHOT_lam > np.nanmax(SPEC_lam_OG))]
mod_PHOT_lam = PHOT_lam[np.logical_or(PHOT_lam < np.nanmin(SPEC_lam_OG), PHOT_lam > np.nanmax(SPEC_lam_OG))]
for filter_file in filter_files:
	lamF,filt = np.loadtxt(filter_file, unpack=True)
	lamF = lamF/(1+redshift)
	filter_avglam.append(np.average(lamF, weights=filt))
filter_avglam = np.array(filter_avglam)


CORR_freq, CORR_power, CORR_epower = np.array([]), np.array([]), np.array([])
for temp_lam, temp_pow in zip(mod_PHOT_lam, mod_PHOT_power):
	filt_file = filter_files[np.argmin(np.abs(filter_avglam-temp_lam))]
	lamF,filt = np.loadtxt(filt_file, unpack=True)
	lamF = lamF/(1+redshift)

	dl = cosmo.luminosity_distance(redshift).to(u.cm)
	wav = np.average(lamF, weights=filt)
	freq = con.c/(wav*10**(-10))
	spec_interp = np.interp(lamF, lamS, specS)
	I1        = simps(spec_interp*filt*lamF,lamF)                     
	I2        = simps( filt/lamF,lamF)
	fnu       = I1/I2 / (con.c*10**10)
	lum = fnu * 4 * np.pi * dl.value**2 * freq * (1+redshift) # <-- What

	temp_specS = (specS*temp_pow/lum)[np.logical_and(lamS > np.min(lamF), lamS < np.max(lamF))]
	temp_especS = (e_specS*temp_pow/lum)[np.logical_and(lamS > np.min(lamF), lamS < np.max(lamF))]
	temp_lamS = lamS[np.logical_and(lamS > np.min(lamF), lamS < np.max(lamF))]
	temp_freqS, temp_powS, temp_epowS = calc_power(redshift, temp_lamS, temp_specS, temp_especS)

	all_masks = [Fe_masks, Emission_masks] # same masks defined for fauxtometry
	data_mask = make_mask(temp_lamS, all_masks)

	temp_lam, temp_flux, temp_eflux = make_fauxtometry(temp_lamS, data_mask, temp_specS, temp_especS)
	temp_freq, temp_power, temp_epower = calc_power(redshift, temp_lam, temp_flux, temp_eflux)
	if len(temp_freq) > 0:
		CORR_freq = np.append(CORR_freq, temp_freq)
		CORR_power = np.append(CORR_power, temp_power)
		CORR_epower = np.append(CORR_epower, temp_epower)


##############
# Final Data #
##############

PHOT_FAUX_freq = np.concatenate((PHOT_freq, FAUX_freq))
_, PHOT_FAUX_power = sort_ybyx(PHOT_FAUX_freq, np.concatenate((PHOT_power, FAUX_power)))
PHOT_FAUX_freq, PHOT_FAUX_epower = sort_ybyx(PHOT_FAUX_freq, np.concatenate((PHOT_epower, FAUX_epower)))

CORR_FAUX_freq = np.concatenate((CORR_freq, FAUX_freq))
_, CORR_FAUX_power = sort_ybyx(CORR_FAUX_freq, np.concatenate((CORR_power, FAUX_power)))
CORR_FAUX_freq, CORR_FAUX_epower = sort_ybyx(CORR_FAUX_freq, np.concatenate((CORR_epower, FAUX_epower)))


# Setup Data #
data = [CORR_FAUX_freq, CORR_FAUX_power, CORR_FAUX_epower] # Combined corrected photometry and fauxtometry



pdata = pd.DataFrame()
pdata['Freq'] = data[0]
pdata['Power'] = data[1]
pdata['ePower'] = data[2]
pdata.to_csv('data/test.csv',index=False)










    
    