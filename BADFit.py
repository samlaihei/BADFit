# BADFit: Code for fitting black hole accretion disks with Sherpa/XSpec
# Author: Samuel Lai, ANU
# Email: samuel.lai (AT) anu (DOT) edu (DOT) au

# Version 1.0
# 2023-02-28

###################
# Version History #
###################
# V1.0 - Forked from AD_Fitting_v1, basic functionality established
# 	
#
#

#########
# To Do #
#########
# AGNSED and AGNSLIM
#
#

#######################################
#  ____          _____  ______ _ _    #
# |  _ \   /\   |  __ \|  ____(_) |   #
# | |_) | /  \  | |  | | |__   _| |_  #
# |  _ < / /\ \ | |  | |  __| | | __| #
# | |_) / ____ \| |__| | |    | | |_  #
# |____/_/    \_\_____/|_|    |_|\__| #
#                                     #
#######################################                                    

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
from uncertainties import ufloat
from uncertainties.umath import *
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps
from sklearn.neighbors import KernelDensity
from astropy.coordinates import SkyCoord
import h5py

# Spectra Modules
from specutils import Spectrum1D
from specutils.manipulation import median_smooth


# Modules for plots
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
rcParams['font.family'] = 'serif'


# Modules for MCMC
import emcee
import corner
from chainconsumer import ChainConsumer # Chainconsumer analysis

# Modules for Sherpa and XSPEC
import sherpa.astro.xspec as xspec

# Modules for dust extintction 
from dustmaps.sfd import SFDQuery
from dust_extinction.parameter_averages import F19
from dust_extinction.parameter_averages import G16


class BADFit():

	#################
	# Main Routines #
	#################
	
	def __init__(self, name, modelChoice, lam, flux, eflux, z, 
				 freq=[], power=[], epower=[],
				 ra=-999, dec=-999, MW_Ebv=0, AGN_Ebv=0):
		"""
		Get input data

		Parameters:
		-----------
		name: string
			name of the output files, same name will overwrite file
			
		modelChoice: string
			choose from ['KERRBB', 'SLIMBH']
			
		lam: 1-D array
			wavelength in unit of Angstrom in rest-frame
 
		flux: 1-D array
			flux density in unit of erg/s/cm^2/Hz

		eflux: 1-D array
			 1 sigma err with the same unit of flux
			 
		freq, power, epower: 1-D arrays
			rest-frame in units of Hz and erg/s, overrides lam, flux, eflux
 
		z: float number
			redshift

		ra, dec: float number, optional 
			the location of the source in degrees, right ascension and declination, used to determine MW_Ebv
			
		MW_Ebv, AGN_Ebv: float numbers, optional 
			Galactic and host galaxy extinction, respectively

		"""
		self.name = name
		self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
		self.modelChoice = modelChoice 
		self.inputLam = np.asarray(lam, dtype=np.float64)
		self.inputFlux = np.asarray(flux, dtype=np.float64)
		self.inputFluxError = np.asarray(eflux, dtype=np.float64)
		self.z = z
		self.ra = ra
		self.dec = dec

		if len(freq) > 0 and all(len(x) == len(freq) for x in [power, epower]):
			self.inputData = np.array([freq, power, epower])
			self.data = np.array([freq, power, epower])
		else:
			self.inputData = self.calcPower(lam, flux, eflux)
			self.data = self.calcPower(lam, flux, eflux)
			
		self.returnModel(self.modelChoice)
		self.MW_Ebv = MW_Ebv
		self.AGN_Ebv = AGN_Ebv

		

	def runMCMC(self, nwalkers=124, niter=256, forceData=False,
				fitWindow=[3.E14, 1.91E15], 
				kde_bandwidth = 0.75, errorFloor = 0.05, errorFactor = 0.05,
				priorMassFunction=False, priorMbh=-999, priorMbhSigma=0.5):
				
		"""
		Run MCMC main routine to produce parameter posteriors

		Parameters:
		-----------
		nwalkers, niter: integers
			Number of walkers and iterations for emcee
			
		forceData: bool
			Use data as is, without extinction or other manipulation

		MW_Ebv: float number
			 Milky Way Galactic extinction IF ra and dec not provided
 
		AGN_Ebv: float number
			AGN host dust extinction

		fitWindow: 1D array, length 2 
			Frequency minimum and maximum for fitting
			
		kde_bandwidth, errorFloor, errorFactor: float numbers
			Modifying the error on datapoints based on proximity
			
		priorMassFunction: bool
			Toggle to use mass function prior
			
		priorMbh, priorMbhSigma: float
			Black hole mass prior and its standard deviation
			
		"""
		self.priorMassFunction = priorMassFunction
		self.priorMbh = priorMbh
		self.priorMbhSigma = priorMbhSigma
		
		if self.priorMassFunction:
			#######################
			# Mass Function Prior #
			#######################
			prior_file = 'prior/mass_function.csv'
			pdata = pd.read_csv(prior_file)
			self.massFunctionMbh = pdata['Mbh'].to_numpy()
			self.massFunctionLnLike = pdata['logLikelihood'].to_numpy()
		
		#####################
		# Data Manipulation #
		#####################
		if forceData:
			self.data = self.inputData
			self.data = self.adjustUncertaintyKDE(self.data, kde_bandwidth, errorFloor, errorFactor)
		else:
			self.data = self.mainDataRoutine(self.inputData, fitWindow, kde_bandwidth, errorFloor, errorFactor)

		##############
		# MCMC Model #
		##############

		# Random init walker position #
		p0 = []
		for i in range(nwalkers):
			temp = []
			for index, par in enumerate(self.parinfo):
				if not par['fixed']:
					current_limits = par['limits']
					spacing = np.linspace(current_limits[0], current_limits[1], nwalkers)
					temp.append(spacing[np.random.randint(0, len(spacing))])
			p0.append(np.array(temp))

		# Main mcmc routine #
		sampler, pos, prob, state = self.mcmcMain(p0, nwalkers, niter, self.ndim, self.likelihood, self.data)

		# Main plotting routine
		return (self.createPlot(sampler.flatchain, sampler.flatlnprobability, self.data, self.z))	


	def h5ReadFile(self, forceData=False, fitWindow=[3.E14, 1.91E15], 
				   kde_bandwidth = 0.75, errorFloor = 0.05, errorFactor = 0.05,
				   createPlot=True, returnMbh=False, returnArrays=False):
		if forceData:
			self.data = self.inputData
		else:
			self.data = self.mainDataRoutine(self.inputData, fitWindow, kde_bandwidth, errorFloor, errorFactor)

		chain_filename = 'output/'+self.name+'.h5'

		with h5py.File(chain_filename, "r") as f:
			samples = np.array(f['mcmc']['chain'])
			likelihoods = np.array(f['mcmc']['log_prob'])
			niter, nwalkers, _ = samples.shape
			ndim = len(samples[0][0])
			samples = np.ndarray.flatten(samples)
			samples = samples.reshape(int(len(samples)/ndim), ndim)
			samples = samples[:int(nwalkers*niter)]
			
			c = ChainConsumer().add_chain(samples, parameters=self.par_labels)
			summary = c.analysis.get_summary()
			print(c.analysis.get_summary())

			if 'log(M$_{\\rm{BH}}$/M$_{\\odot}$)' in list(summary.keys()):
				current_lo, current_med, current_hi = summary['log(M$_{\\rm{BH}}$/M$_{\\odot}$)']    
				if current_lo == None:
					ind = list(summary.keys()).index('log(M$_{\\rm{BH}}$/M$_{\\odot}$)')
					current_med = np.median(np.transpose(samples)[ind]) # Median
					current_lo = np.quantile(np.transpose(samples)[ind], 0.16) # 16th percentile
					current_hi = np.quantile(np.transpose(samples)[ind], 0.84) # 84th percentile
		
		# Return max likelihood
		if returnMbh:
		    return [current_lo, current_med, current_hi]
		elif returnArrays:
		    return samples, likelihoods
		elif createPlot:
		    return (self.createPlot(samples, likelihoods, self.data, self.z))




	# -------------------------------------------------------------- #
	

	#############
	# Functions #
	#############

	def sortYByX(self, x, y):
		x, y = np.array(x), np.array(y)
		temp_y = np.array([i for _, i in sorted(zip(x, y))])
		temp_x = np.array(sorted(x))
		return [temp_x, temp_y]

	def calcPower(self, lam, flux, eflux):
		dl = self.cosmo.luminosity_distance(self.z).to(u.cm)
		freq = con.c/lam*10**10
		power = np.array(flux) * 4 * np.pi * dl.value**2 * np.array(freq)/(1+self.z)
		epower = np.array(eflux) * 4 * np.pi * dl.value**2 * np.array(freq)/(1+self.z)
		_, power = self.sortYByX(freq, power)
		freq, epower = self.sortYByX(freq, epower)
		return [freq, power, epower]
	
	def calcFlux(self, freq, power, epower):
		dl = self.cosmo.luminosity_distance(self.z).to(u.cm)
		lam = con.c/(freq)*10**10
		flux = np.array(power)/(4 * np.pi * dl.value**2 * np.array(freq)/(1+self.z))
		eflux = np.array(epower)/(4 * np.pi * dl.value**2 * np.array(freq)/(1+self.z))
		_, flux = self.sortYByX(lam, flux)
		lam, eflux = self.sortYByX(lam, eflux)
		return [lam, flux, eflux]
	
	def adjustUncertaintyKDE(self, data, kdeBandwidth, errorFloor, errorFactor):
		# Modified Uncertainty #    ! Using Gaussian kernel density estimation !
		kde_freq = [[np.log10(i)] for i in data[0]]
		kde = KernelDensity(kernel="gaussian", bandwidth=kdeBandwidth).fit(kde_freq)
		inv_dens = 1./np.exp(kde.score_samples(kde_freq))
		errorFactors = errorFactor/(inv_dens/np.min(inv_dens))
		data[2] = np.sqrt(data[2]**2 + (errorFactors*data[1])**2 + errorFloor**2) # Additional uncertainty added in quadrature
		return data
	
	def extinctionCorrection(self, data, MW_Ebv, AGN_Ebv):
		# Extinction Correction #
		MW_Rv = 3.1

		AGN_RvA = 2.7
		AGN_fA = 0.2
		AGN_Av = AGN_Ebv * AGN_RvA
		
	
		lam, flux, eflux = self.calcFlux(data[0], data[1], data[2])
	
		lam, flux, eflux = self.extinction_MW(MW_Rv, MW_Ebv, lam, flux, eflux)
		#lam, flux, eflux = self.extinction_AGN(AGN_RvA, AGN_fA, AGN_Av, lam, flux, eflux)
		lam, flux, eflux = self.extinction_GC10_MEC(AGN_Av, lam, flux, eflux)
	
		data = self.calcPower(lam, flux, eflux)
	
		return data
	
	def fitThresholds(self, data, thresholdMin, thresholdMax):
		# Apply cutoffs to data #
		freq, power, epower = data
		power = power[np.logical_and(freq > thresholdMin, freq < thresholdMax)]
		epower = epower[np.logical_and(freq > thresholdMin, freq < thresholdMax)]
		freq = freq[np.logical_and(freq > thresholdMin, freq < thresholdMax)]
		return [freq, power, epower]
		
		
	def mainDataRoutine(self, data, fitWindow=[3.E14, 1.91E15], 
						kde_bandwidth = 0.75, errorFloor = 0.05, errorFactor = 0.05):
		#########################
		# Extinction Correction #
		#########################
		if self.ra > 0  and np.abs(self.dec) < 90:
			sfd = SFDQuery()
			coord = SkyCoord(self.ra, self.dec, frame='icrs', unit=(u.deg, u.deg))
			self.MW_Ebv = sfd(coord) * 0.86 # SFD map with Schlegal 14% recalibration
		data = self.extinctionCorrection(data, self.MW_Ebv, self.AGN_Ebv)

		######################
		# Fitting Thresholds #
		######################
		freq_low, freq_high = fitWindow
		data = self.fitThresholds(data, freq_low, freq_high)

		##############
		# Final Data #
		##############

		# Setup Data #
		data = self.adjustUncertaintyKDE(data, kde_bandwidth, errorFloor, errorFactor)
		
		return data
		

	##################
	# MCMC Functions #
	##################

	def returnModel(self, Choice):

		############
		# Controls #
		############
		# Mbh parameter is required in all models #
		
		## General ##
		redshift = self.z
		prop_dist = self.cosmo.comoving_distance(redshift).to(u.Mpc).value

		mbh = 9.5 # log10(Msun)
		mdot = 90. # in Msun/yr
		edd_rat = 0.6 # Eddington ratio
		a, theta = 0.6, 0.8 # spin, cos(i)
		
		kTe_hot, kTe_warm = 100., 0.2
		Gamma_hot, Gamma_warm = 1.7, 2.7
		R_hot, R_warm, R_out, R_in = 10., 20., 2., -1.
		Htmax = 10.
		

		eta = 0 # disk power from torque over accretion power
		alpha = 0.01 # alpha-viscosity
		fcol = 1. # hardening factor

		rflag, lflag, vflag = 1, 1, 1 # self-irradiation, limb darkening, consider height profile

		norm = 1

		#######################
		## KerrBB Parameters ##
		#######################
		# eta - ratio of disk power from torque over disk (0 <= eta <= 1)
		# a - black hole spin (-1 < a < 1)
		# i - inclination in deg (0 <= i < 85), 0 for face-on
		# Mbh - black hole mass in Msuns
		# Mdd - black hole accretion rate in 10^18 g/s
		# z - redshift
		# fcol - spectral hardening factor Tcol/Teff, 1 for Campitiello
		# rflag - switch for self-irradiation, keep on at 1
		# lflag - switch for limb darkening, keep on at10
		# norm - normalisation

		# Changed to log10(Mbh), dl to z, cos(theta), mdot to Msun/yr. log10(Mdot)
		kerrbb_parinfo = [{'fixed': True, 'limits':(0, 1), 'label':'eta', 'units':''}, # eta
						{'fixed': False, 'limits':(-0.99, 0.99), 'label':'$a$', 'units':'$\\frac{J\,c}{GM^2}$'}, # a
						{'fixed': False, 'limits':(0.4, 1), 'label':'cos$(\\theta_{\\rm{inc}})$', 'units':''}, # cos(i)
						{'fixed': False, 'limits':(8, 11), 'label':'log(M$_{\\rm{BH}}$/M$_{\\odot}$)', 'units':''}, # Mbh
						{'fixed': False, 'limits':(-3, 3), 'label':'log($\\dot{\\rm{M}}$/M$_{\\odot}$ yr$^{-1}$)', 'units':''}, # Mdd
						{'fixed': True, 'limits':(redshift, redshift), 'label':'$z$', 'units':''}, # redshift
						{'fixed': True, 'limits':(0.5, 2.7), 'label':'fcol', 'units':''}, # fcol
						{'fixed': True, 'limits':(0, 1), 'label':'rflag', 'units':''}, # rflag
						{'fixed': True, 'limits':(0, 1), 'label':'lflag', 'units':''}, # lflag
						{'fixed': True, 'limits':(0, 1), 'label':'norm', 'units':''}] # norm
			
		kerrbb_init_params = [eta, a, theta, mbh, mdot, redshift, fcol, rflag, lflag, norm]
		kerrbb_model = xspec.XSkerrbb()
		kerrbb_setup = [kerrbb_parinfo, kerrbb_init_params, kerrbb_model, self.kerrbb_evalModel]
	
		#######################
		## SLIMBH Parameters ##
		#######################
		# Mbh - black hole mass in Msuns
		# a - black hole spin (0 < a < 1)
		# lumin - disk luminosity in Eddington units, (0.05 < lumin < 1.2)
		# alpha - alpha viscosity
		# i - inclination in deg (0 <= i < 85), 0 for face-on
		# z - redshift
		# fcol - spectral hardening factor Tcol/Teff, 1 for Campitiello
		# lflag - switch for limb darkening, keep on at 1
		# vflag - switch for height profile, keep on at 1
		# norm - normalisation

		# Changed to log10(Mbh), dl to z, cos(theta)
		slimbh_parinfo = [{'fixed': False, 'limits':(7, 11), 'label':'log(M$_{\\rm{BH}}$/M$_{\\odot}$)', 'units': ''}, # Mbh
						{'fixed': False, 'limits':(0, 0.99), 'label':'$a$', 'units':'$\\frac{J\,c}{GM^2}$'}, # a
						{'fixed': False, 'limits':(0.05, 1.25), 'label':'L$_{\\rm{disc}}$', 'units':'L$_{\\rm{Edd}}$'}, # lumin
						{'fixed': True, 'limits':(0.005, 0.1), 'label':'$\alpha$', 'units':''}, # alpha
						{'fixed': False, 'limits':(0.4, 1), 'label':'cos$(\\theta_{\\rm{inc}})$', 'units':''}, # cos(i)
						{'fixed': True, 'limits':(redshift, redshift), 'label':'$z$', 'units':''}, # redshift
						{'fixed': True, 'limits':(-1, 2.7), 'label':'fcol', 'units':''}, # fcol
						{'fixed': True, 'limits':(0, 1), 'label':'lflag', 'units':''}, # lflag
						{'fixed': True, 'limits':(0, 1), 'label':'vflag', 'units':''}, # vflag
						{'fixed': True, 'limits':(0, 1), 'label':'norm', 'units':''}] # norm
			
		slimbh_init_params = [mbh, a, edd_rat, alpha, theta, redshift, fcol, lflag, vflag, norm]
		slimbh_model = xspec.XSslimbh()
		slimbh_setup = [slimbh_parinfo, slimbh_init_params, slimbh_model, self.slimbh_evalModel]
	
		
		#######################
		## AGNSED Parameters ##
		#######################
		# Mbh - black hole mass in Msuns
		# dist - comoving (proper) distance in Mpc
		# logmdot - log10 of mass accretion rate in Eddington units
		# a - black hole spin (-1 < a < 0.998)
		# cosi - cosine of the inclination angle i for the warm Comptonising component and the outer disc
		# kTe_hot - electron temperature for the hot Comptonisation component in keV
		# kTe_warm - electron temperature for the warm Comptonisation component in keV
		# Gamma_hot - spectral index of the hot Comptonisation component
		# Gamma_warm - spectral index of the warm Comptonisation component
		# R_hot - outer radius of the hot Comptonisation component in Rg
		# R_warm - outer radius of the warm Comptonisation component in Rg
		# logRout - log of the outer radius of the disc in units of Rg
		# Htmax -  upper limit of the scale height for the hot Comptonisation component in Rg
		# reprocess - whether reprocessing is considered, keep on at 1
		# z - redshift
		# norm - normalisation
		
		
		agnsed_parinfo = [{'fixed': False, 'limits':(8, 11), 'label':'log(M$_{\\rm{BH}}$/M$_{\\odot}$)', 'units': ''}, # Mbh
						 {'fixed': True, 'limits':(0.01, 1E9), 'label':'dist', 'units':'Mpc'}, # dist
						 {'fixed': False, 'limits':(-10, 2), 'label':'logmdot', 'units':''}, # log mdot
						 {'fixed': False, 'limits':(-1, 0.998), 'label':'$a$', 'units':'$\\frac{J\,c}{GM^2}$'}, # a
						 {'fixed': False, 'limits':(0.05, 1.0), 'label':'cos$(\\theta_{\\rm{inc}})$', 'units':''}, # cos(i)
						 {'fixed': True, 'limits':(10, 300), 'label':'kTe_hot', 'units':'keV'}, # kte_hot
						 {'fixed': True, 'limits':(0.1, 0.5), 'label':'kTe_warm', 'units':'keV'}, # kte_warm
						 {'fixed': True, 'limits':(1.3, 3), 'label':'Gamma_hot', 'units':''}, # Gamma_hot
						 {'fixed': True, 'limits':(2, 10), 'label':'Gamma_warm', 'units':''}, # Gamma_warm
						 {'fixed': True, 'limits':(6, 500), 'label':'R_hot', 'units':'R$_{\\rm{g}}$'}, # R_hot
						 {'fixed': True, 'limits':(6, 500), 'label':'R_warm', 'units':'R$_{\\rm{g}}$'}, # R_warm
						 {'fixed': True, 'limits':(-3, 7), 'label':'log(R$_{\\rm{out}}$/R$_{\\rm{g}}$)', 'units':''}, # logRout
						 {'fixed': True, 'limits':(6, 10), 'label':'Htmax', 'units':'R$_{\\rm{g}}$'}, # Htmax
						 {'fixed': True, 'limits':(0, 1), 'label':'reprocess', 'units':''}, # reprocess
						 {'fixed': True, 'limits':(redshift, redshift), 'label':'redshift', 'units':''}, # redshift
						 {'fixed': True, 'limits':(0, 1), 'label':'norm', 'units':''}] # norm
		
		agnsed_init_params = [mbh, prop_dist, np.log10(edd_rat), a, theta, kTe_hot, kTe_warm, Gamma_hot, Gamma_warm, R_hot, R_warm, R_out, Htmax, rflag, redshift, norm]
		agnsed_model = xspec.XSagnsed()
		agnsed_setup = [agnsed_parinfo, agnsed_init_params, agnsed_model, self.agnsed_evalModel]
		
		########################
		## AGNSLIM Parameters ##
		########################
		# Mbh - black hole mass in Msuns
		# dist - comoving (proper) distance in Mpc
		# logmdot - log10 of mass accretion rate in Eddington units
		# a - black hole spin (-1 < a < 0.998)
		# cosi - cosine of the inclination angle i for the warm Comptonising component and the outer disc
		# kTe_hot - electron temperature for the hot Comptonisation component in keV
		# kTe_warm - electron temperature for the warm Comptonisation component in keV
		# Gamma_hot - spectral index of the hot Comptonisation component
		# Gamma_warm - spectral index of the warm Comptonisation component
		# R_hot - outer radius of the hot Comptonisation component in Rg
		# R_warm - outer radius of the warm Comptonisation component in Rg
		# logRout - log of the outer radius of the disc in units of Rg
		# Rin - inner radius of the disc in Rg
		# z - redshift
		# norm - normalisation
		
		
		agnslim_parinfo = [{'fixed': False, 'limits':(8, 11), 'label':'log(M$_{\\rm{BH}}$/M$_{\\odot}$)', 'units': ''}, # Mbh
						 {'fixed': True, 'limits':(0.01, 1E9), 'label':'dist', 'units':'Mpc'}, # dist
						 {'fixed': False, 'limits':(-10, 2), 'label':'logmdot', 'units':''}, # log mdot
						 {'fixed': False, 'limits':(-1, 0.998), 'label':'$a$', 'units':'$\\frac{J\,c}{GM^2}$'}, # a
						 {'fixed': False, 'limits':(0.05, 1.0), 'label':'cos$(\\theta_{\\rm{inc}})$', 'units':''}, # cos(i)
						 {'fixed': True, 'limits':(10, 300), 'label':'kTe_hot', 'units':'keV'}, # kte_hot
						 {'fixed': True, 'limits':(0.1, 0.5), 'label':'kTe_warm', 'units':'keV'}, # kte_warm
						 {'fixed': True, 'limits':(1.3, 3), 'label':'Gamma_hot', 'units':''}, # Gamma_hot
						 {'fixed': True, 'limits':(2, 10), 'label':'Gamma_warm', 'units':''}, # Gamma_warm
						 {'fixed': True, 'limits':(6, 500), 'label':'R_hot', 'units':'R$_{\\rm{g}}$'}, # R_hot
						 {'fixed': True, 'limits':(6, 500), 'label':'R_warm', 'units':'R$_{\\rm{g}}$'}, # R_warm
						 {'fixed': True, 'limits':(-3, 7), 'label':'log(R$_{\\rm{out}}$/R$_{\\rm{g}}$)', 'units':''}, # logRout
						 {'fixed': True, 'limits':(-1, 100), 'label':'Rin', 'units':'R$_{\\rm{g}}$'}, # Rin
						 {'fixed': True, 'limits':(redshift, redshift), 'label':'redshift', 'units':''}, # redshift
						 {'fixed': True, 'limits':(0, 1), 'label':'norm', 'units':''}] # norm
		
		agnslim_init_params = [mbh, prop_dist, np.log10(edd_rat), a, theta, kTe_hot, kTe_warm, Gamma_hot, Gamma_warm, R_hot, R_warm, R_out, R_in, rflag, redshift, norm]
		agnslim_model = xspec.XSagnslim()
		agnslim_setup = [agnslim_parinfo, agnslim_init_params, agnslim_model, self.agnslim_evalModel]
		
		modelID = ['KERRBB', 'SLIMBH', 'AGNSED', 'AGNSLIM']
		models = [kerrbb_setup, slimbh_setup, agnsed_setup, agnslim_setup]

		if Choice in modelID:
			setup = models[modelID.index(Choice)]
		else:
			setup = models[0]
			
		# Get labels and init mcmc params #
		self.parinfo, self.init_params, self.model, self.evalBHModel = setup
		mcmc_params = []
		self.par_labels = []
		self.par_units = []
		for index, par in enumerate(self.parinfo):
			if not par['fixed']:
				mcmc_params.append(self.init_params[index])
				self.par_labels.append(par['label'])
				self.par_units.append(par['units'])

		self.ndim = len(mcmc_params)
		return setup

	def evalModel(self, params, freq):
		ekeV = 10**11. * con.h * 6.242E15
		dl = self.cosmo.luminosity_distance(self.z).to(u.cm)
		freq_kev = (con.h*6.242E15) * freq
		freq_lo = freq_kev - ekeV
		freq_hi = freq_kev + ekeV
		ymodel = self.model.calc(params, freq_lo, freq_hi)/(2*ekeV)*freq_kev*4.*np.pi*dl.value**2*con.h*10**(7)*freq
		return ymodel

	def kerrbb_evalModel(self, bestfitp, freq): # in rest frame
		current_params = self.deriveParams(self.parinfo, self.init_params, bestfitp)
		current_params[2] = np.arccos(current_params[2])*180./np.pi # Convert cos(i) to deg
		current_params[3] = 10**(current_params[3]) # Convert log10(Msun) to Msun
		current_params[4] = 10**(current_params[4]) # Convert log10(Msun/yr) to Msun/yr	
		current_params[4] *= 1.989E33/(10**18)/(3.154E7) # Convert Msun/yr to 10^18 g/s
		current_params[5] = self.cosmo.luminosity_distance(self.z).to(u.kpc).value # Convert redshift to dl in kpc
		ymodel = self.evalModel(current_params, freq)
		return ymodel
	
	def slimbh_evalModel(self, bestfitp, freq): # in rest frame
		current_params = self.deriveParams(self.parinfo, self.init_params, bestfitp)
		current_params[0] = 10**(current_params[0]) # Convert log10(Msun) to Msun
		current_params[4] = np.arccos(current_params[4])*180./np.pi # Convert cos(i) to deg
		current_params[5] = self.cosmo.luminosity_distance(self.z).to(u.kpc).value # Convert redshift to dl in kpc
		ymodel = self.evalModel(current_params, freq)
		return ymodel
		
	def agnsed_evalModel(self, bestfitp, freq): # in rest frame
		current_params = self.deriveParams(self.parinfo, self.init_params, bestfitp)
		ymodel = self.evalModel(current_params, freq)
		return ymodel 
		
	def agnslim_evalModel(self, bestfitp, freq): # in rest frame
		current_params = self.deriveParams(self.parinfo, self.init_params, bestfitp)
		ymodel = self.evalModel(current_params, freq)
		return ymodel
	
	
	def deriveParams(self, parinfo, init_params, new_params):
		current_params, counter = [], 0
		for index, par in enumerate(parinfo):
			if par['fixed']:
				current_params.append(init_params[index])
			else:
				current_params.append(new_params[counter])
				counter += 1
		return current_params
	
	def paramPriors(self, bestfitp): #log
		current_params = self.deriveParams(self.parinfo, self.init_params, bestfitp)
			
		for index, par in enumerate(self.parinfo):
			par_val = current_params[index]
			current_limits = par['limits']
			if par['label'] == 'log(M$_{\\rm{BH}}$/M$_{\\odot}$)':
				mbhIndex = index
			if par_val < current_limits[0] or par_val > current_limits[1]:
				return -np.inf # if priors not satisfied
			
		lnprior = 0.0
		if self.priorMassFunction:
			lnprior = self.massFunctionLnLike[np.argmin(np.abs(self.massFunctionMbh-current_params[mbhIndex]))]
		elif self.priorMbh > 0:
			lnprior = -0.5*((current_params[mbhIndex] - self.priorMbh)/self.priorMbhSigma)**2 
		return lnprior 
	
	def residual(self, bestfitp, data_freq, data_power, data_epower):
		model_yy = self.evalBHModel(bestfitp, data_freq)
	
		# logfit
		model_yy = np.log10(model_yy)
		data_power = [log10(ufloat(data_power[i], data_epower[i])) for i in range(len(data_power))]
		data_epower = np.array([i.s for i in data_power])
		data_power = np.array([i.n for i in data_power])
		return np.sum(-0.5*((data_power-model_yy)/data_epower)**2)
	
	def likelihood(self, bestfitp, data_freq, data_power, data_epower):
		prior_result = self.paramPriors(bestfitp)
		if prior_result == -np.inf:
			return -np.inf
		return prior_result+self.residual(bestfitp, data_freq, data_power, data_epower)
	
	def mcmcMain(self, p0, nwalkers, niter, ndim, lnprob, data):
		filename = 'output/'+self.name+'.h5'
		backend = emcee.backends.HDFBackend(filename)
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, backend=backend)

		print("Running burn-in...")
		p0, _, _ = sampler.run_mcmc(p0, 100)
		sampler.reset()

		print("Running production...")
		pos, prob, state = sampler.run_mcmc(p0, niter)

		return sampler, pos, prob, state

	#####################
	# Probe MCMC Result #
	#####################

	def sampleWalkers(self, nsamples, flattened_chain, xx):
		models = []
		draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
		params = flattened_chain[draw]
		for i in params:
			mod = self.evalBHModel(i, xx)
			models.append(mod)
		spread = np.std(models,axis=0)
		med_model = np.median(models,axis=0)
		return med_model,spread
	


	##############
	# Extinction #
	##############
	def extinction_MW(self, Rv, Ebv, lams, flux, eflux): # lams input in rest frame
		ext = F19(Rv=Rv)
		defined_range = np.array([0.115, 3.3])*10**4*u.angstrom
		lams_obs = lams*(1+self.z)*u.angstrom
		flux_obs = flux[np.logical_and(lams_obs>defined_range[0], lams_obs<defined_range[1])]
		eflux_obs = eflux[np.logical_and(lams_obs>defined_range[0], lams_obs<defined_range[1])]
		lams_obs_ext = lams_obs[np.logical_and(lams_obs>defined_range[0], lams_obs<defined_range[1])]
		flux_noext = flux_obs/ext.extinguish(lams_obs_ext, Ebv=Ebv)
		eflux_noext = eflux_obs/ext.extinguish(lams_obs_ext, Ebv=Ebv)
	
		flux[np.logical_and(lams_obs>defined_range[0], lams_obs<defined_range[1])] = flux_noext
		eflux[np.logical_and(lams_obs>defined_range[0], lams_obs<defined_range[1])] = eflux_noext
	
		return [lams, flux, eflux]
	
	def extinction_AGN(self, RvA, fA, Av, lams, flux, eflux): # lams input in rest frame
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
	
	def extinction_GC10_MEC(self, Av, lams, flux, eflux): # lams input in rest frame
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

	############
	# Plotting #
	############
	def makeCornerPlot(self, flattened_mcmc_chain, labels, units):
		num_params = len(flattened_mcmc_chain[0])
		
		if 'log(M$_{\\rm{BH}}$/M$_{\\odot}$)' in labels:
			argMbh = labels.index('log(M$_{\\rm{BH}}$/M$_{\\odot}$)')
			if argMbh != 0:
				labels[0], labels[argMbh] = labels[argMbh], labels[0]
				units[0], units[argMbh] = units[argMbh], units[0]
				trans_chain = [np.copy(i) for i in np.transpose(flattened_mcmc_chain)]
				trans_chain[0], trans_chain[argMbh] = trans_chain[argMbh], trans_chain[0]
				flattened_mcmc_chain = np.transpose(trans_chain)

		temp_labels_listx = np.copy(labels)
		temp_labels_listy = np.copy(labels)
		fig = plt.figure(figsize=(num_params*2, num_params*2))
		plt.subplots_adjust(wspace= 0.05, hspace= 0.05)

		
		hist_axs = []
		density_axs = []
		fit_plot = []
		c = ChainConsumer().add_chain(flattened_mcmc_chain, parameters=labels)
		cc_summary = c.analysis.get_summary()

		for i in range(num_params):
			hist_axs.append(fig.add_subplot(num_params, num_params, i*num_params+i+1))

		for i in range(num_params):
			for j in range(i):
				density_axs.append(fig.add_subplot(num_params,num_params,i*num_params+j+1))

		for i in range(num_params):
			if (i+1)*num_params-i <= num_params**2/2.:
				fit_plot.append(int((i+1)*num_params-i))
		if len(fit_plot) == 1:
			fit_ax = fig.add_subplot(num_params, num_params, fit_plot[0])
		else:
			fit_plot = sorted(fit_plot)
			fit_ax = fig.add_subplot(num_params, num_params, (fit_plot[0], fit_plot[-1]))

		for ind, ax in enumerate(hist_axs):
			current_dist = np.transpose(flattened_mcmc_chain)[ind]
			sns.histplot(current_dist, kde='False', ax = ax, color='C0', fill=True, element="step")
			ax.tick_params(axis='both', which='both', direction='in', left=False, labelleft=False, labelbottom=False)
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax.spines['left'].set_visible(False)
			xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
			current_med = cc_summary[labels[ind]][1]
			current_lo = cc_summary[labels[ind]][0]
			current_hi = cc_summary[labels[ind]][2]
			if current_lo == None:
				current_med = np.median(current_dist) # Median
				current_lo = np.quantile(current_dist, 0.16) # 16th percentile
				current_hi = np.quantile(current_dist, 0.84) # 84th percentile

			ax.axvline(current_med, color='k')
			ax.axvline(current_lo, linestyle='--', color='k')
			ax.axvline(current_hi, linestyle='--', color='k')
			textstr = labels[ind] + ' = %.2f$^{+%.2f}_{-%.2f}$ ' % (current_med,current_hi-current_med,current_med-current_lo) + units[ind]
			if ind < len(hist_axs)/2:
				ax.text(0.05, 1.05, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='left', rotation=0)
			else:
				ax.text(1.05, 0.05, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='left', rotation=270)
			if ax == hist_axs[-1]:
				ax.tick_params(axis='both', which='both', direction='in', labelbottom=True, rotation=40)
				ax.set_xlabel(temp_labels_listx[num_params-1])
				temp_labels_listx = np.delete(temp_labels_listx, num_params-1)
			
			if False: # !! adds points to the Mbh histogram !!
				if ind == 0:
					mbhs = [10.52, 9.58]
					embhs = [0.08, 0.14]
					mbh_labels = ['MgII', 'H$\\beta$']
					for mbh_ind, (mbh, embh, mbh_label) in enumerate(zip(mbhs, embhs, mbh_labels)):
						ax.errorbar(mbh, 0.4*ax.get_ylim()[1], xerr=embh, c='g', fmt='o', capsize=5.0, linewidth=2.0, capthick=2.0)
						ax.text(mbh+(mbh_ind)%2*0.35, 0.45*ax.get_ylim()[1], mbh_label, ha='center', va='bottom')
				
				
		ycounter = 0
		xcounter = 0
		for ind, ax in enumerate(density_axs):
			if ind <= len(density_axs)-num_params:
				ax.tick_params(axis='both', which='both', direction='in', top=True, labelbottom=False, right=True, rotation=40)
			else:
				ax.set_xlabel(temp_labels_listx[0])
				temp_labels_listx = np.delete(temp_labels_listx, 0)
				ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, rotation=40)
			temp=[0]
			for i in range(num_params):
				temp.append(temp[-1]+i+1)
			temp=np.array(temp)
			if ind not in temp:
				ax.tick_params(axis='both', which='both', direction='in', labelleft=False)
				xdata = np.transpose(flattened_mcmc_chain)[xcounter]
				xcounter += 1
				ax.get_shared_y_axes().join(ax, last_ax)
			else:
				xcounter = 0
				xdata = np.transpose(flattened_mcmc_chain)[xcounter]
				xcounter += 1
				ax.set_ylabel(temp_labels_listy[1])
				last_ax = ax
				ydata = np.transpose(flattened_mcmc_chain)[ycounter+1]
				ycounter += 1
				temp_labels_listy = np.delete(temp_labels_listy, 1)
			corner.hist2d(xdata, ydata, ax=ax, color='C0')
			for i in range(num_params):
				if ind in (temp+i)[i:]:
					ax.get_shared_x_axes().join(ax, hist_axs[i])
					break
		return fit_ax
	

	def createPlot(self, samples, lnlikelihoods, data, redshift):
		freq = np.linspace(0.8*np.min(data[0]), 1.1*np.max(data[0]), 1000)
	
		med_model, spread = self.sampleWalkers(100, samples, freq) # find median model
		param_max  = samples[np.argmax(lnlikelihoods)] # highest likelihood model
		new_bestfit = self.evalBHModel(param_max, freq)

		print(np.nanmax(lnlikelihoods))
		print(param_max)
	
		# Multi parameter plot #
		print('\nPLOTTING...')
		ax = self.makeCornerPlot(samples, self.par_labels, self.par_units)

		ax.plot(freq, new_bestfit, c='r', label='Highest Likelihood Model', zorder=5)
		ax.fill_between(freq,med_model-spread,med_model+spread, color='grey',alpha=0.5,label=r'$1\sigma$ Posterior Spread', zorder=4)

		ax.errorbar(data[0], data[1], yerr=data[2], fmt='none', c='k', zorder=7)
		ax.plot(data[0], data[1], 's', zorder=6, mfc='none', mec='b', mew=1.5, ms=6, lw=0)

		ax.set_xlabel('Rest Frequency [Hz]', loc='right')
		ax.set_ylabel('$\\nu L_{\\nu}$ [erg/s]', loc='top')
	
		ax.set_ylim([ax.get_ylim()[0], 1.1*ax.get_ylim()[1]])
	
		ax.set_xscale('log')
		ax.set_yscale('log')

		ax.text(x=0.05, y=0.95, s='z = %.3f\n$\\chi^2_{\\nu}$ = %.3f' %(self.z, np.nanmax(lnlikelihoods)/(len(data[0])-len(param_max))), transform=ax.transAxes, fontsize=10, 
				verticalalignment='top', horizontalalignment='left',
				bbox=dict(facecolor='white', edgecolor='none'))
		#ax.text(x=0.05, y=0.95, s='z = %.3f' %(self.z), transform=ax.transAxes, fontsize=10, 
		#		verticalalignment='top', horizontalalignment='left',
		#		bbox=dict(facecolor='white', edgecolor='none'))

		ax.tick_params(axis='both', which='both', direction='in', labelbottom=False, labelleft=False, top=True, right=True, labeltop=True, labelright=True)
		ax.tick_params(axis='both', which='minor', labelright=False, labeltop=False)
		plt.savefig('output/'+self.name+'.png', dpi=200, facecolor='white', transparent=False)
		return ax

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    