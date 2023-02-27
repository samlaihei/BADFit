# Example
import BADFit
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import scipy.constants as con
from astropy import units as u
import numpy as np

def calcFlux(redshift, freq, power, epower):
	cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
	dl = cosmo.luminosity_distance(redshift).to(u.cm)
	lam = con.c/(freq)*10**10
	flux = np.array(power)/(4 * np.pi * dl.value**2 * np.array(lam))
	eflux = np.array(epower)/(4 * np.pi * dl.value**2 * np.array(lam))
	return [lam, flux, eflux]

########
# Data #
########

redshift = 4.692
datafile = 'data/example.csv' 

pdata = pd.read_csv(datafile)

inputFreq = pdata['Frequency'].to_numpy()# Hz
inputPower = pdata['Power'].to_numpy() # erg/s
inputPowerError = pdata['e_Power'].to_numpy()

lams, flux, eflux = calcFlux(redshift, inputFreq, inputPower, inputPowerError)



J2157 = BADFit.BADFit('example', 'SLIMBH', lams, flux, eflux, redshift, ra=329.36758, dec=-36.03752)
#J2157.createPlotFromFile('example.h5', 'SLIMBH', 128, 256)
J2157.runMCMC(nwalkers=128, niter=256)











    