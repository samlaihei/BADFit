# Example
import BADFit
import pandas as pd

redshift = 4.692
datafile = 'data/example.csv'

pdata = pd.read_csv(datafile)

lams = pdata['Wavelength'].to_numpy() # rest-frame wavelength in angstrom 
flux = pdata['FluxDensity'].to_numpy() # flux density in erg/s/cm2/Hz
eflux = pdata['eFluxDensity'].to_numpy()


if False:
	datafile = 'data/fauxtometry_test.csv'

	pdata = pd.read_csv(datafile)

	freq = pdata['Freq'].to_numpy() # rest-frame wavelength in angstrom 
	power = pdata['Power'].to_numpy() # flux density in erg/s/cm2/Hz
	epower = pdata['ePower'].to_numpy()
else:
	freq, power, epower = [], [], []

J2157 = BADFit.BADFit('example', 'SLIMBH', lams, flux, eflux, redshift, freq=freq, power=power, epower=epower, ra=329.36758, dec=-36.03752)
#J2157.createPlotFromFile()
J2157.runMCMC()











    