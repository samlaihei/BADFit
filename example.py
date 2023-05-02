# Example
import BADFit
import pandas as pd
import os

os.system("mkdir output")

redshift = 4.692
datafile = 'data/example.csv'

pdata = pd.read_csv(datafile)

lams = pdata['Wavelength'].to_numpy() # rest-frame wavelength in angstrom 
flux = pdata['FluxDensity'].to_numpy() # flux density in erg/s/cm2/Hz
eflux = pdata['eFluxDensity'].to_numpy()


if False:
	datafile = 'data/fauxtometry_test.csv'

	pdata = pd.read_csv(datafile)

	freq = pdata['Freq'].to_numpy() # rest-frame frequency in Hz 
	power = pdata['Power'].to_numpy() # Power in erg/s
	epower = pdata['ePower'].to_numpy()
else:
	freq, power, epower = [], [], []


#testObj = BADFit.BADFit('example', 'KERRBB', lams, flux, eflux, redshift, freq=freq, power=power, epower=epower, ra=329.36758, dec=-36.03752)
testObj = BADFit.BADFit('example', 'KERRBB', lams, flux, eflux, redshift, ra=329.36758, dec=-36.03752)
#testObj.createPlotFromFile()
testObj.runMCMC()












    