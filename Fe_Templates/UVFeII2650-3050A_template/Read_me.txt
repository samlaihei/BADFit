

Here we give the UV Fe II template in 2650-3050 A range, presented in Appendix A of paper Popovic et al. 2018. 
The template is calculated using formula (1) from Popovic et al. 2018, for temperature T = 10000 K and for different widths and shifts of UV Fe II.
The UV Fe II is given in FWHM range from 1000 km/s to 10000 km/s (with step 200 km/s), and in the shift range (relative to laboratory wavelength) from -2000 km/s to 2000 km/s (with step 500 km/s).

The UV Fe II templates for different pairs of shifts and widths are given in 414 folders.

In each folder, there are 5 multiplets (60, 61, 62, 63 and 78) and additional 'I Zw 1 lines' (see the Appendix A) in shape of spectrum (ascii files). They are shown in UVFeII_FWHM.ps
with different colors, and their sum (total template) is denoted with black solid line. The multiplets have arbitary relative intensities, and their relative intensities are the same in all folders.
In process of fitting these multiplets should be multiplated with scalars, in order to fit the intensity of fitted spectrum.

Steps for fitting with this UV Fe II template:

1. Remove the UV pseudocontinuum (power law and Balmer continuum)
2. Cut the spectrum to be within 2650-3050 A range.
3. Fit stimuntaniosly Mg II line and UV Fe II lines in the spectrum. The UV Fe II lines should be fitted with linear combination of 6 multiplets in ascii form for different pairs of FWHM and shift, which are in different folders.
The Mg II line can be fitted with two Gaussian functions - one which fits the core and ne which fits the wings. 
The final result of fit is the solution with the smallest xi2, where you get the best combination of the FWHM and shift for the UV Fe II template.


