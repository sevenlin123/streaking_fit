
##########################################################################
#
# streaking_fit_forpl.py, version 1.5.3
#
# Perform PSF (x) line fitting on input candidate streaks from findstreaks
# and compute metrics. 
# 
# Original code (streaking_fit.py) is from Edward Lin: edlin@gm.astro.ncu.edu.tw
# Modified by F. Masci for use in production.
#
# Synopsis:
# streaking_fit_forpl.py -dimgname <difference image in FITS format>
#                        -subsz <sub frame size; ~ length of streak [pixels]> 
#                        -xpos <x-coord position of streak>
#                        -ypos <y-coord position of streak>
#                        -PA <position angle of strea [deg]>
#
# Example:
# /ptf/pos/ext/anaconda2/bin/python -W ignore streaking_fit_forpl.py -dimgname /stage/ptf_pos_sbx5/frank/streaks_reals/processing_v2/diffprod_d3386_f2_c5/PTF_201405103637_i_p_scie_t084343_u024530586_f02_p003386_c05_pmtchscimref.fits -subsz 27 -xpos 143 -ypos 123 -PA 118.2
#
# v1.2: added PA input; fixed the zero length problem; improved fitting success rate (Modified by E. Lin on 18/02/17)
# v1.3: fixed the inconsistent outputs; fixed the issue of ridiculous fitting uncertainties
# v1.3.1: improve efficiency
# v1.4: updated to process ZTF input difference images (Modified by F. Masci on 16/04/17)
# v1.4.1: fixed the "fitting failure" issue caused by the nan pixel (Modified by E. Lin on 18/04/17)
# v1.4.2: corrected the uncertainties calculations (Modified by E. Lin on 28/10/17)
# v1.5: Generate the covariance matrix of x0, y0, x1, y1 (Modified by E. Lin on 29/10/17)
# v1.5.1: Calculate the covariance matrix of RA, DEC (Modified by E. Lin on 30/10/17)
# v1.5.2: RA, Dec uncertainties in unit "degrees"; output correlation coefficients instead covariances (Modified by E. Lin on 07/02/18)
# v1.5.3: fixed the fitting issue when streak on image edge (Modified by E. Lin on 09/02/18)
##########################################################################

from numpy import *
from scipy import optimize
from scipy.special import erf
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from astropy.io import fits as pyfits
import sys, subprocess

class streak:
	def __init__(self, fits_name):
		self.fits_name = fits_name
		self.image = pyfits.open(self.fits_name)[0]
		self.mag = None
		self.filter_id = None
		self.size = 20. #subframe size (2 x self.size by 2 x self.size)
		self.X = None # Center_X of streak (initial value for fitting) 
		self.Y = None # Center_Y of streak (initial value for fitting)
		self.PA0 = 0 # Position angle (initial value for fitting)
		self.exptime = None
		self.mjd0 = None #shutter open time
		self.mjd1 = None #shutter close time
		self.X0 = None #start x
		self.Y0 = None #start y
		self.X1 = None #end x
		self.Y1 = None #end y
		self.ra0 = None #start ra
		self.ra1 = None	#start dec
		self.dec0 = None #end ra
		self.dec1 = None #end dec
		self.streak_image = None 
		self.params = None
		self.fit_result = None # (flux, length, x, y, sigma, PA, BG)
		self.fit_error = None # uncertainties of (flux, length, x, y, sigma, PA, BG)
		
	def read_fits(self):
		self.mjd0 = float(self.image.header['obsmjd'])
		self.exptime = float(self.image.header['EXPTIME'])
		self.mjd1 = self.mjd0 + (self.exptime)/86400.
		image = zeros([int(2*self.size), int(2*self.size)])
		y_max, x_max = shape(self.image.data)
		y_min, x_min = 0, 0
		ya, yb, xa, xb = int(self.Y-self.size), int(self.Y+self.size), int(self.X-self.size), int(self.X+self.size)
		y0 = max(ya, y_min)
		y1 = min(yb, y_max)
		x0 = max(xa, x_min)
		x1 = min(xb, x_max)
		image0 = self.image.data[y0:y1, x0:x1]
		yoff, xoff = array(shape(image)) - array(shape(image0))
		ysize, xsize = shape(image0)
		if yb > y_max and xb > x_max:
		    image[:ysize, :xsize] += image0
		elif yb > y_max and xb <= x_max:
		    image[:ysize, xoff:] += image0
		elif yb <= y_max and xb > x_max:
		    image[yoff:, :xsize] += image0
		else:
		    image[yoff:, xoff:] += image0
		image[image < -999 ] = 0
		image[isnan(image)] = 0
		self.streak_image = image
	def moments(self):
		"""Returns (flux, length, x, y, sigma, PA, BG) as the initial values of trail function """		
		trail_flux = 1000
		trail_L = self.size
		trail_x = self.size
		trail_y = self.size
		trail_sigma = 1. 
		trail_PA = -self.PA0/180.*pi
		trail_bg = median(self.streak_image)
		self.params = trail_flux, trail_L, trail_x, trail_y, trail_sigma, trail_PA, trail_bg
		#print self.params

	def trail(self, flux, L, center_x, center_y, sigma, PA, BG): #the streak function (Veres et al., 2012; Lin et al., 2015) is defined here 
		"""Returns a trail function with the given parameters"""
		return lambda x,y: BG+flux/(L*(2*sigma*sqrt(2*pi)))*exp(-(-(x-center_x)*sin(PA)+(y-center_y)*cos(PA))**2/(2*sigma**2))*(erf(((x-center_x)*cos(PA)+(y-center_y)*sin(PA)+L/2)/(sigma*(2)**0.5))-erf(((x-center_x)*cos(PA)+(y-center_y)*sin(PA)-L/2)/(sigma*(2)**0.5)))

	def fittrail(self): #Using scipy least square minimization 
		"""Returns (flux, length, x, y, sigma, PA, BG) the trail parameters of a 2D distribution found by a fit"""
		self.errorfunction = lambda p: ravel(self.trail(*p)(*indices(self.streak_image.shape)) - self.streak_image)
		self.fit_result, self.cov, infodict, errmsg, success = optimize.leastsq(self.errorfunction, self.params, full_output=1, col_deriv=1, maxfev=400)
		#print self.fit_result
		#print self.cov
		if self.cov is None:
			return 0
		if self.cov.max() < 99999 :
			self.flux, self.length, self.x, self.y, self.sigma, self.PA, self.BG = self.fit_result
			ximg, yimg = shape(self.streak_image)
			Xin, Yin = mgrid[:ximg, 0:yimg]
			self.streak_model = self.trail(*self.fit_result)(Xin,Yin)
			self.streak_image0 = copy(self.streak_image)
			self.streak_image0[self.streak_model == self.BG] = self.BG.mean()
			self.errorfunction = lambda p: ravel(self.trail(*p)(*indices(self.streak_image.shape)) - self.streak_image0)
			self.fit_result, self.cov, infodict, errmsg, success = optimize.leastsq(self.errorfunction, self.params, col_deriv=1, full_output=1, maxfev=400)		
			return 1	
		else:
			return 0
	def cal_properties(self):
		s_sq = (self.errorfunction(self.fit_result)**2).sum()/(2*self.size)**2
		try:
			self.cov = self.cov * s_sq
			self.flux, self.length, self.x, self.y, self.sigma, self.PA, self.BG = self.fit_result
			rot = matrix([[1, 0, 0.5*sin(self.PA), 0],[0, 1, 0.5*cos(self.PA), 0],[1, 0, -0.5*sin(self.PA), 0],[0, 1, -0.5*cos(self.PA), 0]])
			xyls = matrix([[self.x+self.X-self.size],[self.y+self.Y-self.size],[self.length],[self.sigma]])
			x0, y0, x1, y1 = rot*xyls
			self.X0 = float(x0)
			self.X1 = float(x1)
			self.Y0 = float(y0)
			self.Y1 = float(y1)
			self.fit_err = diag(self.cov)**0.5 
			self.flux_err, self.length_err, self.x_err, self.y_err, self.sigma_err, self.PA_err, self.BG_err = self.fit_err
			xyls_err = matrix([[self.cov[2][2], self.cov[2][3], self.cov[2][1], self.cov[2][4]],\
			                   [self.cov[3][2], self.cov[3][3], self.cov[3][1], self.cov[3][4]],\
			                   [self.cov[1][2], self.cov[1][3], self.cov[1][1], self.cov[1][4]],\
			                   [self.cov[4][2], self.cov[4][3], self.cov[4][1], self.cov[4][4]]])
			                   
			cov = rot * xyls_err  * rot.T
			#self.X0 = self.x+self.length*sin(self.PA)*0.5+self.X-self.size
			#self.X1 = self.x-self.length*sin(self.PA)*0.5+self.X-self.size
			#self.Y0 = self.y+self.length*cos(self.PA)*0.5+self.Y-self.size
			#self.Y1 = self.y-self.length*cos(self.PA)*0.5+self.Y-self.size
			cov = array(cov)
			sigma_x0x0, sigma_x0y0, sigma_x0x1, sigma_x1y1 = cov[0]
			sigma_y0x0, sigma_y0y0, sigma_y0x1, sigma_y0y1 = cov[1]
			sigma_x1x0, sigma_x1y0, sigma_x1x1, sigma_x1y1 = cov[2]
			sigma_y1x0, sigma_y1y0, sigma_y1x1, sigma_y1y1 = cov[3]
			#self.X0_err = (self.x_err**2+(0.5*self.length*sin(self.PA))**2*((self.length_err/self.length)**2+(cos(self.PA)*self.PA_err/sin(self.PA))**2))**0.5
			#self.X1_err = (self.x_err**2+(0.5*self.length*sin(self.PA))**2*((self.length_err/self.length)**2+(cos(self.PA)*self.PA_err/sin(self.PA))**2))**0.5
			#self.Y0_err = (self.y_err**2+(0.5*self.length*cos(self.PA))**2*((self.length_err/self.length)**2+(sin(self.PA)*self.PA_err/cos(self.PA))**2))**0.5
			#self.Y1_err = (self.y_err**2+(0.5*self.length*cos(self.PA))**2*((self.length_err/self.length)**2+(sin(self.PA)*self.PA_err/cos(self.PA))**2))**0.5
			self.X0_err = sigma_x0x0**0.5
			self.Y0_err = sigma_y0y0**0.5
			self.X1_err = sigma_x1x1**0.5
			self.Y1_err = sigma_y1y1**0.5 
			
			self.X0Y0 = sigma_x0y0
			self.X1Y1 = sigma_x1y1
			CRVAL1  = self.image.header['CRVAL1']
			CRVAL2  = self.image.header['CRVAL2']
			CRPIX1  = self.image.header['CRPIX1']
			CRPIX2  = self.image.header['CRPIX2']
			CD11 =  self.image.header['CD1_1']
			CD12 =  self.image.header['CD1_2']
			CD21 =  self.image.header['CD2_1']
			CD22 =  self.image.header['CD2_2']
			CD_MATRIX = matrix([[CD11, CD12], [CD21, CD22]])
			RADEC0 = CD_MATRIX * matrix([[self.X0 - CRPIX1],[self.Y0 - CRPIX2]])
			RADEC1 = CD_MATRIX * matrix([[self.X1 - CRPIX1],[self.Y1 - CRPIX2]])
			#print CRVAL1, CRVAL2, CRPIX1, CRPIX2
			#print CD_MATRIX
			RA0 = array(RADEC0)[0] + CRVAL1
			DEC0 = array(RADEC0)[1] + CRVAL2
			RA1 = array(RADEC1)[0] + CRVAL1
			DEC1 = array(RADEC1)[1] + CRVAL2			
			#print RA0, DEC0, RA1, DEC1
			
			X0Y0_cov = matrix([[sigma_x0x0, sigma_x0y0],[sigma_x0y0, sigma_y0y0]])
			X1Y1_cov = matrix([[sigma_x1x1, sigma_x1y1],[sigma_x1y1, sigma_y1y1]])
			RADEC0_cov = CD_MATRIX * X0Y0_cov * CD_MATRIX.T
			RADEC1_cov = CD_MATRIX * X1Y1_cov * CD_MATRIX.T
			RADEC0_cov = abs(array(RADEC0_cov))
			RADEC1_cov = abs(array(RADEC1_cov))
			#print CD_MATRIX
			#print X0Y0_cov
			#print X1Y1_cov
			self.rara0, self.radec0 = RADEC0_cov[0]
			self.decra0, self.decdec0 = RADEC0_cov[1]
			self.rara1, self.radec1 = RADEC1_cov[0]
			self.decra1, self.decdec1 = RADEC1_cov[1]					
			
			#print self.X0, self.X0_err, sigma_x0x0**0.5
			#print self.Y0, self.Y0_err, sigma_y0y0**0.5
			#print self.X1, self.X1_err, sigma_x1x1**0.5
			#print self.Y1, self.Y1_err, sigma_y1y1**0.5 
			
			ximg, yimg = shape(self.streak_image)
			Xin, Yin = mgrid[:ximg, 0:yimg]
			self.streak_model = self.trail(*self.fit_result)(Xin,Yin)
			self.flux_AP = abs((self.streak_image-self.BG)[self.streak_model != self.BG].sum())
			self.flux_AP_err =  abs((self.streak_image)[self.streak_model != self.BG].sum())**0.5
			self.BG_area = abs(self.streak_image-median(self.streak_image)) < 2*self.streak_image.std()
			self.BG_rms = self.streak_image[self.BG_area].std()
			self.rms = (self.streak_image-self.streak_model)[self.streak_model != self.BG].std()#/self.flux_AP
			self.chisq = (self.rms/self.BG_rms)**2
			self.streak_snr = self.flux_AP/self.flux_AP_err
			if self.flux < 200:
				#print "FLUX LESS THAN 200"
				return 0
			else:
				return 1
		except TypeError:
			return 0
		except ValueError:
			return 0

		
	def cal_ra_dec(self):
		if self.image.header['CTYPE1'] == 'RA---TAN':
			self.image.header['CTYPE1'] = 'RA---TPV'
			self.image.header['CTYPE2'] = 'DEC--TPV'

		w = WCS(self.image.header)
		radec = w.all_pix2world([[self.X0, self.Y0],[self.X1, self.Y1]], 1)
		self.ra0 = radec[0][0]
		self.dec0 = radec[0][1]
		self.ra1 = radec[1][0]
		self.dec1 = radec[1][1]
		
	def cal_mag(self):
		try:
			zp = self.image.header['MAGZP']
		except KeyError:
			zp = 27.
		self.mag = zp-2.5*log10(abs(self.flux))
		self.mag_AP = zp-2.5*log10(abs(self.flux_AP))
		self.mag_err = 2.5*self.flux_err/(abs(self.flux)*log(10))
		self.mag_AP_err = 2.5*self.flux_AP_err/(abs(self.flux_AP)*log(10))
			
	def plot(self):		
		plt.imshow(self.BG_area)
		plt.show()
		plt.imshow(self.streak_model != self.BG)
		plt.show()
		plt.imshow(self.streak_model)
		plt.colorbar()
		plt.show()
		plt.contour(self.streak_model)
		plt.imshow(self.streak_image)
		plt.colorbar()
		plt.show()
		#pp.contour(fit_T)
		plt.imshow(self.streak_image-self.streak_model)
		plt.colorbar()
		plt.show()
		
	
def main():
	asteroid.read_fits() # read sub frame
	asteroid.moments() # generate the initial moments for leaset square fit
	good_fit = asteroid.fittrail() # fit the streak
	good = asteroid.cal_properties()
	#print cos(-asteroid.PA/180.*pi-asteroid.fit_result[5])
	n = 0
	size0 = asteroid.size
	X0 = asteroid.X
	Y0 = asteroid.Y
	while (not good*good_fit or abs(asteroid.fit_result[1]) <= 10. or abs(cos(-asteroid.PA0/180.*pi-asteroid.fit_result[5])) < 0.70) and n < 8:
		n += 1
		x_grid, y_grid = mgrid[-1:2, -1:2]
		x_shift = delete(x_grid.reshape(9), 4) 
		y_shift = delete(y_grid.reshape(9), 4)
		asteroid.X = X0 + 2*x_shift[n-1]
		asteroid.Y = Y0 + 2*y_shift[n-1]
		asteroid.read_fits()
		asteroid.moments()
		good_fit = asteroid.fittrail() 
		#asteroid.plot()
		#print asteroid.size, asteroid.fit_result[1], asteroid.PA0, -asteroid.fit_result[5]*180/pi
		good = asteroid.cal_properties()
		#print asteroid.fit_result[2], asteroid.fit_result[2]
			
	
	if good and abs(asteroid.fit_result[1]) >= 10. and  abs(cos(-asteroid.PA0/180.*pi-asteroid.fit_result[5])) > 0.70:
		
		#asteroid.cal_properties()
		asteroid.cal_mag()
		asteroid.cal_ra_dec()
		#asteroid.plot() # plot the sub frame image before/after subtract the fitting result
		print "# streak fitting results [flux, length, x, y, sigma, PA, BG]:"
		print " ",asteroid.fit_result[0], \
	              asteroid.fit_result[1], \
	              asteroid.fit_result[2], \
	              asteroid.fit_result[3], \
	              asteroid.fit_result[4], \
   	           -asteroid.fit_result[5]*180/pi, \
   	           asteroid.fit_result[6]
	
		print "# uncertainties in the above:"
		print " ",asteroid.fit_err[0], \
   	           asteroid.fit_err[1], \
   	           asteroid.fit_err[2], \
   	           asteroid.fit_err[3], \
   	           asteroid.fit_err[4], \
   	           asteroid.fit_err[5], \
   	           asteroid.fit_err[6]
	
		print "# streak_fit_mag, error_fit_mag:"
		print " ",asteroid.mag, asteroid.mag_err
		
		print "# streak_aperture_flux, error_aperture_flux, streak_aperture_snr:"
		print " ",asteroid.flux_AP, asteroid.flux_AP_err, asteroid.streak_snr
	
		print "# streak_aperture_mag, error_aperture_mag:"
		print " ",asteroid.mag_AP, asteroid.mag_AP_err	
		
		print "# dmag = \"streak_fit_mag - streak_aperture_mag\", error_dmag (RSS of errors):"
		print " ",asteroid.mag-asteroid.mag_AP,sqrt((asteroid.mag_err**2)+(asteroid.mag_AP_err**2)) 
		
		print "# fit_residual_rms, background_rms, chi-square:"
		print " ",asteroid.rms, asteroid.BG_rms, asteroid.chisq
		
		print "# MJDs at start and end of exposure:"
		print " ",asteroid.mjd0, asteroid.mjd1
	
		print "# RA,Dec at endpoint pixel position x,y =",asteroid.X0,asteroid.Y0,":"
		print " ",asteroid.ra0, asteroid.dec0

		print "# RA,Dec at endpoint pixel position x,y =",asteroid.X1,asteroid.Y1,":"
		print " ",asteroid.ra1, asteroid.dec1
		
		print "# Number of refit:"
		print " ",n
		
		print "# RA,Dec uncertainties and correlation coefficient at endpoint x,y =",asteroid.X0,asteroid.Y0,":"
		print " ",asteroid.rara0**0.5, asteroid.decdec0**0.5, asteroid.radec0/(asteroid.rara0**0.5*asteroid.decdec0**0.5)

		print "# RA,Dec uncertainties and correlation coefficient at endpoint x,y =",asteroid.X1,asteroid.Y1,":"
		print " ",asteroid.rara1**0.5, asteroid.decdec1**0.5, asteroid.radec1/(asteroid.rara1**0.5*asteroid.decdec1**0.5)
		
		return 1
	else:
		#asteroid.plot() # plot the sub frame image before/after subtract the fitting result
		print '# fitting failure'
		print '\n'


if __name__ == "__main__":

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-dimgname', dest='dimgname', type=str)
        parser.add_argument('-subsz', dest='subsz', type=float)
        parser.add_argument('-xpos', dest='xpos', type=float)
        parser.add_argument('-ypos', dest='ypos', type=float)
        parser.add_argument('-PA', dest='PA0', type=float)

        args = parser.parse_args()
        asteroid = streak(args.dimgname) #read fits image
        asteroid.size = args.subsz
        asteroid.X = args.xpos
        asteroid.Y = args.ypos
        asteroid.PA0 = args.PA0

        main()

