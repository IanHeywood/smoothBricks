# This takes an image and a list of directions and FWHMs,
# applies inverted Gaussian tapers at the positions in the image
# and produces subimages at the positions with Gaussian tapers.
#
# This is designed to be applied to clean component model images 
# so that the direction independent component can be predicted from
# the original image with inverse tapers, and then dEs can be computed
# from the subimage bricks. The tapering is supposed to make everything
# nice and smooth.
#
# Needs an intallation of Montage as mSubimage is called from the command
# line.
#
# ian.heywood@csiro.au 05.12.16


import numpy
import glob
import os
import pyfits
from astLib import astWCS
from astLib import astCoords as ac


# -----------------------------------------------------------
#
# DEFINE FUNCTIONS
#
# -----------------------------------------------------------


def ri(message):
	# Red terminal message
	print '\033[91m'+message+'\033[0m'


def gi(message):
	# Green terminal message
	print '\033[92m'+message+'\033[0m'


def bi(message):
	# Blue(ish) terminal message
	print '\033[94m\033[1m'+message+'\033[0m'


def getImage(fitsfile):
	# Return the image data from fitsfile as a numpy array
	input_hdu = pyfits.open(fitsfile)[0]
	if len(input_hdu.data.shape) == 2:
		image = numpy.array(input_hdu.data[:,:])	
	elif len(input_hdu.data.shape) == 3:
		image = numpy.array(input_hdu.data[0,:,:])
	else:
		image = numpy.array(input_hdu.data[0,0,:,:])
	return image


def flushFits(newimage,fitsfile):
	# Write numpy array newimage to fitsfile
	# Dimensions must match (obv)
	f = pyfits.open(fitsfile,mode='update')
	input_hdu = f[0]
	if len(input_hdu.data.shape) == 2:
		input_hdu.data[:,:] = newimage
	elif len(input_hdu.data.shape) == 3:
		input_hdu.data[0,:,:] = newimage
	else:
		input_hdu.data[0,0,:,:] = newimage
	f.flush()


def makeCheckerboard(infits,outfits,N):
	# Duplicate infits into outfits and replace the data
	# with a N/N checkerboard pattern
	# imsizes that don't divide cleanly by N will probably fail
	inp_img = getImage(infits)
	nr = inp_img.shape[0]
	nc = inp_img.shape[1]
	xtile = nr/N
	ytile = nc/N
	N = N / 2
	out_img = numpy.kron([[1, 0] * N, [0, 1] * N] * N, numpy.ones((xtile,ytile)))
	os.system('cp '+infits+' '+outfits)
	flushFits(out_img,outfits)
	return outfits


def applyTaper(infits,directions,fwhms,invert,outfits):
	# Apply a 2D Gaussian taper to an image 
	# infits = input fits file
	# centre = [(ra1,dec1),(ra2,dec2), ... ] in degrees
	# fwhms = [fwhm1,fwhm2, ... ] list of FWHMs of Gaussians (degrees)
	# invert = True to multiply by inverse Gaussian
	# outfits = output fits file
	# 
	gi('applyTaper: Processing image '+infits)
	input_hdu = pyfits.open(infits)[0]
	hdr = input_hdu.header
	WCS = astWCS.WCS(hdr,mode='pyfits')
	if len(input_hdu.data.shape) == 2:
		image = numpy.array(input_hdu.data[:,:])	
	elif len(input_hdu.data.shape) == 3:
		image = numpy.array(input_hdu.data[0,:,:])
	else:
		image = numpy.array(input_hdu.data[0,0,:,:])
	raDelta = hdr.get('CDELT1')
	decDelta = hdr.get('CDELT2')
	pixscale = abs(min(raDelta,decDelta))
	imX = image.shape[1]
	imY = image.shape[0]
	finalTaper = numpy.zeros((imY,imX))
	for centre in directions:
		gi('applyTaper: Direction '+str(centre))
		if len(directions) == len(fwhms):
			fwhm = fwhms[directions.index(centre)]
			gi('applyTaper: FWHM '+str(fwhm))
		else:
			fwhm = fwhms[0]
			ri('applyTaper: Length mismatch between directions and fwhms, using fwhms[0] '+str(fwhm))
		fwhm = fwhm/pixscale # fwhm now in pixel units
		sig_x = fwhm/2.3548
		sig_y = sig_x
		centre = WCS.wcs2pix(centre[0],centre[1])
		x0 = centre[0]
		y0 = centre[1]
		bx = numpy.arange(0,image.shape[1],1,float)
		by = bx[0:image.shape[0],numpy.newaxis]
		xPart = ((bx-x0)**2.0)/(2.0*(sig_x**2.0))
		yPart = ((by-y0)**2.0)/(2.0*(sig_y**2.0))
		taperImage = (numpy.exp(-1.0*(xPart+yPart)))
		gi('applyTaper: intermediate taper min,max '+str(numpy.min(taperImage))+','+str(numpy.max(taperImage)))
		finalTaper += taperImage
	if invert:
		finalTaper = 1.0 - finalTaper
	gi('applyTaper: final taper min,max '+str(numpy.min(finalTaper))+','+str(numpy.max(finalTaper)))
	opimage = image*finalTaper
	gi('applyTaper: Writing image '+outfits)
	os.system('cp '+infits+' '+outfits)
	flushFits(opimage,outfits)
	return outfits


def fixMontageHeaders(infile,outfile,axes):
	# Images produced by Montage do not have FREQ or STOKES axes
	# or information about the restoring beam. This confuses things like PyBDSM
	# infile provides the keywords to be written to outfile
	inphdu = pyfits.open(infile)
	inphdr = inphdu[0].header
	outhdu = pyfits.open(outfile,mode='update')
	outhdr = outhdu[0].header
	keywords = ['CTYPE','CRVAL','CDELT','CRPIX']
	for axis in axes: 
		for key in keywords:
			inkey = key+str(axis)
			outkey = key+str(axis)
			afterkey = key+str(axis-1)
			xx = inphdr[inkey]
			outhdr.set(outkey,xx,after=afterkey)
	outhdr.set('BUNIT',inphdr['BUNIT'],after=outkey)
	outhdr.set('BMAJ',inphdr['BMAJ'],after='BUNIT')
	outhdr.set('BMIN',inphdr['BMIN'],after='BMAJ')
	outhdr.set('BPA',inphdr['BPA'],after='BMIN')
	outhdu.flush()


# -----------------------------------------------------------
#
# EXAMPLE
#
# -----------------------------------------------------------

# tar -xzvf example_fits.tar.gz

# Grab a list of FITS files, e.g. model outputs from wsclean
chan_list = sorted(glob.glob('myimg_chan-0*.fits'))
# Give it a list of directions for dE bricks (ra,dec in degrees)
directions = [(336.821884,-42.122744),(335.305521,-43.464774),(328.396897,-44.544670),(335.550094,-47.906527)]
# And a corresponding list of FWHMs (degrees), thumbnails are twice this size
fwhms = [0.5,1.0,0.8,2.0]

# This example lets one of the Gaussians run over the edge of the image. It works, but might not be wise.

for infits in chan_list:
	# Change the line below to adjust the output filename
	opfits = infits.replace('.fits','_tapered.fits')
	applyTaper(infits,directions,fwhms,True,opfits)	
	for mydir in directions:
		fwhm = fwhm = fwhms[directions.index(mydir)]
		extent = 2.0*fwhm
		dirstr = str(mydir[0]).replace('.','p')+'_'+str(mydir[1]).replace('.','p')
		subim = infits.replace('.fits','_dir'+dirstr+'.fits')
		# Change the line below to adjust the thumbnail filename
		subim_tapered = subim.replace('.fits','_tapered.fits')
		syscall = 'mSubimage '+infits+' '+subim+' '+str(mydir[0])+' '+str(mydir[1])+' '+str(extent)
		os.system(syscall)
		applyTaper(subim,[mydir],[fwhm],False,subim_tapered)
		os.system('rm '+subim)
		# Montage strips the freq and stokes axes, so this function copies them from the master image
		# assuming in this example they are axes 3 and 4
		fixMontageHeaders(infits,subim_tapered,[3,4])

