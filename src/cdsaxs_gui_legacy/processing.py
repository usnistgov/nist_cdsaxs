# -*- coding: utf-8 -*-
"""
This is part of the CDSAXS Data Processing GUI.
This module contains the objects that contain the various stages of processed data.
"""

from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import *
from future.moves.itertools import zip_longest
from future.utils import native_str_to_bytes

import glob
import os
import traceback
import sys
import itertools
import csv
from multiprocessing import cpu_count
from collections import defaultdict

from qtpy import QtWidgets
from namedlist import namedtuple
from PIL import Image
import tifffile
import numpy as np
from scipy.special import erf
from scipy import interpolate
from concurrent import futures
import statsmodels.api as sm
from statsmodels.robust.norms import Hampel

USE_ASTROPY = False  # astropy doesn't mix well with pyinstaller and currently (1.2.1) fails to load FITS files sometimes
if USE_ASTROPY:
    from astropy.io import fits
    import pyfits  # lets you load pickles saved with pyfits
else:
    import pyfits as fits

from cdsaxs_gui_legacy import diffraction, base, matplotlibwidget

# settings that apply globally, overwritten by GUI
PEAKS = {'Vertical up': 0, 'Horizontal left': 1, 'Vertical down': 2, 'Horizontal right': 3}
EV_NM = 1239.84
NOISE_REGION_STR = ':20, :'  # str, region of Qxz-Qy to subtract from rest of image
OVERLAP_DET_THETA_THRESH = 0.2  # in deg, sfs must have detector_theta this much apart or closer to consider cropping one of them in Qz-Qx
OVERLAP_SAMPLE_THETA_THRESH = 2  # in deg, sfs must have sample_theta this much apart or closer to consider cropping one of them in Qz-Qx
FIND_ANGLE_MAX = 10  # in deg, max integration box angle
FIND_ANGLE_MIN = -10  # in deg, min integration box angle
FIND_ANGLE_MAX_CHANGE = 1.0  # in deg, max change in box angle of adjacent sfs before setting box angle to avg of sfs on either side
FIND_ANGLE_PRINT = False  # bool
FIND_ANGLE_MASK = 0.004  # in Qxz, region near direct beam to ignore when finding angle
RECT_BOTTOM_BELOWCENTER = 20  # int, bottom pixel of the box below the beamcenter, if using both top and bottom of beamcenter to fit angle
USE_INPUT_ANGLE = True  # bool, whether to use inputted angle to assist in finding auto angle
FIX_ANGLE_ENERGY_THRESH = 0.1  # adjacent sfs must have energy this much apart or closer to consider fixing angle
FIX_ANGLE_SAMPLE_THETA_THRESH = 10  # adjacent sfs must have sample_theta this much apart or closer to consider fixing angle

if sys.version_info >= (3, 0, 0):
    opencsv_kwargs = {'mode': 'w', 'newline': ''}
else:
    opencsv_kwargs = {'mode': 'wb'}

# settings that apply to a DataQxzQy object
Params = namedtuple('Params', [
    'pixel_um',  # pixel size
    'SDD_cm',  # sample-detector distance
    'peaks_direction',  # see PEAKS above
    'center_px',  # [rows, cols], center of image in pixels without corrections for detector theta, x, and scale
                  # should be the same for all images in dataset
    'normalize_exposure',  # bool
    'normalize_I0',  # bool
    'multiply_intensity',  # float
    'subtract_bottom',  # bool, whether to subtract the `NOISE_REGION_STR` region of Qxz-Qy image from rest of image
    'detector_theta0',  # in deg, value of detector_theta for sample normal to beam
    'detector_x0',  # in mm, value of detector_x for sample normal to beam
    'detector_thetascale',  # float, value to multiple detector_theta by
    'normalize_current',  # bool
], default=False)
# tiff:
# python beam center qxz = igor beam x
# python beam center qy = 1920 - igor beam y
# fits:
# python beam center qxz = igor beam y
# python beam center qy = igor beam x

# settings used to find integration box on DataQxzQy
RectDims = namedtuple('RectDims', [
    'left', 'bottom', 'width', 'height',  # in px, integer left and bottom means box edge is at center of pixel
    'angle_deg',  # in deg, counter clockwise, 0 is vertical box
    'threshold',  # float, for auto angle, if max value of a row is less than threshold*mean, ignore it
    'width_peaks'  # for auto angle, box width in px to search for peaks in
])

# settings that apply to an IvsQxz object
AnalysisParams = namedtuple('AnalysisParams', [
    'subtract_value',  # float to subtract from each column(qxz) of DataRot to make IvsQxz, or 'min' to subtract minimum
    'reduce_fn',  # np.nanmean or np.nansum
    'samplethetaoffset',  # in deg, CCW
    'footprintcorr',  # bool
    'samplesize',  # in mm, for sample size correction
    'beamcenter',  # in mm, 0 for beam center == sample center, for sample size correction
    'beamfwhm',  # in mm, for sample size correction
    'substratethickness',  # in um, for absorption correction
    'substrateattenuation',  # in 1/um, for absorption correction, http://henke.lbl.gov/optical_constants/atten2.html
    'frac_polarized_y',  # float between 0 and 1, the remainder is polarized in x-z plane, for polarization correction
    'samplethickness',  # in um, for sample absorption correction
    'sampleattenuation',  # in 1/um, for sample absorption correction
    'samplesizecorr',  # bool
    'abscorr',  # bool
    'sampleabscorr',  # bool
    'polarizationcorr',  # bool
    # int, if not 0, files with same group number will not be cropped by each other in QzQx even if detector_thetas are different
    ('groupnumber', 0),
], default=False)


class DatasetTIFF(object):
    """Make sf object for a list of files or each image in folder (optional range),
    pass the relevant row of datfile

    Attributes:
        filelist: sorted list of .tif files
        datfile: list of strings containing each line of .dat file
        scatteringfilelist: list of sf objects

    Args:
        params: namedtuple of user parameters
        datfilestr: the loaded .dat file
        folder: the folder containing .tif files
        part: part of .dat file to load, starting and ending with '#S', defaults to None to load all parts.
        start: first file specified in part to load, defaults to 0.
        stop: last+1 file specified in part to load, defaults to None.
    """
    def __init__(self, params, datfilestr, folder, part=None, start=0, stop=None):
        if not os.path.isdir(folder):
            raise FileNotFoundError('Folder with .tif files is missing.')
        to_bytes = lambda x: [native_str_to_bytes(i) for i in x]
        print('Made dataset from ' + datfilestr)
        sys.stdout.flush()
        self.datfilestr = os.path.abspath(datfilestr)
        self.folder = os.path.abspath(folder)
        pattern = os.path.join(self.folder, '*.tif')
        # sorts tif files in folder in alphabetical order, should also be order listed in .dat file
        self.filelist = sorted(glob.glob(pattern))
        # opens dat file, converts each line in file to an item in list, closes file
        self.datfile = list(open(self.datfilestr, 'r'))
        start_index = 0
        part_counter = 0
        datasetinfo_keys = []
        datasetinfo_values = []
        # generates infoarray, a structured array containing per-tif-file instrument parameter values, length=number of tif files
        for index, line in enumerate(self.datfile):
            if line.startswith('#O'):  # Lines starting with #O contain parameter names specific to dataset
                datasetinfo_keys += line.split()[1:]
            elif line.startswith('#P'):  # Lines starting with #P contain parameter values specific to dataset
                datasetinfo_values += [float(i) for i in line.split()[1:]]
            elif line.startswith('#L'):
                names = line.split()[1:]  # Line starting with #L contains per-tif-file instrument parameter names
                # note: if part=None, this will end up with the names from the last #L line,
                # if it doesn't have Sample_phi there will be an error
            elif line.startswith('#S'):  # End of part
                if part_counter == part:  # Load current part per-tif-file instrument parameter values
                    infoarray = np.genfromtxt(to_bytes(self.datfile[start_index: index + 1]), names=names)
                    break
                part_counter += 1
                start_index = index
        else:  # if haven't broken by the end of datfile
            if part_counter == part:  # Load last part per-tif-file instrument parameter values
                infoarray = np.genfromtxt(to_bytes(self.datfile[start_index: index + 1]), names=names)
            elif part is None:  # Load all per-tif-file instrument parameter values
                infoarray = np.genfromtxt(to_bytes(self.datfile), names=names)
                # by default genfromtxt ignores all lines starting with '#'
        datasetinfo = dict(list(zip(datasetinfo_keys, datasetinfo_values)))
        self.scatteringfilelist = []
        if infoarray.shape == ():  # if only a single TIFF file
            enum = enumerate([infoarray[()]])
            print('Expecting 1 file...')
        else:
            enum = enumerate(infoarray[start: stop])
            print('Expecting {0} files...'.format(len(infoarray[start: stop])))
        for (imgnum, sample_desc) in enum:
            # for each tif file, make info dict and str and make ScatteringFile
            info = {key: sample_desc[key] for key in sample_desc.dtype.names}
            info['mono_act'] = datasetinfo['mono_act']
            infostr = '--- Specific to dataset ---\n'
            infostr += '\n'.join(['{0}: {1}'.format(key, datasetinfo[key]) for key in sorted(datasetinfo)])
            infostr += '\n--- Specific to one image file ---\n'
            infostr += '\n'.join(['{0}: {1}'.format(key, info[key]) for key in sorted(info)])
            try:
                self.scatteringfilelist.append(ScatteringFile('tiff', self.filelist[imgnum], params, info, infostr))
            except IndexError:
                print('Warning: not enough .tif files in folder, and the loaded files could have misassigned metadata')
            except Exception as exception:
                print(type(exception), exception)
                print('Error, skipping file ' + self.filelist[imgnum])


class DatasetGeneralTIFF(object):
    """Make sf object for a list of .tif files

    Attributes:
        filelist: list of .tif files
        scatteringfilelist: list of sf objects
        folder: folder with .tif files

    Args:
        params: namedtuple of user parameters
        filenames_tif: list of .tif files
        filename_csv: path of .csv metadata file
    """
    def __init__(self, params, filenames_tif, filename_csv):
        self.filelist = sorted(filenames_tif)
        self.folder, _ = os.path.split(filenames_tif[0])
        print('Made dataset from ' + self.folder)
        self.scatteringfilelist = []
        infoarray = np.genfromtxt(filename_csv, delimiter=',', skip_header=1)
        if infoarray.ndim == 1:
            infoarray = [infoarray]
        for imgnum, row in enumerate(infoarray):
            # for each tif file, make info dict and str and make ScatteringFile
            info = {key: row[col] for col, key in enumerate(['Sample Theta', 'mono_act', 'Seconds', 'IC_cntr1'])}
            infostr = '--- Specific to one image file ---\n'
            infostr += '\n'.join(['{0}: {1}'.format(key, info[key]) for key in sorted(info)])
            try:
                self.scatteringfilelist.append(ScatteringFile('gentiff', self.filelist[imgnum], params, info, infostr))
            except IndexError:
                print('Warning: not enough .tif files specified, and the loaded files could have misassigned metadata')
            except Exception as exception:
                print(type(exception), exception)
                print('Error, skipping file ' + self.filelist[imgnum])


class DatasetGeneralCSV_TIFF(object):
    """Make sf object for a list of .tif files
    TODO: this will become the sole future general tiff loader

    List of accepted CSV column headers and definitions:
        Required
        --------
        filename
        sample_phi_deg : sample rotation angle during cd-saxs measurement in degrees
        energy_eV : source energy in eV (cannot be used with wavelength_nm)
        wavelength_nm : source wavelength in nm (cannot be used with energy_eV)
        exposure_time_s : exposture time in s

        Optional
        --------
        sample_label : user-specified sample label
        sdd_cm : sample-to-detector distance in cm
        sample_chi_deg : rotation in the sample xy plane about the z axis

    Attributes:
        filelist: list of .tif files
        scatteringfilelist: list of sf objects
        folder: folder with .tif files

    Args:
        params: namedtuple of user parameters
        filenames_tif: list of .tif files
        filename_csv: path of .csv metadata file
    """
    def __init__(self, filepath_csv, params):

        # load the csv metadata file that includes the scattering filenames
        csv_data = np.loadtxt(filepath_csv, dtype='str', delimiter=',')
        header = csv_data[0, :]
        csv_data = csv_data[1:, :]
        label2index = {keyword: i for i, keyword in enumerate(header)}

        # extract data directory and list of filepaths to scattering files 
        self.folder, _ = os.path.split(filepath_csv)
        print('Made dataset from ' + self.folder)

        self.filelist = []
        self.scatteringfilelist = []
        # even if only one row, csv_data will always be a 2D array
        for i, row in enumerate(csv_data):
            # for each tif file, make info dict and str and make ScatteringFile
            info = {}
            for label in [x for x in header if x != 'filename']:
                info[label] = float(row[label2index[label]]) if label != 'sample_label' else row[label2index[label]]
            infostr = '--- Specific to one image file ---\n'
            infostr += '\n'.join(['{0}: {1}'.format(key, info[key]) for key in sorted(info)])

            self.filelist.append(os.path.join(self.folder, row[label2index['filename']]))
            self.scatteringfilelist.append(ScatteringFile('gencsvtiff', self.filelist[-1], params, info, infostr))
        

class DatasetBIN_INFO(object):
    def __init__(self, params, filenames_bin):
        self.filelist = sorted(filenames_bin)
        self.folder, _ = os.path.split(filenames_bin[0])
        print('Made dataset from ' + self.folder)
        self.scatteringfilelist = []
        for (imgnum, filename_bin) in enumerate(self.filelist):
            # for each bin file, make info dict and str and make ScatteringFile
            filename_info = os.path.splitext(filename_bin)[0] + '.info'
            info = np.genfromtxt(filename_info, delimiter='=', skip_header=1, dtype=str)
            info = {key: value for key, value in info}
            info['Sample Theta'] = float(info['Sample Theta ']) if 'Sample Theta ' in info else float(info['Theta '])
            info['mono_act'] = 24200
            info['Seconds'] = float(info['LiveTime '])
            infostr = '--- Specific to one image file ---\n'
            infostr += '\n'.join(['{0}: {1}'.format(key, info[key]) for key in sorted(info)])
            self.scatteringfilelist.append(ScatteringFile('bin', self.filelist[imgnum], params, info, infostr))


class DatasetFITS(object):
    """Make ScatteringFile object for a list of files or each image in folder (optional range),
    pass the headers from each image file

    Attributes:
        scatteringfilelist: list of ScatteringFile objects

    Args:
        params: namedtuple of user parameters
        filelist: list of .fits files, defaults to None to load from folder
        folder: folder containing .fits files, defaults to None
        start: first file in folder to load, defaults to 0.
        stop: last+1 file in folder to load, defaults to None.
    """
    def __init__(self, params, filelist=None, folder=None, start=0, stop=None):
        if filelist is not None:
            self.folder, _ = os.path.split(filelist[0])
            self.filelist = sorted(filelist)
        else:
            self.folder = os.path.abspath(folder)
            pattern = os.path.join(self.folder, '*.fits')
            self.filelist = sorted(glob.glob(pattern))
        print('Made dataset from ' + self.folder)
        sys.stdout.flush()
        self.scatteringfilelist = []
        for fullfilename in self.filelist[start: stop]:
            with open(fullfilename, 'rb') as f:
                hdulist = fits.open(f)
            # first hdu: main info, second: table of Izeros info, third: image info
            info = [hdu.header for hdu in hdulist]
            # fix wrong metadata from ALS 11.0.1.2
            if round(info[0]['EPU Polarization']) in {200, 380}:
                info[0]['EPU Polarization'] /= 2
                info[0]['Beam Current'] /= 2
                info[0]['AI 3 Izero'] /= 2
                info[0]['AI 6 BeamStop'] /= 2
                print('Corrected ALS metadata')
            infostr = ''
            for i, header in enumerate(info):
                infostr += '\n--- Header {0} ---\n'.format(i)
                infostr += '\n'.join(['{0} = {1} / {2}'.format(key, header[key], header.cards[key].comment)
                                      for key in sorted(header)])
            self.scatteringfilelist.append(ScatteringFile('fits', fullfilename, params, info, infostr))


class ScatteringFile(object):
    """Top object corresponding to one data file, intialize with Dataset.__init__
    Creates dataqxzqy object

    Attributes:
        dirname, filename, scaling_factor, sample_theta, lambda_nm, energy_ev, dataqxzqy, datarot, ivsqxz

    Args:
        fileformat: 'tiff' or 'fits' or 'gentiff' or 'gencsvtiff' or 'bin'
        fullfilename: full path of data file
        params: namedtuple of user parameters
        info: dict of instrument parameters for one image, from one row of .dat file, or fits header card
        infostr: info in nice readable format
    """
    def __init__(self, fileformat, fullfilename, params, info, infostr, div_photodiode=1):
        self.fileformat = fileformat
        self.fullfilename = fullfilename
        self.info = info
        self.infostr = infostr

        if (fileformat == 'tiff' or fileformat == 'gentiff' or fileformat == 'bin'):
            self.energy_ev = info['mono_act']
            self.lambda_nm = EV_NM / self.energy_ev
        elif fileformat == 'gencsvtiff':
            if 'energy_eV' in info.keys():
                self.energy_ev = info['energy_eV']
                self.lambda_nm = EV_NM / self.energy_ev
            elif 'wavelength_nm' in info.keys():
                self.lambda_nm = info['wavelength_nm']
                self.energy_ev = EV_NM / self.lambda_nm
        else:
            self.energy_ev = info[0]['Beamline Energy']
            self.lambda_nm = EV_NM / self.energy_ev

        self.dirname, self.filename = os.path.split(self.fullfilename)
        self.div_photodiode = div_photodiode
        self.scaling_factor = params.multiply_intensity / div_photodiode

        if fileformat == 'tiff':
            self.sample_theta = info['Sample_phi']  # sample_theta=0 should be normal to substrate
            if params.normalize_exposure:
                self.scaling_factor /= info['Seconds']
            if params.normalize_I0:
                self.scaling_factor /= info['IC_cntr1']
            if params.normalize_current:
                self.scaling_factor /= info['SRcurrent']
            self.dataqxzqy = DataQxzQy(fileformat, self.sample_theta, fullfilename, params, self.lambda_nm, self.scaling_factor)
        elif fileformat == 'fits':
            self.sample_theta = -info[0]['Sample Theta'] - 90  # sample_theta=0 should be normal to substrate
            if params.normalize_exposure:
                self.scaling_factor /= info[0]['EXPOSURE']
            if params.normalize_I0:
                self.scaling_factor /= info[0]['AI 3 Izero']
            if params.normalize_current:
                self.scaling_factor /= info[0]['Beam Current']
            self.dataqxzqy = DataQxzQy(fileformat, self.sample_theta, fullfilename, params, self.lambda_nm, self.scaling_factor,
                                       detector_theta=info[0]['CCD Theta'], detector_x=info[0]['CCD X'])
        elif fileformat == 'gentiff':
            self.sample_theta = info['Sample Theta']
            if params.normalize_exposure:
                self.scaling_factor /= info['Seconds']
            if params.normalize_I0:
                self.scaling_factor /= info['IC_cntr1']
            self.dataqxzqy = DataQxzQy(fileformat, self.sample_theta, fullfilename, params, self.lambda_nm, self.scaling_factor)
        elif fileformat == 'gencsvtiff':
            self.sample_theta = info['sample_phi_deg']
            if params.normalize_exposure:
                self.scaling_factor /= info['exposture_time_s']
            params._replace(SDD_cm=info['sdd_cm'])
            self.dataqxzqy = DataQxzQy(fileformat, self.sample_theta, fullfilename, params, self.lambda_nm, self.scaling_factor)
        elif fileformat == 'bin':
            self.sample_theta = info['Sample Theta']
            if params.normalize_exposure:
                self.scaling_factor /= info['Seconds']
            self.dataqxzqy = DataQxzQy(fileformat, self.sample_theta, fullfilename, params, self.lambda_nm, self.scaling_factor)
        else:
            raise TypeError('Invalid file type')


class DataQxzQy(object):
    """Object that loads Qxz-Qy image, user parameters, initialize with ScatteringFile.__init__

    Attributes: center_qxz_on_img, center_qy_on_img, subtract_bottom_value
        optional: imgdata, qxzs, qys

    Args: fileformat, sample_theta, fullfilename, params, lambda_nm, scaling_factor, detector_theta=None, detector_x=None
    """
    def __init__(self, fileformat, sample_theta, fullfilename, params, lambda_nm, scaling_factor,
                 detector_theta=None, detector_x=None):
        self.fileformat = fileformat
        self.sample_theta = sample_theta
        self.fullfilename = fullfilename
        self.params = params
        self.lambda_nm = lambda_nm
        self.scaling_factor = scaling_factor
        self.detector_theta = detector_theta if fileformat == 'fits' else None
        self.detector_x = detector_x if fileformat == 'fits' else None
        # center of image in pixels with corrections for detector theta, x, and scale
        if self.fileformat == 'tiff' or self.fileformat == 'gentiff' or self.fileformat == 'gencsvtiff' or self.fileformat == 'bin':
            self.center_qxz_on_img = self.params.center_px[0]
            self.center_qy_on_img = self.params.center_px[1]
        elif self.fileformat == 'fits':
            self.center_qxz_on_img = diffraction.center_px_qxz_on_img(
                params.center_px[0], params.pixel_um, params.SDD_cm, params.detector_thetascale, self.detector_theta, params.detector_theta0)
            self.center_qy_on_img = diffraction.center_px_qy_on_img(
                params.center_px[1], params.pixel_um, self.detector_x, params.detector_x0)

    def load_save_data(self, keepinmemory):
        """Returns data from file or from memory, optionally saves to memory"""
        if not keepinmemory:
            loaded_data = self.load_data()
        elif hasattr(self, 'imgdata'):
            loaded_data = (self.imgdata, self.qxzs, self.qys)
        else:
            loaded_data = self.load_data()
            (self.imgdata, self.qxzs, self.qys) = loaded_data
        return loaded_data

    def load_data(self):
        """Loads from file and creates image array and axes vectors

        Returns: imgdata, qxzs, qys
        """
        num_cw = PEAKS[self.params.peaks_direction]
        if self.fileformat == 'tiff' or self.fileformat == 'gentiff' or self.fileformat == 'gencsvtiff':
            try:
                imgdata = np.flipud(Image.open(self.fullfilename)) * self.scaling_factor
            except:
                imgdata = np.flipud(tifffile.imread(self.fullfilename)) * self.scaling_factor
        elif self.fileformat == 'bin':
            imgdata = np.fromfile(self.fullfilename, dtype=np.float64)
            if imgdata.size == 287626:
                imgdata = imgdata[1:]
            imgdata = imgdata.reshape(195, 1475)
        elif self.fileformat == 'fits':
            with open(self.fullfilename, 'rb') as f:
                imgdata = fits.getdata(f, ext=2) * self.scaling_factor
        imgdata = np.rot90(imgdata, k=num_cw)  # rot90: CCW normally, CW if origin='lower'
        if self.params.subtract_bottom:
            self.subtract_bottom_value = eval('imgdata[' + NOISE_REGION_STR + ']').mean()
            imgdata -= self.subtract_bottom_value  # subtract average of bottom 20 rows
        qxz_pixels = np.array(range(np.shape(imgdata)[0])) - self.params.center_px[0]
        qy_pixels = np.array(range(np.shape(imgdata)[1])) - self.params.center_px[1]
        if self.fileformat == 'tiff' or self.fileformat == 'gentiff' or self.fileformat == 'bin' or self.fileformat == 'gencsvtiff':
            qxzs = diffraction.qxz_pixels_to_qxz(qxz_pixels, self.lambda_nm, self.params.pixel_um, self.params.SDD_cm)
            qys = diffraction.qy_pixels_to_qy(qy_pixels, self.lambda_nm, self.params.pixel_um, self.params.SDD_cm)
        elif self.fileformat == 'fits':
            qxzs = diffraction.qxz_pixels_to_qxz(qxz_pixels, self.lambda_nm, self.params.pixel_um, self.params.SDD_cm,
                                                 self.params.detector_thetascale, self.detector_theta, self.params.detector_theta0)
            qys = diffraction.qy_pixels_to_qy(qy_pixels, self.lambda_nm, self.params.pixel_um, self.params.SDD_cm,
                                              self.detector_x, self.params.detector_x0)
        return imgdata, qxzs, qys

    def find_angle(self, rect_dims, loaded_data, mode):
        """Finds angle of box containing peaks by finding peaks in initial larger box and performing linear regression

        Args:
            rect_dims: processing.RectDims ['left', 'bottom', 'width', 'height', 'angle_deg', 'threshold', 'width_peaks']
            loaded_data: from load_save_data
            mode: 'fitangleleft', 'fitangle_calcleft', or 'fitangle_calcleft_bothsides'

        Returns:
            angle_deg: CCW from vertical
            left: of box, in pixels
            qxz_peaks, qy_peaks: locations of peaks in A^-1
        """
        if mode not in ['fitangleleft', 'fitangle_calcleft', 'fitangle_calcleft_bothsides']:
            raise ValueError
        rect_dims_wide = {}
        if (not USE_INPUT_ANGLE) or (mode == 'fitangle_calcleft_bothsides'):
            rect_dims_wide['angle_deg'] = 0  # vertical box to find peaks in
        # box to find peaks in is wider than integration box
        if mode == 'fitangleleft':
            rect_dims_wide['left'] = rect_dims.left - (rect_dims.width_peaks - rect_dims.width) / 2
        else:
            rect_dims_wide['left'] = self.center_qy_on_img - rect_dims.width_peaks / 2
        rect_dims_wide['width'] = rect_dims.width_peaks
        imgdata, qxzs, qys = loaded_data
        _, data_masked, qxz_masked, qy_masked = base.make_masked(imgdata, rect_dims._replace(**rect_dims_wide), qxzs, qys)

        if mode == 'fitangle_calcleft_bothsides':  # another box on the other side of beam center
            rect_dims_wide['bottom'] = RECT_BOTTOM_BELOWCENTER
            rect_dims_wide['height'] = 2 * self.center_qxz_on_img - rect_dims.bottom - RECT_BOTTOM_BELOWCENTER
            _, data_masked2, qxz_masked2, qy_masked2 = base.make_masked(imgdata, rect_dims._replace(**rect_dims_wide), qxzs, qys)
            data_masked = np.append(data_masked, data_masked2, axis=0)
            qxz_masked = np.append(qxz_masked, qxz_masked2)
            qy_masked = np.append(qy_masked, qy_masked2)

        # find peak indices: if max point in each row is higher than threshold*mean of row
        subtract_bottom_value = self.subtract_bottom_value if hasattr(self, 'subtract_bottom_value') else 0
        peak_indices_bool = (np.nanmax(data_masked, axis=1) + subtract_bottom_value) >= (
            (np.nanmean(data_masked, axis=1) + subtract_bottom_value) * rect_dims.threshold)
        # if beamstop is on image, ignore region near direct beam
        peak_indices_bool = peak_indices_bool & ((qxz_masked > FIND_ANGLE_MASK) | (qxz_masked < -FIND_ANGLE_MASK))
        # find qy and qxz of peaks
        qy_peaks = qy_masked[np.nanargmax(data_masked[peak_indices_bool], axis=1)]
        qxz_peaks = qxz_masked[peak_indices_bool]
        if mode == 'fitangleleft':
            indep = sm.add_constant(qxz_peaks)  # fit slope and intercept
        else:
            indep = qxz_peaks  # fit slope only

        # robust linear regression, qy is dependent variable, qxz is independent variable
        try:
            rlmresults = sm.RLM(qy_peaks, indep, M=Hampel()).fit()
            if FIND_ANGLE_PRINT:
                print(rlmresults.summary())
            if mode == 'fitangleleft':
                qy_intercept, slope = rlmresults.params
            else:
                qy_intercept = None
                slope = rlmresults.params[0]
        except:
            print('sample_theta={0}, detector_theta={1}: box angle not found, using initial'.format(
                self.sample_theta, self.detector_theta))
            return rect_dims.angle_deg, rect_dims.left, (), ()

        # angle from slope
        angle_deg = -np.arctan(slope) * 180 / np.pi
        if not FIND_ANGLE_MIN <= angle_deg <= FIND_ANGLE_MAX:
            print('sample_theta={0}, detector_theta={1}: box angle outside limits, using initial'.format(
                self.sample_theta, self.detector_theta))
            return rect_dims.angle_deg, rect_dims.left, (), ()
        # left from slope, qxz of box bottom, and qy_intercept
        if mode == 'fitangleleft':
            left_qy = slope * base.find_interp_value(qxzs, rect_dims.bottom) + qy_intercept
        else:
            left_qy = slope * base.find_interp_value(qxzs, rect_dims.bottom)
        left_qy_pixels = base.find_interp_index(qys, left_qy) - rect_dims.width / 2
        if FIND_ANGLE_PRINT:
            print('qy_intercept=', qy_intercept)
            print('slope=', slope)
            print('angle_deg=', angle_deg)
            print('left_qy=', left_qy)
            print('left_qy_pixels=', left_qy_pixels)
        return angle_deg, left_qy_pixels, qxz_peaks, qy_peaks

    @staticmethod
    def plot_self(mplwidget, loaded_data):
        """Plots Qxz-Qy"""
        imgdata, qxzs, qys = loaded_data
        mplwidget.pcolorfast(imgdata, cols=qys, rows=qxzs, xlabel='$q_y (\AA^{-1})$', ylabel='$q_{xz} (\AA^{-1})$')


class DataRot(object):
    """Object containing rotated Qy-Qxz image

    Attributes: rect_dims, rect (drawn in DataQxzQy), data_rotated, qxzs, qys

    Args: rect_dims, lambda_nm, sample_theta, imgdata, imgdata_qxz, imgdata_qy
    """
    def __init__(self, rect_dims, lambda_nm, sample_theta, imgdata, imgdata_qxz, imgdata_qy):
        self.rect_dims = rect_dims
        self.lambda_nm = lambda_nm
        self.sample_theta = sample_theta
        self.rect, data_masked, self.qxzs, self.qys = base.make_masked(imgdata, rect_dims, imgdata_qxz, imgdata_qy)
        self.data_rotated = np.flipud(np.rot90(data_masked))

    def plot_self(self, mplwidget):
        """Plots rotated Qy-Qxz"""
        mplwidget.pcolorfast(self.data_rotated, cols=self.qxzs, rows=self.qys,
                             xlabel='$q_{xz} (\AA^{-1})$', ylabel='$q_y (\AA^{-1})$')


class IvsQxz(object):
    """Object containing I vs Qxz plot

    Attributes:
        px_theta_rad_vector: theta in radians for each point
        I, qy0, qxzs, qxs, footprintfactor, absfactor, polarizationfactor

    Args: data_rotated, qys, qxzs, aparams, lambda_nm, sample_theta
    """
    def __init__(self, data_rotated, qys, qxzs, aparams, lambda_nm, sample_theta):
        if aparams.subtract_value == 'min':
            subtract_value = np.nanmin(data_rotated)
        else:
            subtract_value = aparams.subtract_value
        # qy0 is the extrapolated value of qy at qxz=0, nonzero qy0 is treated as real
        # while nonzero (qys - qy0) is due to sample misalignment and corrected out when finding self.qxzs
        _, qy_tiled = np.meshgrid(qxzs, qys)
        qy_tiled[np.isnan(data_rotated)] = np.nan  # ignore points outside box in qy median calculation
        qy_medians = np.nanmedian(qy_tiled, axis=0)
        qxzs_ = qxzs[~np.isnan(qy_medians)]
        qy_medians = qy_medians[~np.isnan(qy_medians)]  # UnivariateSpline can't handle nan
        self.qy0 = interpolate.UnivariateSpline(qxzs_, qy_medians, k=1)(0)
        self.I, _, self.qxzs = base.to1D(data_rotated, qys - self.qy0, qxzs, 0, subtract_value, aparams.reduce_fn)
        self.lambda_nm = lambda_nm
        self.sample_theta = sample_theta
        self.apply_correction(aparams)

    def apply_correction(self, aparams):
        self.aparams = aparams
        self.qzs, self.qxs, self.px_theta_rad_vector, sample_theta_corr_rad = diffraction.qxz_to_qz_qx(
            self.qxzs, self.qy0, self.lambda_nm, self.sample_theta, aparams.samplethetaoffset)

        # footprintfactor = I(path length=sample thickness) / I(path length) = (t/cos(0)) / (t/cos(theta)) = cos(theta)
        cos_sample_theta = np.cos(sample_theta_corr_rad)
        self.footprintfactor = cos_sample_theta if aparams.footprintcorr else 1

        # see project meeting 2015-07-22.pptx
        if aparams.samplesizecorr:
            sigma_times_sqrt2 = 1e-99 + aparams.beamfwhm / (2 * np.sqrt(np.log(2)))
            self.samplesizefactor = (
                (erf((aparams.beamcenter + aparams.samplesize / 2) / sigma_times_sqrt2) -
                 erf((aparams.beamcenter - aparams.samplesize / 2) / sigma_times_sqrt2)) /
                (erf((aparams.beamcenter + aparams.samplesize / 2) * cos_sample_theta / sigma_times_sqrt2) -
                 erf((aparams.beamcenter - aparams.samplesize / 2) * cos_sample_theta / sigma_times_sqrt2) + 1e-99))
        else:
            self.samplesizefactor = 1

        # absfactor = I(path length=substrate thickness) / I(path length)
        # = (I0 exp(-mu*t)) / (I0 exp(-mu*t/cos(theta))) = exp(-mu*t*(1-1/cos(theta)))
        if aparams.abscorr:
            self.absfactor = np.exp(-aparams.substratethickness * aparams.substrateattenuation * (1 - 1 / cos_sample_theta))
        else:
            self.absfactor = 1

        # see http://pd.chem.ucl.ac.uk/pdnn/diff2/polar.htm
        # polarizationfactor = I(px_theta=0) / I(px_theta) = 1 / (frac_y * cos(2*px_theta_y)**2 + (1 - frac_y) * cos(2*px_theta_xz)**2)
        if aparams.polarizationcorr:
            # px_thetas_qxz_rad = np.arcsin(self.qxzs * self.lambda_nm / 0.4 / np.pi) * 2
            # px_thetas_qy_rad = np.arcsin(self.qy0 * self.lambda_nm / 0.4 / np.pi) * 2
            # self.polarizationfactor = 1 / (
            #     aparams.frac_polarized_y * np.cos(px_thetas_qy_rad) ** 2 +
            #     (1 - aparams.frac_polarized_y) * np.cos(px_thetas_qxz_rad) ** 2)
            q = np.sqrt(self.qxzs ** 2 + self.qy0 ** 2)
            det_azimuth = np.arctan(self.qy0 / self.qxzs)
            px_thetas = np.arcsin(q * self.lambda_nm / 0.4 / np.pi) * 2
            self.polarizationfactor = 1 / (
                aparams.frac_polarized_y * (1 - np.sin(det_azimuth) ** 2 * np.sin(px_thetas) ** 2) +
                (1 - aparams.frac_polarized_y) * (1 - np.cos(det_azimuth) ** 2 * np.sin(px_thetas) ** 2))
        else:
            self.polarizationfactor = 1

        # sum_scattered = int_0^t I0 exp(-mu*x/cos(theta)) exp(-mu*(t-x)/cos(theta+px_theta))
        # sampleabsfactor = I(theta=0,px_theta=0) / I(theta,px_theta) = I0*t*exp(-mu*t) / I(theta,px_theta)
        if aparams.sampleabscorr:
            tu = aparams.samplethickness * aparams.sampleattenuation
            cos_sample_px_theta = np.cos(sample_theta_corr_rad + self.px_theta_rad_vector)
            self.sampleabsfactor = (
                np.exp(-tu) * tu * (1 / cos_sample_theta - 1 / cos_sample_px_theta) /
                (-np.exp(-tu / cos_sample_theta) + np.exp(-tu / cos_sample_px_theta)))
        else:
            self.sampleabsfactor = 1

        self.I = self.I * self.footprintfactor * self.samplesizefactor * self.absfactor * self.sampleabsfactor * self.polarizationfactor

    def reapply_correction(self, aparams):
        self.I = self.I / self.footprintfactor / self.samplesizefactor / self.absfactor / self.sampleabsfactor / self.polarizationfactor
        self.apply_correction(aparams)

    def plot_self(self, mplwidget, plot_qx=False):
        """Plots I vs Qxz"""
        if not plot_qx:
            mplwidget.plot(self.qxzs, self.I, xlabel='$q_{xz} (\AA^{-1})$', ylabel='I (counts)', yscale='log', label='I')
        else:
            mplwidget.plot(self.qxs, self.I, xlabel='$q_x (\AA^{-1})$', ylabel='I (counts)', yscale='log', label='I')


class ReducedData(object):
    """Top object corresponding to one set of sfs reduced to Qz-Qx
    Creates dataqzqx object

    Attributes:
        filenamelist: list of sf names
        dataqzqx, dataqzqxinterp

    Args:
        scatteringfilelist
        title: used as title of plots and in qlistwidget
        omit_overlap_det_theta=True: whether to skip portions of files with overlapping detector theta in Qz-Qx
    """
    def __init__(self, scatteringfilelist, title, omit_overlap_det_theta=True):
        self.filenamelist = [sf.fullfilename for sf in scatteringfilelist]
        self.dataqzqx = DataQzQx(scatteringfilelist, title, omit_overlap_det_theta)


class DataQzQx(object):
    """Object containing Qz-Qx data, initialize with ReducedData.__init__

    Attributes:
        sample_thetas: list of unique sample_thetas from scatteringfilelist
        px_theta_listoflists, detector_theta_listoflists, qz_listoflists, qx_listoflists, qy_listoflists, I_listoflists, Itrue_listoflists, corr_listoflists:
            for each sample theta, list of points in Qz-Qx, I_listoflists is what will be plotted

    Args:
        scatteringfilelist, title='',
        omit_overlap_det_theta=True: for each scattering file sf, if there is another scattering file sf1
            with sf1.sample_theta ~ sf.sample_theta and sf1.detector_theta > sf.detector_theta,
            only use the portion of sf with px_theta < sf1.px_theta[0]
            if multiple sf1's, only consider sf1 with lowest detector_theta
    """
    def __init__(self, scatteringfilelist, title='', omit_overlap_det_theta=True):
        self.title = title
        self.omit_overlap_det_theta = omit_overlap_det_theta
        px_theta_dict = defaultdict(lambda: [])
        detector_theta_dict = defaultdict(lambda: [])
        qz_dict = defaultdict(lambda: [])
        qx_dict = defaultdict(lambda: [])
        qy_dict = defaultdict(lambda: [])
        I_dict = defaultdict(lambda: [])
        corr_dict = defaultdict(lambda: [])
        if self.omit_overlap_det_theta:
            if all(sf.dataqxzqy.detector_theta is None for sf in scatteringfilelist):
                self.omit_overlap_det_theta = False  # only FITS files have detector_theta
            else:
                max_detector_theta = max(sf.dataqxzqy.detector_theta for sf in scatteringfilelist
                                         if sf.dataqxzqy.detector_theta is not None)
        for sf in scatteringfilelist:
            sample_theta = sf.sample_theta
            detector_theta = sf.dataqxzqy.detector_theta
            px_theta_rad_vector = sf.ivsqxz.px_theta_rad_vector
            groupnumber = sf.ivsqxz.aparams.groupnumber
            if self.omit_overlap_det_theta and sf.dataqxzqy.detector_theta != max_detector_theta:
                # find any detector_thetas greater than the one for the current sf
                matches = [sf1 for sf1 in scatteringfilelist if (sf1.dataqxzqy.detector_theta - detector_theta) > OVERLAP_DET_THETA_THRESH and
                           abs(sf1.sample_theta - sample_theta) <= OVERLAP_SAMPLE_THETA_THRESH and
                           ((groupnumber == 0) or (groupnumber != sf1.ivsqxz.aparams.groupnumber))]
                matches.sort(key=lambda match: match.dataqxzqy.detector_theta)
            else:
                matches = []
            px_theta_values = px_theta_rad_vector * 180 / np.pi
            qz_values = sf.ivsqxz.qzs
            qx_values = sf.ivsqxz.qxs
            I_values = sf.ivsqxz.I
            corr_values = (np.ones_like(sf.ivsqxz.I) * sf.ivsqxz.footprintfactor * sf.ivsqxz.samplesizefactor *
                           sf.ivsqxz.absfactor * sf.ivsqxz.sampleabsfactor * sf.ivsqxz.polarizationfactor)
            if len(matches) > 0:  # crop sf with lowest detector_theta greater than the one for the current sf
                bool_array = px_theta_rad_vector < matches[0].ivsqxz.px_theta_rad_vector[0]
                px_theta_values = px_theta_values[bool_array]
                qz_values = qz_values[bool_array]
                qx_values = qx_values[bool_array]
                I_values = I_values[bool_array]
                corr_values = corr_values[bool_array]
            # make list for sample_theta if it doesn't already exist, otherwise add to list
            detector_theta_dict[sample_theta].extend([detector_theta] * len(qz_values))
            px_theta_dict[sample_theta].extend(px_theta_values)
            qz_dict[sample_theta].extend(qz_values)
            qx_dict[sample_theta].extend(qx_values)
            qy_dict[sample_theta].extend([sf.ivsqxz.qy0] * len(I_values))
            I_dict[sample_theta].extend(I_values)
            corr_dict[sample_theta].extend(corr_values)

        # keep using list of lists instead of defaultdict so that old pickles can be loaded
        self.sample_thetas = sorted(qz_dict.keys())
        if all(sf.dataqxzqy.detector_theta is None for sf in scatteringfilelist):
            self.detector_theta_listoflists = None
        else:
            self.detector_theta_listoflists = [item[1] for item in sorted(detector_theta_dict.items())]
        self.px_theta_listoflists = [item[1] for item in sorted(px_theta_dict.items())]
        self.qz_listoflists = [item[1] for item in sorted(qz_dict.items())]
        self.qx_listoflists = [item[1] for item in sorted(qx_dict.items())]
        self.qy_listoflists = [item[1] for item in sorted(qy_dict.items())]
        self.I_listoflists = [item[1] for item in sorted(I_dict.items())]
        self.Itrue_listoflists = [item[1] for item in sorted(I_dict.items())]
        self.corr_listoflists = [item[1] for item in sorted(corr_dict.items())]

    def plot_self(self, mplwidget, mode='I'):
        """Draws non-interpolated plots

        Args:
            mplwidget
            mode:
                'I' for I vs Qz-Qx
                'detector_theta' for detector_theta vs Qz-Qx
                'sample_theta' for sample_theta vs Qz-Qx
                'px_theta' for px_theta vs Qz-Qx
                'I-theta' for I vs px_theta-sample_theta
                'corrections_only'
        """
        if not hasattr(self, 'Itrue_listoflists'):
            self.Itrue_listoflists = self.I_listoflists
        if mode == 'detector_theta' and self.detector_theta_listoflists is not None:
            self.I_listoflists = self.detector_theta_listoflists
        elif mode == 'sample_theta':
            self.I_listoflists = [[sample_theta] * len(qzs) for sample_theta, qzs in zip(self.sample_thetas, self.qz_listoflists)]
        elif mode == 'px_theta':
            self.I_listoflists = self.px_theta_listoflists
        elif mode == 'I-theta':
            self.I_listoflists = self.Itrue_listoflists
            colors = np.fromiter(itertools.chain(*self.I_listoflists), dtype=float)
            sample_theta_values = []
            for sample_theta, qzs in zip(self.sample_thetas, self.qz_listoflists):
                sample_theta_values.extend([sample_theta] * len(qzs))
            px_theta_values = np.fromiter(itertools.chain(*self.px_theta_listoflists), dtype=float)
            mplwidget.scatter(
                sample_theta_values, px_theta_values, xlabel='sample_theta (deg)', ylabel='px_theta (deg)',
                title=self.title, s=10, c=colors, linewidths=0)
            return
        elif mode == 'corrections_only':
            self.I_listoflists = self.corr_listoflists
        elif mode == 'I':
            self.I_listoflists = self.Itrue_listoflists
        else:
            print('Nothing plotted')
            return
        # flatten a list of lists
        colors = np.fromiter(itertools.chain(*self.I_listoflists), dtype=float)
        qx_values = np.fromiter(itertools.chain(*self.qx_listoflists), dtype=float)
        qz_values = np.fromiter(itertools.chain(*self.qz_listoflists), dtype=float)
        mplwidget.scatter(qx_values, qz_values, xlabel='$q_x (\AA^{-1})$', ylabel='$q_z (\AA^{-1})$',
                          title=self.title, s=10, c=colors, linewidths=0)

    def export_to_csv(self, csvstr):
        qx_values = np.fromiter(itertools.chain(*self.qx_listoflists), dtype=float)
        qz_values = np.fromiter(itertools.chain(*self.qz_listoflists), dtype=float)
        I_values = np.fromiter(itertools.chain(*self.I_listoflists), dtype=float)
        if hasattr(self, 'qy_listoflists'):
            qy_values = np.fromiter(itertools.chain(*self.qy_listoflists), dtype=float)
        else:
            qy_values = np.copy(I_values) * np.nan
        with open(csvstr, **opencsv_kwargs) as f:
            writer = csv.writer(f)
            writer.writerow(['qx', 'qy', 'qz', 'I'])
            writer.writerows(zip(qx_values, qy_values, qz_values, I_values))


class DataQzQxInterp(object):
    """Object containing bilinearly interpolated Qz-Qx image
       First interpolates points on each curve with regular qx interval, then linear interp on regular qz interval

    Attributes:
        interp_qx, interp_qz, interp_data

    Args: qx_listoflists, qz_listoflists, I_listoflists, qx_interp_size, qz_interp_size, title
    """
    def __init__(self, qx_listoflists, qz_listoflists, I_listoflists, qx_interp_size, qz_interp_size, title=''):
        qx_values = np.fromiter(itertools.chain(*qx_listoflists), dtype=float)
        qz_values = np.fromiter(itertools.chain(*qz_listoflists), dtype=float)
        self.interp_qx = np.arange(round(min(qx_values) / qx_interp_size) * qx_interp_size,
                                   round(max(qx_values) / qx_interp_size) * qx_interp_size + 1e-9, qx_interp_size)
        self.interp_qz = np.arange(round(min(qz_values) / qz_interp_size) * qz_interp_size,
                                   round(max(qz_values) / qz_interp_size) * qz_interp_size + 1e-9, qz_interp_size)
        self.interp_data = base.bilinear_arbitrary_axes(I_listoflists, qx_listoflists,
                                                        qz_listoflists, self.interp_qx, self.interp_qz)
        self.title = title + '_interp'

    def plot_self(self, mplwidget):
        """Plots interpolated Qz-Qx"""
        mplwidget.pcolorfast(self.interp_data, cols=self.interp_qx, rows=self.interp_qz,
                             xlabel='$q_x (\AA^{-1})$', ylabel='$q_z (\AA^{-1})$', title=self.title)


class IvsQz(object):
    """Vertical slice of Qz-Qx, initialize with DataQzQxInterp.makeivsqz

    Attributes:
        interp_I, interp_qz: slice of interpolated Qz-Qx, flattened to 1-D using mean
        I_values, qz_values, qx_values, qy_values: set of points in non-interpolated Qz-Qx within slice
        mean_theta, mean_qy, mean_qz, mean_I: for each scattering file, mean of points within slice in non-interpolated Qz-Qx
        rect, rect_bg: drawn in DataQzQxInterp

    Args: qx_midpt, qx_width, qx_bg, dataqzqx, dataqzqxinterp

    """
    def __init__(self, qx_midpt, qx_width, qx_bg, dataqzqx, dataqzqxinterp):
        # TODO: make qx_bg apply for non-interp also
        self.title = dataqzqx.title
        self.qx_midpt = qx_midpt
        self.qx_width = qx_width
        self.qx_bg = qx_bg
        min_qx = qx_midpt - qx_width / 2
        max_qx = qx_midpt + qx_width / 2

        # slice of dataqzqxinterp
        data_cropped, qz_cropped, qx_cropped, self.rect, self.rect_bg = base.make_rects(
            dataqzqxinterp.interp_data, dataqzqxinterp.interp_qz, dataqzqxinterp.interp_qx, 1, qx_bg, qx_midpt, qx_width)
        self.interp_I, self.interp_qz, _ = base.to1D(data_cropped, qz_cropped, qx_cropped, 1, None, np.mean)

        # slice of dataqzqx
        qx_values = np.fromiter(itertools.chain(*dataqzqx.qx_listoflists), dtype=float)
        qy_values = np.fromiter(itertools.chain(*dataqzqx.qy_listoflists), dtype=float)
        qz_values = np.fromiter(itertools.chain(*dataqzqx.qz_listoflists), dtype=float)
        I_values = np.fromiter(itertools.chain(*dataqzqx.I_listoflists), dtype=float)
        bool_array = (qx_values >= min_qx) & (qx_values <= max_qx)
        self.I_values = I_values[bool_array]
        self.qx_values = qx_values[bool_array]
        self.qy_values = qy_values[bool_array]
        self.qz_values = qz_values[bool_array]

        # mean non-interpolated data
        self.mean_theta = []
        self.mean_qy = []
        self.mean_qz = []
        self.mean_I = []
        for sample_theta, qx, qy, qz, I in zip(dataqzqx.sample_thetas, dataqzqx.qx_listoflists, dataqzqx.qy_listoflists, dataqzqx.qz_listoflists, dataqzqx.I_listoflists):
            qx_array = np.array(qx)
            qy_array = np.array(qy)
            qz_array = np.array(qz)
            I_array = np.array(I)
            bool_array = (qx_array >= min_qx) & (qx_array <= max_qx)
            mean_I = np.mean(I_array[bool_array])
            if not np.isnan(mean_I):
                self.mean_theta.append(sample_theta)
                self.mean_qy.append(np.mean(qy_array[bool_array]))
                self.mean_qz.append(np.mean(qz_array[bool_array]))
                self.mean_I.append(mean_I)

    def plot_interp(self, mplwidget, keep_old=True, num_decades=0):
        """Plots interpolated I vs Qz"""
        mplwidget.plot(self.interp_qz, 10 ** num_decades * self.interp_I, xlabel='$q_z (\AA^{-1})$', ylabel='I (interpolated)', yscale='log',
                       title='interp I vs Qz', keep_old=keep_old, label='qx = ' + str(self.qx_midpt))

    def plot_theta(self, mplwidget, keep_old=True, num_decades=0):
        """Plots non-interpolated I vs theta"""
        mplwidget.plot(self.mean_theta, 10 ** num_decades * np.asarray(self.mean_I),
                       xlabel='sample theta (deg)', ylabel='I (non-interpolated)', yscale='log',
                       title='I vs theta', keep_old=keep_old, label='qx = ' + str(self.qx_midpt))

    def plot_qz(self, mplwidget, keep_old=True, num_decades=0):
        """Plots non-interpolated I vs qz"""
        mplwidget.plot(self.mean_qz, 10 ** num_decades * np.asarray(self.mean_I),
                       xlabel='$q_z (\AA^{-1})$', ylabel='I (non-interpolated)', yscale='log',
                       title='I vs Qz', keep_old=keep_old, label='qx = ' + str(self.qx_midpt))


class IvsQx(object):
    """Horizontal slice of Qz-Qx

    Attributes:
        interp_I, interp_qx: slice of interpolated Qz-Qx, flattened to 1-D using mean
        I_values, qz_values, qx_values, qy_values: set of points in non-interpolated Qz-Qx within slice
        rect, rect_bg: drawn in DataQzQxInterp

    Args:
        qz_midpt, qz_width, qz_bg, dataqzqx, dataqzqxinterp
    """
    def __init__(self, qz_midpt, qz_width, qz_bg, dataqzqx, dataqzqxinterp):
        self.title = dataqzqx.title
        self.qz_midpt = qz_midpt
        self.qz_width = qz_width
        self.qz_bg = qz_bg
        min_qz = qz_midpt - qz_width / 2
        max_qz = qz_midpt + qz_width / 2

        # slice of dataqzqxinterp
        data_cropped, qz_cropped, qx_cropped, self.rect, self.rect_bg = base.make_rects(
            dataqzqxinterp.interp_data, dataqzqxinterp.interp_qz, dataqzqxinterp.interp_qx, 0, qz_bg, qz_midpt, qz_width)
        self.interp_I, self.interp_qx, _ = base.to1D(data_cropped, qz_cropped, qx_cropped, 0, None, np.mean)

        # slice of dataqzqx
        qx_values = np.fromiter(itertools.chain(*dataqzqx.qx_listoflists), dtype=float)
        qy_values = np.fromiter(itertools.chain(*dataqzqx.qy_listoflists), dtype=float)
        qz_values = np.fromiter(itertools.chain(*dataqzqx.qz_listoflists), dtype=float)
        I_values = np.fromiter(itertools.chain(*dataqzqx.I_listoflists), dtype=float)
        bool_array = (qz_values >= min_qz) & (qz_values <= max_qz)
        self.I_values = I_values[bool_array]
        self.qx_values = qx_values[bool_array]
        self.qy_values = qy_values[bool_array]
        self.qz_values = qz_values[bool_array]

    def plot_interp(self, mplwidget, keep_old=True, num_decades=0):
        """Plots interpolated I vs Qx"""
        mplwidget.plot(self.interp_qx, 10 ** num_decades * self.interp_I,
                       xlabel='$q_x (\AA^{-1})$', ylabel='I (interpolated)', yscale='log',
                       title='interp I vs Qx', keep_old=keep_old, label='qz = ' + str(self.qz_midpt))


class DataQxE(object):
    """Object containing Qx-E data"""
    def __init__(self, scatteringfilelist, title=''):
        self.title = title + '_QxE'
        self.qx_listoflists = []
        self.E_listoflists = []
        self.I_listoflists = []
        for sf in scatteringfilelist:
            self.qx_listoflists.append(list(sf.ivsqxz.qxs))
            self.E_listoflists.append(list(np.ones(len(sf.ivsqxz.qxs)) * sf.energy_ev))
            self.I_listoflists.append(list(sf.ivsqxz.I))

    def plot_self(self, mplwidget):
        E_values = np.fromiter(itertools.chain(*self.E_listoflists), dtype=float)
        qx_values = np.fromiter(itertools.chain(*self.qx_listoflists), dtype=float)
        I_values = np.fromiter(itertools.chain(*self.I_listoflists), dtype=float)
        mplwidget.scatter(E_values, qx_values, xlabel='E (eV)', ylabel='$q_x (\AA^{-1})$',
                          title=self.title, s=10, c=I_values, linewidths=0)


class DataQxEInterp(object):
    """Object containing Qx-E data interpolated in Qx direction
    qx_interp_size: interpolated qx pixel size in A^-1"""
    def __init__(self, E_listoflists, qx_listoflists, I_listoflists, qx_interp_size=0.0001, title=''):
        self.interp_E = np.array([i[0] for i in E_listoflists])
        qx_values = np.fromiter(itertools.chain(*qx_listoflists), dtype=float)
        self.interp_qx = np.arange(round(min(qx_values) / qx_interp_size) * qx_interp_size,
                                   round(max(qx_values) / qx_interp_size) * qx_interp_size + 1e-9, qx_interp_size)
        self.interp_data = base.bilinear_arbitrary_axes(I_listoflists, E_listoflists, qx_listoflists,
                                                        self.interp_E, self.interp_qx, interp_y_first=True)
        self.title = title + '_QxE_interp'

    def plot_self(self, mplwidget):
        mplwidget.pcolorfast(self.interp_data, cols=self.interp_E, rows=self.interp_qx,
                             xlabel='E (eV)', ylabel='$q_x (\AA^{-1})$', title=self.title)


class QxE_IvsQx(object):
    """Vertical slice of Qx-E

    Attributes:
        interp_I, interp_qx: slice of interpolated Qx-E, flattened to 1-D using mean
        mean_qx, mean_I: for each dataqxe, mean of points within slice in non-interpolated Qx-E
        rect, rect_bg: drawn in DataQxEInterp

    Args:
        E_midpt, E_width, E_bg, dataqxe, dataqxeinterp
    """
    def __init__(self, E_midpt, E_width, E_bg, dataqxe, dataqxeinterp):
        self.title = dataqxe.title
        self.E_midpt = E_midpt
        self.E_width = E_width
        self.E_bg = E_bg
        min_E = E_midpt - E_width / 2
        max_E = E_midpt + E_width / 2

        # slice of dataqxeinterp
        data_cropped, qx_cropped, E_cropped, self.rect, self.rect_bg = base.make_rects(
            dataqxeinterp.interp_data, dataqxeinterp.interp_qx, dataqxeinterp.interp_E, 1, E_bg, E_midpt, E_width)
        self.interp_I, self.interp_qx, _ = base.to1D(data_cropped, qx_cropped, E_cropped, 1, None, np.mean)

        # slice of dataqxe

        # mean non-interpolated data
        self.mean_qx = []
        self.mean_I = []
        for E, qx, I in zip(zip(*dataqxe.E_listoflists), zip(*dataqxe.qx_listoflists), zip(*dataqxe.I_listoflists)):
            E_array = np.array(E)
            qx_array = np.array(qx)
            I_array = np.array(I)
            bool_array = (E_array >= min_E) & (E_array <= max_E)
            self.mean_qx.append(np.mean(qx_array[bool_array]))
            self.mean_I.append(np.mean(I_array[bool_array]))

    def plot_interp(self, mplwidget, keep_old=True):
        """Plots interpolated I vs Qx"""
        mplwidget.plot(self.interp_qx, self.interp_I, xlabel='$q_x (\AA^{-1})$', ylabel='I (interpolated)', yscale='log',
                       title='interp I vs Qx', keep_old=keep_old, label='E = ' + str(self.E_midpt), marker='D')

    def plot_noninterp(self, mplwidget, keep_old=True):
        """Plots non-interpolated I vs Qx"""
        mplwidget.plot(self.mean_qx, self.mean_I, xlabel='$q_x (\AA^{-1})$', ylabel='I (non-interpolated)', yscale='log',
                       title='I vs Qx', keep_old=keep_old, label='E = ' + str(self.E_midpt), marker='D')


class QxE_IvsE(object):
    """Horizontal slice of Qx-E

    Attributes:
        interp_I, interp_E: slice of interpolated Qx-E, flattened to 1-D using mean
        mean_E, mean_I: for each dataqxe, mean of points within slice in non-interpolated Qx-E
        rect, rect_bg: drawn in DataQxEInterp

    Args:
        qx_midpt, qx_width, qx_bg, dataqxe, dataqxeinterp
    """
    def __init__(self, qx_midpt, qx_width, qx_bg, dataqxe, dataqxeinterp):
        self.title = dataqxe.title
        self.qx_midpt = qx_midpt
        self.qx_width = qx_width
        self.qx_bg = qx_bg
        min_qx = qx_midpt - qx_width / 2
        max_qx = qx_midpt + qx_width / 2

        # slice of dataqxeinterp
        data_cropped, qx_cropped, E_cropped, self.rect, self.rect_bg = base.make_rects(
            dataqxeinterp.interp_data, dataqxeinterp.interp_qx, dataqxeinterp.interp_E, 0, qx_bg, qx_midpt, qx_width)
        self.interp_I, self.interp_E, _ = base.to1D(data_cropped, qx_cropped, E_cropped, 0, None, np.mean)

        # slice of dataqxe

        # mean non-interpolated data
        self.mean_E = []
        self.mean_I = []
        for E, qx, I in zip(dataqxe.E_listoflists, dataqxe.qx_listoflists, dataqxe.I_listoflists):
            E_array = np.array(E)
            qx_array = np.array(qx)
            I_array = np.array(I)
            bool_array = (qx_array >= min_qx) & (qx_array <= max_qx)
            self.mean_E.append(np.mean(E_array[bool_array]))
            self.mean_I.append(np.mean(I_array[bool_array]))

    def plot_interp(self, mplwidget, keep_old=True):
        """Plots interpolated I vs E"""
        mplwidget.plot(self.interp_E, self.interp_I, xlabel='E (eV)', ylabel='I (interpolated)', yscale='log',
                       title='interp I vs E', keep_old=keep_old, label='qx = ' + str(self.qx_midpt), marker='D')

    def plot_noninterp(self, mplwidget, keep_old=True):
        """Plots non-interpolated I vs E"""
        mplwidget.plot(self.mean_E, self.mean_I, xlabel='E (eV)', ylabel='I (non-interpolated)', yscale='log',
                       title='I vs E', keep_old=keep_old, label='qx = ' + str(self.qx_midpt), marker='D')


class PlotsContainer(object):
    def __init__(self):
        pass


def export_slices_to_csv(full_ivsq_list, qx_True, indices, csvstr):
    """Exports three csv files each containing len(indices) sets of columns sorted from smallest to largest q
    Args:
        full_ivsq_list: list of IvsQz or IvsQx objects
        qx_True: True if list contains IvsQx objects, False if list contains IvsQz objects
        indices: list of indices of full_ivsq_list to export
        csvstr: string containing csv path and base filename
    """
    csv_prefix, _ = os.path.splitext(csvstr)
    # make ivsq_list by selecting indices from full_ivsq_list, sort it by q
    ivsq_list = [full_ivsq_list[index] for index in indices]
    ivsq_list.sort(key=lambda ivsq: ivsq.qz_midpt if qx_True else ivsq.qx_midpt)
    q_str = 'qz=' if qx_True else 'qx='

    # the first row of the mean and values csv files (4 columns per ivsq)
    firstrow = [i for ivsq in ivsq_list for i in [
                '{0}, {1}'.format(ivsq.title, q_str),
                '{0}'.format(ivsq.qz_midpt if qx_True else ivsq.qx_midpt),
                '',
                '']]
    # the first row of the interp csv file (2 columns per ivsq)
    firstrow_interp = [i for ivsq in ivsq_list for i in [
                       '{0}, {1}'.format(ivsq.title, q_str),
                       '{0}'.format(ivsq.qz_midpt if qx_True else ivsq.qx_midpt)]]

    # write the 3 csv files
    csv_filename = csv_prefix + '_values.csv'
    with open(csv_filename, **opencsv_kwargs) as f:
        writer = csv.writer(f)
        writer.writerow(firstrow)
        writer.writerow(['qz_values', 'qx_values', 'qy_values', 'I_values'] * len(ivsq_list))
        if hasattr(ivsq_list[0], 'qy_values'):
            writer.writerows(zip_longest(
                *[i for ivsq in ivsq_list for i in [ivsq.qz_values, ivsq.qx_values, ivsq.qy_values, ivsq.I_values]]))
        else:
            writer.writerows(zip_longest(
                *[i for ivsq in ivsq_list for i in [ivsq.qz_values, ivsq.qx_values, np.asarray(ivsq.qx_values) * np.nan, ivsq.I_values]]))

    if not qx_True:  # only save this file for vertical slices
        csv_filename = csv_prefix + '_mean.csv'
        with open(csv_filename, **opencsv_kwargs) as f:
            writer = csv.writer(f)
            writer.writerow(firstrow)
            writer.writerow(['mean_theta', 'mean_qz', 'mean_qy', 'mean_I'] * len(ivsq_list))
            if hasattr(ivsq_list[0], 'mean_qy'):
                writer.writerows(zip_longest(
                    *[i for ivsq in ivsq_list for i in [ivsq.mean_theta, ivsq.mean_qz, ivsq.mean_qy, ivsq.mean_I]]))
            else:
                writer.writerows(zip_longest(
                    *[i for ivsq in ivsq_list for i in [ivsq.mean_theta, ivsq.mean_qz, np.asarray(ivsq.mean_qz) * np.nan, ivsq.mean_I]]))

    csv_filename = csv_prefix + '_interp.csv'
    with open(csv_filename, **opencsv_kwargs) as f:
        writer = csv.writer(f)
        writer.writerow(firstrow_interp)
        if qx_True:
            writer.writerow(['interp_qx', 'interp_I'] * len(ivsq_list))
            writer.writerows(zip_longest(
                *[i for ivsq in ivsq_list for i in [ivsq.interp_qx, ivsq.interp_I]]))
        else:
            writer.writerow(['interp_qz', 'interp_I'] * len(ivsq_list))
            writer.writerows(zip_longest(
                *[i for ivsq in ivsq_list for i in [ivsq.interp_qz, ivsq.interp_I]]))


def return_datarot_ivsqxz(dataqxzqy, rect_dims, aparams, find_angle_mode, keepinmemory):
    """For a dataqxzqy, makes new datarot and ivsqxz, modifies dataqxzqy if keepinmemory
    Returns:
        sample_theta
        loaded_data, datarot, ivsqxz: new objects
    """
    try:
        loaded_data = dataqxzqy.load_save_data(keepinmemory)
        if find_angle_mode is not None:
            rect_dims_update = {}
            rect_dims_update['angle_deg'], rect_dims_update['left'], _, _ = dataqxzqy.find_angle(
                rect_dims, loaded_data, find_angle_mode)
            rect_dims = rect_dims._replace(**rect_dims_update)
        datarot = DataRot(rect_dims, dataqxzqy.lambda_nm, dataqxzqy.sample_theta, *loaded_data)
        ivsqxz = IvsQxz(datarot.data_rotated, datarot.qys, datarot.qxzs, aparams, datarot.lambda_nm, datarot.sample_theta)
#        dataqxzqy_new = dataqxzqy if keepinmemory else None
        return dataqxzqy.sample_theta, loaded_data, datarot, ivsqxz
    except:
        traceback.print_exc()


def parallel_modify_scatteringfilelist(scatteringfilelist, rect_dims, aparams, find_angle_mode,
                                       gui_window=None, loadfromitem=True, keepinmemory=False):
    """Creates a process pool and gives it a task for each sf
    As each task completes the sf is modified with the result
    If loadfromitem: use rect_dims and aparams from each sf if available, else use aparams argument
    """
    with futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        # http://stackoverflow.com/questions/20776189/concurrent-futures-vs-multiprocessing-in-python-3
        futures_dict = {}
        for index, sf in enumerate(scatteringfilelist):
            rect_dims_from_item = sf.datarot.rect_dims if (loadfromitem and hasattr(sf, 'datarot')) else rect_dims
            aparams_from_item = sf.ivsqxz.aparams if (loadfromitem and hasattr(sf, 'ivsqxz')) else aparams
            # Gives process pool a task for each sf
            futures_dict[executor.submit(return_datarot_ivsqxz, sf.dataqxzqy, rect_dims_from_item,
                                         aparams_from_item, find_angle_mode, keepinmemory)] = index
        for future in futures.as_completed(futures_dict):
            # Overwrites datarot, ivsqxz with new objects
            sample_theta, loaded_data, datarot, ivsqxz = future.result()
            index = futures_dict[future]
            sf = scatteringfilelist[index]
            sf.datarot = datarot
            sf.ivsqxz = ivsqxz
            # Shows progress
            message = ' '.join(['Processing sample_theta =', str(sample_theta), ';',
                                str(index + 1), 'out of', str(len(scatteringfilelist))])
            if gui_window is None:
                print(message)
                continue
            if gui_window.checkBox_showplotswhilereducing.isChecked():
                sf.dataqxzqy.plot_self(gui_window.m_QxzQy, loaded_data)
                gui_window.m_QxzQy.draw_collection((sf.datarot.rect,))
                sf.datarot.plot_self(gui_window.m_Rot)
                sf.ivsqxz.plot_self(gui_window.m_IvsQxz, gui_window.radioButton_plot_IvsQx.isChecked())
            gui_window.statusBar().showMessage(message)
            QtWidgets.QApplication.processEvents()  # forces update


def nonparallel_modify_scatteringfilelist(scatteringfilelist, rect_dims, aparams, find_angle_mode,
                                          gui_window=None, loadfromitem=True, keepinmemory=False):
    """Modifies scatteringfilelist
    """
    for index, sf in enumerate(scatteringfilelist):
        rect_dims_from_item = sf.datarot.rect_dims if (loadfromitem and hasattr(sf, 'datarot')) else rect_dims
        aparams_from_item = sf.ivsqxz.aparams if (loadfromitem and hasattr(sf, 'ivsqxz')) else aparams
        _, loaded_data, sf.datarot, sf.ivsqxz = return_datarot_ivsqxz(
            sf.dataqxzqy, rect_dims_from_item, aparams_from_item, find_angle_mode, keepinmemory)
        # Shows progress
        message = ' '.join(['Processing sample_theta =', str(sf.sample_theta), ';',
                            str(index + 1), 'out of', str(len(scatteringfilelist))])
        if gui_window is None:
            print(message)
            continue
        if gui_window.checkBox_showplotswhilereducing.isChecked():
            # gui_window.treeWidget_QxzQy.setCurrentItem(
            #     gui_window.treeWidget_QxzQy.itemBelow(gui_window.treeWidget_QxzQy.currentItem()))
            sf.dataqxzqy.plot_self(gui_window.m_QxzQy, loaded_data)
            gui_window.m_QxzQy.draw_collection((sf.datarot.rect,))
            sf.datarot.plot_self(gui_window.m_Rot)
            sf.ivsqxz.plot_self(gui_window.m_IvsQxz, gui_window.radioButton_plot_IvsQx.isChecked())
        gui_window.statusBar().showMessage(message)
        QtWidgets.QApplication.processEvents()  # forces update


def scatteringfilelist_fix_angles(scatteringfilelist):
    """If box angle of a scattering file is too far from its neighbors, fix it and remake datarot and ivsqxz"""
    for i, sf in enumerate(scatteringfilelist):
        if i == 0 or i == len(scatteringfilelist) - 1:
            continue
        previous_sf = scatteringfilelist[i - 1]
        next_sf = scatteringfilelist[i + 1]
        if (abs(sf.energy_ev - previous_sf.energy_ev) > FIX_ANGLE_ENERGY_THRESH) or (abs(sf.energy_ev - next_sf.energy_ev) > FIX_ANGLE_ENERGY_THRESH):
            continue
        if (abs(sf.sample_theta - previous_sf.sample_theta) > FIX_ANGLE_SAMPLE_THETA_THRESH) or (abs(sf.sample_theta - next_sf.sample_theta) > FIX_ANGLE_SAMPLE_THETA_THRESH):
            continue
        angle_avg = np.mean([previous_sf.datarot.rect_dims.angle_deg,
                             next_sf.datarot.rect_dims.angle_deg])
        angle_old = sf.datarot.rect_dims.angle_deg
        left_avg = np.mean([previous_sf.datarot.rect_dims.left,
                            next_sf.datarot.rect_dims.left])
        left_old = sf.datarot.rect_dims.left
        if abs(angle_old - angle_avg) > FIND_ANGLE_MAX_CHANGE:
            print('box angle corrected: sample_theta={0}, detector_theta={1}: angle:{2:.3f}->{3:.3f}, left:{4:.3f}->{5:.3f}'.format(
                sf.sample_theta, sf.dataqxzqy.detector_theta, angle_old, angle_avg, left_old, left_avg))
            rect_dims_update = {}
            rect_dims_update['angle_deg'] = angle_avg
            rect_dims_update['left'] = left_avg
            loaded_data = sf.dataqxzqy.load_data()
            sf.datarot = DataRot(sf.datarot.rect_dims._replace(**rect_dims_update), sf.lambda_nm, sf.sample_theta, *loaded_data)
            sf.ivsqxz = IvsQxz(sf.datarot.data_rotated, sf.datarot.qys, sf.datarot.qxzs,
                               sf.ivsqxz.aparams, sf.lambda_nm, sf.sample_theta)


def scatteringfilelist_fix_multiply_intensity(scatteringfilelist, intensities):
    """Set params.multiply_intensity of each sf to intensity from intensities vector"""
    for i, sf in enumerate(scatteringfilelist):
        params_new = sf.dataqxzqy.params._replace(multiply_intensity=intensities[i])
        sf.__init__(sf.fileformat, sf.fullfilename, params_new, sf.info, sf.infostr)


def calculations_test():
    params = Params(pixel_um=27, SDD_cm=5.04, peaks_direction='Vertical up', center_px=[620, 744],
                    normalize_exposure=True, normalize_I0=True, normalize_current=True,
                    multiply_intensity=1, subtract_bottom=True,
                    detector_theta0=-2.14, detector_x0=94.4, detector_thetascale=0.984,)
    dataset = DatasetFITS(params, folder='C:/Users/cdl/Desktop/2014 04 12')
    rect_dims = RectDims(left=740, bottom=30, width=30, height=950, angle_deg=1, threshold=2, width_peaks=40)
    aparams = AnalysisParams(
        subtract_value=0, reduce_fn=np.nanmean, samplethetaoffset=0.25,
        footprintcorr=True, samplesize=2, beamcenter=0, beamfwhm=0,
        substratethickness=850, substrateattenuation=0.001558, frac_polarized_y=1)
    # nonparallel_modify_scatteringfilelist(dataset.scatteringfilelist, rect_dims, aparams)
    parallel_modify_scatteringfilelist(dataset.scatteringfilelist, rect_dims, aparams)
    rd = ReducedData(dataset.scatteringfilelist, 'title')
    rd.dataqzqxinterp = DataQzQxInterp(rd.dataqzqx.qx_listoflists, rd.dataqzqx.qz_listoflists,
                                       rd.dataqzqx.I_listoflists, 0.0001, 0.0001)
    for qx_midpt in [0.0065, 0.013, 0.0195, 0.026]:
        rd.ivsqz_list.append(IvsQz(qx_midpt, 0.0007, 0, rd.dataqzqx, rd.dataqzqxinterp))
    return dataset, rd


def plots_calculations_test():
    c = PlotsContainer()
    c.mplwidget_qxzqy = matplotlibwidget.MatplotlibWidget2D()
    c.mplwidget_rot = matplotlibwidget.MatplotlibWidget2D()
    c.mplwidget_ivsqxz = matplotlibwidget.MatplotlibWidget1D()
    c.mplwidget_qzqx = matplotlibwidget.MatplotlibWidget2D()
    c.mplwidget_interp = matplotlibwidget.MatplotlibWidget2D()
    c.mplwidget_ivsqz = matplotlibwidget.MatplotlibWidget1D()
    dataset, rd = calculations_test()
    dataqxzqy = dataset.scatteringfilelist[0].dataqxzqy
    datarot = dataset.scatteringfilelist[0].datarot
    ivsqxz = dataset.scatteringfilelist[0].ivsqxz
    loaded_data = dataqxzqy.load_save_data(keepinmemory=False)
    dataqxzqy.plot_self(c.mplwidget_qxzqy, loaded_data)
    c.mplwidget_qxzqy.draw_collection((datarot.rect,))
    datarot.plot_self(c.mplwidget_rot)
    ivsqxz.plot_self(c.mplwidget_ivsqxz)
    rd.dataqzqx.plot_self(c.mplwidget_qzqx)
    rd.dataqzqxinterp.plot_self(c.mplwidget_interp)
    rects = list(itertools.chain(*[(ivsqz.rect, ivsqz.rect_bg) for ivsqz in rd.ivsqz_list]))
    c.mplwidget_interp.draw_collection(rects)
    for ivsqz in rd.ivsqz_list:
        ivsqz.plot_interp(c.mplwidget_ivsqz, keep_old=True)
    c.mplwidget_ivsqz.draw_legend()
    return c, dataset, rd


if __name__ == '__main__':
    c, dataset, rd = plots_calculations_test()

"""
%load_ext line_profiler
from cdsaxs_gui_legacy import processing
%lprun -f processing.plots_calculations_test -f processing.calculations_test -f processing.DataQzQx.__init__ -f \
processing.DataQzQxInterp.__init__ processing.plots_calculations_test()
"""
