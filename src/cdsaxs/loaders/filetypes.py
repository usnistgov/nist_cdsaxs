import os

from astropy.io import fits
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
import tifffile

import cdsaxs.loaders._loader_tools as loader_tools


def read_tiff(filepath):
    """
    Load an image and header from a tiff file.

    Parameters
    ----------
    filepath : str, path
        Path to the tiff file to be loaded.

    Returns
    -------
    NDArray
        Two-dimensional numpy array that contains the image data.
    str
        Formatted filepath used to load the data.
    dict
        Dictionary of the header information where the key: value paris
        correpond to the tag.name: tag.value pairs of the header tags.
    """
    filepath = loader_tools.clean_filepath(filepath=filepath)

    try:
        image = Image.open(filepath)
        image = np.array(image).astype(np.float64)
        header = {
            TAGS[key]: image.tag[key] for key in image.tag_v2
            if key in TAGS.keys()
            }
    except:
        image = tifffile.imread(filepath).astype(np.float64)
        with tifffile.TiffFile(filepath) as tif:
            header = {
                tag.name: tag.value
                for tag in tif.pages[0].tags}

    return image, filepath, header


def read_nist_bin(filepath):
    """
    Load an image and metadata from a NIST-formatted bin/info file pair
    from the CD-SAXS instrument in group 642.06.

    Parameters
    ----------
    filepath : str, path
        Path to the bin file to be loaded.
        The paired info file should be in the same directory and have
        the same filename (apart from the different extension).

    Returns
    -------
    NDArray
        Two-dimensional numpy array that contains the image data.
    str
        Formatted filepath used to load the data.
    dict
        Dictionary with metadata keyword: value pairs.
    """

    filepath = loader_tools.clean_filepath(filepath=filepath)

    # read the image from the .bin file first
    image = np.fromfile(filepath, dtype=np.float64)[1:].reshape(195, 1475)

    # read the .info file
    infopath = os.path.join(
        os.path.dirname(filepath),
        os.path.basename(filepath)[:-4] + ".info"
    )
    file = open(infopath)
    sample_meta = file.readlines()
    sample_meta = {x.split('=')[0]: x.split('=')[1] for x in sample_meta}
    file.close()

    # extact required information and insert into clean dictionary
    metadata = {}
    metadata['wavelength_nm'] = float(sample_meta['Wavelength (nm) '])
    metadata['exposure_time_s'] = float(sample_meta['LiveTime '])
    metadata['pixel_size_um'] = float(sample_meta['Pixel Size '])

    return image, filepath, metadata


def read_fits(filepath):
    """
    Load an image and header from a fits file.

    Parameters
    ----------
    filepath : str, path
        Path to the fits file to be loaded.

    Returns
    -------
    NDArray
        Two-dimensional numpy array that contains the image data.
    str
        Formatted filepath used to load the data.
    dict
        Dictionary of the header information where the key: value pairs.
    """

    filepath = loader_tools.clean_filepath(filepath=filepath)

    # load image
    image = fits.getdata(filepath, ext=2).astype(np.float64)

    # header dictionary
    info = [hdu.header for hdu in fits.open(filepath)][0]
    header = dict(info)

    return image, filepath, header


def read_als_11_0_1_2(filepath):

    """
    Load an image and metadata from data collected at beamline 11.0.1.2
    at ALS.

    CAUTION: This loader assumes data collected in 2025 or earlier
    uses a detector with a 27 um pixel size. Any data collected in
    2026 or later is assumed to have a pixel size of 9 um.

    Parameters
    ----------
    filepath : str, path
        Path to the fits file to be loaded.


    Returns
    -------
    NDArray
        Two-dimensional numpy array that contains the image data.
    str
        Formatted filepath used to load the data.
    dict
        Dictionary with metadata keyword: value pairs.
    """
    filepath = loader_tools.clean_filepath(filepath=filepath)

    image, filepath, header = read_fits(filepath)


    metadata = {}
    metadata['energy_ev'] = header['Beamline Energy']
    metadata['sample_phi_deg'] = header['Sample Theta'] + 90
    metadata['I0'] = header['AI 3 Izero']
    metadata['exposure_time_s'] = header['EXPOSURE']
    metadata['beam_current'] = header['Beam Current']
    metadata['detector_phi_deg'] = header['CCD Theta']
    metadata['detector_y_mm'] = header['CCD X']
    metadata['CCD Y'] = header['CCD Y']
    metadata['epu_polarization'] = header['EPU Polarization']
    metadata['beam_stop_position'] = header['Beam Stop']
    date = header['DATE']
    if float(date[:4]) <= 2025:
        metadata['pixel_size_um'] = 27
        # orient the detector image with our coordinate system
        image = np.flipud(np.rot90(image, 3))
    else:
        metadata['pixel_size_um'] = 9
        # orient the detector image with our coordinate system
        image = np.flipud(np.rot90(image, 2))

    return image, filepath, metadata


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



class ScatteringFile


def __init__(self, fileformat, fullfilename, params, info, infostr, div_photodiode=1):
    self.fileformat = fileformat
    self.fullfilename = fullfilename
    self.info = info
    self.infostr = infostr

    if (fileformat == 'tiff' or fileformat == 'gentiff' or fileformat == 'bin'):
        self.energy_ev = info['mono_act']
        self.lambda_nm = EV_NM / self.energy_ev
    elif fileformat == 'gencsvtiff':
        if 'energy_ev' in info.keys():
            self.energy_ev = info['energy_ev']
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
            self.scaling_factor /= info['exposure_time_s']
        # this isn't quite correct, just a patch for SMI data
        # TODO: fix SMI bpm normalization
        if params.normalize_I0:
            if 'bpm' in info.keys():
                self.scaling_factor /= info['bpm']
            elif 'IO' in info.keys():
                self.scaling_factor /= info['IO']
            #TODO: Implement error handling if bpm or IO not given but box selected. Currently left commented out as place holder.
            #else:
                #warning.warn("No incident beam intensity found in csv. Data not normalized by IO.")
        if 'sdd_cm' in info.keys():
            params = params._replace(SDD_cm=info['sdd_cm'])
        self.dataqxzqy = DataQxzQy(fileformat, self.sample_theta, fullfilename, params, self.lambda_nm, self.scaling_factor)
    elif fileformat == 'bin':
        self.sample_theta = info['Sample Theta']
        if params.normalize_exposure:
            self.scaling_factor /= info['Seconds']
        self.dataqxzqy = DataQxzQy(fileformat, self.sample_theta, fullfilename, params, self.lambda_nm, self.scaling_factor)
    else:
        raise TypeError('Invalid file type')




class DataQxzQy(object):
    Object that loads Qxz-Qy image, user parameters, initialize with ScatteringFile.__init__

    Attributes: center_qxz_on_img, center_qy_on_img, subtract_bottom_value
        optional: imgdata, qxzs, qys

    Args: fileformat, sample_theta, fullfilename, params, lambda_nm, scaling_factor, detector_theta=None, detector_x=None
    
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
        Returns data from file or from memory, optionally saves to memory
        if not keepinmemory:
            loaded_data = self.load_data()
        elif hasattr(self, 'imgdata'):
            loaded_data = (self.imgdata, self.qxzs, self.qys)
        else:
            loaded_data = self.load_data()
            (self.imgdata, self.qxzs, self.qys) = loaded_data
        return loaded_data

    def load_data(self):
        Loads from file and creates image array and axes vectors

        Returns: imgdata, qxzs, qys
        
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
        Finds angle of box containing peaks by finding peaks in initial larger box and performing linear regression

        Args:
            rect_dims: processing.RectDims ['left', 'bottom', 'width', 'height', 'angle_deg', 'threshold', 'width_peaks']
            loaded_data: from load_save_data
            mode: 'fitangleleft', 'fitangle_calcleft', or 'fitangle_calcleft_bothsides'

        Returns:
            angle_deg: CCW from vertical
            left: of box, in pixels
            qxz_peaks, qy_peaks: locations of peaks in A^-1
    
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
        Plots Qxz-Qy
        imgdata, qxzs, qys = loaded_data
        mplwidget.pcolorfast(imgdata, cols=qys, rows=qxzs, xlabel='$q_y (\AA^{-1})$', ylabel='$q_{xz} (\AA^{-1})$')


"""
