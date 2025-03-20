# -*- coding: utf-8 -*-
"""
This is part of the CDSAXS Data Processing GUI.
This module contains the MainWindow class, which inherits from the automatically generated Ui_MainWindow.
It contains the methods that run when widgets in the GUI are manipulated.
It should not contain data processing code but instead call another module to do that.
"""

from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import *

import os
import sys
# import cPickle as pickle
import pickle
import json
import itertools
import warnings
import io

from qtpy import QtCore, QtGui, QtWidgets, QtWebEngineWidgets, uic
from qtpy.QtCore import Slot
import numpy as np
import pandas as pd
from IPython.lib import guisupport
from scipy import interpolate

USE_ASTROPY = False  # astropy doesn't mix well with pyinstaller and currently (1.2.1) fails to load FITS files sometimes
if USE_ASTROPY:
    from astropy.io import fits
    import pyfits  # lets you load pickles saved with pyfits
else:
    import pyfits as fits

from cdsaxs_gui_legacy import base, processing, matplotlibwidget, ALS_run_generator

warnings.filterwarnings('ignore')
datasets = []
reduceddatas = []
ivsqx_list = []
ivsqz_list = []
photodiode_list = []
qxe_list = []
qxe_ivsqx_list = []
qxe_ivse_list = []
ALS_frame = pd.DataFrame()
ALS_exposure_frame = pd.DataFrame(columns=(
    'Energy', 'EPU', 'Sample theta (-90 normal)', 'CCD Theta', 'First peak order', 'Time', 'Max I', 'Calc best time'))
NEWDIRNAME = os.path.abspath('E:/saxs/APS July 2015/hs104')  # desktop
NEWDIRNAME = os.path.abspath('C:/Users/Chris/Desktop/ALS Oct 2016')  # laptop


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui_dir = os.path.dirname(os.path.realpath(__file__))
        if getattr(sys, 'frozen', False):
            self.html_file = os.path.join(self.ui_dir, 'readme_processing.md.html')
        else:
            self.html_file = os.path.join(self.ui_dir, 'readme_processing.md.html')
        uic.loadUi(os.path.join(self.ui_dir, 'CDSAXS_gui_ui.ui'), self)
#        self.setupUi(self)
        self.selecteddirectory = os.path.expanduser('~')

        with open(self.html_file, 'r', encoding='utf8') as f:
            html = f.read()
        # the modified markdeep script makes the MathJax scale 200% because QWebView is bad at rendering
        html = html.replace('<script src="https://casual-effects.com/markdeep/latest/markdeep.min.js"></script>',
                            '<script src="markdeep_qwebview"></script>')
        if getattr(sys, 'frozen', False):
            self.webView.setHtml(html, baseUrl=QtCore.QUrl.fromLocalFile(os.path.join(self.ui_dir, 'mainwindow')))
        else:
            self.webView.setHtml(html, baseUrl=QtCore.QUrl.fromLocalFile(self.ui_dir))

        # custom connections
        self.actionExit.triggered.connect(self.close)
        self.m_QzQx_interp.canvas.mpl_connect('button_press_event', self.m_QzQx_interp_clicked)
        self.actionImport_FITS.triggered.connect(self.qfiledialog_wrapper)
        self.actionImport_TIFF.triggered.connect(self.qfiledialog_wrapper)
        self.actionImport_TIFF_PIL.triggered.connect(self.qfiledialog_wrapper)
        self.actionImport_BIN.triggered.connect(self.qfiledialog_wrapper)
        self.actionSave_sample_parameters.triggered.connect(self.qfiledialog_wrapper)
        self.actionLoad_sample_parameters.triggered.connect(self.qfiledialog_wrapper)
        self.actionExport_reduced_data_to_CSV.triggered.connect(self.qfiledialog_wrapper)
        self.actionExport_slices_to_CSV.triggered.connect(self.qfiledialog_wrapper)
        self.actionLoad_from_pickle.triggered.connect(self.qfiledialog_wrapper)
        self.actionSave_to_pickle.triggered.connect(self.qfiledialog_wrapper)
        self.pushButton_photodiode_load.clicked.connect(self.qfiledialog_wrapper)
        self.m_QxEobj_interp.canvas.mpl_connect('button_press_event', self.m_QxEobj_interp_clicked)
        self.listWidget_QzQx.model().rowsMoved.connect(self.listWidget_QzQx_model_rowsMoved)
        self.listWidget_IvsQ.model().rowsMoved.connect(self.listWidget_IvsQ_model_rowsMoved)
        self.pushButton_gen_loadexposures.clicked.connect(self.qfiledialog_wrapper)
        self.pushButton_gen_savetxt.clicked.connect(self.qfiledialog_wrapper)
        self.pushButton_gen_loadtxt.clicked.connect(self.qfiledialog_wrapper)
        self.plainTextEdit_gen_runtext.focusOutEvent = self.plainTextEdit_gen_runtext_customhandler
        # allows editing of item names
        self.listwidgetitemflags = (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable |
                                    QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsDragEnabled)
        self.showMaximized()
        # hides base methods from tab completion if %config IPCompleter.limit_to__all__ = True
#        self.__all__ = [i for i in dir(self) if i not in dir(QtWidgets.QMainWindow)]
        qactiongroup = QtWidgets.QActionGroup(self)
        # only one of these 4 can be checked
        qactiongroup.addAction(self.actionManual_angle_and_left)
        qactiongroup.addAction(self.actionFit_angle_and_left)
        qactiongroup.addAction(self.actionFit_angle)
        qactiongroup.addAction(self.actionFit_both_sides)

        self.linspacearangedialog = SliceCenterPitchDialog(__file__, parent=self)
        self.miscsettingsdialog = MiscSettingsDialog(__file__, parent=self)
        if getattr(sys, 'frozen', False):
            app_icon = QtGui.QIcon(os.path.join(self.ui_dir, 'images', 'icon.png'))
        else:
            app_icon = QtGui.QIcon(os.path.join(self.ui_dir, '..', 'images', 'icon.png'))
        self.setWindowIcon(app_icon)

    def make_params(self, fileformat):
        """makes params dict from values in the GUI"""
        params = processing.Params(
            pixel_um=self.doubleSpinBox_pixel_um.value(),
            SDD_cm=self.doubleSpinBox_SDD_cm.value(),
            peaks_direction=self.comboBox_peaksdirection.currentText(),
            center_px=[self.doubleSpinBox_centerY_px.value(), self.doubleSpinBox_centerX_px.value()],
            normalize_exposure=self.checkBox_normalizeexposure.isChecked(),
            normalize_I0=self.checkBox_normalizeI0.isChecked(),
            normalize_current=self.checkBox_normalizecurrent.isChecked(),
            multiply_intensity=float(self.lineEdit_multiplyintensity.text()),
            subtract_bottom=self.checkBox_subtractbottom.isChecked(),
            detector_theta0=self.doubleSpinBox_detector_theta0.value() if fileformat == 'fits' else None,
            detector_x0=self.doubleSpinBox_detector_x0.value() if fileformat == 'fits' else None,
            detector_thetascale=self.doubleSpinBox_detector_thetascale.value() if fileformat == 'fits' else None)
        return params

    def make_rect_dims(self):
        rect_dims = processing.RectDims(
            bottom=self.doubleSpinBox_boxbottom_px.value(),
            width=self.doubleSpinBox_boxwidth_px.value(),
            left=self.doubleSpinBox_boxleft_px.value(),
            height=self.doubleSpinBox_boxheight_px.value(),
            angle_deg=self.doubleSpinBox_boxangle.value() - 90,
            threshold=self.doubleSpinBox_threshold.value(),
            width_peaks=self.spinBox_widthpeaks_px.value())
        return rect_dims

    def make_aparams(self):
        aparams = processing.AnalysisParams(
            subtract_value=float(self.lineEdit_subtractvalue.text()) if self.radioButton_subtractvalue.isChecked()
            else 'min',
            reduce_fn=np.nanmean if self.radioButton_integratemean.isChecked() else np.nansum,
            samplethetaoffset=self.doubleSpinBox_samplethetaoffset.value(),
            footprintcorr=self.checkBox_footprintcorr.isChecked(),
            samplesize=self.doubleSpinBox_samplesize.value(),
            beamcenter=self.doubleSpinBox_beamcenter.value(),
            beamfwhm=self.doubleSpinBox_beamfwhm.value(),
            substratethickness=self.doubleSpinBox_substratethickness.value(),
            substrateattenuation=self.doubleSpinBox_substrateattenuation.value(),
            samplethickness=self.doubleSpinBox_samplethickness.value(),
            sampleattenuation=self.doubleSpinBox_sampleattenuation.value(),
            frac_polarized_y=self.doubleSpinBox_fracpolarizedy.value(),
            samplesizecorr=self.checkBox_samplesizecorr.isChecked(),
            abscorr=self.checkBox_abscorr.isChecked(),
            sampleabscorr=self.checkBox_sampleabscorr.isChecked(),
            polarizationcorr=self.checkBox_polarizationcorr.isChecked(),
            groupnumber=self.spinBox_groupnumber.value(),
        )
        return aparams

    def get_find_angle_mode(self):
        modes = np.array([None, 'fitangleleft', 'fitangle_calcleft', 'fitangle_calcleft_bothsides'])
        index = np.array([self.actionManual_angle_and_left.isChecked(), self.actionFit_angle_and_left.isChecked(),
                          self.actionFit_angle.isChecked(), self.actionFit_both_sides.isChecked()])
        return modes[index][0]

    def modify_gui_TIFF(self):
        self.doubleSpinBox_pixel_um.setEnabled(True)
        self.label_detector_theta0.hide()
        self.doubleSpinBox_detector_theta0.hide()
        self.label_detector_x0.hide()
        self.doubleSpinBox_detector_x0.hide()
        self.label_detector_thetascale.hide()
        self.doubleSpinBox_detector_thetascale.hide()
        self.label_center_qxz_on_img.hide()
        self.label_center_qy_on_img.hide()
        self.label_center_qxz_on_img_2.hide()
        self.label_center_qy_on_img_2.hide()

    def modify_gui_FITS(self):
        self.doubleSpinBox_pixel_um.setValue(27)
        self.doubleSpinBox_pixel_um.setEnabled(False)
        self.label_detector_theta0.show()
        self.doubleSpinBox_detector_theta0.show()
        self.label_detector_x0.show()
        self.doubleSpinBox_detector_x0.show()
        self.label_detector_thetascale.show()
        self.doubleSpinBox_detector_thetascale.show()
        self.label_center_qxz_on_img.show()
        self.label_center_qy_on_img.show()
        self.label_center_qxz_on_img_2.show()
        self.label_center_qy_on_img_2.show()

    def get_scatteringfilelist(self):
        selectedrows = get_sel_qmodelindices(self.treeWidget_QxzQy)
        scatteringfilelist = [datasets[qmodelindex.parent().row()].scatteringfilelist[qmodelindex.row()]
                              for qmodelindex in selectedrows if qmodelindex.parent().isValid()]
        return scatteringfilelist

# %% Menus

    @Slot()
    def on_actionAbout_triggered(self):
        with open(self.html_file, 'r', encoding='utf8') as f:
            f.readline()
            f.readline()
            f.readline()
            version = f.readline()
        QtWidgets.QMessageBox.about(self, 'About CDSAXS Data Processing GUI', version +
                                    'Christopher Liman\n'
                                    'National Institute of Standards and Technology')

    @Slot()
    def on_actionTest_triggered(self):
        filename = os.path.abspath('Z:/my documents/CDSAXSMCMCProject/python_code/params/2014-12-13.json')
        self.actionLoad_sample_parameters_triggered(filename)
        filename = os.path.abspath('C:/Users/cdl/Desktop/2014 12 13/specfiles/intel158a_a1.dat')
        self.actionImport_TIFF_triggered(filename)
        self.on_pushButton_integrate_correct_files_clicked()
        self.on_pushButton_new_reduced_clicked()
        values = np.arange(0.004672, 0.127367, 0.0011575)
        self.plainTextEdit_listQ_tointegrate.setPlainText('\n'.join([str(i) for i in values]))
        self.on_pushButton_integrateto_IvsQ_clicked()

    def qfiledialog_wrapper(self):
        """Instead of directly connecting slots that call QFileDialog functions, this allows these slots
        to also be called by tests more easily"""
        sender = self.sender()
        if sender is self.actionImport_FITS:
            filefn = QtWidgets.QFileDialog.getOpenFileNames
            caption = 'Choose ALS .fits files to load:'
            filefilter = '*.fits'
            slot = self.actionImport_FITS_triggered
        elif sender is self.actionImport_TIFF:
            filefn = QtWidgets.QFileDialog.getOpenFileName
            caption = 'Choose Argonne .dat file to load:'
            filefilter = '*.dat'
            slot = self.actionImport_TIFF_triggered
        elif sender is self.actionImport_TIFF_PIL:
            filefn = QtWidgets.QFileDialog.getOpenFileName
            caption = 'Choose Argonne .dat file to load:'
            filefilter = '*.dat'
            slot = self.actionImport_TIFF_PIL_triggered            
        elif sender is self.actionImport_BIN:
            filefn = QtWidgets.QFileDialog.getOpenFileNames
            caption = 'Choose .bin files to load:'
            filefilter = '*.bin'
            slot = self.actionImport_BIN_triggered
        elif sender is self.actionSave_sample_parameters:
            filefn = QtWidgets.QFileDialog.getSaveFileName
            caption = 'Choose .json file to save to:'
            filefilter = '*.json'
            slot = self.actionSave_sample_parameters_triggered
        elif sender is self.actionLoad_sample_parameters:
            filefn = QtWidgets.QFileDialog.getOpenFileName
            caption = 'Choose .json file to load:'
            filefilter = '*.json'
            slot = self.actionLoad_sample_parameters_triggered
        elif sender is self.actionExport_reduced_data_to_CSV:
            filefn = QtWidgets.QFileDialog.getSaveFileName
            caption = 'Choose .csv file to save to:'
            filefilter = '*.csv'
            slot = self.rd.dataqzqx.export_to_csv
        elif sender is self.actionExport_slices_to_CSV:
            filefn = QtWidgets.QFileDialog.getSaveFileName
            caption = 'Choose file prefix to save to:'
            filefilter = '*.csv'
            slot = self.actionExport_slices_to_CSV_triggered
        elif sender is self.actionLoad_from_pickle:
            filefn = QtWidgets.QFileDialog.getOpenFileName
            caption = 'Choose .p file to load:'
            filefilter = '*.p'
            slot = self.actionLoad_from_pickle_triggered
        elif sender is self.actionSave_to_pickle:
            filefn = QtWidgets.QFileDialog.getSaveFileName
            caption = 'Choose .p file to save to:'
            filefilter = '*.p'
            slot = self.actionSave_to_pickle_triggered
        elif sender is self.pushButton_photodiode_load:
            filefn = QtWidgets.QFileDialog.getOpenFileNames
            caption = 'Choose .txt files to load:'
            filefilter = '*.txt'
            slot = self.pushButton_photodiode_load_clicked
        elif sender is self.pushButton_gen_loadexposures:
            filefn = QtWidgets.QFileDialog.getOpenFileNames
            caption = 'Choose ALS .fits files to load:'
            filefilter = '*.fits'
            slot = self.pushButton_gen_loadexposures_clicked
        elif sender is self.pushButton_gen_savetxt:
            filefn = QtWidgets.QFileDialog.getSaveFileName
            caption = 'Choose .txt file to save to:'
            filefilter = '*.txt'
            slot = self.pushButton_gen_savetxt_clicked
        elif sender is self.pushButton_gen_loadtxt:
            filefn = QtWidgets.QFileDialog.getOpenFileName
            caption = 'Choose .txt file to load:'
            filefilter = '*.txt'
            slot = self.pushButton_gen_loadtxt_clicked
        else:
            self.statusBar().showMessage('Error in file dialog wrapper.')
            return
        filename = filefn(parent=self, caption=caption, directory=self.selecteddirectory, filter=filefilter)
        if not (QtCore.QT_VERSION >> 16) == 4:
            filename = filename[0]
        if len(filename) == 0:
            return
        filename1 = filename[0] if isinstance(filename, list) else filename
        self.selecteddirectory = QtCore.QFileInfo(filename1).absolutePath()
        slot(filename)

    def actionImport_FITS_triggered(self, filelist):
        self.statusBar().showMessage('Importing FITS...')
        dataset = processing.DatasetFITS(self.make_params('fits'), filelist)
        self.addto_treeWidget_QxzQy(dataset)
        self.statusBar().showMessage('Importing FITS done.')

    def actionImport_BIN_triggered(self, filelist):
        self.statusBar().showMessage('Importing BIN...')
        dataset = processing.DatasetBIN_INFO(self.make_params('bin'), filelist)
        self.addto_treeWidget_QxzQy(dataset)
        self.statusBar().showMessage('Importing BIN done.')

    def actionImport_TIFF_triggered(self, fullfilename):
        dirname, filename = os.path.split(fullfilename)
        folder_auto = os.path.abspath(os.path.join(dirname, '..', 'hs104', os.path.splitext(filename)[0]))
        dialog = QtWidgets.QDialog(parent=self)
        layout = QtWidgets.QGridLayout(dialog)
        layout.addWidget(QtWidgets.QLabel(text='folder that contains TIFF files'))
        folder = QtWidgets.QLineEdit(text=folder_auto)
        layout.addWidget(folder)
        layout.addWidget(QtWidgets.QLabel(text="1-based index of block (starting with #S) of .dat file to load, or 'all'"))
        part = QtWidgets.QLineEdit(text='all')
        layout.addWidget(part)
        layout.addWidget(QtWidgets.QLabel(text='1-based index of first file to load from the block'))
        start = QtWidgets.QLineEdit(text='1')
        layout.addWidget(start)
        layout.addWidget(QtWidgets.QLabel(text="1-based index of last file to load from the block, or 'all'"))
        stop = QtWidgets.QLineEdit(text='all')
        layout.addWidget(stop)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        ok_or_cancel = dialog.exec_()
        if ok_or_cancel == 0:
            return
        folder = folder.text()
        part = int(part.text()) if part.text() != 'all' else None
        start = int(start.text()) - 1
        stop = int(stop.text()) if stop.text() != 'all' else None
        # make objects
        self.statusBar().showMessage('Importing TIFF...')
        dataset = processing.DatasetTIFF(self.make_params('tiff'), fullfilename, folder, part, start, stop)
        self.addto_treeWidget_QxzQy(dataset)
        self.statusBar().showMessage('Importing TIFF done.')

    def actionImport_TIFF_PIL_triggered(self, fullfilename):
        dirname, filename = os.path.split(fullfilename)
        folder_auto = os.path.abspath(os.path.join(dirname, '..', 'pilatus861', os.path.splitext(filename)[0]))
        dialog = QtWidgets.QDialog(parent=self)
        layout = QtWidgets.QGridLayout(dialog)
        layout.addWidget(QtWidgets.QLabel(text='folder that contains TIFF files'))
        folder = QtWidgets.QLineEdit(text=folder_auto)
        layout.addWidget(folder)
        layout.addWidget(QtWidgets.QLabel(text="1-based index of block (starting with #S) of .dat file to load, or 'all'"))
        part = QtWidgets.QLineEdit(text='all')
        layout.addWidget(part)
        layout.addWidget(QtWidgets.QLabel(text='1-based index of first file to load from the block'))
        start = QtWidgets.QLineEdit(text='1')
        layout.addWidget(start)
        layout.addWidget(QtWidgets.QLabel(text="1-based index of last file to load from the block, or 'all'"))
        stop = QtWidgets.QLineEdit(text='all')
        layout.addWidget(stop)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        ok_or_cancel = dialog.exec_()
        if ok_or_cancel == 0:
            return
        folder = folder.text()
        part = int(part.text()) if part.text() != 'all' else None
        start = int(start.text()) - 1
        stop = int(stop.text()) if stop.text() != 'all' else None
        # make objects
        self.statusBar().showMessage('Importing TIFF...')
        dataset = processing.DatasetTIFF(self.make_params('tiff'), fullfilename, folder, part, start, stop)
        self.addto_treeWidget_QxzQy(dataset)
        self.statusBar().showMessage('Importing TIFF done.')

    @Slot()
    def on_actionImport_general_TIFF_triggered(self):
        filenames_tif = QtWidgets.QFileDialog.getOpenFileNames(parent=self, caption='Choose general .tif files to load:',
                                                               directory=self.selecteddirectory, filter='*.tif')
        if not (QtCore.QT_VERSION >> 16) == 4:
            filenames_tif = filenames_tif[0]
        if len(filenames_tif) == 0:
            return
        filename1 = filenames_tif[0] if isinstance(filenames_tif, list) else filenames_tif
        self.selecteddirectory = QtCore.QFileInfo(filename1).absolutePath()
        filename_csv = QtWidgets.QFileDialog.getOpenFileName(parent=self, caption='Choose metadata .csv file to load:',
                                                             directory=self.selecteddirectory, filter='*.csv')
        if not (QtCore.QT_VERSION >> 16) == 4:
            filename_csv = filename_csv[0]
        if len(filename_csv) == 0:
            return
        filename1 = filename_csv[0] if isinstance(filename_csv, list) else filename_csv
        self.selecteddirectory = QtCore.QFileInfo(filename1).absolutePath()
        self.statusBar().showMessage('Importing general TIFF...')
        dataset = processing.DatasetGeneralTIFF(self.make_params('tiff'), filenames_tif, filename_csv)
        self.addto_treeWidget_QxzQy(dataset)
        self.statusBar().showMessage('Importing general TIFF done.')

    @Slot()
    def on_actionImport_general_CSV_TIFF_triggered(self):
        filename_csv = QtWidgets.QFileDialog.getOpenFileName(parent=self, caption='Choose metadata .csv file to load (same directory as your tif files):',
                                                             directory=self.selecteddirectory, filter='*.csv')
        if not (QtCore.QT_VERSION >> 16) == 4:
            filename_csv = filename_csv[0]
        if len(filename_csv) == 0:
            return
        filename1 = filename_csv[0] if isinstance(filename_csv, list) else filename_csv
        self.selecteddirectory = QtCore.QFileInfo(filename1).absolutePath()
        self.statusBar().showMessage("Importing general TIFF selected in CSV...")
        dataset = processing.DatasetGeneralCSV_TIFF(filename_csv, self.make_params('gencsvtiff'))
        self.addto_treeWidget_QxzQy(dataset)
        self.statusBar().showMessage('Importing general CSV/TIFF done.')
        

    def addto_treeWidget_QxzQy(self, dataset):
        datasets.append(dataset)
        top_item = QtWidgets.QTreeWidgetItem([dataset.folder])
        self.treeWidget_QxzQy.addTopLevelItem(top_item)
        top_item.setExpanded(1)
        for sf in dataset.scatteringfilelist:
            # tiff files will have empty columns
            if sf.fileformat == 'tiff':
                item = QtWidgets.QTreeWidgetItem([
                    str(sf.filename),
                    '{0:.4g}'.format(sf.sample_theta),
                    '{0:.4g}'.format(sf.energy_ev),
                    '{0:.4g}'.format(sf.info['Seconds']),
                    '{0:.4g}'.format(sf.info['IC_cntr1']),
                    '{0:.4g}'.format(sf.info['BS_cntr3']),
                    '{0:.4g}'.format(sf.info['SRcurrent']),
                ])
            elif sf.fileformat == 'fits':
                item = QtWidgets.QTreeWidgetItem([
                    str(sf.filename),
                    '{0:.4g}'.format(sf.sample_theta),
                    '{0:.4g}'.format(sf.energy_ev),
                    '{0:.4g}'.format(sf.info[0]['EXPOSURE']),
                    '{0:.4g}'.format(sf.info[0]['AI 3 Izero']),
                    '{0:.4g}'.format(sf.info[0]['AI 6 BeamStop']),
                    '{0:.4g}'.format(sf.info[0]['Beam Current']),
                    '{0:.4g}'.format(sf.info[0]['EPU Polarization']),
                    '{0:.4g}'.format(sf.info[0]['CCD Theta']),
                    '{0:.4g}'.format(sf.info[0]['CCD X']),
                ])
            elif sf.fileformat == 'gentiff':
                item = QtWidgets.QTreeWidgetItem([
                    str(sf.filename),
                    '{0:.4g}'.format(sf.sample_theta),
                    '{0:.4g}'.format(sf.energy_ev),
                    '{0:.4g}'.format(sf.info['Seconds']),
                    '{0:.4g}'.format(sf.info['IC_cntr1']),
                ])
            elif sf.fileformat == 'gencsvtiff':
                item = QtWidgets.QTreeWidgetItem([
                    str(sf.filename),
                    '{0:.4g}'.format(sf.sample_theta),
                    '{0:.4g}'.format(sf.energy_ev),
                    '{0:.4g}'.format(sf.info['Seconds']),
                ])
            elif sf.fileformat == 'bin':
                item = QtWidgets.QTreeWidgetItem([
                    str(sf.filename),
                    '{0:.4g}'.format(sf.sample_theta),
                    '{0:.4g}'.format(sf.energy_ev),
                    '{0:.4g}'.format(sf.info['Seconds']),
                ])
            else:
                self.statusBar().showMessage('Error adding to tree.')
                return
            top_item.addChild(item)

    def actionSave_sample_parameters_triggered(self, filename):
        self.statusBar().showMessage('Saving JSON...')
        params_json = {}
        params_json['data_file_type'] = 'FITS' if self.doubleSpinBox_detector_theta0.isVisible() else 'TIFF'
        # list of all GUI elements with changeable values, except those in matplotlibwidgets that have no objectName
        children = self.findChildren(
            (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox, QtWidgets.QCheckBox, QtWidgets.QRadioButton, QtWidgets.QComboBox,
             QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit),
            QtCore.QRegExp('\w'))

        for child in children:
            name = child.objectName()
            if child.parent() is self.linspacearangedialog:
                key = 'linspacearangedialog.' + name
                root = self.linspacearangedialog
            elif child.parent() is self.miscsettingsdialog:
                key = 'miscsettingsdialog.' + name
                root = self.miscsettingsdialog
            else:
                key = name
                root = self

            if name.startswith(('doubleSpinBox', 'spinBox')):
                params_json[key] = getattr(root, name).value()
            elif name.startswith(('checkBox', 'radioButton')):
                params_json[key] = getattr(root, name).isChecked()
            elif name.startswith('comboBox'):
                params_json[key] = getattr(root, name).currentIndex()
            elif name.startswith('lineEdit'):
                params_json[key] = getattr(root, name).text()
            elif name.startswith('plainTextEdit'):
                params_json[key] = getattr(root, name).toPlainText()

        with open(filename, 'w') as f:
            json.dump(params_json, f, sort_keys=True, separators=(',\n', ': '))
        self.statusBar().showMessage('Saving JSON done.')

    def actionLoad_sample_parameters_triggered(self, filename):
        self.statusBar().showMessage('Loading JSON...')
        with open(filename, 'r') as f:
            params_json = json.load(f)

        for key, value in params_json.items():
            try:
                if key.count('.') == 1:
                    root, key = key.split('.')
                    root = getattr(self, root)
                else:
                    root = self

                if key.startswith(('doubleSpinBox', 'spinBox')):
                    getattr(root, key).setValue(value)
                elif key.startswith(('checkBox', 'radioButton')):
                    getattr(root, key).setChecked(value)
                elif key.startswith('comboBox'):
                    getattr(root, key).setCurrentIndex(value)
                elif key.startswith('lineEdit'):
                    getattr(root, key).setText(value)
                elif key.startswith('plainTextEdit'):
                    getattr(root, key).setPlainText(value)
                elif key == 'data_file_type' and value == 'FITS':
                    self.modify_gui_FITS()
                elif key == 'data_file_type' and value == 'TIFF':
                    self.modify_gui_TIFF()
                else:
                    print('{0} in json file ignored'.format(key))
            except AttributeError:
                print('{0} in json file ignored'.format(key))
        self.statusBar().showMessage('Loading JSON done.')

    def actionExport_slices_to_CSV_triggered(self, filename):
        indices = [qmodelindex.row() for qmodelindex in get_sel_qmodelindices(self.listWidget_IvsQ)]
        if len(indices) < 1:
            self.statusBar().showMessage('No slices to save.')
            return
        self.statusBar().showMessage('Exporting slices to CSV...')
        if self.radioButton_integrateQz.isChecked():
            processing.export_slices_to_csv(ivsqx_list, True, indices, filename)
        else:
            processing.export_slices_to_csv(ivsqz_list, False, indices, filename)
        self.statusBar().showMessage('Exporting slices to CSV done.')

    def actionLoad_from_pickle_triggered(self, filename):
        self.statusBar().showMessage('Loading from pickle...')
        with open(filename, 'rb') as f:
            globals_list = pickle.load(f)
        # previous versions may not contain all elements of globals_list
        try:
            for dataset in globals_list[0]:
                self.addto_treeWidget_QxzQy(dataset)
            for rd in globals_list[1]:
                reduceddatas.append(rd)
                listwidgetitem = QtWidgets.QListWidgetItem(rd.dataqzqx.title)
                listwidgetitem.setFlags(self.listwidgetitemflags)
                self.listWidget_QzQx.addItem(listwidgetitem)
            ivsqx_list.extend(globals_list[2])
            ivsqz_list.extend(globals_list[3])
            if self.radioButton_integrateQx.isChecked():
                self.on_radioButton_integrateQx_toggled(True)
            else:
                self.on_radioButton_integrateQz_toggled(True)
            for photodiode in globals_list[4]:
                self.add_to_photodiode_tableWidget(*photodiode)
            for dataqxe, dataqxeinterp in globals_list[5]:
                qxe_list.append((dataqxe, dataqxeinterp))
                self.listWidget_QxEobj.addItem(dataqxe.title.replace('_QxE', ''))
            qxe_ivsqx_list.extend(globals_list[6])
            qxe_ivse_list.extend(globals_list[7])
            if self.radioButton_QxEslice_integrateQx.isChecked():
                self.on_radioButton_QxEslice_integrateQx_toggled(True)
            else:
                self.on_radioButton_QxEslice_integrateE_toggled(True)
        except IndexError:
            pass
        self.statusBar().showMessage('Loading from pickle done.')

    def actionSave_to_pickle_triggered(self, filename):
        self.statusBar().showMessage('Saving to pickle...')
        globals_list = (datasets, reduceddatas, ivsqx_list, ivsqz_list,
                        photodiode_list, qxe_list, qxe_ivsqx_list, qxe_ivse_list)
        with open(filename, 'wb') as f:
            pickle.dump(globals_list, f, pickle.HIGHEST_PROTOCOL)
        self.statusBar().showMessage('Saving to pickle done.')

    @Slot()
    def on_actionMisc_settings_triggered(self):
        ok = self.miscsettingsdialog.exec_()
        if ok == 1:
            processing.NOISE_REGION_STR = self.miscsettingsdialog.lineEdit_noise_slice.text()
            processing.OVERLAP_SAMPLE_THETA_THRESH = self.miscsettingsdialog.doubleSpinBox_overlap_sample_theta_thresh.value()
            processing.FIND_ANGLE_MIN = self.miscsettingsdialog.doubleSpinBox_find_angle_min.value() - 90
            processing.FIND_ANGLE_MAX = self.miscsettingsdialog.doubleSpinBox_find_angle_max.value() - 90
            processing.FIND_ANGLE_MAX_CHANGE = self.miscsettingsdialog.doubleSpinBox_find_angle_max_change.value()
            processing.RECT_BOTTOM_BELOWCENTER = self.miscsettingsdialog.spinBox_bottom_belowcenter.value()
            processing.FIND_ANGLE_PRINT = self.miscsettingsdialog.checkBox_verbose_output.isChecked()
            processing.USE_INPUT_ANGLE = self.miscsettingsdialog.checkBox_use_input_angle.isChecked()
            matplotlibwidget.mpl.rcParams['font.size'] = self.miscsettingsdialog.spinBox_fontsize.value()
            global NEWDIRNAME
            NEWDIRNAME = self.miscsettingsdialog.pushButton_missingfolder.text()

# %% First tab (Qxz-Qy) column 1

    @Slot()
    def on_treeWidget_QxzQy_itemSelectionChanged(self):
        """When a new scattering file object is selected, make and plot dataqxzqy, datarot, ivsqxz"""
        # reassign self.dataset and self.sf
        try:
            qmodelindex = get_sel_qmodelindices(self.treeWidget_QxzQy)[-1]
        except IndexError:
            return
        if not qmodelindex.parent().isValid():
            return
        self.statusBar().showMessage('Loading and plotting...')
        self.dataset = datasets[qmodelindex.parent().row()]
        self.sf = self.dataset.scatteringfilelist[qmodelindex.row()]
        # for when data files are moved
        if not os.path.isfile(self.sf.fullfilename):
            dirname, filename = os.path.split(self.sf.fullfilename)
            rootname, tailname = os.path.split(dirname)
            newfullfilename = os.path.join(NEWDIRNAME, tailname, filename)
            self.statusBar().showMessage('{0} is missing, moved to {1}'.format(self.sf.fullfilename, newfullfilename))
            self.sf.fullfilename = newfullfilename
            self.sf.dataqxzqy.fullfilename = newfullfilename
        # enable/disable input widgets based on fileformat
        if self.sf.fileformat == 'tiff' or self.sf.fileformat == 'gentiff' or self.sf.fileformat == 'bin':
            self.modify_gui_TIFF()
        elif self.sf.fileformat == 'fits':
            self.modify_gui_FITS()
        # restore values if they exist to input widgets
        self.load_params()
        # plot dataqxzqy
        loaded_data = self.sf.dataqxzqy.load_save_data(self.actionKeep_Qxz_Qy_in_memory.isChecked())
        self.sf.dataqxzqy.plot_self(self.m_QxzQy, loaded_data)
        # make and plot datarot, ivsqxz
        try:
            self.makerotated(loaded_data)
        except IndexError:
            self.statusBar().showMessage('Error: box extends past bounds of image.')
            return
        self.on_pushButton_integrateto_IvsQxz_clicked()
        self.statusBar().showMessage('Loading and plotting done.')

    def load_params(self):
        """Loads parameters from self.sf into input widgets"""
        params = self.sf.dataqxzqy.params
        self.doubleSpinBox_pixel_um.setValue(params.pixel_um)
        self.label_lambda_nm.setText('{0:.7g}'.format(self.sf.lambda_nm))
        self.label_energy_ev.setText('{0:.6g}'.format(self.sf.energy_ev))
        self.doubleSpinBox_SDD_cm.setValue(params.SDD_cm)
        self.comboBox_peaksdirection.setCurrentIndex(self.comboBox_peaksdirection.findText(params.peaks_direction))
        self.doubleSpinBox_centerY_px.setValue(params.center_px[0])
        self.doubleSpinBox_centerX_px.setValue(params.center_px[1])
        self.checkBox_normalizeexposure.setChecked(params.normalize_exposure)
        self.checkBox_normalizeI0.setChecked(params.normalize_I0)
        self.checkBox_normalizecurrent.setChecked(params.normalize_current)
        self.lineEdit_multiplyintensity.setText('{0:.4g}'.format(params.multiply_intensity))
        self.checkBox_subtractbottom.setChecked(params.subtract_bottom)
        # hack to load old pickle files
        if not hasattr(self.sf, 'div_photodiode'):
            self.sf.div_photodiode = 1
        self.label_dividebyphotodiode.setText('{0:.4g}'.format(self.sf.div_photodiode))
        # load file format dependent parameters
        if self.sf.fileformat == 'fits':
            self.doubleSpinBox_detector_theta0.setValue(params.detector_theta0)
            self.doubleSpinBox_detector_x0.setValue(params.detector_x0)
            self.doubleSpinBox_detector_thetascale.setValue(params.detector_thetascale)
            self.label_center_qxz_on_img.setText('{0:.6g}'.format(self.sf.dataqxzqy.center_qxz_on_img))
            self.label_center_qy_on_img.setText('{0:.6g}'.format(self.sf.dataqxzqy.center_qy_on_img))
        if hasattr(self.sf, 'datarot'):
            rect_dims = self.sf.datarot.rect_dims
            self.doubleSpinBox_boxbottom_px.setValue(rect_dims.bottom)
            self.doubleSpinBox_boxwidth_px.setValue(rect_dims.width)
            self.doubleSpinBox_boxleft_px.setValue(rect_dims.left)
            self.doubleSpinBox_boxheight_px.setValue(rect_dims.height)
            self.doubleSpinBox_boxangle.setValue(90 + rect_dims.angle_deg)
            if rect_dims.threshold is not None:
                self.doubleSpinBox_threshold.setValue(rect_dims.threshold)
            if rect_dims.width_peaks is not None:
                self.spinBox_widthpeaks_px.setValue(rect_dims.width_peaks)
        if hasattr(self.sf, 'ivsqxz'):
            aparams = self.sf.ivsqxz.aparams
            remake = False  # hack to load old pickle files
            new_values = []
            for value in aparams:
                if value is None:
                    remake = True
                    new_values.append(False)
                else:
                    new_values.append(value)
            if remake:
                aparams = processing.AnalysisParams(*new_values)

            if aparams.reduce_fn == np.nanmean:
                self.radioButton_integratemean.setChecked(True)
            elif aparams.reduce_fn == np.nansum:
                self.radioButton_integratesum.setChecked(True)
            if aparams.subtract_value == 'min':
                self.label_subtractminimum.setText('{0:.4g}'.format(np.nanmin(self.sf.datarot.data_rotated)))
                self.radioButton_subtractminimum.setChecked(True)
            else:
                self.lineEdit_subtractvalue.setText('{0:.4g}'.format(aparams.subtract_value))
                self.radioButton_subtractvalue.setChecked(True)

            self.doubleSpinBox_samplethetaoffset.setValue(aparams.samplethetaoffset)
            self.doubleSpinBox_samplesize.setValue(aparams.samplesize)
            self.doubleSpinBox_beamcenter.setValue(aparams.beamcenter)
            self.doubleSpinBox_beamfwhm.setValue(aparams.beamfwhm)
            self.doubleSpinBox_substratethickness.setValue(aparams.substratethickness)
            self.doubleSpinBox_substrateattenuation.setValue(aparams.substrateattenuation)
            self.doubleSpinBox_samplethickness.setValue(aparams.samplethickness)
            self.doubleSpinBox_sampleattenuation.setValue(aparams.sampleattenuation)
            self.doubleSpinBox_fracpolarizedy.setValue(aparams.frac_polarized_y)

            self.checkBox_footprintcorr.setChecked(aparams.footprintcorr)
            self.checkBox_samplesizecorr.setChecked(aparams.samplesizecorr)
            self.checkBox_abscorr.setChecked(aparams.abscorr)
            self.checkBox_sampleabscorr.setChecked(aparams.sampleabscorr)
            self.checkBox_polarizationcorr.setChecked(aparams.polarizationcorr)
            self.spinBox_groupnumber.setValue(aparams.groupnumber)

    @Slot()
    def on_pushButton_showfileinfo_clicked(self):
        """Make window displaying sf.infostr if sf is selected, or dataset.datfile if DatasetTIFF is selected"""
        try:
            qmodelindex = get_sel_qmodelindices(self.treeWidget_QxzQy)[-1]
        except IndexError:
            return
        if not qmodelindex.parent().isValid():
            dataset = datasets[qmodelindex.row()]
            if not hasattr(dataset, 'datfile'):
                return
            infostr = ''.join(dataset.datfile)
        else:
            infostr = self.sf.infostr
        dialog = QtWidgets.QDialog(parent=self)
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowMinMaxButtonsHint)
        layout = QtWidgets.QGridLayout(dialog)
        infowidget = QtWidgets.QPlainTextEdit(infostr, parent=self)
        infowidget.setReadOnly(True)
        layout.addWidget(infowidget)
        dialog.show()

    @Slot()
    def on_pushButton_files_clearselected_clicked(self):
        self.treeWidget_QxzQy.itemSelectionChanged.disconnect()
        for qmodelindex in sorted(get_sel_qmodelindices(self.treeWidget_QxzQy), key=lambda i: i.row(), reverse=True):
            if qmodelindex.parent().isValid():  # is a sf
                datasets[qmodelindex.parent().row()].scatteringfilelist.pop(qmodelindex.row())
                self.treeWidget_QxzQy.topLevelItem(qmodelindex.parent().row()).takeChild(qmodelindex.row())
            else:  # is a dataset
                datasets.pop(qmodelindex.row())
                self.treeWidget_QxzQy.takeTopLevelItem(qmodelindex.row())
        self.treeWidget_QxzQy.itemSelectionChanged.connect(self.on_treeWidget_QxzQy_itemSelectionChanged)

    @Slot()
    def on_pushButton_files_clearall_clicked(self):
        datasets.clear()
        self.treeWidget_QxzQy.clear()

    @Slot(int)
    def on_spinBox_selectnrows_valueChanged(self, value):
        """select every n rows starting with last currently selected row"""
        self.treeWidget_QxzQy.itemSelectionChanged.disconnect()
        model = self.treeWidget_QxzQy.selectionModel()
        qmodelindex = model.selectedRows()[-1]
        num_rows = self.treeWidget_QxzQy.topLevelItem(qmodelindex.parent().row()).childCount()
        for i in range(num_rows):
            qmodelindex = self.treeWidget_QxzQy.indexBelow(qmodelindex)
            if (i + 1) % value == 0:
                model.select(qmodelindex, model.Rows | model.Select)
        self.treeWidget_QxzQy.itemSelectionChanged.connect(self.on_treeWidget_QxzQy_itemSelectionChanged)

    @Slot()
    def on_pushButton_params_applytoselected_clicked(self):
        """Save current values in input widgets to each selected scattering file object"""
        scatteringfilelist = self.get_scatteringfilelist()
        for sf in scatteringfilelist:
            sf.__init__(sf.fileformat, sf.fullfilename, self.make_params(sf.fileformat), sf.info, sf.infostr)
        self.on_treeWidget_QxzQy_itemSelectionChanged()

# %% First tab column 2

    @Slot()
    def on_pushButton_makerotated_clicked(self):
        self.statusBar().showMessage('Making rotated QxzQy...')
        loaded_data = self.sf.dataqxzqy.load_save_data(self.actionKeep_Qxz_Qy_in_memory.isChecked())
        try:
            self.makerotated(loaded_data)
        except IndexError:
            self.statusBar().showMessage('Error: box extends past bounds of image.')
            return
        self.statusBar().showMessage('Making rotated QxzQy done.')

    def makerotated(self, loaded_data):
        self.m_QxzQy.draw_collection([])
        find_angle_mode = self.get_find_angle_mode()
        if find_angle_mode is not None:
            angle_deg, left, qxz_peaks, qy_peaks = self.sf.dataqxzqy.find_angle(
                self.make_rect_dims(), loaded_data, find_angle_mode)
            self.m_QxzQy.axis.scatter(qy_peaks, qxz_peaks, c='r')
            self.doubleSpinBox_boxangle.setValue(90 + angle_deg)
            self.doubleSpinBox_boxleft_px.setValue(left)
        self.sf.datarot = processing.DataRot(self.make_rect_dims(), self.sf.lambda_nm, self.sf.sample_theta, *loaded_data)
        self.m_QxzQy.draw_collection((self.sf.datarot.rect,), keep_old=True)
        self.sf.datarot.plot_self(self.m_Rot)
        self.label_subtractminimum.setText('{0:.4g}'.format(np.nanmin(self.sf.datarot.data_rotated)))

    @Slot()
    def on_pushButton_integrateto_IvsQxz_clicked(self):
        self.statusBar().showMessage('Integrating to IvsQxz...')
        aparams = self.make_aparams()
        self.sf.ivsqxz = processing.IvsQxz(self.sf.datarot.data_rotated, self.sf.datarot.qys, self.sf.datarot.qxzs,
                                           aparams, self.sf.lambda_nm, self.sf.sample_theta)
        self.sf.ivsqxz.plot_self(self.m_IvsQxz, self.radioButton_plot_IvsQx.isChecked())
        self.statusBar().showMessage('Integrating to IvsQxz done.')

    @Slot(bool)
    def on_radioButton_plot_IvsQxz_toggled(self, value):
        if value and hasattr(self, 'sf') and hasattr(self.sf, 'ivsqxz'):
            self.sf.ivsqxz.plot_self(self.m_IvsQxz, False)

    @Slot(bool)
    def on_radioButton_plot_IvsQx_toggled(self, value):
        if value and hasattr(self, 'sf') and hasattr(self.sf, 'ivsqxz'):
            self.sf.ivsqxz.plot_self(self.m_IvsQxz, True)

    @Slot(bool)
    def on_checkBox_samplesizecorr_toggled(self, value):
        if value:
            self.doubleSpinBox_samplesize.setEnabled(True)
            self.doubleSpinBox_beamcenter.setEnabled(True)
            self.doubleSpinBox_beamfwhm.setEnabled(True)
        else:
            self.doubleSpinBox_samplesize.setEnabled(False)
            self.doubleSpinBox_beamcenter.setEnabled(False)
            self.doubleSpinBox_beamfwhm.setEnabled(False)

    @Slot(bool)
    def on_checkBox_abscorr_toggled(self, value):
        if value:
            self.doubleSpinBox_substratethickness.setEnabled(True)
            self.doubleSpinBox_substrateattenuation.setEnabled(True)
        else:
            self.doubleSpinBox_substratethickness.setEnabled(False)
            self.doubleSpinBox_substrateattenuation.setEnabled(False)

    @Slot(bool)
    def on_checkBox_sampleabscorr_toggled(self, value):
        if value:
            self.doubleSpinBox_samplethickness.setEnabled(True)
            self.doubleSpinBox_sampleattenuation.setEnabled(True)
        else:
            self.doubleSpinBox_samplethickness.setEnabled(False)
            self.doubleSpinBox_sampleattenuation.setEnabled(False)

    @Slot(bool)
    def on_checkBox_polarizationcorr_toggled(self, value):
        if value:
            self.doubleSpinBox_fracpolarizedy.setEnabled(True)
        else:
            self.doubleSpinBox_fracpolarizedy.setEnabled(False)

# %% First tab column 3

    @Slot()
    def on_pushButton_integrate_correct_files_clicked(self):
        """For each scattering file object (sf) concurrently, modify sf (make rotated and integrate)
        use aparams for each sf if they exist, else the values currently in the gui"""
        self.statusBar().showMessage('Rotating, integrating, and correcting selected files...')
        scatteringfilelist = self.get_scatteringfilelist()
        if len(scatteringfilelist) == 0:
            self.statusBar().showMessage('Select at least 1 file.')
            return
        loadfromitem = not self.checkBox_sameperfile.isChecked()
        args = (scatteringfilelist, self.make_rect_dims(), self.make_aparams(), self.get_find_angle_mode(),
                self, loadfromitem, self.actionKeep_Qxz_Qy_in_memory.isChecked())
        if self.miscsettingsdialog.checkBox_use_threads.isChecked():
            processing.parallel_modify_scatteringfilelist(*args)
        else:
            processing.nonparallel_modify_scatteringfilelist(*args)
        processing.scatteringfilelist_fix_angles(scatteringfilelist)
        self.statusBar().showMessage('Rotating, integrating, and correcting selected files done.')

    @Slot()
    def on_pushButton_recorrect_files_clicked(self):
        """Undoes sample theta offset and corrections and applies new ones to selected sf objects.
        All selected objects must already be integrated (have IvsQxz object)"""
        self.statusBar().showMessage('Recorrecting selected files...')
        scatteringfilelist = self.get_scatteringfilelist()
        for sf in scatteringfilelist:
            if not hasattr(sf, 'ivsqxz'):
                self.statusBar().showMessage('Need to integrate all selected files first.')
                return
            else:
                sf.ivsqxz.reapply_correction(self.make_aparams())
        self.statusBar().showMessage('Recorrecting selected files done.')

    @Slot()
    def on_pushButton_new_reduced_clicked(self):
        """Make new reduced dataset from selected sf objects (or all if only 1 selected)
        All selected objects must already be integrated (have IvsQxz object)"""
        self.statusBar().showMessage('Combining selected to scatterplot...')
        scatteringfilelist = self.get_scatteringfilelist()
        if len(scatteringfilelist) == 0:
            self.statusBar().showMessage('Select at least 1 file.')
            return
        for sf in scatteringfilelist:
            if not hasattr(sf, 'ivsqxz'):
                self.statusBar().showMessage('Need to integrate all selected files first.')
                return
        # generate reduceddata and dataqzqx
        rd = processing.ReducedData(scatteringfilelist, self.lineEdit_nametocreate_QzQx.text())
        # generate dataqxqzinterp object
        self.statusBar().showMessage('Interpolating...')
        rd.dataqzqxinterp = processing.DataQzQxInterp(
            rd.dataqzqx.qx_listoflists, rd.dataqzqx.qz_listoflists, rd.dataqzqx.I_listoflists,
            qx_interp_size=self.doubleSpinBox_qx_interp_size.value(),
            qz_interp_size=self.doubleSpinBox_qz_interp_size.value(),
            title=self.lineEdit_nametocreate_QzQx.text())
        listwidgetitem = QtWidgets.QListWidgetItem(self.lineEdit_nametocreate_QzQx.text())
        listwidgetitem.setFlags(self.listwidgetitemflags)
        self.listWidget_QzQx.addItem(listwidgetitem)
        reduceddatas.append(rd)
        # calls on_listWidget_QzQx_itemSelectionChanged
        self.listWidget_QzQx.setCurrentRow(self.listWidget_QzQx.count() - 1)
        self.tabWidget.setCurrentIndex(1)

    @Slot()
    def on_pushButton_reduction_plotI0_clicked(self):
        scatteringfilelist = self.get_scatteringfilelist()
        if scatteringfilelist[0].fileformat == 'tiff' or scatteringfilelist[0].fileformat == 'gentiff':
            I0 = [sf.info['IC_cntr1'] for sf in scatteringfilelist]
        elif scatteringfilelist[0].fileformat == 'fits':
            I0 = [sf.info[0]['AI 3 Izero'] for sf in scatteringfilelist]
        else:
            self.statusBar().showMessage('No I0 data.')
            return
        widget = matplotlibwidget.MatplotlibWidget1D()
        widget.plot(range(len(I0)), I0, ylabel='I0', marker='D')

    @Slot()
    def on_pushButton_reduction_plotbeamstop_clicked(self):
        scatteringfilelist = self.get_scatteringfilelist()
        if scatteringfilelist[0].fileformat == 'tiff':
            beamstop = [sf.info['BS_cntr3'] for sf in scatteringfilelist]
        elif scatteringfilelist[0].fileformat == 'fits':
            beamstop = [sf.info[0]['AI 6 BeamStop'] for sf in scatteringfilelist]
        else:
            self.statusBar().showMessage('No beamstop data.')
            return
        widget = matplotlibwidget.MatplotlibWidget1D()
        widget.plot(range(len(beamstop)), beamstop, ylabel='beamstop', marker='D')

    @Slot()
    def on_pushButton_reduction_plotleft_clicked(self):
        scatteringfilelist = self.get_scatteringfilelist()
        left = [sf.datarot.rect_dims.left for sf in scatteringfilelist]
        widget = matplotlibwidget.MatplotlibWidget1D()
        widget.plot(range(len(left)), left, ylabel='box left', marker='D')
        widget.axis.ticklabel_format(useOffset=False)
        widget.figure.canvas.draw()

    @Slot()
    def on_pushButton_reduction_plotangle_clicked(self):
        scatteringfilelist = self.get_scatteringfilelist()
        angle = [sf.datarot.rect_dims.angle_deg for sf in scatteringfilelist]
        widget = matplotlibwidget.MatplotlibWidget1D()
        widget.plot(range(len(angle)), angle, ylabel='box angle', marker='D')

# %% Second tab (Qz-Qx) column 1

    @Slot(QtWidgets.QListWidgetItem)
    def on_listWidget_QzQx_itemChanged(self, item):
        """Double click list widget item to change name"""
        self.rd._groupstr = item.text()
        self.rd.dataqzqx.title = item.text()
        self.rd.dataqzqxinterp.title = item.text()

    @Slot()
    def on_listWidget_QzQx_itemSelectionChanged(self):
        """Replots dataqzqx, dataqzqxinterp"""
        try:
            self.rd = reduceddatas[self.listWidget_QzQx.currentRow()]
        except IndexError:
            self.statusBar().showMessage('Invalid index of reduceddatas.')
            return
        self.statusBar().showMessage('Plotting QzQx of ' + self.rd.dataqzqx.title)
        if self.radioButton_QzQx_plotI.isChecked():
            mode = 'I'
        elif self.radioButton_QzQx_plotpxsample.isChecked():
            mode = 'I-theta'
        elif self.radioButton_QzQx_plotsampletheta.isChecked():
            mode = 'sample_theta'
        elif self.radioButton_QzQx_plotdetectortheta.isChecked():
            mode = 'detector_theta'
        elif self.radioButton_QzQx_plotpxtheta.isChecked():
            mode = 'px_theta'
        elif self.radioButton_QzQx_correctionsonly.isChecked():
            mode = 'corrections_only'
        self.rd.dataqzqx.plot_self(self.m_QzQx, mode)
        self.statusBar().showMessage('Plotting interpolated QzQx of ' + self.rd.dataqzqx.title)
        try:
            self.doubleSpinBox_qx_interp_size.setValue(self.rd.dataqzqxinterp.interp_qx[1] - self.rd.dataqzqxinterp.interp_qx[0])
            self.doubleSpinBox_qz_interp_size.setValue(self.rd.dataqzqxinterp.interp_qz[1] - self.rd.dataqzqxinterp.interp_qz[0])
            self.rd.dataqzqxinterp.plot_self(self.m_QzQx_interp)
        except:
            self.statusBar().showMessage('Missing interpolated QzQx of ' + self.rd.dataqzqx.title)
            return
        self.statusBar().showMessage('Done plotting ' + self.rd.dataqzqx.title)

    def listWidget_QzQx_model_rowsMoved(self, sourceParent, sourceStart, sourceEnd, destinationParent, destinationRow):
        rd_moved = reduceddatas.pop(sourceStart)
        if sourceStart < destinationRow:
            reduceddatas.insert(destinationRow - 1, rd_moved)
        else:
            reduceddatas.insert(destinationRow, rd_moved)

    @Slot(bool)
    def on_radioButton_QzQx_plotI_toggled(self, value):
        if value:
            self.on_listWidget_QzQx_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_QzQx_plotpxsample_toggled(self, value):
        if value:
            self.on_listWidget_QzQx_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_QzQx_plotpxtheta_toggled(self, value):
        if value:
            self.on_listWidget_QzQx_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_QzQx_plotdetectortheta_toggled(self, value):
        if value:
            self.on_listWidget_QzQx_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_QzQx_plotsampletheta_toggled(self, value):
        if value:
            self.on_listWidget_QzQx_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_QzQx_correctionsonly_toggled(self, value):
        if value:
            self.on_listWidget_QzQx_itemSelectionChanged()

    @Slot()
    def on_pushButton_QzQx_showfiles_clicked(self):
        dialog = QtWidgets.QDialog(parent=self)
        layout = QtWidgets.QGridLayout(dialog)
        infostr = '\n'.join(self.rd.filenamelist)
        infowidget = QtWidgets.QPlainTextEdit(infostr, parent=self)
        infowidget.setReadOnly(True)
        layout.addWidget(infowidget)
        dialog.show()

    @Slot()
    def on_pushButton_QzQx_clearselected_clicked(self):
        self.listWidget_QzQx.itemSelectionChanged.disconnect()
        for qmodelindex in sorted(get_sel_qmodelindices(self.listWidget_QzQx), key=lambda i: i.row(), reverse=True):
            reduceddatas.pop(qmodelindex.row())
            self.listWidget_QzQx.takeItem(qmodelindex.row())
        self.listWidget_QzQx.itemSelectionChanged.connect(self.on_listWidget_QzQx_itemSelectionChanged)

    @Slot()
    def on_pushButton_QzQx_reinterpolate_clicked(self):
        self.statusBar().showMessage('Reinterpolating QzQx...')
        self.rd.dataqzqxinterp = processing.DataQzQxInterp(
            self.rd.dataqzqx.qx_listoflists, self.rd.dataqzqx.qz_listoflists, self.rd.dataqzqx.I_listoflists,
            qx_interp_size=self.doubleSpinBox_qx_interp_size.value(), qz_interp_size=self.doubleSpinBox_qz_interp_size.value(),
            title=self.lineEdit_nametocreate_QzQx.text())
        self.on_listWidget_QzQx_itemSelectionChanged()


# %% Second tab column 2

    def m_QzQx_interp_clicked(self, event):
        """Adds line with qz or qx value to text box"""
        if self.checkBox_clicktoadd.isChecked():
            if self.radioButton_integrateQx.isChecked():
                self.plainTextEdit_listQ_tointegrate.insertPlainText('{0:.4g}, '.format(event.xdata))
            else:
                self.plainTextEdit_listQ_tointegrate.insertPlainText('{0:.4g}, '.format(event.ydata))

    @Slot(bool)
    def on_radioButton_integrateQx_toggled(self, value):
        """Updates listWidget_IvsQ"""
        if not value:
            return
        self.listWidget_IvsQ.clear()
        for ivsqz in ivsqz_list:
            self.listWidget_IvsQ.addItem('{0}, qx={1}, width={2}, bg={3}'.format(
                ivsqz.title, ivsqz.qx_midpt, ivsqz.qx_width, ivsqz.qx_bg))

    @Slot(bool)
    def on_radioButton_integrateQz_toggled(self, value):
        """Updates listWidget_IvsQ"""
        if not value:
            return
        self.listWidget_IvsQ.clear()
        for ivsqx in ivsqx_list:
            self.listWidget_IvsQ.addItem('{0}, qz={1}, width={2}, bg={3}'.format(
                ivsqx.title, ivsqx.qz_midpt, ivsqx.qz_width, ivsqx.qz_bg))

    @Slot()
    def on_pushButton_generatevalues_clicked(self):
        if self.linspacearangedialog.exec_():
            self.linspacearangedialog.ok_clicked(self.plainTextEdit_listQ_tointegrate)

    @Slot()
    def on_pushButton_cleartextbox_clicked(self):
        self.plainTextEdit_listQ_tointegrate.clear()

    @Slot()
    def on_pushButton_drawslices_clicked(self):
        self.integrateto_IvsQ(add_to_list=False)

    @Slot()
    def on_pushButton_integrateto_IvsQ_clicked(self):
        self.integrateto_IvsQ(add_to_list=True)

    def integrateto_IvsQ(self, add_to_list=True):
        """Removes old rectangles, draws rectangles, adds to ivsqx_list or ivsqz_list"""
        # removes rectangles from interpolated image
        self.m_QzQx_interp.draw_collection([])
        if len(self.m_QzQx.axis.collections) > 1:
            # removes rectangles from non-interpolated image but keeps scatter data
            self.m_QzQx.axis.collections[1].remove()
            self.m_QzQx.canvas.draw()
        text_list = self.plainTextEdit_listQ_tointegrate.toPlainText().split(',')
        if len(text_list) == 0:
            return

        self.statusBar().showMessage('Integrating to I vs Q...')
        q_midpts = []
        for item in text_list:
            try:
                q_midpts.append(float(item))
            except ValueError:
                continue
        q_width = self.doubleSpinBox_boxwidth_QzQx.value()
        bg_q = self.doubleSpinBox_background_QzQx.value()
        # adds to ivsqx_list or ivsqz_list
        if self.radioButton_integrateQx.isChecked():
            new_ivsq_list = [processing.IvsQz(q_midpt, q_width, bg_q, self.rd.dataqzqx, self.rd.dataqzqxinterp)
                             for q_midpt in q_midpts]
            if add_to_list:
                ivsqz_list.extend(new_ivsq_list)
                self.on_radioButton_integrateQx_toggled(True)
        else:
            new_ivsq_list = [processing.IvsQx(q_midpt, q_width, bg_q, self.rd.dataqzqx, self.rd.dataqzqxinterp)
                             for q_midpt in q_midpts]
            if add_to_list:
                ivsqx_list.extend(new_ivsq_list)
                self.on_radioButton_integrateQz_toggled(True)

        self.statusBar().showMessage('Drawing rectangles...')
        # makes flat list of rects
        rects = list(itertools.chain(*[(ivsq.rect, ivsq.rect_bg) for ivsq in new_ivsq_list]))
        # draws new rectangles
        self.m_QzQx_interp.draw_collection(rects)
        self.m_QzQx.draw_collection(rects, keep_old=True)
        self.statusBar().showMessage('Done integrating to I vs Q.')

# %% Second tab column 3

    @Slot(bool)
    def on_radioButton_plotinterp_toggled(self, value):
        if value:
            self.on_listWidget_IvsQ_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_plotnoninterp_theta_toggled(self, value):
        if value:
            self.on_listWidget_IvsQ_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_plotnoninterp_qz_toggled(self, value):
        if value:
            self.on_listWidget_IvsQ_itemSelectionChanged()

    @Slot(float)
    def on_doubleSpinBox_IvsQ_decades_valueChanged(self, value):
        self.on_listWidget_IvsQ_itemSelectionChanged()

    @Slot(int)
    def on_spinBox_IvsQ_numlines_valueChanged(self, value):
        self.on_listWidget_IvsQ_itemSelectionChanged()

    def listWidget_IvsQ_model_rowsMoved(self, sourceParent, sourceStart, sourceEnd, destinationParent, destinationRow):
        print(sourceParent, sourceStart, sourceEnd, destinationParent, destinationRow)
        if self.radioButton_integrateQx.isChecked():
            ivsq_list = ivsqz_list
        else:
            ivsq_list = ivsqx_list
        rd_moved = ivsq_list.pop(sourceStart)
        if sourceStart < destinationRow:
            ivsq_list.insert(destinationRow - 1, rd_moved)
        else:
            ivsq_list.insert(destinationRow, rd_moved)

    @Slot()
    def on_listWidget_IvsQ_itemSelectionChanged(self):
        """Clears I vs Q widget and plots selected curves"""
        if self.radioButton_integrateQx.isChecked():
            ivsq_list = ivsqz_list
        else:
            ivsq_list = ivsqx_list
        if not self.radioButton_plotinterp.isChecked() and self.radioButton_integrateQz.isChecked():
            self.statusBar().showMessage("Can't plot vs theta when Qz is checked.")
            return
        self.statusBar().showMessage('Plotting slices...')
        selected_ivsq_list = [ivsq_list[qmodelindex.row()]
                              for qmodelindex in get_sel_qmodelindices(self.listWidget_IvsQ)]
        decades = self.doubleSpinBox_IvsQ_decades.value()
        numlines_in_group = self.spinBox_IvsQ_numlines.value()
        for i, ivsq in enumerate(selected_ivsq_list):
            keep_old = False if i == 0 else True
            if self.radioButton_plotinterp.isChecked():
                ivsq.plot_interp(self.m_IvsQ, keep_old=keep_old, num_decades=i // numlines_in_group * decades)
            elif self.radioButton_plotnoninterp_theta.isChecked():
                ivsq.plot_theta(self.m_IvsQ, keep_old=keep_old, num_decades=i // numlines_in_group * decades)
            elif self.radioButton_plotnoninterp_qz.isChecked():
                ivsq.plot_qz(self.m_IvsQ, keep_old=keep_old, num_decades=i // numlines_in_group * decades)
        if len(selected_ivsq_list) > 0:
            self.m_IvsQ.draw_legend()
            self.m_IvsQ.set_legend_font_size(self.miscsettingsdialog.spinBox_legendfontsize.value())
        self.m_IvsQ.canvas.draw()
        self.statusBar().showMessage('Done plotting slices.')

    @Slot()
    def on_pushButton_IvsQ_clearselected_clicked(self):
        self.listWidget_IvsQ.itemSelectionChanged.disconnect()
        for qmodelindex in sorted(get_sel_qmodelindices(self.listWidget_IvsQ), key=lambda i: i.row(), reverse=True):
            if self.radioButton_integrateQx.isChecked():
                ivsqz_list.pop(qmodelindex.row())
            else:
                ivsqx_list.pop(qmodelindex.row())
            self.listWidget_IvsQ.takeItem(qmodelindex.row())
        self.listWidget_IvsQ.itemSelectionChanged.connect(self.on_listWidget_IvsQ_itemSelectionChanged)

    @Slot()
    def on_pushButton_IvsQ_clearall_clicked(self):
        self.listWidget_IvsQ.clear()
        if self.radioButton_integrateQx.isChecked():
            ivsqz_list.clear()
        else:
            ivsqx_list.clear()

# %% Third tab (Photodiode)

    @Slot()
    def on_tableWidget_photodiode_itemSelectionChanged(self):
        selected_photodiode_list = [photodiode_list[qmodelindex.row()]
                                    for qmodelindex in get_sel_qmodelindices(self.tableWidget_photodiode)]
        self.statusBar().showMessage('Plotting...')
        try:
            for i, photodiode in enumerate(selected_photodiode_list):
                if self.radioButton_photodiode_plotphotodiode.isChecked():
                    ycorr = photodiode[0]['Photodiode'].copy()
                    if self.checkBox_photodiode_normalizecurrent.isChecked():
                        ycorr = ycorr / photodiode[0]['Beam Current']
                    if self.checkBox_photodiode_I0.isChecked():
                        ycorr = ycorr / photodiode[0]['AI 3 Izero']
                elif self.radioButton_photodiode_plotI0.isChecked():
                    ycorr = photodiode[0]['AI 3 Izero']
                elif self.radioButton_photodiode_plotbeamcurrent.isChecked():
                    ycorr = photodiode[0]['Beam Current']
                elif self.radioButton_photodiode_plotTEY.isChecked():
                    ycorr = photodiode[0]['TEY signal']
                keep_old = False if i == 0 else True
                self.m_photodiode.plot(
                    ycorr.index, ycorr, xlabel=ycorr.index.name, ylabel='', yscale='linear',
                    keep_old=keep_old, label=photodiode[1], marker='D')
        except KeyError:
            self.statusBar().showMessage('Missing data, try unchecking normalize or plotting something else.')
            return
        if len(selected_photodiode_list) > 0:
            self.m_photodiode.draw_legend()
            self.m_photodiode.set_legend_font_size(self.miscsettingsdialog.spinBox_legendfontsize.value())
        self.m_photodiode.canvas.draw()
        self.statusBar().showMessage('Done plotting.')

    @Slot(int, int)
    def on_tableWidget_photodiode_cellChanged(self, row, col):
        """Double click table widget item to change name"""
        if col == 1:
            photodiode_list[row][1] = self.tableWidget_photodiode.item(row, 1).text()

    @Slot(bool)
    def on_checkBox_photodiode_normalizecurrent_toggled(self, value):
        if value:
            self.on_tableWidget_photodiode_itemSelectionChanged()

    @Slot(bool)
    def on_checkBox_photodiode_I0_toggled(self, value):
        if value:
            self.on_tableWidget_photodiode_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_photodiode_plotphotodiode_toggled(self, value):
        if value:
            self.on_tableWidget_photodiode_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_photodiode_plotI0_toggled(self, value):
        if value:
            self.on_tableWidget_photodiode_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_photodiode_plotbeamcurrent_toggled(self, value):
        if value:
            self.on_tableWidget_photodiode_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_photodiode_plotTEY_toggled(self, value):
        if value:
            self.on_tableWidget_photodiode_itemSelectionChanged()

    def pushButton_photodiode_load_clicked(self, filelist):
        self.statusBar().showMessage('Loading photodiode file...')
        for filename in filelist:
            with open(filename) as f:
                frame = pd.read_table(f, skiprows=14, index_col=2)
            frame.index = np.round(frame.index, 1)
            #TODO: find better solution
            self.add_to_photodiode_tableWidget(frame, filename)
        self.statusBar().showMessage('Done loading photodiode file.')

    @Slot()
    def on_pushButton_photodiode_clear_clicked(self):
        self.tableWidget_photodiode.itemSelectionChanged.disconnect()
        for qmodelindex in sorted(get_sel_qmodelindices(self.tableWidget_photodiode), key=lambda i: i.row(), reverse=True):
            photodiode_list.pop(qmodelindex.row())
            self.tableWidget_photodiode.removeRow(qmodelindex.row())
        self.tableWidget_photodiode.itemSelectionChanged.connect(self.on_tableWidget_photodiode_itemSelectionChanged)

    @Slot()
    def on_pushButton_photodiode_apply_clicked(self):
        """Apply selected photodiode file to selected scattering files as div_photodiode"""
        self.statusBar().showMessage('Dividing selected 2D scans by selected photodiode file...')
        selected_photodiode_list = [photodiode_list[qmodelindex.row()]
                                    for qmodelindex in get_sel_qmodelindices(self.tableWidget_photodiode)]
        if len(selected_photodiode_list) != 1:
            self.statusBar().showMessage('One photodiode file must be selected.')
            return
        photodiode = selected_photodiode_list[0]
        scatteringfilelist = self.get_scatteringfilelist()
        # TODO: only apply to 2D scans with E matching points in photodiode file
        if len(scatteringfilelist) != len(photodiode[0]):
            self.statusBar().showMessage('Number of selected 2D scans and points in photodiode file must match.')
            return
        ycorr = photodiode[0]['Photodiode']
        if self.checkBox_photodiode_normalizecurrent.isChecked():
            ycorr = ycorr / photodiode[0]['Beam Current']
        if self.checkBox_photodiode_I0.isChecked():
            ycorr = ycorr / photodiode[0]['AI 3 Izero']
        for i, sf in enumerate(scatteringfilelist):
            sf.__init__(sf.fileformat, sf.fullfilename, sf.dataqxzqy.params, sf.info, sf.infostr, div_photodiode=ycorr.iloc[i])
        self.statusBar().showMessage('Done dividing selected 2D scans by selected photodiode file.')

    @Slot()
    def on_pushButton_photodiode_reset_clicked(self):
        """Reset div_photodiode to 1 for selected scattering files"""
        self.statusBar().showMessage('Resetting div_photodiode of selected 2D scans...')
        for i, sf in enumerate(self.get_scatteringfilelist()):
            sf.__init__(sf.fileformat, sf.fullfilename, sf.dataqxzqy.params, sf.info, sf.infostr, div_photodiode=1)
        self.statusBar().showMessage('Done resetting div_photodiode of selected 2D scans.')

    @Slot()
    def on_pushButton_photodiode_add_clicked(self):
        """Add calculated photodiode scan to tablewidget using user formula, points where x-axis does not match will be NaN"""
        self.statusBar().showMessage('Adding calculated photodiode scan to list...')
        # automatically generated horizontal headers in QTableWidget have indices starting at 1
        if self.checkBox_photodiode_normalizecurrent.isChecked() and self.checkBox_photodiode_I0.isChecked():
            p_dict = {'p' + str(index + 1): photodiode[0]['Photodiode'] / photodiode[0]['Beam Current'] / photodiode[0]['AI 3 Izero']
                      for index, photodiode in enumerate(photodiode_list) if 'Photodiode' in photodiode[0].keys()}
            t_dict = {'t' + str(index + 1): photodiode[0]['TEY signal'] / photodiode[0]['Beam Current'] / photodiode[0]['AI 3 Izero']
                      for index, photodiode in enumerate(photodiode_list) if 'TEY signal' in photodiode[0].keys()}
        elif self.checkBox_photodiode_normalizecurrent.isChecked():
            p_dict = {'p' + str(index + 1): photodiode[0]['Photodiode'] / photodiode[0]['Beam Current']
                      for index, photodiode in enumerate(photodiode_list) if 'Photodiode' in photodiode[0].keys()}
            t_dict = {'t' + str(index + 1): photodiode[0]['TEY signal'] / photodiode[0]['Beam Current']
                      for index, photodiode in enumerate(photodiode_list) if 'TEY signal' in photodiode[0].keys()}
        elif self.checkBox_photodiode_I0.isChecked():
            p_dict = {'p' + str(index + 1): photodiode[0]['Photodiode'] / photodiode[0]['AI 3 Izero']
                      for index, photodiode in enumerate(photodiode_list) if 'Photodiode' in photodiode[0].keys()}
            t_dict = {'t' + str(index + 1): photodiode[0]['TEY signal'] / photodiode[0]['AI 3 Izero']
                      for index, photodiode in enumerate(photodiode_list) if 'TEY signal' in photodiode[0].keys()}
        else:
            p_dict = {'p' + str(index + 1): photodiode[0]['Photodiode']
                          for index, photodiode in enumerate(photodiode_list) if 'Photodiode' in photodiode[0].keys()}
            t_dict = {'t' + str(index + 1): photodiode[0]['TEY signal']
                      for index, photodiode in enumerate(photodiode_list) if 'TEY signal' in photodiode[0].keys()}
        x_dict = {'x' + str(index + 1): photodiode[0].index for index, photodiode in enumerate(photodiode_list)}
        p_dict.update(x_dict)
        p_dict.update(t_dict)
        frame = pd.DataFrame({
            'Photodiode': eval(self.lineEdit_photodiode_formula.text(), globals(), p_dict).dropna(),
            'AI 3 Izero': 1,
            'Beam Current': 1})
        file = self.lineEdit_photodiode_name.text() + ' ' + self.lineEdit_photodiode_formula.text()
        self.add_to_photodiode_tableWidget(frame, file)
        self.statusBar().showMessage('Done adding calculated photodiode scan to list.')

    def add_to_photodiode_tableWidget(self, frame, file):
        """Items are [dataframe, filename] lists"""
        photodiode_list.append([frame, file])
        row = self.tableWidget_photodiode.rowCount()
        self.tableWidget_photodiode.insertRow(row)
        self.tableWidget_photodiode.setItem(row, 1, QtWidgets.QTableWidgetItem(file))
        self.tableWidget_photodiode.setItem(row, 0, QtWidgets.QTableWidgetItem(str(len(frame))))

# %% Fourth tab (Qx-E), top

    @Slot()
    def on_listWidget_QxEobj_itemSelectionChanged(self):
        """Replots dataqxe, dataqxeinterp"""
        try:
            self.qxe = qxe_list[self.listWidget_QxEobj.currentRow()]
        except IndexError:
            return
        self.statusBar().showMessage('Plotting...')
        self.qxe[0].plot_self(self.m_QxEobj)
        self.qxe[1].plot_self(self.m_QxEobj_interp)
        self.statusBar().showMessage('Done.')

    def m_QxEobj_interp_clicked(self, event):
        """Adds line with qx or E value to text box"""
        if self.checkBox_QxEslice_clicktoadd.isChecked():
            if self.radioButton_QxEslice_integrateE.isChecked():
                self.plainTextEdit_QxEslice_list_tointegrate.insertPlainText('{0:.4g}, '.format(event.xdata))
            else:
                self.plainTextEdit_QxEslice_list_tointegrate.insertPlainText('{0:.4g}, '.format(event.ydata))

    @Slot()
    def on_pushButton_QxEobj_new_clicked(self):
        """items in Qxe_list are [dataqxe, dataqxeinterp] lists"""
        name = self.lineEdit_QxEobj_name.text()
        dataqxe = processing.DataQxE(self.get_scatteringfilelist(), title=name)
        dataqxeinterp = processing.DataQxEInterp(
            dataqxe.E_listoflists, dataqxe.qx_listoflists, dataqxe.I_listoflists, title=name)
        qxe_list.append([dataqxe, dataqxeinterp])
        self.listWidget_QxEobj.addItem(name)

    @Slot()
    def on_pushButton_QxEobj_clear_clicked(self):
        qxe_list.pop(self.listWidget_QxEobj.currentRow())
        self.listWidget_QxEobj.takeItem(self.listWidget_QxEobj.currentRow())

# %% Fourth tab, bottom

    @Slot(bool)
    def on_radioButton_QxEslice_integrateE_toggled(self, value):
        """Updates listWidget_QxEslice"""
        if not value:
            return
        self.listWidget_QxEslice.clear()
        for qxe_ivsqx in qxe_ivsqx_list:
            self.listWidget_QxEslice.addItem('{0}, E={1}'.format(qxe_ivsqx.title, qxe_ivsqx.E_midpt))

    @Slot(bool)
    def on_radioButton_QxEslice_integrateQx_toggled(self, value):
        """Updates listWidget_QxEslice"""
        if not value:
            return
        self.listWidget_QxEslice.clear()
        for qxe_ivse in qxe_ivse_list:
            self.listWidget_QxEslice.addItem('{0}, qx={1}'.format(qxe_ivse.title, qxe_ivse.qx_midpt))

    @Slot()
    def on_pushButton_QxEslice_generatevalues_clicked(self):
        if self.linspacearangedialog.exec_():
            self.linspacearangedialog.ok_clicked(self.plainTextEdit_QxEslice_list_tointegrate)

    @Slot()
    def on_pushButton_QxEslice_cleartextbox_clicked(self):
        self.plainTextEdit_QxEslice_list_tointegrate.clear()

    @Slot()
    def on_pushButton_QxEslice_integrate_clicked(self):
        """Removes old rectangles, draws rectangles, adds to qxe_ivsqx_list or qxe_ivse_list"""
        # removes rectangles from interpolated image
        self.m_QxEobj_interp.draw_collection([])
        if len(self.m_QxEobj.axis.collections) > 1:
            # removes rectangles from non-interpolated image but keeps scatter data
            self.m_QxEobj.axis.collections[1].remove()
            self.m_QxEobj.canvas.draw()
        text_list = self.plainTextEdit_QxEslice_list_tointegrate.toPlainText().split(',')
        if len(text_list) == 0:
            return

        q_midpts = []
        self.statusBar().showMessage('Integrating...')
        for item in text_list:
            try:
                q_midpts.append(float(item))
            except ValueError:
                continue
        q_width = self.doubleSpinBox_QxEslice_boxwidth.value()
        bg_q = self.doubleSpinBox_QxEslice_background.value()
        # adds to qxe_ivsqx_list or qxe_ivse_list
        if self.radioButton_QxEslice_integrateE.isChecked():
            new_ivsq_list = [processing.QxE_IvsQx(q_midpt, q_width, bg_q, self.qxe[0], self.qxe[1])
                             for q_midpt in q_midpts]
            qxe_ivsqx_list.extend(new_ivsq_list)
            self.on_radioButton_QxEslice_integrateE_toggled(True)
        else:
            new_ivsq_list = [processing.QxE_IvsE(q_midpt, q_width, bg_q, self.qxe[0], self.qxe[1])
                             for q_midpt in q_midpts]
            qxe_ivse_list.extend(new_ivsq_list)
            self.on_radioButton_QxEslice_integrateQx_toggled(True)

        self.statusBar().showMessage('Drawing rectangles...')
        # makes flat list of rects
        rects = list(itertools.chain(*[(ivsq.rect, ivsq.rect_bg) for ivsq in new_ivsq_list]))
        # draws new rectangles
        self.m_QxEobj_interp.draw_collection(rects)
        self.m_QxEobj.draw_collection(rects, keep_old=True)
        self.statusBar().showMessage('Done.')

    @Slot()
    def on_listWidget_QxEslice_itemSelectionChanged(self):
        """Clears I vs Q widget and plots selected curves"""
        if self.radioButton_QxEslice_integrateE.isChecked():
            ivsq_list = qxe_ivsqx_list
        else:
            ivsq_list = qxe_ivse_list
        self.statusBar().showMessage('Plotting...')
        selected_ivsq_list = [ivsq_list[qmodelindex.row()]
                              for qmodelindex in get_sel_qmodelindices(self.listWidget_QxEslice)]
        for i, ivsq in enumerate(selected_ivsq_list):
            keep_old = False if i == 0 else True
            if self.radioButton_QxEslice_plotinterp.isChecked():
                ivsq.plot_interp(self.m_QxEslice, keep_old=keep_old)
            elif self.radioButton_QxEslice_plotnoninterp.isChecked():
                ivsq.plot_noninterp(self.m_QxEslice, keep_old=keep_old)
        if len(selected_ivsq_list) > 0:
            self.m_QxEslice.draw_legend()
            self.m_QxEslice.set_legend_font_size(self.miscsettingsdialog.spinBox_legendfontsize.value())
        self.m_QxEslice.canvas.draw()
        self.statusBar().showMessage('Done.')

    @Slot(bool)
    def on_radioButton_QxEslice_plotnoninterp_toggled(self, value):
        if value:
            self.on_listWidget_QxEslice_itemSelectionChanged()

    @Slot(bool)
    def on_radioButton_QxEslice_plotinterp_toggled(self, value):
        if value:
            self.on_listWidget_QxEslice_itemSelectionChanged()

    @Slot()
    def on_pushButton_QxEslice_clearselected_clicked(self):
        self.listWidget_QxEslice.itemSelectionChanged.disconnect()
        for qmodelindex in sorted(get_sel_qmodelindices(self.listWidget_QxEslice), key=lambda i: i.row(), reverse=True):
            if self.radioButton_QxEslice_integrateE.isChecked():
                qxe_ivsqx_list.pop(qmodelindex.row())
            else:
                qxe_ivse_list.pop(qmodelindex.row())
            self.listWidget_QxEslice.takeItem(qmodelindex.row())
        self.listWidget_QxEslice.itemSelectionChanged.connect(self.on_listWidget_QxEslice_itemSelectionChanged)

    @Slot()
    def on_pushButton_QxEslice_clearall_clicked(self):
        self.listWidget_QxEslice.clear()
        if self.radioButton_QxEslice_integrateE.isChecked():
            qxe_ivsqx_list.clear()
        else:
            qxe_ivse_list.clear()

# %% Fifth tab (ALS Run Generator)

    def pushButton_gen_loadexposures_clicked(self, filelist):
        """Loads short scans used to calculate exposure time"""
        MAX_COUNTS = 40000
        self.statusBar().showMessage('Loading exposure scans...')
        params = self.make_params('fits')
        pitch_nm = self.doubleSpinBox_gen_pitch.value()
        for filename in filelist:
            with open(filename, 'rb') as f:
                hdulist = fits.open(f)
                # first hdu: main info, second: table of Izeros info, third: image info
                info = [hdu.header for hdu in hdulist]
                f.seek(0)
                imgdata = hdulist[2].data
            peak_order = ALS_run_generator.find_peak_order(
                info[0]['Beamline Energy'], info[0]['Sample Theta'], info[0]['CCD Theta'], pitch_nm=pitch_nm, pixel_um=params.pixel_um,
                SDD_cm=params.SDD_cm, beamcenter_xz_px=params.center_px[0], detector_theta0=params.detector_theta0,
                detector_thetascale=params.detector_thetascale, qxz_px_start=50)
            max_I = eval('imgdata[' + self.lineEdit_gen_regionformaxI.text() + ']').max()
            items = [
                ('Energy', [info[0]['Beamline Energy']]),
                ('EPU', [info[0]['EPU Polarization']]),
                ('Sample theta (-90 normal)', [info[0]['Sample Theta']]),
                ('CCD Theta', [info[0]['CCD Theta']]),
                ('First peak order', [peak_order]),
                ('Time', [info[0]['EXPOSURE']]),
                ('Max I', [max_I]),
                ('Calc best time', [info[0]['EXPOSURE'] * MAX_COUNTS / max_I]),  # so counts don't go over 65535
            ]
            items[0][1][0] = np.round(items[0][1][0], 1)  # Energy rounded to 1 decimal place
            for item in items:
                item[1][0] = np.round(item[1][0], 2)  # Others rounded to 2 decimal places
            row = pd.DataFrame.from_items(items)
            global ALS_exposure_frame
            ALS_exposure_frame = ALS_exposure_frame.append(row)
            self.tableWidget_gen_exposures.setRowCount(self.tableWidget_gen_exposures.rowCount() + 1)
            for col, item in enumerate(items):
                self.tableWidget_gen_exposures.setItem(
                    self.tableWidget_gen_exposures.rowCount() - 1, col, QtWidgets.QTableWidgetItem(str(item[1][0])))
            if self.checkBox_gen_plusminusqz.isChecked():
                neg_sample_theta = ALS_run_generator.peak_order_to_neg_sample_thetas(
                    peak_order, info[0]['Beamline Energy'], info[0]['Sample Theta'], pitch_nm=pitch_nm)
                items[2][1][0] = np.round(neg_sample_theta, 2)
                row = pd.DataFrame.from_items(items)
                ALS_exposure_frame = ALS_exposure_frame.append(row)
                self.tableWidget_gen_exposures.setRowCount(self.tableWidget_gen_exposures.rowCount() + 1)
                for col, item in enumerate(items):
                    self.tableWidget_gen_exposures.setItem(
                        self.tableWidget_gen_exposures.rowCount() - 1, col, QtWidgets.QTableWidgetItem(str(item[1][0])))
        self.statusBar().showMessage('Done loading exposure scans.')

    @Slot()
    def on_pushButton_gen_clearexposures_clicked(self):
        global ALS_exposure_frame
        ALS_exposure_frame = pd.DataFrame(columns=(
            'Energy', 'EPU', 'Sample theta (-90 normal)', 'CCD Theta', 'First peak order', 'Time', 'Max I', 'Calc best time'))
        self.tableWidget_gen_exposures.setRowCount(0)

    def plainTextEdit_gen_runtext_customhandler(self, event):
        """Adds custom behavior to focus out"""
        self.statusBar().showMessage('ALS_frame global updated.')
        QtWidgets.QPlainTextEdit.focusOutEvent(self.plainTextEdit_gen_runtext, event)
        global ALS_frame
        with io.StringIO(self.plainTextEdit_gen_runtext.toPlainText()) as f:
            ALS_frame = ALS_run_generator.load_frame(f)

    @Slot(bool)
    def on_radioButton_gen_plotsampletheta_toggled(self, value):
        if value:
            self.on_pushButton_gen_plot_clicked()

    @Slot(bool)
    def on_radioButton_gen_plotdetectortheta_toggled(self, value):
        if value:
            self.on_pushButton_gen_plot_clicked()

    @Slot()
    def on_pushButton_gen_appendrun_clicked(self):
        """Adds scans based on parameters in input boxes to ALS run text box"""
        MIN_EXPOSURE = 0.1
        MAX_EXPOSURE = 60
        self.statusBar().showMessage('Adding scans to ALS run text box...')
        sample_thetas = list(np.arange(
            self.spinBox_gen_min.value(), self.spinBox_gen_max.value() + 0.01, self.spinBox_gen_step.value()))
        # distance from normal must continuously increase within each group to avoid decreasing detector_theta
        normal_index = base.find_closest_index(sample_thetas, -90)
        sample_thetas = list(reversed(sample_thetas[:normal_index])) + sample_thetas[normal_index:]
        sample_x = self.doubleSpinBox_gen_x.value()
        sample_ys, sample_zs = ALS_run_generator.make_sample_ys_zs(
            y90=self.doubleSpinBox_gen_y.value(), z0=self.doubleSpinBox_gen_z0.value(),
            z180=self.doubleSpinBox_gen_z180.value(), sample_thetas=sample_thetas)
        avoid_peaks = [self.spinBox_gen_peakorder.value()] * len(sample_thetas)
        exposures = [self.doubleSpinBox_gen_exposure.value()] * len(sample_thetas)
        energy_ev = self.doubleSpinBox_gen_energy.value()
        polarization = self.spinBox_gen_EPU.value()
        suppressors = [self.doubleSpinBox_gen_suppressor.value()] * len(sample_thetas)
        if self.checkBox_gen_varydettheta.isChecked():
            detector_thetas = ALS_run_generator.make_detector_thetas_new(
                sample_thetas, avoid_peaks, energy_ev, self.doubleSpinBox_gen_pitch.value())  # detector theta floored to 1 degree
        else:
            detector_thetas = self.doubleSpinBox_gen_fixeddettheta.value()
        frame = ALS_run_generator.make_frame(
            energy_ev, polarization, sample_x, sample_ys, sample_zs, sample_thetas,
            detector_thetas, suppressors, exposures)
        if self.checkBox_gen_exposures.isChecked():
            filtered = ALS_exposure_frame[
                (ALS_exposure_frame['Energy'] == energy_ev) &
                (ALS_exposure_frame['EPU'] == polarization) &
                (ALS_exposure_frame['First peak order'] == self.spinBox_gen_peakorder.value())]
            filtered = filtered.drop_duplicates('Sample theta (-90 normal)').sort_values('Sample theta (-90 normal)')
            try:
                fun = interpolate.UnivariateSpline(filtered['Sample theta (-90 normal)'], filtered['Calc best time'], ext=3)
                calc_best_time_raw = fun(sample_thetas)
                calc_best_time = np.fmin(calc_best_time_raw, MAX_EXPOSURE)
                calc_best_time = np.fmax(calc_best_time, MIN_EXPOSURE)
                calc_best_time = np.round(calc_best_time, 1)
                frame[''] = calc_best_time  # round best time to nearest 0.1s because we don't trust precision of shutter
                # if calc_best_time_raw less than 0.1s, add 1 to suppressor
                for i in range(frame.shape[0]):
                    if calc_best_time_raw[i] < 0.1:
                        frame['Higher Order Suppressor'].iloc[i] += 1
            except:
                self.statusBar().showMessage(
                    'Error in calculating exposures from scans, are you sure exposure scans '
                    'have the same energy, EPU, and first peak order as the run you want to create?')
        global ALS_frame
        ALS_frame = ALS_frame.append(frame, ignore_index=True)
        with io.StringIO() as f:
            ALS_run_generator.save_frame(ALS_frame, f)
            self.plainTextEdit_gen_runtext.setPlainText(f.getvalue())
        self.statusBar().showMessage('Done adding scans to ALS run text box.')

    @Slot()
    def on_pushButton_gen_insertref_clicked(self):
        """Inserts extra scans to track beam intensity
         Extra scans have same parameters as the normal incidence scan with their energy and EPU
         Scans are inserted at beginning and end, at jumps in sample theta, and optionally at intervals between those jumps"""
        self.statusBar().showMessage('Adding exposure scans between jumps in sample theta to ALS run text box...')
        global ALS_frame
        ALS_frame = ALS_run_generator.insert_ref(ALS_frame, self.spinBox_gen_numrefscans.value())
        with io.StringIO() as f:
            ALS_run_generator.save_frame(ALS_frame, f)
            self.plainTextEdit_gen_runtext.setPlainText(f.getvalue())
        self.statusBar().showMessage('Done adding exposure scans between jumps in sample theta to ALS run text box.')

    @Slot()
    def on_pushButton_gen_insertfiller_clicked(self):
        self.statusBar().showMessage('Adding filler scans between EPU changes to ALS run text box...')
        filler_ys, filler_zs = ALS_run_generator.make_sample_ys_zs(
            y90=self.doubleSpinBox_gen_fillery.value(), z0=self.doubleSpinBox_gen_z0.value(),
            z180=self.doubleSpinBox_gen_z180.value(), sample_thetas=[-90])
        global ALS_frame
        ALS_frame = ALS_run_generator.insert_filler(
            ALS_frame, self.doubleSpinBox_gen_fillerexposure.value(), self.doubleSpinBox_gen_fillerx.value(), filler_ys[0], filler_zs[0])
        with io.StringIO() as f:
            ALS_run_generator.save_frame(ALS_frame, f)
            self.plainTextEdit_gen_runtext.setPlainText(f.getvalue())
        self.statusBar().showMessage('Done adding filler scans.')

    @Slot()
    def on_pushButton_gen_plot_clicked(self):
        self.statusBar().showMessage('Plotting ALS run...')
        detector_theta_plot = self.widget_gen_QzQx if self.radioButton_gen_plotdetectortheta.isChecked() else False
        sample_theta_plot = self.widget_gen_QzQx if self.radioButton_gen_plotsampletheta.isChecked() else False
        filtered = ALS_frame[
            (ALS_frame['Beamline Energy'] == self.doubleSpinBox_gen_energy.value()) &
            (ALS_frame['EPU Polarization'] == self.spinBox_gen_EPU.value())]
        params = self.make_params('fits')
        ALS_run_generator.plot_qz_qx(
            filtered, self.doubleSpinBox_gen_pitch.value(), num_slices=13, pixel_um=params.pixel_um,
            SDD_cm=params.SDD_cm, beamcenter_xz_px=params.center_px[0], detector_theta0=params.detector_theta0,
            detector_thetascale=params.detector_thetascale, qxz_px_start=50, qxz_px_end=970,
            detector_theta_plot=detector_theta_plot, sample_theta_plot=sample_theta_plot)
        self.statusBar().showMessage('Done plotting ALS run.')

    @Slot()
    def on_pushButton_gen_plotexposures_clicked(self):
        """Plot counts per second vs sample theta of exposure scans, for single energy, EPU, and peak order from input boxes
        Also plots interpolated best time using sample thetas from input boxes"""
        MIN_EXPOSURE = 0.1
        MAX_EXPOSURE = 60
        self.statusBar().showMessage('Plotting counts per second of exposure scans...')
        sample_thetas = list(np.arange(
            self.spinBox_gen_min.value(), self.spinBox_gen_max.value() + 0.01, self.spinBox_gen_step.value()))
        filtered = ALS_exposure_frame[
            (ALS_exposure_frame['Energy'] == self.doubleSpinBox_gen_energy.value()) &
            (ALS_exposure_frame['EPU'] == self.spinBox_gen_EPU.value()) &
            (ALS_exposure_frame['First peak order'] == self.spinBox_gen_peakorder.value())]
        filtered = filtered.drop_duplicates('Sample theta (-90 normal)').sort_values('Sample theta (-90 normal)')
        #fun = interpolate.UnivariateSpline(filtered['Sample theta (-90 normal)'], filtered['Max I'] / filtered['Time'], ext=3)
        #I_interp = fun(sample_thetas)
        fun = interpolate.UnivariateSpline(filtered['Sample theta (-90 normal)'], filtered['Calc best time'], ext=3)
        calc_best_time = fun(sample_thetas)
        calc_best_time = np.fmin(calc_best_time, MAX_EXPOSURE)
        calc_best_time = np.fmax(calc_best_time, MIN_EXPOSURE)
        calc_best_time = np.round(calc_best_time, 1)
        widget = matplotlibwidget.MatplotlibWidget1D()
        widget.plot(sample_thetas, calc_best_time, xlabel='Sample theta (-90 normal)', ylabel='Calc best time (s)')
        widget = matplotlibwidget.MatplotlibWidget1D()
        widget.plot(filtered['Sample theta (-90 normal)'], filtered['Max I'] / filtered['Time'],
                    xlabel='Sample theta (-90 normal)', ylabel='Max I / Time', marker='o', linestyle='None')
        self.statusBar().showMessage('Done plotting counts per second of exposure scans.')

    def pushButton_gen_savetxt_clicked(self, filename):
        self.statusBar().showMessage('Saving ALS_frame...')
        ALS_run_generator.save_frame(ALS_frame, filename)
        self.statusBar().showMessage('Done saving ALS_frame.')

    def pushButton_gen_loadtxt_clicked(self, filename):
        self.statusBar().showMessage('Loading ALS_frame...')
        global ALS_frame
        ALS_frame = ALS_run_generator.load_frame(filename)
        with open(filename) as f:
            self.plainTextEdit_gen_runtext.setPlainText(f.read())
        self.statusBar().showMessage('Done loading ALS_frame.')


class SliceCenterPitchDialog(QtWidgets.QDialog):
    def __init__(self, __file__, parent=None):
        super(SliceCenterPitchDialog, self).__init__(parent)
        uic.loadUi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'slice_center_pitch_ui.ui'), self)

    def ok_clicked(self, plaintextedit):
        first = self.spinBox_first.value()
        last = self.spinBox_last.value()
        offset = self.doubleSpinBox_offset.value()
        pitch = self.doubleSpinBox_pitch.value()

        values = np.round(np.arange(first, last + 1) * 2 * np.pi / pitch + offset, 5)
        plaintextedit.setPlainText(', '.join([str(i) for i in values]) + ', ')


class MiscSettingsDialog(QtWidgets.QDialog):
    def __init__(self, __file__, parent=None):
        super(MiscSettingsDialog, self).__init__(parent)
        uic.loadUi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'misc_settings_ui.ui'), self)

    @Slot()
    def on_pushButton_missingfolder_clicked(self):
        directoryname = QtWidgets.QFileDialog.getExistingDirectory(
            parent=self, caption='Choose existing directory')
        if len(directoryname) == 0:
            return
        self.pushButton_missingfolder.setText(directoryname)


def get_sel_qmodelindices(widget):
    """Returns a list of QModelIndex in the order that the rows were selected in
    need to sort from highest to lowest QModelIndex.row() if taking items out"""
    return widget.selectionModel().selectedRows()


def main():
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = guisupport.get_app_qt4()
#    app.setStyle('cleanlooks')
    global window
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    if getattr(sys, 'frozen', False):
        app.exec_()
    else:
        guisupport.start_event_loop_qt4(app)
    return window

if __name__ == '__main__':
    main()
