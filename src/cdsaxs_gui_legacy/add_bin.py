# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:33:52 2016

@author: cdl
"""

import numpy as np
from qtpy import QtCore, QtWidgets
from IPython.lib import guisupport


def main():
    filenames_bin = QtWidgets.QFileDialog.getOpenFileNames(
        caption='Choose .bin files to add together:', filter='*.bin')
    if not (QtCore.QT_VERSION >> 16) == 4:
        filenames_bin = filenames_bin[0]
    if len(filenames_bin) == 0:
        return
    filename_new = QtWidgets.QFileDialog.getSaveFileName(
        caption='Choose new filename, you have to type the .bin', filter='*.bin')
    if not (QtCore.QT_VERSION >> 16) == 4:
        filename_new = filename_new[0]
    if len(filename_new) == 0:
        return
    final_array = None
    for filename in filenames_bin:
        current_image = np.fromfile(filename, dtype=np.float64)
        if current_image.size == 287626:
            current_image = current_image[1:]
        current_image = current_image.reshape(195, 1475)
        if final_array is None:
            final_array = np.empty(current_image.shape)
        final_array += current_image
    final_array.tofile(filename_new)
    print('New image saved.')

if __name__ == '__main__':
    app = guisupport.get_app_qt4()
    main()
