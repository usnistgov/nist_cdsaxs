# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:50:57 2016

@author: cdl
"""

import numpy as np
from qtpy import QtCore, QtWidgets
from PIL import Image
from IPython.lib import guisupport


def main():
    filenames_tif = QtWidgets.QFileDialog.getOpenFileNames(
        caption='Choose .tif files to add together:', filter='*.tif')
    if not (QtCore.QT_VERSION >> 16) == 4:
        filenames_tif = filenames_tif[0]
    if len(filenames_tif) == 0:
        return
    filename_new = QtWidgets.QFileDialog.getSaveFileName(
        caption='Choose new filename, you have to type the .tif', filter='*.tif')
    if not (QtCore.QT_VERSION >> 16) == 4:
        filename_new = filename_new[0]
    if len(filename_new) == 0:
        return
    final_array = None
    for filename in filenames_tif:
        with Image.open(filename) as current_image:
            if final_array is None:
                width, height = current_image.size
                final_array = np.empty((height, width))
            final_array += np.asarray(current_image)
    final_image = Image.fromarray(final_array)
    final_image.save(filename_new, format='tiff')
    print('New image saved.')

if __name__ == '__main__':
    app = guisupport.get_app_qt4()
    main()
