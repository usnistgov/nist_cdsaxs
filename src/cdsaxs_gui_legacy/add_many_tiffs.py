# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:50:57 2016
Modified on Thur Mar 26, 2025

@author: cdl and cmw
"""

import numpy as np
from qtpy import QtCore, QtWidgets
from PIL import Image
from IPython.lib import guisupport
import os


def main():
    csv_path = QtWidgets.QFileDialog.getOpenFileName(
        caption='Choose .csv file that lists files to add together:',
    )
    if not (QtCore.QT_VERSION >> 16) == 4:
        csv_path = csv_path[0]
    if len(csv_path) == 0:
        return
    # if csv_path[-4:] != '.csv':
    #     csv_path += '.csv'

    data_folder = QtWidgets.QFileDialog.getExistingDirectory(
        caption='Select the folder containing the data files:')
    # if not (QtCore.QT_VERSION >> 16) == 4:
    #     data_folder = data_folder[0]
    if len(data_folder) == 0:
        return
    print(data_folder)

    save_folder = QtWidgets.QFileDialog.getExistingDirectory(
        caption='Select a folder to save the combined data files:')
    # if not (QtCore.QT_VERSION >> 16) == 4:
    #     save_folder = save_folder[0]
    if len(save_folder) == 0:
        return
    print(save_folder)

    csv_table = np.loadtxt(csv_path, dtype=str, delimiter=',')
    new_filenames = np.unique(csv_table[:, 1])

    final_array = None
    for new in new_filenames:
        select = csv_table[:,1] == new
        grab_files = csv_table[select, 0]
        final_array = None
        for filename in grab_files:
            with Image.open(os.path.join(data_folder, filename)) as current_image:
                if final_array is None:
                    width, height = current_image.size
                    final_array = np.empty((height, width))
                final_array += np.asarray(current_image)
        final_image = Image.fromarray(final_array)
        final_image.save(os.path.join(save_folder, new), format='tiff')
        print("New image saved: ", new)

if __name__ == '__main__':
    app = guisupport.get_app_qt4()
    main()
