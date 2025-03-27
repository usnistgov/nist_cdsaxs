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

    """
    This script can be used to combine multible TIFF images into a single
    image. The images are simply summed together, so be cautious of
    whether this makes sense for your data. 

    To run the script from the cdsaxs_legacy Python environment:
    python path/to/add_many_tiffs.py

    You will be prompted to select the location of the CSV input file,
    the folder in which the original images are stored, and the folder
    where you would like the combined images exported.

    The CSV input file should contain two columns and no headers.
    The first column is a list of all original filenames. The second
    column contains the filename of the combined image the original
    file should be added to. For example:

    file_sampleA_001.tif, combined_file_sampleA.tif
    file_sampleA_002.tif, combined_file_sampleA.tif
    file_sampleB_001.tif, combined_file_sampleB.tif
    file_sampleB_002.tif, combined_file_sampleB.tif
    file_sampleB_003.tif, combined_file_sampleB.tif
    file_sampleA_003.tif, combined_file_sampleA.tif

    In the above sample, the files in rows 1, 2, and 6 will be combined
    into a single image with the filename 'combined_file_sampleA.tif'
    while the files in rows 3, 4, and 5 will be combined into a single
    image with the filename 'combined_file_sampleB.tif'. This
    demonstrates that you don't have to list the files in any particular
    order, only that the filename for the combined file be
    carefully applied to the appropriate rows.

    """
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
