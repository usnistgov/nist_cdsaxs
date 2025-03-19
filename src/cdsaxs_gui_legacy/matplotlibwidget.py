# -*- coding: utf-8 -*-
"""
This is part of the CDSAXS Data Processing GUI.
This module has Qt widgets which contain matplotlib figures and additions to make them more interactive.
http://stackoverflow.com/questions/12459811/how-to-embed-matplotib-in-pyqt-for-dummies
"""

from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import *
from future.moves.itertools import zip_longest

import copy
import re
# import cPickle as pickle
import pickle
import os
import sys
import subprocess
import tempfile
import csv
import itertools
import warnings

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Slot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.lib import guisupport
if (QtCore.QT_VERSION >> 16) == 4:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from cdsaxs_gui_legacy.base import find_closest_index, find_closest_index_2D

# sns.set_style('dark')
# mpl.rcParams['axes.facecolor'] = 'black'
mpl.rcParams['figure.facecolor'] = 'white'
if hasattr(mpl.cm, 'viridis'):
    mpl.rcParams['image.cmap'] = 'viridis'
    mpl.cm.viridis.set_over('r')
else:
    mpl.rcParams['image.cmap'] = 'YlGnBu_r'
    mpl.cm.YlGnBu_r.set_over('r')
mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['legend.frameon'] = False
try:
    mpl.rcParams['patch.force_edgecolor'] = True
    mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
except KeyError:
    pass
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0
COORD_SEPARATE_LINE = True  # if window is small, necessary to see coords

if sys.version_info >= (3, 0, 0):
    opencsv_kwargs = {'mode': 'w', 'newline': ''}
else:
    opencsv_kwargs = {'mode': 'wb'}


def move_to_dock(qwidget, position=-1):
    """Moves qwidget to inside a dockwidget inside dockwidgetwindow
    position: index of new tab, QtCore.Qt.Vertical, or QtCore.Qt.Horizontal"""
    warnings.filterwarnings('ignore')
    global dockwidgetwindow
    if 'dockwidgetwindow' not in globals():
        dockwidgetwindow = QtWidgets.QMainWindow()
        dockwidgetwindow.resize(800, 600)
        dockwidgetwindow.setDockNestingEnabled(True)
        dockwidgetwindow.setWindowTitle('DockWidgetWindow')
        dockwidgetwindow.show()

    docks = dockwidgetwindow.findChildren(QtWidgets.QDockWidget)
    dock = QtWidgets.QDockWidget(dockwidgetwindow)
    dock.setWindowTitle(qwidget.windowTitle())

    if position is QtCore.Qt.Vertical:
        dockwidgetwindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), dock, QtCore.Qt.Vertical)
    elif position is QtCore.Qt.Horizontal:
        dockwidgetwindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), dock, QtCore.Qt.Horizontal)
    else:
        dockwidgetwindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), dock)
        if len(docks) > 0:
            dockwidgetwindow.tabifyDockWidget(docks[position], dock)
    dock.setWidget(qwidget)


class MplWithSliderWidget(QtWidgets.QWidget):
    """Contains a MatplotlibWidget2D with pcolorfast and a slider for plotting 2D slices of 3D arrays
    Optionally also plots a 1D slice if cursor hovers over 2D plot"""
    def __init__(self, data3d, axis0, axis1, axis2, axes_labels,
                 init_axis, init_value, logz=None, title='', parent=None):
        super(MplWithSliderWidget, self).__init__(parent)

        self.data3d = data3d
        self.axes = [axis0, axis1, axis2]
        self.axes_labels = axes_labels
        self.logz = logz
        self.title = title
        self.setWindowTitle(self.title)

        self.m_widget2d = MatplotlibWidget2D(self, allow_slice=True)
        self.slider = QtWidgets.QSlider(self)
        self.slider.setMaximum(np.shape(self.data3d)[init_axis] - 1)
        self.slider.setPageStep(1)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider.setValue(find_closest_index(self.axes[init_axis], init_value))
        self.combobox = QtWidgets.QComboBox(self)
        self.combobox.addItem(self.axes_labels[0])
        self.combobox.addItem(self.axes_labels[1])
        self.combobox.addItem(self.axes_labels[2])
        self.combobox.setCurrentIndex(init_axis)
        self.label = QtWidgets.QLabel('{0:4.4g}'.format(init_value), self)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.combobox)
        hbox.addWidget(self.label)
        hbox.addWidget(self.slider)
        hbox.setContentsMargins(0, 0, 0, 0)

        hbox_fig = QtWidgets.QHBoxLayout()
        hbox_fig.addWidget(self.m_widget2d)
        hbox_fig.setContentsMargins(0, 0, 0, 0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(hbox_fig)
        layout.addLayout(hbox)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.show()
        self.on_slider_valueChanged(0)
#        self.on_slider_sliderReleased()
        self.logz = None  # after initialization logz is controlled by GUI
        self.combobox.currentIndexChanged.connect(self.on_combobox_currentIndexChanged)
        self.slider.valueChanged.connect(self.on_slider_valueChanged)
#        self.slider.sliderReleased.connect(self.on_slider_sliderReleased)

    def save_to_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(
                (self.data3d, self.axes[0], self.axes[1], self.axes[2], self.axes_labels,
                 self.slider.value(), float(self.label.text()), self.logz, self.title),
                f, pickle.HIGHEST_PROTOCOL)

    def on_combobox_currentIndexChanged(self, index):
        self.slider.setMaximum(np.shape(self.data3d)[index] - 1)
        self.slider.setValue(0)
        self.on_slider_valueChanged(0)
#        self.on_slider_sliderReleased()

    def on_slider_valueChanged(self, index):
        axis = self.combobox.currentIndex()
        self.label.setText('{0:.4g}'.format(self.axes[axis][index]))
        indices = [slice(None)] * 3
        indices[axis] = index
        axes_2d = copy.copy(self.axes)
        axes_2d.pop(axis)
        axes_labels_2d = copy.copy(self.axes_labels)
        axes_labels_2d.pop(axis)
        self.m_widget2d.pcolorfast(self.data3d[indices], cols=axes_2d[1], rows=axes_2d[0], logz=self.logz,
                                   xlabel=axes_labels_2d[1], ylabel=axes_labels_2d[0], title=self.title)

#    def on_slider_sliderReleased(self):
#        axis = self.combobox.currentIndex()
#        indices = [slice(None)] * 3
#        indices[axis] = self.slider.value()
#        axes_2d = copy.copy(self.axes)
#        axes_2d.pop(axis)
#        axes_labels_2d = copy.copy(self.axes_labels)
#        axes_labels_2d.pop(axis)
#        self.m_widget2d.pcolorfast(self.data3d[indices], cols=axes_2d[1], rows=axes_2d[0], logz=self.logz,
#                                   xlabel=axes_labels_2d[1], ylabel=axes_labels_2d[0], title=self.title)


class MatplotlibWidget(QtWidgets.QWidget):
    """Base for MatplotlibWidget1D, MatplotlibWidget2D

    Attributes: figure, axis, canvas, toolbar, hbox, coordlabel
        layout() is QVboxLayout with toolbar, canvas, hbox with coordlabel

    Args:
        parent=None
        *args, **kwargs: used in Figure call, http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
    """
    def __init__(self, parent=None, *args, **kwargs):
        if not getattr(sys, 'frozen', False):
            app = guisupport.get_app_qt4()
        super(MatplotlibWidget, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.figure = mpl.figure.Figure(*args, tight_layout=True, **kwargs)
        self.axis = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = MyNavigationToolbar(self.canvas, self, coordinates=(not COORD_SEPARATE_LINE))
        self.hbox = QtWidgets.QHBoxLayout()
        self.checkBox_keeplimits = QtWidgets.QCheckBox('keep limits')
        self.coordlabel = QtWidgets.QLabel(self)
        # hides base methods from tab completion if %config IPCompleter.limit_to__all__ = True
        # self.__all__ = [i for i in dir(self) if i not in dir(QtWidgets.QWidget)]

        # set the layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(self.hbox)
        if COORD_SEPARATE_LINE:
            self.hbox.addWidget(self.coordlabel)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setStretchFactor(self.canvas, 1)
        self.setLayout(layout)
        self.show()

    def keyPressEvent(self, event):
        """Ctrl+C to copy bitmap to clipboard or Ctrl+Alt+C to copy png to clipboard
        Generates temporary files but you can delete them after you paste the png"""
        if event.key() == QtCore.Qt.Key_C and (QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ControlModifier) and (
                QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.AltModifier):
            print('300dpi PNG copied to clipboard.')
            clipboard = QtWidgets.QApplication.clipboard()
            mplwidget_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'mplwidget')
            os.makedirs(mplwidget_dir, exist_ok=True)
            t = tempfile.mkdtemp(dir=mplwidget_dir)
            i = 0
            while os.path.exists('img{}.png'.format(i)):
                i += 1
            t_png = os.path.join(t, 'img{}.png'.format(i))
            # t_svg = os.path.join(t, 'temp.svg')
            # t_emf = os.path.join(t, 'temp.emf')
            self.figure.savefig(t_png, dpi=300)
            # command = '"C:/Program Files/Inkscape/inkscape.exe" -f {0} -M {1}'.format(t_svg, t_emf)
            # subprocess.run(command)
            # os.remove(t_svg)
            data = QtCore.QMimeData()
            url = QtCore.QUrl.fromLocalFile(t_png)
            data.setUrls([url])
            clipboard.setMimeData(data)
        elif event.key() == QtCore.Qt.Key_C and QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            print('Bitmap copied to clipboard.')
            clipboard = QtWidgets.QApplication.clipboard()
            if (QtCore.QT_VERSION >> 16) == 4:
                clipboard.setPixmap(QtGui.QPixmap.grabWidget(self.canvas))
            else:
                clipboard.setPixmap(QtWidgets.QWidget.grab(self.canvas))

    def get_limits(self, keep_old=False):
        restore_limits_bool = False
        old_limits = {}
        if self.checkBox_keeplimits.isChecked():
            restore_limits_bool = True
            old_limits['left'], old_limits['right'] = self.axis.get_xlim()
            old_limits['bottom'], old_limits['top'] = self.axis.get_ylim()
            old_limits['xscale'] = self.axis.get_xscale()
            old_limits['yscale'] = self.axis.get_yscale()
        if hasattr(self, 'colorbar'):
            old_limits['vmin'], old_limits['vmax'] = self.colorbar.get_clim()
        if not keep_old:
            for ax in self.figure.axes:
                self.figure.delaxes(ax)
            self.axis = self.figure.add_subplot(1, 1, 1)
        return restore_limits_bool, old_limits

    def restore_limits(self, restore_limits_bool, old_limits):
        self.axis.axis('tight')
        if restore_limits_bool:
            self.axis.set_xlim(old_limits['left'], old_limits['right'])
            self.axis.set_ylim(old_limits['bottom'], old_limits['top'])
            self.axis.set_xscale(old_limits['xscale'])
            self.axis.set_yscale(old_limits['yscale'])

    def final_draw(self):
        self.axis.set_xlabel(self.xlabel)
        self.axis.set_ylabel(self.ylabel)
        self.axis.set_title(self.title)
        self.setWindowTitle(self.title)
#        self.cursor = mpl.widgets.Cursor(self.figure.axes[0], useblit=True, color='red')
        self.canvas.draw()

    def draw_legend(self):
        self.axis.legend()
        self.axis.legend_.draggable(True)
        self.canvas.draw()

    def set_font_size(self, font_size):
        for ax in self.figure.axes:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_size)
        self.canvas.draw()

    def set_legend_font_size(self, font_size):
        for ax in self.figure.axes:
            if ax.legend_ is not None:
                ax.legend(prop={'size': font_size})
                ax.legend_.draggable(True)
        self.canvas.draw()


class MatplotlibWidget1D(MatplotlibWidget):
    """QWidget containing matplotlib figure for 1D plots

    Attributes: figure, axis, canvas, toolbar, hbox, coordlabel
        layout() is QVboxLayout with toolbar, canvas, hbox with checkBox_keeplimits and coordlabel

    Args:
        parent=None
        *args, **kwargs: used in Figure call, http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
    """
    def __init__(self, parent=None, *args, **kwargs):
        super(MatplotlibWidget1D, self).__init__(parent, *args, **kwargs)
        self.xs_list = []
        self.ys_list = []
        self.label_list = []

        # define widgets
        self.checkBox_marker = QtWidgets.QCheckBox('draw markers', self)
        self.pushButton_savedata = QtWidgets.QPushButton('Save data', self)
        self.pushButton_savedata.clicked.connect(self.save_data)

        # set hbox2
        self.hbox2 = QtWidgets.QHBoxLayout()
        self.hbox2.addWidget(self.checkBox_marker)
        self.hbox2.addWidget(self.checkBox_keeplimits)
        self.hbox2.addWidget(self.pushButton_savedata)
        self.hbox2.setContentsMargins(0, 0, 0, 0)
        self.layout().addLayout(self.hbox2)

    def format_coord_wrapper1D(self):
        """returns format_coord which is required to have two arguments"""
        def format_coord(xcursor, ycursor):
            """Displays the following when hovering: index, xlabel, xvalue, ylabel, yvalue, linelabel"""
            index = find_closest_index(self.xs, xcursor)
            x_val = self.xs[index]
            y_val = self.ys[index]
            xlabel = self.xlabel.replace('$', '').split()
            xlabel = xlabel[0] if len(xlabel) > 0 else 'xlabel'
            ylabel = self.ylabel.replace('$', '').split()
            ylabel = ylabel[0] if len(ylabel) > 0 else 'ylabel'
            return 'px={0} {1}={2:.4g} {3}={4:.4g} {5}'.format(
                index, xlabel, x_val, ylabel, y_val, self.label)
        return format_coord

    def plot_frame(self, frame, title):
        for name, series in frame.iteritems():
            self.plot(
                series.index, series.values, xlabel=frame.index.name, ylabel=frame.columns.name,
                title=title, keep_old=True, label=name)

    def plot(self, xs, ys, xlabel=None, ylabel=None, title=None, keep_old=False, xscale=None, yscale=None, *args, **kwargs):
        """plots line(s) in existing figure

        Args:
            xs, ys: first line, used for cursor coordinates
            xlabel='x', ylabel='y', title='1D plot', keep_old=False, xscale=None, yscale=None, *args, **kwargs

        Some *args:
            format string for first line
            further xs, ys pairs with optional format strings

        Some **kwargs: http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.plot
            label: goes in legend
            marker, linestyle
        """

        if xlabel is not None:
            self.xlabel = xlabel
        elif hasattr(self, 'xlabel'):
            pass
        else:
            self.xlabel = 'x'
        if ylabel is not None:
            self.ylabel = ylabel
        elif hasattr(self, 'ylabel'):
            pass
        else:
            self.ylabel = 'y'
        if title is not None:
            self.title = title
        elif hasattr(self, 'title'):
            pass
        else:
            self.title = ''
        # must be arrays due to things like this bug: https://github.com/pydata/pandas/issues/8383
        self.xs = np.asarray(xs)
        self.ys = np.asarray(ys)

        if (not np.isfinite(np.nanmax(self.ys))) or np.nanmax(self.ys) <= 0:
            yscale = 'linear'
        if (not np.isfinite(np.nanmax(self.xs))) or np.nanmax(self.xs) <= 0:
            xscale = 'linear'

        restore_limits_bool, old_limits = self.get_limits(keep_old)
        if self.checkBox_marker.isChecked():
            lines = self.axis.plot(self.xs, self.ys, *args, marker='o', **kwargs)
        else:
            lines = self.axis.plot(self.xs, self.ys, *args, **kwargs)
        if xscale is not None:
            try:
                if xscale == 'log':
                    self.axis.set_xscale(xscale, nonposx='mask')
                else:
                    self.axis.set_xscale(xscale)
            except ValueError:
                print('Could not set xscale={}'.format(xscale))
                print(np.nanmax(self.xs))
                self.axis.set_xscale('linear')
        if yscale is not None:
            try:
                if yscale == 'log':
                    self.axis.set_yscale(yscale, nonposy='mask')
                else:
                    self.axis.set_yscale(yscale)
            except ValueError:
                print('Could not set yscale={}'.format(yscale))
                print(np.nanmax(self.ys))
                self.axis.set_yscale('linear')
        self.axis.format_coord = self.format_coord_wrapper1D()
        self.restore_limits(restore_limits_bool, old_limits)
        if hasattr(self, 'mycursor'):
            self.mycursor.disconnect_events()  # needed to garbage collect old cursor
        try:
            self.mycursor = SnapCursor(1, 1, self.xs, self.ys, self.axis.format_coord, self.axis, useblit=True, color='red')
        except:
            pass
        self.final_draw()

        if 'label' in kwargs:
            self.label = kwargs['label']
        else:
            self.label = self.ylabel
        if keep_old:
            self.xs_list.append(self.xs)
            self.ys_list.append(self.ys)
            self.label_list.append(self.label)
        else:
            self.xs_list = [self.xs]
            self.ys_list = [self.ys]
            self.label_list = [self.label]
        self.cur_line = len(self.xs_list) - 1
        return lines

    def remove_line(self, xs_list_index):
        for line in self.axis.lines:
            if self.label_list[xs_list_index] is not None and line.get_label() == self.label_list[xs_list_index]:
                print(self.label_list[xs_list_index] + ' removed from plot')
                line.remove()
                self.xs_list.pop(xs_list_index)
                self.ys_list.pop(xs_list_index)
                self.label_list.pop(xs_list_index)
                self.canvas.draw()
                break

    def change_cursor(self):
        self.xs = self.xs_list[self.cur_line]
        self.ys = self.ys_list[self.cur_line]
        self.label = self.label_list[self.cur_line]
        if hasattr(self, 'mycursor'):
            self.mycursor.disconnect_events()  # needed to garbage collect old cursor
        self.mycursor = SnapCursor(1, 1, self.xs, self.ys, self.axis.format_coord, self.axis, useblit=True, color='red')
        self.canvas.draw()

    def keyPressEvent(self, event):
        """Up/Down to select active line, Delete to delete active line"""
        if event.key() == QtCore.Qt.Key_Down:
            self.cur_line = self.cur_line + 1 if self.cur_line < len(self.xs_list) - 1 else 0
            self.change_cursor()
        elif event.key() == QtCore.Qt.Key_Up:
            self.cur_line = self.cur_line - 1 if self.cur_line > 0 else len(self.xs_list) - 1
            self.change_cursor()
        elif event.key() == QtCore.Qt.Key_Delete:
            self.remove_line(self.cur_line)
        else:
            super(MatplotlibWidget1D, self).keyPressEvent(event)

    def save_data(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(parent=self, caption='Choose .csv file to save', filter='*.csv')
        if not (QtCore.QT_VERSION >> 16) == 4:
            filename = filename[0]
        if len(filename) == 0:
            return
        with open(filename, **opencsv_kwargs) as f:
            writer = csv.writer(f)
            writer.writerow([label for ylabel in self.label_list for label in (self.xlabel, ylabel)])
            writer.writerows(zip_longest(
                *itertools.chain(*zip(self.xs_list, self.ys_list))))


class MatplotlibWidget2D(MatplotlibWidget):
    """QWidget containing matplotlib figure for 2D plots

    Attributes: figure, axis, canvas, toolbar, hbox, hbox2, coordlabel, slice_fig
        layout() is QVboxLayout with toolbar, canvas,
            hbox2 with label_min, lineEdit_min, label_max, lineEdit_max, checkBox_log, checkBox_keeplimits, checkBox_keepzlimits, pushButton
            hbox with coordlabel

    Args:
        parent=None, allow_slice=False
        *args, **kwargs: used in Figure call, http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
    """
    def __init__(self, parent=None, allow_slice=False, *args, **kwargs):
        super(MatplotlibWidget2D, self).__init__(parent, *args, **kwargs)

        # define widgets
        label_min = QtWidgets.QLabel('min', self)
        self.lineEdit_min = QtWidgets.QLineEdit(self)
        label_max = QtWidgets.QLabel('max', self)
        self.lineEdit_max = QtWidgets.QLineEdit(self)
        self.checkBox_log = QtWidgets.QCheckBox('log', self)
        self.checkBox_log.setChecked(True)
        self.checkBox_keepzlimits = QtWidgets.QCheckBox('keep z limits', self)
        self.pushButton = QtWidgets.QPushButton('Apply', self)
        self.pushButton.clicked.connect(self.switch_loglin)
        self.slice_fig = None
        if allow_slice:
            self.checkbox_showslice = QtWidgets.QCheckBox('show slice', self)
            self.checkbox_showslice.toggled.connect(self.show_slice)
            self.checkbox_plotcolumn = QtWidgets.QCheckBox('plot column', self)

        # set hbox2
        self.hbox2 = QtWidgets.QHBoxLayout()
        self.hbox2.addWidget(label_min)
        self.hbox2.addWidget(self.lineEdit_min)
        self.hbox2.addWidget(label_max)
        self.hbox2.addWidget(self.lineEdit_max)
        self.hbox2.addWidget(self.checkBox_log)
        self.hbox2.addWidget(self.checkBox_keeplimits)
        self.hbox2.addWidget(self.checkBox_keepzlimits)
        self.hbox2.addWidget(self.pushButton)
        if allow_slice:
            self.hbox2.addWidget(self.checkbox_plotcolumn)
            self.hbox2.addWidget(self.checkbox_showslice)
        self.hbox2.setContentsMargins(0, 0, 0, 0)
        self.layout().addLayout(self.hbox2)

    def format_coord_wrapper_image(self):
        """returns format_coord which is required to have two arguments
        also updates slice 1D plot if self.slice_fig is not None"""
        yscale = 'log' if self.checkBox_log.isChecked() else 'linear'

        def format_coord(xcursor, ycursor):
            """Displays the following when hovering: xlabel_px, xindex, ylabel_px, yindex,
            xlabel, xvalue, ylabel, yvalue, zvalue
            http://matplotlib.org/examples/api/image_zcoord.html"""
            col = find_closest_index(self.cols, xcursor)
            row = find_closest_index(self.rows, ycursor)
            I = self.data[row, col]
            if self.slice_fig is not None:
                if self.checkbox_plotcolumn.isChecked():
                    self.slice_fig.plot(self.rows, self.data[:, col], xlabel=self.ylabel,
                                        ylabel='Intensity (a.u.)', yscale=yscale, title='{0}={1}'.format(self.xlabel, self.cols[col]))
                else:
                    self.slice_fig.plot(self.cols, self.data[row], xlabel=self.xlabel,
                                        ylabel='Intensity (a.u.)', yscale=yscale, title='{0}={1}'.format(self.ylabel, self.rows[row]))
            xlabel = self.xlabel.replace('$', '').split()
            xlabel = xlabel[0] if len(xlabel) > 0 else 'xlabel'
            ylabel = self.ylabel.replace('$', '').split()
            ylabel = ylabel[0] if len(ylabel) > 0 else 'ylabel'
            return '{0}_px={1} {2}_px={3} {0}={4:.4g} {2}={5:.4g} I={6:.4g}'.format(
                xlabel,
                col,
                ylabel,
                row,
                self.cols[col],
                self.rows[row],
                I)
        return format_coord

#    def format_coord_wrapper_image(self):
#        """returns format_coord which is required to have two arguments"""
#        def format_coord(xcursor, ycursor):
#            """Displays intensity (z coordinate) when hovering: http://matplotlib.org/examples/api/image_zcoord.html"""
#            col = find_closest_index(self.cols, xcursor)
#            row = find_closest_index(self.rows, ycursor)
#            I = self.data[row, col]
#            return '{0}_px={1} {2}_px={3} {0}={4:.4g} {2}={5:.4g} I={6:.4g}'.format(
#                self.xlabel.split()[0], col, self.ylabel.split()[0], row, self.cols[col], self.rows[row], I)
#        return format_coord

    def format_coord_wrapper_scatter(self):
        """returns format_coord which is required to have two arguments"""
        def format_coord(xcursor, ycursor):
            """Displays the following when hovering: index, xlabel, xvalue, ylabel, yvalue, zvalue"""
            index = find_closest_index_2D(self.xs, self.ys, xcursor, ycursor)
            x = self.xs[index]
            y = self.ys[index]
            I = self.c[index]
            xlabel = self.xlabel.replace('$', '').split()
            xlabel = xlabel[0] if len(xlabel) > 0 else 'xlabel'
            ylabel = self.ylabel.replace('$', '').split()
            ylabel = ylabel[0] if len(ylabel) > 0 else 'ylabel'
            return 'i={0} {1}={2:.4g} {3}={4:.4g} I={5:.4g}'.format(
                index, xlabel, x, ylabel, y, I)
        return format_coord

    def get_norm(self, logz, restore_zlimits_bool, old_limits):
        if logz is None:
            logz = self.checkBox_log.isChecked()
        else:
            self.checkBox_log.setChecked(logz)
        if logz:
            if restore_zlimits_bool:
                norm = mpl.colors.LogNorm(old_limits['vmin'], old_limits['vmax'])
            else:
                norm = mpl.colors.LogNorm()
        else:
            if restore_zlimits_bool:
                norm = mpl.colors.Normalize(old_limits['vmin'], old_limits['vmax'])
            else:
                norm = mpl.colors.Normalize()
        return norm

    def make_colorbar(self, image):
        self.colorbar = self.figure.colorbar(image)
        minvalue, maxvalue = self.colorbar.get_clim()
        self.lineEdit_min.setText('{0:.4g}'.format(minvalue))
        self.lineEdit_max.setText('{0:.4g}'.format(maxvalue))

    def pcolorfast(self, data, cols=None, rows=None, col_edges=None, row_edges=None, logz=None, xlabel='x', ylabel='y',
                   title='', keep_old=False, **kwargs):
        """plots image in existing figure

        Args:
            data: 2D
            cols=None, rows=None: 1D or 2D, specify centers of pixels
            col_edges=None, row_edges=None: specify edges of pixels, if None assumes evenly spaced
                1D shape=2, or 1D shape=data.shape[0]+1 and data.shape[1]+1, or 2D shape=data.shape+1 (must be this or None for edgecolors)
            logz: True, False, or None to get from GUI
            xlabel='x', ylabel='y', title='', **kwargs

        Some **kwargs: http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.pcolorfast
            edgecolors: if present, uses pcolormesh instead
        """
        self.plottype = 'pcolorfast'
        self.data = np.asarray(data)
        self.cols = cols
        self.rows = rows
        self.col_edges = col_edges
        self.row_edges = row_edges
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.kwargs = kwargs

        restore_limits_bool, old_limits = self.get_limits(keep_old)
#        kwargs['interpolation'] = 'nearest'
#        kwargs['extent'] = find_imshow_extents(cols, rows)
#        kwargs['origin'] = 'lower'
        if (not np.isfinite(np.nanmax(self.data))) or np.nanmax(self.data) <= 0:
            logz = False
        self.kwargs['norm'] = self.get_norm(logz, self.checkBox_keepzlimits.isChecked(), old_limits)
#        kwargs['aspect'] = 'auto'
        if self.cols is None:
            self.cols = list(range(np.shape(self.data)[1]))
        if self.rows is None:
            self.rows = list(range(np.shape(self.data)[0]))
        if self.col_edges is None or self.row_edges is None:
            cols_back = [2 * self.cols[0] - self.cols[1]] + list(self.cols)
            cols_forw = list(self.cols) + [2 * self.cols[-1] - self.cols[-2]]
            self.col_edges = [(i + j) / 2 for i, j in zip(cols_back, cols_forw)]
        if self.row_edges is None:
            rows_back = [2 * self.rows[0] - self.rows[1]] + list(self.rows)
            rows_forw = list(self.rows) + [2 * self.rows[-1] - self.rows[-2]]
            self.row_edges = [(i + j) / 2 for i, j in zip(rows_back, rows_forw)]
        if 'edgecolors' in self.kwargs:
            # X, Y = np.meshgrid(self.col_edges, self.row_edges)
            image = self.axis.pcolormesh(np.asarray(self.col_edges), np.asarray(self.row_edges), self.data, **self.kwargs)
        else:
            image = self.axis.pcolorfast(np.asarray(self.col_edges), np.asarray(self.row_edges), self.data, **self.kwargs)
        if not keep_old:
            self.make_colorbar(image)
        self.axis.patch.set_facecolor('black')
        if len(np.asarray(self.cols).shape) == 2:
            self.xs = self.cols.ravel()
            self.ys = self.rows.ravel()
            self.c = self.data.ravel()
            self.axis.format_coord = self.format_coord_wrapper_scatter()
        else:
            self.axis.format_coord = self.format_coord_wrapper_image()
        self.restore_limits(restore_limits_bool, old_limits)
        if hasattr(self, 'mycursor'):
            self.mycursor.disconnect_events()  # needed to garbage collect old cursor
        if len(np.asarray(self.cols).shape) == 2:
            self.mycursor = SnapCursor(1, 1, self.xs, self.ys, self.axis.format_coord, self.axis, useblit=True, color='red')
        else:
            self.mycursor = SnapCursor(1, 3, self.cols, self.rows, self.axis.format_coord, self.axis, useblit=True, color='red')
        self.final_draw()

    def scatter(self, xs, ys, logz=None, xlabel='x', ylabel='y', title='', keep_old=False, **kwargs):
        """plots scatterplot in existing figure

        Args:
            xs, ys, logz=None, xlabel='x', ylabel='y', title='', keep_old=False, **kwargs

        Some **kwargs: http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.scatter
            c: list of numbers to be mapped to colors
            s: size of points
            linewidths: scalar
        """
        self.plottype = 'scatter'
        self.xs = np.asarray(xs)
        self.ys = np.asarray(ys)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.kwargs = kwargs

        restore_limits_bool, old_limits = self.get_limits(keep_old)
        self.kwargs['norm'] = self.get_norm(logz, self.checkBox_keepzlimits.isChecked(), old_limits)
        image = self.axis.scatter(self.xs, self.ys, **self.kwargs)
        if 'c' in self.kwargs:
            self.c = np.asarray(self.kwargs['c'])
            self.make_colorbar(image)
        else:
            self.c = np.ones(self.xs.shape)
        self.axis.patch.set_facecolor('black')
        self.axis.format_coord = self.format_coord_wrapper_scatter()
        self.restore_limits(restore_limits_bool, old_limits)
        if hasattr(self, 'mycursor'):
            self.mycursor.disconnect_events()  # needed to garbage collect old cursor
        self.cursor = SnapCursor(1, 1, self.xs, self.ys, self.axis.format_coord, self.axis, useblit=True, color='red')
        self.final_draw()

    def draw_collection(self, rects, keep_old=False):
        """deletes all rectangles in figure and draws ones in given list"""
        if not keep_old:
            while len(self.axis.collections) > 0:
                self.axis.collections[0].remove()
        if rects is None or len(rects) == 0:
            self.canvas.draw()
            return
        rectlist = [copy.copy(rect) for rect in rects if rect is not None]
        rectlist.append(mpl.patches.Rectangle((0, 0), 0, 0))  # draw invisible rectangle
        # for some reason drawing a collection of one patch causes it to disappear if you zoom in too much
        # and if you call add_patch on a transformed patch you need to apply ax.transData manually
        self.axis.add_collection(mpl.collections.PatchCollection(rectlist, match_original=True))
        self.canvas.draw()

    def draw_rects(self, rects, keep_old=False):
        """deletes all rectangles in figure and draws ones in given list"""
        if not keep_old:
            while len(self.axis.patches) > 0:
                self.axis.patches[0].remove()
        if rects is None or len(rects) == 0:
            self.canvas.draw()
            return
        for rect in rects:
            if rect is not None:
                # once rect is plotted it contains a reference to data
                rectcopy = copy.copy(rect)  # do this to avoid growing rect in calling object
                rectcopy.set_transform(rectcopy.get_transform() + self.axis.transData)
                self.axis.add_patch(rectcopy)
        self.canvas.draw()

    def switch_loglin(self):
        """updates mplwidget given min, max, log widgets
        assumes image and colorbar already exists
        use set_norm if log/lin doesn't change, otherwise delete all axes and replot"""
        def replot():
            if self.plottype == 'scatter':
                self.scatter(self.xs, self.ys, xlabel=self.xlabel, ylabel=self.ylabel, title=self.title, **self.kwargs)
            else:
                self.pcolorfast(self.data, self.cols, self.rows, self.col_edges, self.row_edges,
                                xlabel=self.xlabel, ylabel=self.ylabel, title=self.title, **self.kwargs)

        minvalue = float(self.lineEdit_min.text())
        maxvalue = float(self.lineEdit_max.text())
        if len(self.axis.images) > 0:
            images = self.axis.images
        elif len(self.axis.collections) > 0:
            images = self.axis.collections
        else:
            print('error replotting')
            return
        for image in images:
            if isinstance(image.norm, mpl.colors.LogNorm):
                if self.checkBox_log.isChecked():
                    image.set_norm(mpl.colors.LogNorm(minvalue, maxvalue))
                    self.canvas.draw()
                else:
                    replot()
            elif isinstance(image.norm, mpl.colors.Normalize):
                if not self.checkBox_log.isChecked():
                    image.set_norm(mpl.colors.Normalize(minvalue, maxvalue))
                    self.canvas.draw()
                else:
                    replot()

    @Slot(bool)
    def show_slice(self, value):
        if value:
            self.slice_fig = MatplotlibWidget1D()
        else:
            self.slice_fig = None


class MatplotlibWidget2DWithSlice(MatplotlibWidget2D):
    def __init__(self, parent=None, *args, **kwargs):
        super(MatplotlibWidget2DWithSlice, self).__init__(parent, allow_slice=True, *args, **kwargs)


class SnapCursor(mpl.widgets.Cursor):
    """Subclass of matplotlib cursor that snaps to centers of pixels

    Args:
        x_index, y_index: indices of splitted format_coord output string that contain values
        xs, ys: lists or arrays containing centers of pixels
        format_coord: function returning string containing pixels being hovered on
        *args, **kwargs: ax, horizOn=True, vertOn=True, useblit=False, **lineprops
    """
    def __init__(self, x_index, y_index, xs, ys, format_coord, *args, **kwargs):
        super(SnapCursor, self).__init__(*args, **kwargs)
        self.y_index = y_index
        self.x_index = x_index
        self.xs = xs
        self.ys = ys
        self.format_coord = format_coord

    def onmove(self, event):
        """on mouse motion draw the cursor if visible"""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            self.linev.set_visible(False)
            self.lineh.set_visible(False)

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True
        if not self.visible:
            return

        # ----- this part is changed from the base class
        # self.linev.set_xdata((event.xdata, event.xdata))
        # self.lineh.set_ydata((event.ydata, event.ydata))

        index_str = self.format_coord(event.xdata, event.ydata)
        index_list = re.split(' |=', index_str)
        xdata = self.xs[int(index_list[self.x_index])]
        ydata = self.ys[int(index_list[self.y_index])]
        self.linev.set_xdata((xdata, xdata))
        self.lineh.set_ydata((ydata, ydata))
        # -----

        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_visible(self.visible and self.horizOn)
        self._update()


class MyNavigationToolbar(NavigationToolbar):
    """Optionally puts text of cursor coordinates somewhere else"""
    def set_message(self, s):
        self.message.emit(s)
        if self.coordinates:
            # puts coordinates in default place at top right of navigation toolbar
            self.locLabel.setText(s)
        else:
            # puts coordinates in matplotlibwidget.coordlabel instead
            self.parent.coordlabel.setText(s)


def close_plots(plots):
    for i, plot in enumerate(plots):
        try:
            plot.close()
            if isinstance(plot.parent(), QtWidgets.QDockWidget):
                plot.parent().close()
        except:
            plt.close(plot)
        plots.pop(i)

if __name__ == '__main__':
    win = MatplotlibWidget1D()
    win.plot([1,2],[3,4])
