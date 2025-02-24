# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:42:07 2015

@author: cdl
"""

from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import *

import sys
import functools
import copy
import time

from qtpy import QtWidgets
import numpy as np
import lmfit
from lmfit.models import Model


def powerexp(x, amplitude, exponent, growth):
    return amplitude * x ** exponent * np.exp(growth * x)


class PowerExpModel(Model):
    def __init__(self, *args, **kwargs):
        super(PowerExpModel, self).__init__(powerexp, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        try:
            expon, amp = np.polyfit(np.log(x+1.e-14), np.log(data+1.e-14), 1)
        except:
            expon, amp = 1, np.log(abs(max(data)+1.e-9))

        pars = self.make_params(amplitude=np.exp(amp), exponent=expon, growth=0)
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)


def fit_peaks(y, x, model_classes, prefixes, params_dict={}, params_start=None, mplwidget=None,
              full_x=None, logx=False, logy=False, fit_kws={}):
    """Fits x-y data with given models, optionally plots fit

    Args:
        model_classes: list of Models
        prefixes: list of prefix strings to initialize models
        params_dict={}: {'param': {param_dict} or value, ...} overwrites guesses
        params_start=None: overwrites guesses and params_dict
        mplwidget=None: mplwidget to plot fit into
        full_x=None: x vector to use for plotting fit instead of x
        logx=False, logy=False: x or y arguments are the log of the data being plotted, so plot 10**x or 10**y
        fit_kws: passed to models.fit(), see http://lmfit.github.io/lmfit-py/model.html#model.Model

    Returns:
        modelfit: lmfit.model.ModelFit instance
        models: lmfit.model.CompositeModel instance
    """
    models = []
    for model_class, prefix in zip(model_classes, prefixes):
        models.append(model_class(prefix=prefix))
    params = lmfit.Parameters()
    for model in models:
        params.update(model.guess(y, x=x))
    for param_key, param_value in params_dict.items():
        if isinstance(param_value, dict):
            params[param_key].set(**param_value)
        else:
            params[param_key].set(value=param_value)
    if params_start is not None:
        params.update(params_start)
    models = functools.reduce(lambda i, j: i + j, models)
    modelfit = models.fit(y, params, x=x, **fit_kws)
#    integ = np.trapz(y, x)
    label = ''.join(['fit_'] + prefixes)
    if mplwidget:
        if full_x is None:
            plotx = 10 ** x if logx else x
            ploty = 10 ** modelfit.best_fit if logy else modelfit.best_fit
        else:
            plotx = 10 ** full_x if logx else full_x
            ploty = 10 ** models.eval(x=full_x, params=modelfit.params) if logy else models.eval(x=full_x, params=modelfit.params)
        mplwidget.plot(plotx, ploty, keep_old=True, color='red', label=label)
    return modelfit, models


def fit_peaks_two_steps(y, x, y2, x2, model_classes, prefixes, model_classes2, prefixes2, params_dict={},
                        params_dict2={}, hold_first=True, params_start=None, params_start2=None, mplwidget=None,
                        full_x=None, logx=False, logy=False, fit_kws={}, fit_kws2={}):
    """Fits background region with given models, then fits peak region
    with first model parameters held constant and new model parameters varied, optionally plots fit"""
    modelfit, models = fit_peaks(y, x, model_classes, prefixes, params_dict, params_start, mplwidget, full_x, logx, logy, fit_kws)

    models2 = []
    for model_class, prefix in zip(model_classes2, prefixes2):
        models2.append(model_class(prefix=prefix))
    params2 = lmfit.Parameters()
    for model in models2:
        params2.update(model.guess(y2, x=x2))
    for param_key, param_value in params_dict2.items():
        if isinstance(param_value, dict):
            params2[param_key].set(**param_value)
        else:
            params2[param_key].set(value=param_value)
    if params_start2 is not None:
        params2.update(params_start2)

    params_copy = copy.deepcopy(modelfit.params)
    if hold_first:
        for param in params_copy.values():
            param.set(vary=False)
    params2.update(params_copy)
    models2 = functools.reduce(lambda i, j: i + j, [models] + models2)
    modelfit2 = models2.fit(y2, params2, x=x2, **fit_kws2)

    label = ''.join(['fit_'] + prefixes + prefixes2)
    if mplwidget:
        if full_x is None:
            plotx = 10 ** x2 if logx else x2
            ploty = 10 ** modelfit2.best_fit if logy else modelfit2.best_fit
        else:
            plotx = 10 ** full_x if logx else full_x
            ploty = 10 ** models2.eval(x=full_x, params=modelfit2.params) if logy else models2.eval(x=full_x, params=modelfit2.params)
        mplwidget.plot(plotx, ploty, keep_old=True, color='red', label=label)
    return modelfit, models, modelfit2, models2


def fit_datas(ys, xs, model_classes, prefixes, params_dict={}, params=None,
              feedback=True, window=None, wait=False, full_xs=None, logx=False, logy=False, fit_kws={}):
    """Fits multiple sets of x-y data individually with the same model

    Args:
        ys, xs: list of y vectors, x vectors
        feedback=True: set params of each set of data to best fit of previous set
        window=None: instance of CDSAXS_gui MainWindow in which to cycle through data, fit plots
        other args: see fit_peaks()
    """
    modelfit_list = []
    models_list = []
    for i, (y, x) in enumerate(zip(ys, xs)):
        full_x = full_xs[i] if full_xs is not None else None
        mplwidget = window.m_IvsQxz if window else None
        modelfit, models = fit_peaks(y, x, model_classes, prefixes, params_dict, params, mplwidget, full_x, logx, logy, fit_kws)
        if feedback:
            params = modelfit.params
        modelfit_list.append(modelfit)
        models_list.append(models)
        if window:
            QtWidgets.QApplication.processEvents()
            if wait:
                while 1:
                    QtWidgets.QApplication.processEvents()
                    if not window.m_IvsQxz.checkBox_keeplimits.isChecked():
                        window.m_IvsQxz.checkBox_keeplimits.setChecked(True)
                        break
            if i < (len(ys) - 1):
                window.treeWidget_QxzQy.setCurrentItem(window.treeWidget_QxzQy.itemBelow(window.treeWidget_QxzQy.currentItem()))
    return modelfit_list, models_list


def fit_datas_two_steps(ys, xs, ys2, xs2, model_classes, prefixes, model_classes2, prefixes2,
                        params_dict={}, params_dict2={}, params=None, params2=None,
                        hold_first=True, feedback=True, window=None, wait=False, full_xs=None,
                        logx=False, logy=False, fit_kws={}, fit_kws2={}):
    """Fits multiple sets of x-y data individually with the same model"""
    modelfit_list = []
    models_list = []
    modelfit2_list = []
    models2_list = []
    for i, (y, x, y2, x2) in enumerate(zip(ys, xs, ys2, xs2)):
        full_x = full_xs[i] if full_xs is not None else None
        mplwidget = window.m_IvsQxz if window else None
        modelfit, models, modelfit2, models2 = \
            fit_peaks_two_steps(y, x, y2, x2, model_classes, prefixes, model_classes2, prefixes2, params_dict, params_dict2,
                                hold_first, params, params2, mplwidget, full_x, logx, logy, fit_kws, fit_kws2)
        if feedback:
            params = modelfit.params
            params2 = modelfit2.params
        modelfit_list.append(modelfit)
        models_list.append(models)
        modelfit2_list.append(modelfit2)
        models2_list.append(models2)
        if window:
            QtWidgets.QApplication.processEvents()
            if wait:
                while 1:
                    QtWidgets.QApplication.processEvents()
                    if not window.m_IvsQxz.checkBox_keeplimits.isChecked():
                        window.m_IvsQxz.checkBox_keeplimits.setChecked(True)
                        break
            if i < (len(ys) - 1):
                window.treeWidget_QxzQy.setCurrentItem(window.treeWidget_QxzQy.itemBelow(window.treeWidget_QxzQy.currentItem()))
    return modelfit_list, models_list, modelfit2_list, models2_list

if __name__ == '__main__':
    from lmfit.models import LinearModel, ExponentialModel, GaussianModel, PowerLawModel, ConstantModel, Model
    from CDSAXS_gui import matplotlibwidget

    x = np.arange(-50,50)
    y = np.random.rand(100) + 3 * np.exp(-(x - 10) ** 2 / 10) + 2 * np.exp(-(x - 20) ** 2 / 30)
    w = matplotlibwidget.MatplotlibWidget1D()
    w.plot(x, y)
    guess = {'g1_center': 10, 'g2_center': 20,}
    modelfit, models = fit_peaks(y, x, [GaussianModel, GaussianModel, ConstantModel], ['g1_', 'g2_', 'const_'], guess, mplwidget=w)
    print(modelfit.fit_report())