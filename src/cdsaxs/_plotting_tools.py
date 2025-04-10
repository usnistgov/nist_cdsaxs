"""
Helpful plotting tools for cdsaxs.
"""

import numpy as np


def create_even_q_ticks(q, num=6, includes_zero=True):
    """
    Produces evenly spaced indices and corresponding q values
    for scattering image axes.

    The scattering vector array, q, must be sorted. It doesn't have to
    be increasing or decreasing as long as it's in order.

    If you want to include q=0 as one of the enforced tick marks,
    includes_zero should be set to True.

    TODO: figure out what happens if includes_zero=True and 0 is not in q
    """
    if not includes_zero:
        ticks_index = np.arange(0, len(q))
        ticks_index = np.linspace(ticks_index, num=num)
        ticks_q = q[ticks_index]
        return ticks_index, ticks_q

    else:
        spacing_exp = np.ceil(np.log10((np.nanmax(q)-np.nanmin(q))/(num-1)))
        spacing = 10**spacing_exp

        start = np.ceil(np.nanmin(q)/spacing)*spacing
        stop = np.floor(np.nanmax(q)/spacing)*spacing

        ticks_q = np.round(np.arange(start, stop+spacing, spacing),
                           int(np.abs(min(0, spacing_exp))))
        while len(ticks_q) < (num-1):
            spacing /= 2
            spacing_exp = np.floor(np.log10(spacing))

            start = np.ceil(np.nanmin(q)/spacing)*spacing
            stop = np.floor(np.nanmax(q)/spacing)*spacing
            ticks_q = np.round(np.arange(start, stop+spacing, spacing),
                               int(np.abs(min(0, spacing_exp))))

        ticks_interp = []
        ticks_index = np.arange(0, len(q))
        for val in ticks_q:
            if q[0] > q[-1]:
                ticks_interp.append(
                    np.interp(val, np.flip(q), np.flip(ticks_index)))
            else:
                ticks_interp.append(np.interp(val, q, ticks_index))

        return ticks_interp, ticks_q


def generate_axis_label_units(q_axis):
    """
    Generate formatted axis label with units based on the axis string.
    This only does anything with q axes currently; everythign else it
    just returns back to you.

    """

    if q_axis[0] == 'q':
    
        units = r" $(\AA^{-1})$"

        subscript = r"$_{" + q_axis[1]
        if len(q_axis) > 2:
            for var in q_axis[2:]:
                subscript += f",{var}"
        subscript += r"}$"

        label = r"q" + subscript

        return label + units
    
    else:
        return q_axis

