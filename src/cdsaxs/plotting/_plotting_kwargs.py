import inspect
import matplotlib
import matplotlib.pyplot as plt

"""
Keyword arguments for matplotlib.pyplot.errorbar.
"""
ERRORBAR_KWARGS = [
    param.name for param in inspect.signature(plt.errorbar).parameters.values()
    if param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY)
    and param.default is not inspect.Parameter.empty
]
# kwargs go to matplotlib.axes.Axes.plot
ERRORBAR_KWARGS.extend([
    param.name for param in inspect.signature(
        matplotlib.axes.Axes.plot).parameters.values()
    if param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY)
    and param.default is not inspect.Parameter.empty
])
# kwargs for matplotlib.axes.Axes.plot go to matplotlib.lines.Line2D properties
ERRORBAR_KWARGS.extend(
    [x[4:]if x[:4] == 'set_' else x[5:]
     for x, y in matplotlib.lines.Line2D.__dict__.items() if 'set_' in x]
)
ERRORBAR_KWARGS = list(set(ERRORBAR_KWARGS))

"""
Keyword arguments for matplotlib.pyplot.imshow.
"""
IMSHOW_KWARGS = [
    param.name for param in inspect.signature(plt.imshow).parameters.values()
    if param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY)
    and param.default is not inspect.Parameter.empty
]
# kwargs go to matplotlib.axes.Axes.plot
IMSHOW_KWARGS.extend([
    param.name for param in inspect.signature(
        matplotlib.image.AxesImage).parameters.values()
    if param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY)
    and param.default is not inspect.Parameter.empty
])
# kwargs for matplotlib.axes.Axes.plot go to matplotlib.lines.Line2D properties
IMSHOW_KWARGS.extend(
    [x[4:]if x[:4] == 'set_' else x[5:]
     for x, y in matplotlib.artist.Artist.__dict__.items() if 'set_' in x]
)
IMSHOW_KWARGS = list(set(IMSHOW_KWARGS))

"""
Keyword arguments for matplotlib.pyplot.scatter.
"""
SCATTER_KWARGS = [
    param.name for param in inspect.signature(plt.scatter).parameters.values()
    if param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY)
    and param.default is not inspect.Parameter.empty
]
# kwargs go to matplotlib.axes.Axes.plot
SCATTER_KWARGS.extend([
    param.name for param in inspect.signature(
        matplotlib.collections.PathCollection).parameters.values()
    if param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY)
    and param.default is not inspect.Parameter.empty
])
# kwargs for matplotlib.axes.Axes.plot go to matplotlib.lines.Line2D properties
SCATTER_KWARGS.extend(
    [x[4:]if x[:4] == 'set_' else x[5:]
     for x, y in matplotlib.collections.Collection.__dict__.items() if 'set_' in x]
)

SCATTER_KWARGS = list(set(SCATTER_KWARGS))
