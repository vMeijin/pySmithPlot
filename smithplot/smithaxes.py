# -*- coding: utf-8 -*-
# last edit: 10.06.2025
'''
Library for plotting fully automatic a Smith Chart with various customizable
parameters and well selected default values. It also provides the following 
modifications and features:

    - circle shaped drawing area with labels placed around 
    - :meth:`plot` accepts single real and complex numbers and numpy.ndarray's
    - plotted lines can be interpolated
    - start/end markers of lines can be modified and rotate tangential
    - gridlines are 3-point arcs to improve space efficiency of exported plots
    - 'fancy' option for adaptive grid generation
    - own tick locators for nice axis labels

For making a Smith Chart plot it is sufficient to import :mod:`smithplot` and
create a new subplot with projection set to 'smith'. Parameters can be set 
either with keyword arguments or :meth:`update_Params`.

Example:

    # creating a new plot and modify parameters afterwards
    import smithplot
    from smithplot import SmithAxes
    from matplotlib import pyplot as pp
    ax = pp.subplot('111', projection='smith')
    SmithAxes.update_scParams(ax, reset=True, grid_major_enable=False)
    ## or in short form direct
    #ax = pp.subplot('111', projection='smith', grid_major_enable=False)
    pp.plot([25, 50 + 50j, 100 - 50j], datatype=SmithAxes.Z_PARAMETER)
    pp.show()
    
Note: Supplying parameters to :meth:`subplot` may not always work as
expected, because subplot uses an index for the axes with a key created
of all given parameters. This does not work always, especially if the
parameters are array-like types (e.g. numpy.ndarray).
'''
# changed by FG for python 3.10 
# matplotlib==3.10.3
# numpy==2.2.6

try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
    
from numbers import Number
from types import MethodType, FunctionType

import matplotlib as mp
import numpy as np
from matplotlib.axes import Axes
from matplotlib.axis import XAxis
from matplotlib.cbook import simple_linear_interpolation as linear_interpolation
from matplotlib.legend_handler import HandlerLine2DCompound
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle, Arc
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.ticker import Formatter, AutoMinorLocator, Locator
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
from scipy.interpolate import fitpack

from . import smithhelper
from .smithhelper import EPSILON, TWO_PI, ang_to_c, z_to_xy


class SmithAxes(Axes):
    '''
    The :class:`SmithAxes` provides an implementation of :class:`matplotlib.axes.Axes`
    for drawing a full automatic Smith Chart it also provides own implementations for
     
        - :class:`matplotlib.transforms.Transform`
            -> :class:`MoebiusTransform`
            -> :class:`InvertedMoebiusTransform`
            -> :class:`PolarTranslate`
        - :class:`matplotlib.ticker.Locator`
            -> :class:`RealMaxNLocator`
            -> :class:`ImagMaxNLocator`
            -> :class:`SmithAutoMinorLocator`
        - :class:`matplotlib.ticker.Formatter`
            -> :class:`RealFormatter`
            -> :class:`ImagFormatter`
    '''

    name = 'smith'

    # data types
    S_PARAMETER = "S"
    Z_PARAMETER = "Z"
    Y_PARAMETER = "Y"
    _datatypes = [S_PARAMETER, Z_PARAMETER, Y_PARAMETER]

    # constants used for indicating values near infinity, which are all transformed into one point
    _inf = smithhelper.INF
    _near_inf = 0.9 * smithhelper.INF
    _ax_lim_x = 2 * _inf  # prevents missing labels in special cases
    _ax_lim_y = 2 * _inf  # prevents missing labels in special cases

    # default parameter, see update_scParams for description
    scDefaultParams = {"plot.zorder": 4,
                       "plot.marker.hack": True,
                       "plot.marker.rotate": True,
                       "plot.marker.start": "s",
                       "plot.marker.default": "o",
                       "plot.marker.end": "^",
                       "plot.default.interpolation": 5,
                       "plot.default.datatype": S_PARAMETER,
                       "grid.zorder": 1,
                       "grid.locator.precision": 2,
                       "grid.major.enable": True,
                       "grid.major.linestyle": '-',
                       "grid.major.linewidth": 1,
                       "grid.major.color": "0.2",
                       "grid.major.xmaxn": 10,
                       "grid.major.ymaxn": 16,
                       "grid.major.fancy": True,
                       "grid.major.fancy.threshold": (100, 50),
                       "grid.minor.enable": True,
                       "grid.minor.capstyle": "round",
                       "grid.minor.dashes": [0.2, 2],
                       "grid.minor.linewidth": 0.75,
                       "grid.minor.color": "0.4",
                       "grid.minor.xauto": 4,
                       "grid.minor.yauto": 4,
                       "grid.minor.fancy": True,
                       "grid.minor.fancy.dividers": [0, 1, 2, 3, 5, 10, 20],
                       "grid.minor.fancy.threshold": 35,
                       "axes.xlabel.rotation": 90,
                       "axes.xlabel.fancybox": {"boxstyle": "round,pad=0.2,rounding_size=0.2",
                                                "facecolor": 'w',
                                                "edgecolor": "w",
                                                "mutation_aspect": 0.75,
                                                "alpha": 1},
                       "axes.ylabel.correction": (-1, 0, 0),
                       "axes.radius": 0.44,
                       "axes.impedance": 50,
                       "axes.normalize": True,
                       "axes.normalize.label": True,
                       "symbol.infinity": "∞ ",  # BUG: symbol gets cut off without end-space
                       "symbol.infinity.correction": 8,
                       "symbol.ohm": "Ω"}

    @staticmethod
    def update_scParams(sc_dict=None, instance=None, filter_dict=False, reset=True, **kwargs):
        '''
        Method for updating the parameters of a SmithAxes instance. If no instance
        is given, the changes are global, but affect only instances created
        afterwards. Parameter can be passed as dictionary or keyword arguments.
        If passed as keyword, the seperator '.' must be  replaced with '_'.

        Note: Parameter changes are not always immediate (e.g. changes to the
        grid). It is not recommended to modify parameter after adding anything to
        the plot. For a reset call :meth:`cla`.

        Example:
            update_scParams({grid.major: True})
            update_scParams(grid_major=True)

        Valid parameters with default values and description:

            plot.zorder: 5
                Zorder of plotted lines.
                Accepts: integer

            plot.marker.hack: True
                Enables the replacement of start and endmarkers.
                Accepts: boolean
                Note: Uses ugly code injection and may causes unexpected behavior.

            plot.marker.rotate: True
                Rotates the endmarker in the direction of its line.
                Accepts: boolean
                Note: needs plot.marker.hack=True

            plot.marker.start: 's',
                Marker for the first point of a line, if it has more than 1 point.
                Accepts: None or see matplotlib.markers.MarkerStyle()
                Note: needs plot.marker.hack=True

            plot.marker.default: 'o'
                Marker used for linepoints.
                Accepts: None or see matplotlib.markers.MarkerStyle()

            plot.marker.end: '^',
                Marker for the last point of a line, if it has more than 1 point.
                Accepts: None or see matplotlib.markers.MarkerStyle()
                Note: needs plot.marker.hack=True

            plot.default.interpolation: 5
                Default number of interpolated steps between two points of a
                line, if interpolation is used.
                Accepts: integer

            plot.default.datatype: SmithAxes.S_PARAMETER
                Default datatype for plots.
                Accepts: SmithAxes.[S_PARAMETER,Z_PARAMETER,Y_PARAMETER]

            grid.zorder : 1
                Zorder of the gridlines.
                Accepts: integer
                Note: may not work as expected

            grid.locator.precision: 2
                Sets the number of significant decimals per decade for the
                Real and Imag MaxNLocators. Example with precision 2:
                    1.12 -> 1.1, 22.5 -> 22, 135 -> 130, ...
                Accepts: integer
                Note: value is an orientation, several exceptions are implemented

            grid.major.enable: True
                Enables the major grid.
                Accepts: boolean

            grid.major.linestyle: 'solid'
                Major gridline style.
                Accepts: see matplotlib.patches.Patch.set_linestyle()

            grid.major.linewidth: 1
                Major gridline width.
                Accepts: float

            grid.major.color: '0.2'
                Major gridline color.
                Accepts: matplotlib color

            grid.major.xmaxn: 10
                Maximum number of spacing steps for the real axis.
                Accepts: integer

            grid.major.ymaxn: 16
                Maximum number of spacing steps for the imaginary axis.
                Accepts: integer

            grid.major.fancy: True
                Draws a fancy major grid instead of the standard one.
                Accepts: boolean

            grid.major.fancy.threshold: (100, 50)
                Minimum distance times 1000 between two gridlines relative to
                total plot size 2x2. Either tuple for individual real and
                imaginary distances or single value for both.
                Accepts: (float, float) or float

            grid.minor.enable: True
                Enables the minor grid.
                Accepts: boolean

            grid.minor.capstyle: 'round'
                Minor dashes capstyle
                Accepts: 'round', 'butt', 'miter', 'projecting'

            grid.minor.dashes: (0.2, 2)
                Minor gridline dash style.
                Accepts: tuple

            grid.minor.linewidth: 0.75
                Minor gridline width.
                Accepts: float

            grid.minor.color: 0.4
                Minor gridline color.
                Accepts: matplotlib color

            grid.minor.xauto: 4
                Maximum number of spacing steps for the real axis.
                Accepts: integer

            grid.minor.yauto: 4
                Maximum number of spacing steps for the imaginary axis.
                Accepts: integer

            grid.minor.fancy: True
                Draws a fancy minor grid instead the standard one.
                Accepts: boolean

            grid.minor.fancy.dividers: [1, 2, 3, 5, 10, 20]
                Divisions for the fancy minor grid, which are selected by
                comparing the distance of gridlines with the threshold value.
                Accepts: list of integers

            grid.minor.fancy.threshold: 25
                Minimum distance for using the next bigger divider. Value times
                1000 relative to total plot size 2.
                Accepts: float

            axes.xlabel.rotation: 90
               Rotation of the real axis labels in degree.
               Accepts: float

            axes.xlabel.fancybox: {"boxstyle": "round4,pad=0.3,rounding_size=0.2",
                                               "facecolor": 'w',
                                               "edgecolor": "w",
                                               "mutation_aspect": 0.75,
                                               "alpha": 1},
                FancyBboxPatch parameters for the x-label background box.
                Accepts: dictionary with rectprops

            axes.ylabel.correction: (-1, 0, 0)
                Correction in x, y, and radial direction for the labels of the imaginary axis.
                Usually needs to be adapted when fontsize changes 'font.size'.
                Accepts: (float, float, float)

            axes.radius: 0.44
                Radius of the plotting area. Usually needs to be adapted to
                the size of the figure.
                Accepts: float

            axes.impedance: 50
                Defines the reference impedance for normalisation.
                Accepts: float

            axes.normalize: True
                If True, the Smith Chart is normalized to the reference impedance.
                Accepts: boolean

            axes.normalize.label: True
                If 'axes.normalize' and True, a textbox with 'Z_0: ... Ohm' is put in
                the lower left corner.
                Accepts: boolean

            symbol.infinity: "∞ "
                Symbol string for infinity.
                Accepts: string

                Note: Without the trailing space the label might get cut off.

            symbol.infinity.correction: 8
                Correction of size for the infinity symbol, because normal symbol
                seems smaller than other letters.
                Accepts: float

            symbol.ohm "Ω"
                Symbol string for the resistance unit (usually a large Omega).
                Accepts: string

        Note: The keywords are processed after the dictionary and override
        possible double entries.
        '''
        scParams = SmithAxes.scDefaultParams if instance is None else instance.scParams

        if sc_dict is not None:
            for key, value in sc_dict.items():
                if key in scParams:
                    scParams[key] = value
                else:
                    raise KeyError("key '%s' is not in scParams" % key)

        remaining = kwargs.copy()
        for key in kwargs:
            key_dot = key.replace("_", ".")
            if key_dot in scParams:
                scParams[key_dot] = remaining.pop(key)

        if not filter_dict and len(remaining) > 0:
            raise KeyError("Following keys are invalid SmithAxes parameters: '%s'" % ",".join(remaining.keys()))

        if reset and instance is not None:
            instance.cla()

        if filter_dict:
            return remaining

    def __init__(self, *args, **kwargs):
        '''
        Builds a new :class:`SmithAxes` instance. For futher details see:
        
            :meth:`update_scParams`
            :class:`matplotlib.axes.Axes`
        '''
        # define new class attributes
        self._majorarcs = None
        self._minorarcs = None
        self._impedance = None
        self._normalize = None
        self._current_zorder = None
        self.scParams = self.scDefaultParams.copy()

        # seperate Axes parameter
        Axes.__init__(self, *args, **SmithAxes.update_scParams(instance=self, filter_dict=True, reset=False, **kwargs))
        self.set_aspect(1, adjustable='box', anchor='C')

        # remove all ticks
        self.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)

    def _get_key(self, key):
        '''
        Get a key from the local parameter dictionary or from global 
        matplotlib rcParams.
        
        Keyword arguments:
            
            *key*:
                Key to get from scParams or matplotlib.rcParams
                Accepts: string
            
        Returns:
        
            *value*:
                Value got from scParams or rcParams with key
        '''
        if key in self.scParams:
            return self.scParams[key]
        elif key in mp.rcParams:
            return mp.rcParams[key]
        else:
            raise KeyError("%s is not a valid key" % key)

    def _init_axis(self):
        self.xaxis = mp.axis.XAxis(self)
        self.yaxis = mp.axis.YAxis(self)
        self._update_transScale()

    def cla(self):
        self._majorarcs = []
        self._minorarcs = []

        # deactivate grid function when calling base class
        tgrid = self.grid

        def dummy(*args, **kwargs):
            pass

        self.grid = dummy
        # Don't forget to call the base class
        Axes.cla(self)
        self.grid = tgrid

        self._normbox = None
        self._impedance = self._get_key("axes.impedance")
        self._normalize = self._get_key("axes.normalize")
        self._current_zorder = self._get_key("plot.zorder")

        self.xaxis.set_major_locator(self.RealMaxNLocator(self, self._get_key("grid.major.xmaxn")))
        self.yaxis.set_major_locator(self.ImagMaxNLocator(self, self._get_key("grid.major.ymaxn")))

        self.xaxis.set_minor_locator(self.SmithAutoMinorLocator(self._get_key("grid.minor.xauto")))
        self.yaxis.set_minor_locator(self.SmithAutoMinorLocator(self._get_key("grid.minor.yauto")))

        self.xaxis.set_ticks_position('none')
        self.yaxis.set_ticks_position('none')

        Axes.set_xlim(self, 0, self._ax_lim_x)
        Axes.set_ylim(self, -self._ax_lim_y, self._ax_lim_y)

        for label in self.get_xticklabels():
            label.set_verticalalignment("center")
            label.set_horizontalalignment('center')
            label.set_rotation_mode("anchor")
            label.set_rotation(self._get_key("axes.xlabel.rotation"))
            label.set_bbox(self._get_key("axes.xlabel.fancybox"))
            self.add_artist(label)  # if not readded, labels are drawn behind grid

        for tick, loc in zip(self.yaxis.get_major_ticks(),
                             self.yaxis.get_majorticklocs()):
            # workaround for fixing to small infinity symbol
            if abs(loc) > self._near_inf:
                tick.label1.set_fontsize(tick.label1.get_fontsize() +
                                    self._get_key("symbol.infinity.correction"))

            tick.label1.set_verticalalignment('center')

            x = np.real(self._moebius_z(loc * 1j))
            if x < -0.1:
                tick.label1.set_horizontalalignment('right')
            elif x > 0.1:
                tick.label1.set_horizontalalignment('left')
            else:
                tick.label1.set_horizontalalignment('center')

        self.yaxis.set_major_formatter(self.ImagFormatter(self))
        self.xaxis.set_major_formatter(self.RealFormatter(self))

        if self._get_key("axes.normalize") and self._get_key("axes.normalize.label"):
            x, y = z_to_xy(self._moebius_inv_z(-1 - 1j))
            box = self.text(x, y, "Z$_\mathrm{0}$ = %d$\,$%s" % (self._impedance, self._get_key("symbol.ohm")), ha="left", va="bottom")

            px = self._get_key("ytick.major.pad")
            py = px + 0.5 * box.get_size()
            box.set_transform(self._yaxis_correction + Affine2D().translate(-px, -py))

        for grid in ['major', "minor"]:
            self.grid(b=self._get_key("grid.%s.enable" % grid), which=grid)

    def _set_lim_and_transforms(self):
        r = self._get_key("axes.radius")
        self.transProjection = self.MoebiusTransform(self)  # data space  -> moebius space
        self.transAffine = Affine2D().scale(r, r).translate(0.5, 0.5)  # moebius space -> axes space
        self.transDataToAxes = self.transProjection + self.transAffine
        self.transAxes = BboxTransformTo(self.bbox)  # axes space -> drawing space
        self.transMoebius = self.transAffine + self.transAxes
        self.transData = self.transProjection + self.transMoebius

        self._xaxis_pretransform = Affine2D().scale(1, 2 * self._ax_lim_y).translate(0, -self._ax_lim_y)
        self._xaxis_transform = self._xaxis_pretransform + self.transData
        self._xaxis_text1_transform = Affine2D().scale(1.0, 0.0) + self.transData

        self._yaxis_stretch = Affine2D().scale(self._ax_lim_x, 1.0)
        self._yaxis_correction = self.transData + Affine2D().translate(*self._get_key("axes.ylabel.correction")[:2])
        self._yaxis_transform = self._yaxis_stretch + self.transData
        self._yaxis_text1_transform = self._yaxis_stretch + self._yaxis_correction

    def get_xaxis_transform(self, which='grid'):
        assert which in ['tick1', 'tick2', 'grid']
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pixelPad):
        return self._xaxis_text1_transform, 'center', 'center'

    def get_yaxis_transform(self, which='grid'):
        assert which in ['tick1', 'tick2', 'grid']
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pixelPad):
        if hasattr(self, 'yaxis') and len(self.yaxis.majorTicks) > 0:
            #font_size = self.yaxis.majorTicks[0].label.get_size()
            font_size = self.yaxis.majorTicks[0].label1.get_fontsize()
        else:
            font_size = self._get_key("font.size")

        offset = self._get_key("axes.ylabel.correction")[2]
        return self._yaxis_text1_transform + self.PolarTranslate(self, pad=pixelPad + offset, font_size=font_size), 'center', 'center'

    def _gen_axes_patch(self):
        return Circle((0.5, 0.5), self._get_key("axes.radius") + 0.015)

    def _gen_axes_spines(self, locations=None, offset=0.0, units='inches'):
        return {SmithAxes.name: Spine.circular_spine(self, (0.5, 0.5), self._get_key("axes.radius"))}

    def set_xscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError()
        Axes.set_xscale(self, *args, **kwargs)

    def set_yscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError()
        Axes.set_yscale(self, *args, **kwargs)

    def set_xlim(self, *args, **kwargs):
        '''xlim is immutable and always set to (0, infinity)'''
        Axes.set_xlim(self, 0, self._ax_lim_x)

    def set_ylim(self, *args, **kwargs):
        '''ylim is immutable and always set to (-infinity, infinity)'''
        Axes.set_ylim(self, -self._ax_lim_y, self._ax_lim_y)

    def format_coord(self, re, im):
        sgn = "+" if im > 0 else "-"
        return "%.5f %s %.5fj" % (re, sgn, abs(im)) if re > 0 else ""

    def get_data_ratio(self):
        return 1.0

    # disable panning and zoom in matplotlib figure viewer
    def can_zoom(self):
        return False

    def start_pan(self, x, y, button):
        pass

    def end_pan(self):
        pass

    def drag_pan(self, button, key, x, y):
        pass

    def _moebius_z(self, *args, normalize=None):
        '''
        Basic transformation. 
        
        Arguments:

            *z*: 
                Complex number or numpy.ndarray with dtype=complex

            *x, y*:
                Float numbers or numpy.ndarray's with dtype not complex

            *normalize*:
                If True, the values are normalized to self._impedance.
                If None, self._normalize determines behaviour.
                Accepts: boolean or None
                
        Returns:

            *w*:
                Performs w = (z - k) / (z + k) with k = 'axes.scale' 
                Type: Complex number or numpy.ndarray with dtype=complex
        '''
        normalize = self._normalize if normalize is None else normalize
        norm = 1 if normalize else self._impedance
        return smithhelper.moebius_z(*args, norm=norm)

    def _moebius_inv_z(self, *args, normalize=None):
        '''
        Basic inverse transformation. 
        
        Arguments:

            *z*: 
                Complex number or numpy.ndarray with dtype=complex

            *x, y*:
                Float numbers or numpy.ndarray's with dtype not complex

            *normalize*:
                If True, the values are normalized to self._impedance.
                If None, self._normalize determines behaviour.
                Accepts: boolean or None

        Returns:

            *w*:
                Performs w = k * (1 - z) / (1 + z) with k = 'axes.scale' 
                Type: Complex number or numpy.ndarray with dtype=complex
        '''
        normalize = self._normalize if normalize is None else normalize
        norm = 1 if normalize else self._impedance
        return smithhelper.moebius_inv_z(*args, norm=norm)

    def real_interp1d(self, x, steps):
        '''
        Interpolates the given vector as real numbers in the way, that they 
        are evenly spaced after a transformation with imaginary part 0.
        
        Keyword Arguments
        
            *x*:
                Real values to interpolate.
                Accepts: 1D iterable (e.g. list or numpy.ndarray)
                
            *steps*:
                Number of steps between two points.
                Accepts: integer
        '''
        return self._moebius_inv_z(linear_interpolation(self._moebius_z(np.array(x)), steps))

    def imag_interp1d(self, y, steps):
        '''
        Interpolates the given vector as imaginary numbers in the way, that 
        they are evenly spaced after a transformation with real part 0.
        
        Keyword Arguments
        
            *y*:
                Imaginary values to interpolate.
                Accepts: 1D iterable (e.g. list or numpy.ndarray)

            *steps*:
                Number of steps between two points.
                Accepts: integer
        '''
        angs = np.angle(self._moebius_z(np.array(y) * 1j)) % TWO_PI
        i_angs = linear_interpolation(angs, steps)
        return np.imag(self._moebius_inv_z(ang_to_c(i_angs)))

    def legend(self, *args, **kwargs):
        this_axes = self

        class SmithHandlerLine2D(HandlerLine2DCompound):
            def create_artists(self, legend, orig_handle,
                               xdescent, ydescent, width, height, fontsize,
                               trans):
                legline, legline_marker = HandlerLine2DCompound.create_artists(self, legend, orig_handle, xdescent, ydescent,
                                                                       width, height, fontsize, trans)

                if hasattr(orig_handle, "_markerhacked"):
                    this_axes._hack_linedraw(legline_marker, True)
                return legline, legline_marker

        return Axes.legend(self, *args, handler_map={Line2D: SmithHandlerLine2D()}, **kwargs)

    def plot(self, *args, **kwargs):
        '''
        Plot the given data into the Smith Chart. Behavior similar to basic 
        :meth:`matplotlib.axes.Axes.plot`, but with some extensions:
        
            - Additional support for real and complex data. Complex values must be
            either of type 'complex' or a numpy.ndarray with dtype=complex.
            - If 'zorder' is not provided, the current default value is used.
            - If 'marker' is not providet, the default value is used.
            - Extra keywords are added.
        
        Extra keyword arguments:
        
            *datatype*:
                Specifies the input data format. Must be either 'S', 'Z' or 'Y'.
                Accepts: SmithAxes.[S_PARAMETER,Z_PARAMETER,Y_PARAMETER]
                Default: 'plot.default.datatype'
                
            *markerhack*:
                If set, activates the manipulation of start and end markern 
                of the created line.
                Accepts: boolean
                Default: 'plot.marker.hack'
                
            *rotate_marker*:
                If *markerhack* is active, rotates the endmarker in direction
                of the corresponding path.
                Accepts: boolean
                Default: 'plot.rotatemarker'

            *interpolate*:
                If 'value' >0 the given data is interpolated linearly by 'value'
                steps in SmithAxes cooardinate space. 'markevery', if specified,
                will be modified accordingly. If 'True' the 'plot.default_intperpolation'
                value is used.
                Accepts: boolean or integer
                Default: False

            *equipoints*:
                If 'value' >0 the given data is interpolated linearly by equidistant
                steps in SmithAxes cooardinate space. Cannot be used with 'interpolate'
                enabled.
                Accepts: boolean
                Default: False


                
        See :meth:`matplotlib.axes.Axes.plot` for mor details
        '''
        # split input into real and imaginary part if complex
        new_args = ()
        for arg in args:
            # check if argument is a string or already an ndarray
            # if not, try to convert to an ndarray
            if not (isinstance(arg, str) or isinstance(arg, np.ndarray)):
                try:
                    if isinstance(arg, Iterable):
                        arg = np.array(arg)
                    elif isinstance(arg, Number):
                        arg = np.array([arg])
                except TypeError:
                    pass

            # if (converted) arg is an ndarray of complex type, split it
            if isinstance(arg, np.ndarray) and arg.dtype in [complex, np.complex128]:
                new_args += z_to_xy(arg)
            else:
                new_args += (arg,)

        # ensure newer plots are above older ones
        if 'zorder' not in kwargs:
            kwargs['zorder'] = self._current_zorder
            self._current_zorder += 0.001

        # extract or load non-matplotlib keyword arguments from parameters
        kwargs.setdefault("marker", self._get_key("plot.marker.default"))
        interpolate = kwargs.pop("interpolate", False)
        equipoints = kwargs.pop("equipoints", False)
        datatype = kwargs.pop("datatype", self._get_key("plot.default.datatype"))
        markerhack = kwargs.pop("markerhack", self._get_key("plot.marker.hack"))
        rotate_marker = kwargs.pop("rotate_marker", self._get_key("plot.marker.rotate"))

        if datatype not in self._datatypes:
            raise ValueError("'datatype' must be either '%s'" % ",".join(self._datatypes))

        if interpolate is not False:
            if equipoints > 0:
                raise ValueError("Interpolation is not available with equidistant markers")

            if interpolate is True:
                interpolate = self._get_key("plot.default.interpolation")
            elif interpolate < 0:
                raise ValueError("Interpolation is only for positive values possible!")

            if "markevery" in kwargs:
                mark = kwargs["markevery"]
                if isinstance(mark, Iterable):
                    mark = np.asarray(mark) * (interpolate + 1)
                else:
                    mark *= interpolate + 1
                kwargs["markevery"] = mark

        lines = Axes.plot(self, *new_args, **kwargs)
        for line in lines:
            cdata = smithhelper.xy_to_z(line.get_data())

            if datatype == SmithAxes.S_PARAMETER:
                z = self._moebius_inv_z(cdata)
            elif datatype == SmithAxes.Y_PARAMETER:
                z = 1 / cdata
            elif datatype == SmithAxes.Z_PARAMETER:
                z = cdata
            else:
                raise ValueError("'datatype' must be '%s', '%s' or '%s'" % (SmithAxes.S_PARAMETER, SmithAxes.Z_PARAMETER, SmithAxes.Y_PARAMETER))

            if self._normalize and datatype != SmithAxes.S_PARAMETER:
                z /= self._impedance

            line.set_data(z_to_xy(z))

            if interpolate or equipoints:
                z = self._moebius_z(*line.get_data())
                if len(z) > 1:
                    spline, t0 = fitpack.splprep(z_to_xy(z), s=0)
                    ilen = (interpolate + 1) * (len(t0) - 1) + 1
                    if equipoints == 1:
                        t = np.linspace(0, 1, ilen)
                    elif equipoints > 1:
                        t = np.linspace(0, 1, equipoints)
                    else:
                        t = np.zeros(ilen)
                        t[0], t[1:] = t0[0], np.concatenate([np.linspace(i0, i1, interpolate + 2)[1:] for i0, i1 in zip(t0[:-1], t0[1:])])

                    z = self._moebius_inv_z(*fitpack.splev(t, spline))
                    line.set_data(z_to_xy(z))

            if markerhack:
                self._hack_linedraw(line, rotate_marker)

        return lines

    def grid(self,
             b=None,
             which='major',
             fancy=None,
             dividers=None,
             threshold=None,
             **kwargs):
        '''
        Complete rewritten grid function. Gridlines are replaced with Arcs, 
        which reduces the amount of points to store and increases speed. The
        grid consist of a minor and major part, which can be drawn either as 
        standard with lines from axis to axis, or fancy with dynamic spacing
        and length adaption. 
        
        Keyword arguments:
        
            *b*:
                Enables or disables the selected grid.
                Accepts: boolean
                
            *which*:
                The grid to be drawn.
                Accepts: ['major', 'minor', 'both']
                
            *axis*:
                The axis to be drawn. x=real and y=imaginary
                Accepts: ['x', 'y', 'both']
                Note: if fancy is set, only 'both' is valid
                
            *fancy*:
                If set to 'True', draws the grid on the fancy way.
                Accepts: boolean

            *dividers*:
                Adaptive divisions for the minor fancy grid.
                Accepts: array with integers
                Note: has no effect on major and non-fancy grid
                
            *threshold*:
                Threshold for dynamic adaption of spacing and line length. Can
                be specified for both axis together or each seperatly.
                Accepts: float or (float, float)
                
            **kwargs*:
                Keyword arguments passed to the gridline creator.
                Note: Gridlines are :class:`matplotlib.patches.Patch` and does 
                not accept all arguments :class:`matplotlib.lines.Line2D` 
                accepts.
        '''
        assert which in ["both", "major", "minor"]
        assert fancy in [None, False, True]

        def get_kwargs(grid):
            kw = kwargs.copy()
            kw.setdefault('zorder', self._get_key("grid.zorder"))
            kw.setdefault("alpha", self._get_key("grid.alpha"))

            for key in ["linestyle", "linewidth", "color"]:
                if grid == "minor" and key == "linestyle":
                    if "linestyle" not in kw:
                        kw.setdefault("dash_capstyle", self._get_key("grid.minor.capstyle"))
                        kw.setdefault("dashes", self._get_key("grid.minor.dashes"))
                else:
                    kw.setdefault(key, self._get_key("grid.%s.%s" % (grid, key)))

            return kw

        def check_fancy(yticks):
            # checks if the imaginary axis is symetric
            len_y = (len(yticks) - 1) // 2
            if not (len(yticks) % 2 == 1 and (yticks[len_y:] + yticks[len_y::-1] < EPSILON).all()):
                raise ValueError(
                    "fancy minor grid is only supported for zero-symetric imaginary grid - e.g. ImagMaxNLocator")
            return yticks[len_y:]

        def split_threshold(threshold):
            if isinstance(threshold, tuple):
                thr_x, thr_y = threshold
            else:
                thr_x = thr_y = threshold

            assert thr_x > 0 and thr_y > 0

            return thr_x / 1000, thr_y / 1000

        def add_arc(ps, p0, p1, grid, type):
            assert grid in ["major", "minor"]
            assert type in ["real", "imag"]
            assert p0 != p1
            arcs = self._majorarcs if grid == "major" else self._minorarcs
            if grid == "minor":
                param["zorder"] -= 1e-9
            arcs.append((type, (ps, p0, p1), self._add_gridline(ps, p0, p1, type, **param)))

        def draw_nonfancy(grid):
            if grid == "major":
                xticks = self.xaxis.get_majorticklocs()
                yticks = self.yaxis.get_majorticklocs()
            else:
                xticks = self.xaxis.get_minorticklocs()
                yticks = self.yaxis.get_minorticklocs()

            xticks = np.round(xticks, 7)
            yticks = np.round(yticks, 7)

            for xs in xticks:
                if xs < self._near_inf:
                    add_arc(xs, -self._near_inf, self._inf, grid, "real")

            for ys in yticks:
                if abs(ys) < self._near_inf:
                    add_arc(ys, 0, self._inf, grid, "imag")

        # set fancy parameters
        if fancy is None:
            fancy_major = self._get_key("grid.major.fancy")
            fancy_minor = self._get_key("grid.minor.fancy")
        else:
            fancy_major = fancy_minor = fancy

        # check parameters
        if "axis" in kwargs and kwargs["axis"] != "both":
            raise ValueError("Only 'both' is a supported value for 'axis'")

        # plot major grid
        if which in ['both', 'major']:
            for _, _, arc in self._majorarcs:
                arc.remove()
            self._majorarcs = []

            if b:
                param = get_kwargs('major')
                if fancy_major:
                    xticks = np.sort(self.xaxis.get_majorticklocs())
                    yticks = np.sort(self.yaxis.get_majorticklocs())
                    assert len(xticks) > 0 and len(yticks) > 0
                    yticks = check_fancy(yticks)

                    if threshold is None:
                        threshold = self._get_key("grid.major.fancy.threshold")

                    thr_x, thr_y = split_threshold(threshold)

                    # draw the 0 line
                    add_arc(yticks[0], 0, self._inf, "major", "imag")

                    tmp_yticks = yticks.copy()
                    for xs in xticks:
                        k = 1
                        while k < len(tmp_yticks):
                            y0, y1 = tmp_yticks[k - 1:k + 1]
                            if abs(self._moebius_z(xs, y0) - self._moebius_z(xs, y1)) < thr_x:
                                add_arc(y1, 0, xs, "major", "imag")
                                add_arc(-y1, 0, xs, "major", "imag")
                                tmp_yticks = np.delete(tmp_yticks, k)
                            else:
                                k += 1

                    for i in range(1, len(yticks)):
                        y0, y1 = yticks[i - 1:i + 1]
                        k = 1
                        while k < len(xticks):
                            x0, x1 = xticks[k - 1:k + 1]
                            if abs(self._moebius_z(x0, y1) - self._moebius_z(x1, y1)) < thr_y:
                                add_arc(x1, -y0, y0, "major", "real")
                                xticks = np.delete(xticks, k)
                            else:
                                k += 1
                else:
                    draw_nonfancy("major")

        # plot minor grid
        if which in ['both', 'minor']:
            # remove the old grid
            for _, _, arc in self._minorarcs:
                arc.remove()
            self._minorarcs = []

            if b:
                param = get_kwargs("minor")

                if fancy_minor:
                    # 1. Step: get x/y grid data
                    xticks = np.sort(self.xaxis.get_majorticklocs())
                    yticks = np.sort(self.yaxis.get_majorticklocs())
                    assert len(xticks) > 0 and len(yticks) > 0
                    yticks = check_fancy(yticks)

                    if dividers is None:
                        dividers = self._get_key("grid.minor.fancy.dividers")
                    assert len(dividers) > 0
                    dividers = np.sort(dividers)

                    if threshold is None:
                        threshold = self._get_key("grid.minor.fancy.threshold")

                    thr_x, thr_y = split_threshold(threshold)
                    len_x, len_y = len(xticks) - 1, len(yticks) - 1

                    # 2. Step: calculate optimal gridspacing for each quadrant
                    d_mat = np.ones((len_x, len_y, 2))

                    # TODO: optimize spacing algorithm
                    for i in range(len_x):
                        for k in range(len_y):
                            x0, x1 = xticks[i:i + 2]
                            y0, y1 = yticks[k:k + 2]

                            xm = self.real_interp1d([x0, x1], 2)[1]
                            ym = self.imag_interp1d([y0, y1], 2)[1]

                            x_div = y_div = dividers[0]

                            for div in dividers[1:]:
                                if abs(self._moebius_z(x1 - (x1 - x0) / div, ym) - self._moebius_z(x1, ym)) > thr_x:
                                    x_div = div
                                else:
                                    break

                            for div in dividers[1:]:
                                if abs(self._moebius_z(xm, y1) - self._moebius_z(xm, y1 - (y1 - y0) / div)) > thr_y:
                                    y_div = div
                                else:
                                    break

                            d_mat[i, k] = [x_div, y_div]

                    # 3. Steps: optimize spacing
                    # ensure the x-spacing declines towards infinity
                    d_mat[:-1, 0, 0] = list(map(np.max, zip(d_mat[:-1, 0, 0], d_mat[1:, 0, 0])))

                    # find the values which are near (0, 0.5) on the plot
                    idx = np.searchsorted(xticks, self._moebius_inv_z(0)) + 1
                    idy = np.searchsorted(yticks, self._moebius_inv_z(1j).imag)

                    # extend the values around the center towards the border
                    if idx > idy:
                        for d in range(idy):
                            delta = idx - idy + d
                            d_mat[delta, :d + 1] = d_mat[:delta, d] = d_mat[delta, 0]
                    else:
                        for d in range(idx):
                            delta = idy - idx + d
                            d_mat[:d + 1, delta] = d_mat[d, :delta] = d_mat[d, 0]

                    # 4. Step: gather and optimize the lines
                    x_lines, y_lines = [], []

                    for i in range(len_x):
                        x0, x1 = xticks[i:i + 2]

                        for k in range(len_y):
                            y0, y1 = yticks[k:k + 2]

                            x_div, y_div = d_mat[i, k]
                            #print('*'*33+'\n',x0, x1, x_div, y0, y1, y_div)
                            (x0, x1, x_div, y0, y1, y_div)=(float(x0), float(x1), int(x_div), float(y0), float(y1), int(y_div))
                            for xs in np.linspace(x0, x1, x_div + 1)[1:]:
                                x_lines.append([xs, y0, y1])
                                x_lines.append([xs, -y1, -y0])

                            for ys in np.linspace(y0, y1, y_div + 1)[1:]:
                                y_lines.append([ys, x0, x1])
                                y_lines.append([-ys, x0, x1])

                    # round values to prevent float inaccuarcy
                    x_lines = np.round(np.array(x_lines), 7)
                    y_lines = np.round(np.array(y_lines), 7)

                    # remove lines which overlap with the major grid
                    for tp, lines in [("real", x_lines), ("imag", y_lines)]:
                        for i in range(len(lines)):
                            ps, p0, p1 = lines[i]
                            if p0 > p1:
                                p0, p1 = p1, p0

                            for tq, (qs, q0, q1), _ in self._majorarcs:
                                if tp == tq and abs(ps - qs) < EPSILON and p1 > q0 and p0 < q1:
                                    lines[i, :] = np.nan
                                    break

                        lines = lines[~np.isnan(lines[:, 0])]
                        lines = lines[np.lexsort(lines[:, 1::-1].transpose())]

                        ps, p0, p1 = lines[0]
                        for qs, q0, q1 in lines[1:]:
                            if ps != qs or p1 != q0:
                                add_arc(ps, p0, p1, "minor", tp)
                                ps, p0, p1 = qs, q0, q1
                            else:
                                p1 = q1

                else:
                    draw_nonfancy("minor")

    def _hack_linedraw(self, line, rotate_marker=None):
        '''
        Modifies the draw method of a :class:`matplotlib.lines.Line2D` object
        to draw different stard and end marker.
        
        Keyword arguments:
            
            *line*:
                Line to be modified
                Accepts: Line2D 
                
            *rotate_marker*:
                If set, the end marker will be rotated in direction of their
                corresponding path. 
                Accepts: boolean
        '''
        assert isinstance(line, Line2D)

        def new_draw(self_line, renderer):
            def new_draw_markers(self_renderer, gc, marker_path, marker_trans, path, trans, rgbFace=None):
                # get the drawn path for determining the rotation angle
                line_vertices = self_line._get_transformed_path().get_fully_transformed_path().vertices
                vertices = path.vertices

                if len(vertices) == 1:
                    line_set = [[default_marker, vertices]]
                else:
                    if rotate_marker:
                        dx, dy = np.array(line_vertices[-1]) - np.array(line_vertices[-2])
                        end_rot = MarkerStyle(end.get_marker())
                        end_rot._transform += Affine2D().rotate(np.arctan2(dy, dx) - np.pi / 2)
                    else:
                        end_rot = end

                    if len(vertices) == 2:
                        line_set = [[start, vertices[0:1]], [end_rot, vertices[1:2]]]
                    else:
                        line_set = [[start, vertices[0:1]], [default_marker, vertices[1:-1]], [end_rot, vertices[-1:]]]

                for marker, points in line_set:
                    transform = marker.get_transform() + Affine2D().scale(self_line._markersize)
                    old_draw_markers(gc, marker.get_path(), transform, Path(points), trans, rgbFace)

            old_draw_markers = renderer.draw_markers
            renderer.draw_markers = MethodType(new_draw_markers, renderer)
            old_draw(renderer)
            renderer.draw_markers = old_draw_markers

        default_marker = line._marker
        # check if marker is set and visible
        if default_marker:
            start = MarkerStyle(self._get_key("plot.marker.start"))
            if start.get_marker() is None:
                start = default_marker

            end = MarkerStyle(self._get_key("plot.marker.end"))
            if end.get_marker() is None:
                end = default_marker

            if rotate_marker is None:
                rotate_marker = self._get_key("plot.marker.rotate")

            old_draw = line.draw
            line.draw = MethodType(new_draw, line)
            line._markerhacked = True

    def _add_gridline(self, ps, p0, p1, type, **kwargs):
        '''
        Add a gridline for a real axis circle.

        Keyword arguments:

            *ps*:
                Axis value
                Accepts: float

            *p0*:
                Start point 
                Accepts: float

            *p1*:
                End Point
                Accepts: float

            **kwargs*:
                Keywords passed to the arc creator
        '''
        assert type in ["real", "imag"]

        if type == "real":
            assert ps >= 0

            line = Line2D(2 * [ps], [p0, p1], **kwargs)
            line.get_path()._interpolation_steps = "x_gridline"
        else:
            assert 0 <= p0 < p1

            line = Line2D([p0, p1], 2 * [ps], **kwargs)

            if abs(ps) > EPSILON:
                line.get_path()._interpolation_steps = "y_gridline"

        return self.add_artist(line)

    class MoebiusTransform(Transform):
        '''
        Class for transforming points and paths to Smith Chart data space.
        '''
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, axes):
            assert isinstance(axes, SmithAxes)
            Transform.__init__(self)
            self._axes = axes

        def transform_non_affine(self, data):
            def _moebius_xy(_xy):
                return z_to_xy(self._axes._moebius_z(*_xy))

            if isinstance(data[0], Iterable):
                return list(map(_moebius_xy, data))
            else:
                return _moebius_xy(data)

        def transform_path_non_affine(self, path):
            vertices = path.vertices
            codes = path.codes

            linetype = path._interpolation_steps
            if linetype in ["x_gridline", "y_gridline"]:
                assert len(vertices) == 2

                x, y = np.array(list(zip(*vertices)))
                z = self._axes._moebius_z(x, y)

                if linetype == "x_gridline":
                    assert x[0] == x[1]
                    zm = 0.5 * (1 + self._axes._moebius_z(x[0]))
                else:
                    assert y[0] == y[1]
                    scale = 1j * (1 if self._axes._normalize else self._axes._impedance)
                    zm = 1 + scale / y[0]

                d = 2 * abs(zm - 1)
                ang0, ang1 = np.angle(z - zm, deg=True) % 360

                reverse = ang0 > ang1
                if reverse:
                    ang0, ang1 = ang1, ang0

                arc = Arc(z_to_xy(zm), d, d, theta1=ang0, theta2=ang1, transform=self._axes.transMoebius)
                arc._path = Path.arc(ang0, ang1)  # fix for Matplotlib 2.1+
                arc_path = arc.get_patch_transform().transform_path(arc.get_path())

                if reverse:
                    new_vertices = arc_path.vertices[::-1]
                else:
                    new_vertices = arc_path.vertices

                new_codes = arc_path.codes
            elif linetype == 1:
                new_vertices = self.transform_non_affine(vertices)
                new_codes = codes
            else:
                raise NotImplementedError("Value for 'path_interpolation' cannot be interpreted.")

            return Path(new_vertices, new_codes)

        def inverted(self):
            return SmithAxes.InvertedMoebiusTransform(self._axes)

    class InvertedMoebiusTransform(Transform):
        '''
        Inverse transformation for points and paths in Smith Chart data space.
        '''
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, axes):
            assert isinstance(axes, SmithAxes)
            Transform.__init__(self)
            self._axes = axes

        def transform_non_affine(self, data):
            def _moebius_inv_xy(_xy):
                return z_to_xy(self._axes._moebius_inv_z(*_xy))

            return list(map(_moebius_inv_xy, data))

        def inverted(self):
            return SmithAxes.MoebiusTransform(self._axes)

    class PolarTranslate(Transform):
        '''
        Transformation for translating points away from the center by a given
        padding. 
        
        Keyword arguments:
            
            *axes*:
                Parent :class:`SmithAxes`
                Accepts: SmithAxes instance
                
            *pad*:
                Distance to translate away from center for x and y values.
                
            *font_size*:
                y values are shiftet 0.5 * font_size further away.
        '''
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, axes, pad, font_size):
            Transform.__init__(self, shorthand_name=None)
            self.axes = axes
            self.pad = pad
            self.font_size = font_size

        def transform_non_affine(self, xy):
            def _translate(_xy):
                x, y = _xy
                ang = np.angle(complex(x - x0, y - y0))
                return x + np.cos(ang) * self.pad, y + np.sin(ang) * (self.pad + 0.5 * self.font_size)

            x0, y0 = self.axes.transAxes.transform([0.5, 0.5])
            if isinstance(xy[0], Iterable):
                return list(map(_translate, xy))
            else:
                return _translate(xy)

    class RealMaxNLocator(Locator):
        ''' 
        Locator for the real axis of a SmithAxes. Creates a nicely rounded
        spacing with maximum n values. The transformed center value is
        always included.
        
        Keyword arguments:
            
            *axes*:
                Parent SmithAxes
                Accepts: SmithAxes instance
                
            *n*:
                Maximum number of divisions
                Accepts: integer
                
            *precision*:
                Maximum number of significant decimals
                Accepts: integer
        '''

        def __init__(self, axes, n, precision=None):
            assert isinstance(axes, SmithAxes)
            assert n > 0

            Locator.__init__(self)
            self.steps = n
            if precision is None:
                self.precision = axes._get_key("grid.locator.precision")
            else:
                self.precision = precision
            assert self.precision > 0

            self.ticks = None
            self.axes = axes

        def __call__(self):
            if self.ticks is None:
                self.ticks = self.tick_values(0, self.axes._inf)
            return self.ticks

        def nice_round(self, num, down=True):
            # normalize to 'precision' decimals befor comma
            exp = np.ceil(np.log10(np.abs(num) + EPSILON))
            if exp < 1:  # fix for leading 0
                exp += 1
            norm = 10 ** -(exp - self.precision)

            num_normed = num * norm
            # increase precision by 0.5, if normed value is smaller than 1/3
            # of its decade range
            if num_normed < 3.3:
                norm *= 2
            # decrease precision by 1, if normed value is bigger than 1/2
            elif num_normed > 50:
                norm /= 10

            # select rounding function
            if not 1 < num_normed % 10 < 9:
                # round to nearest value, if last digit is 1 or 9
                if abs(num_normed % 10 - 1) < EPSILON:
                    num -= 0.5 / norm
                f_round = np.round
            else:
                f_round = np.floor if down else np.ceil

            return f_round(np.round(num * norm, 1)) / norm

        def tick_values(self, vmin, vmax):
            tmin, tmax = self.transform(vmin), self.transform(vmax)
            mean = self.transform(self.nice_round(self.invert(0.5 * (tmin + tmax))))

            result = [tmin, tmax, mean]
            d0 = abs(tmin - tmax) / (self.steps + 1)
            # calculate values above and below mean, adapt delta
            for sgn, side, end in [[1, False, tmax], [-1, True, tmin]]:
                d, d0 = d0, None
                last = mean
                while True:
                    new = last + d * sgn
                    if self.out_of_range(new) or abs(end - new) < d / 2:
                        break

                    # round new value to the next nice display value
                    new = self.transform(self.nice_round(self.invert(new), side))
                    d = abs(new - last)
                    if d0 is None:
                        d0 = d

                    last = new
                    result.append(last)

            return np.sort(self.invert(np.array(result)))

        def out_of_range(self, x):
            return abs(x) > 1

        def transform(self, x):
            return self.axes._moebius_z(x)

        def invert(self, x):
            return self.axes._moebius_inv_z(x)

    class ImagMaxNLocator(RealMaxNLocator):
        def __init__(self, axes, n, precision=None):
            SmithAxes.RealMaxNLocator.__init__(self, axes, n // 2, precision)

        def __call__(self):
            if self.ticks is None:
                tmp = self.tick_values(0, self.axes._inf)
                self.ticks = np.concatenate((-tmp[:0:-1], tmp))
            return self.ticks

        def out_of_range(self, x):
            return not 0 <= x <= np.pi

        def transform(self, x):
            return np.pi - np.angle(self.axes._moebius_z(x * 1j))

        def invert(self, x):
            return np.imag(-self.axes._moebius_inv_z(ang_to_c(np.pi + np.array(x))))

    class SmithAutoMinorLocator(AutoMinorLocator):
        '''
        AutoLocator for SmithAxes. Returns linear spaced intermediate ticks 
        depending on the major tickvalues. 
        
        Keyword arguments:
            
            *n*:
                Number of intermediate ticks
                Accepts: positive integer
        '''

        def __init__(self, n=4):
            assert isinstance(n, int) and n > 0
            AutoMinorLocator.__init__(self, n=n)
            self._ticks = None

        def __call__(self):
            if self._ticks is None:
                locs = self.axis.get_majorticklocs()
                self._ticks = np.concatenate(
                    [np.linspace(p0, p1, self.ndivs + 1)[1:-1] for (p0, p1) in zip(locs[:-1], locs[1:])])
            return self._ticks

    class RealFormatter(Formatter):
        ''' 
        Formatter for the real axis of a SmithAxes. Prints the numbers as 
        float and removes trailing zeros and commata. Special returns:
            '' for 0. 

        Keyword arguments:

            *axes*:
                Parent axes 
                Accepts: SmithAxes instance
        '''

        def __init__(self, axes, *args, **kwargs):
            assert isinstance(axes, SmithAxes)
            Formatter.__init__(self, *args, **kwargs)
            self._axes = axes

        def __call__(self, x, pos=None):
            if x < EPSILON or x > self._axes._near_inf:
                return ""
            else:
                return ('%f' % x).rstrip('0').rstrip('.')

    class ImagFormatter(RealFormatter):
        ''' 
        Formatter for the imaginary axis of a SmithAxes. Prints the numbers 
        as  float and removes trailing zeros and commata. Special returns:
            - '' for minus infinity 
            - 'symbol.infinity' from scParams for plus infinity
            - '0' for value near zero (prevents -0)
            
        Keyword arguments:

            *axes*:
                Parent axes 
                Accepts: SmithAxes instance
        '''

        def __call__(self, x, pos=None):
            if x < -self._axes._near_inf:
                return ""
            elif x > self._axes._near_inf:
                return self._axes._get_key("symbol.infinity")  # utf8 infinity symbol
            elif abs(x) < EPSILON:
                return "0"
            else:
                return ("%f" % x).rstrip('0').rstrip('.') + "j"

    # update docstrings for all methode not set
    for key, value in locals().copy().items():
        if isinstance(value, FunctionType):
            if value.__doc__ is None and hasattr(Axes, key):
                value.__doc__ = getattr(Axes, key).__doc__


__author__ = "Paul Staerke"
__copyright__ = "Copyright 2018, Paul Staerke"
__license__ = "BSD"
__version__ = "0.3"
__maintainer__ = "Paul Staerke"
__email__ = "paul.staerke@gmail.com"
__status__ = "Prototype"
