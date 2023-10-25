# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from typing import List
import matplotlib as mpl
import matplotlib.pyplot as plt

from .plot_scatter import plot_scatter
from .plot_hist import plot_hist
__all__ = ['plot_scatter', 'plot_hist']

# default options
mpl.rcParams['lines.linewidth']  = 3
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['font.size']        = 14
mpl.rcParams['axes.labelsize']   = 'x-large'
mpl.rcParams['legend.fontsize']  = 'large'
mpl.rcParams['figure.titlesize'] = 'x-large'
mpl.rcParams['figure.figsize']   = [10.0, 9.0]
mpl.rcParams['figure.dpi']       = 100
mpl.rcParams['savefig.dpi']      = 300

# VISUALIZE FUNCTIONS
def get_subplot_dim(num_subplots: int, rowfirst: bool = True) -> tuple:
    """
    Compute the number of rows and columns (nrows, ncols) for a figure with multiple subplots.
    The function returns number of rows and columns given num_subplots.
    Those numbers are computed sequentially until nrows * ncols >= num_subplots.
    By default, the function adds a new row first if the number of subplots has not been reached, then adds a new column.
    By passing rowfirst=False, the function will add a new column first if the number of subplot has not been reached, then a new row.

    nrows and ncols are initialized to 1. If num_subplots==1, then subplots are not needed, and the function returns nrows=ncols=1.
    The command fig.add_subplot(nrows,ncols,1) generates a normal plot (no subplots).

    Parameters
    ----------
    num_subplots : int
                   number of subplots the figure should contain
    rowfirst     : Boolean
                   whether to add a new row first or a new column first to increase the number of rows and columns if necessary.
                   Default is rowfirst=True.
    
    Returns
    -------
    nrows : int
            number of subplots along the rows (vertical axis) of the figure
    ncols : int
            number of subplots along the columns (horizontal axis) of the figure

    Example
    -------
    >>> import numpy as np
    >>> states = np.random.randn(1000,5) # let us consider a state vector with 5 dimensions, and 1000 values of the states (one for each time step)
    >>> n_states = states.shape[-1]     # get the number of states (5)
    >>> nrows, ncols = get_subplot_dim(n_states)
    >>> print(get_subplot_dim(n_states)) # 3, 2
    (3, 2)
    >>> print(get_subplot_dim(n_states, rowfirst=False)) # 2, 3
    (2, 3)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(nrows, ncols, 1)

    # ...
    """
    nrows, ncols = 1, 1  # initialize number of rows and cols to 1.
    if rowfirst:
        while nrows * ncols < num_subplots:         
            nrows += 1
            if nrows * ncols < num_subplots:        
                ncols += 1
    else:
        while nrows * ncols < num_subplots:         
            ncols += 1
            if nrows * ncols < num_subplots:        
                nrows += 1
    return nrows, ncols

def set_plot_options(opt: dict) -> dict:
    """
    Set default plot options by integrating the options specified by the user in 'opt'
    The visualize library works with specific values to generate the plots.
    if those are not specified by the users, this function assign them their default values.

    Parameters
    ----------
    opt : dictionary of plot options. Acceptable entries are:\n
          * 'figsize' : tuple of 2 floats, width and height of the figure
          * 'compact' : bool, whether to plot a "compact" figure. If compact, all time series are displayed in one plot (multiple colored lines)
          * 'xlabel'  : str, label for the x-axis. Default is 'time'
          * 'ylabel'  : str, label for the y-axis. Default is 'state'
          * 'title'   : str or empty list or None, plot title. Default is empty list (no title)
          * 'title_fontsize' : str or float, plot title fontsize. Default is 'x-large'
          * 'suptitle'       : str or empty list or None, plot suptitle. Default is empty list (no suptitle)
          * 'ticklabel_fontsize' : str or float, tick label font sizes. Default is 'small'
          * 'tight_layout' : Boolean, whether to use tight layout (minimize figure blank space around the graph)
          * 'display_labels' : str, whether to display x and y-labels in the figure.
          * 'keys' : list of keys to plot. If not provided, all keys in the series are plotted.

    Returns
    -------
    opt : dict
        dict of plot options with default values added.

    Example:
    -------
    >>> opt = {}
    >>> opt = set_plot_options(opt)
    >>> print(opt)
    {'figsize': (10, 9), 'compact': True, 'xlabel': 'time', 'ylabel': 'state', 'title': [], 'title_fontsize': 'x-large', 'suptitle': [], 'ticklabel_fontsize': 'small', 'tight_layout': False, 'display_labels': 'all', 'label_fontsize': 'x-large', 'xticks': [], 'yticks': []}

    """
    # Set up plot options
    # Get list of options (if provided)
    try:
        opt_list = list(opt.keys())
    except Exception:
        opt, opt_list = {}, []
    
    # Fill out all options if not provided
    if 'figsize' not in opt_list:
        opt['figsize'] = (10, 9)
    if 'compact' not in opt_list:
        opt['compact'] = True
    if 'xlabel' not in opt_list:
        opt['xlabel'] = 'time'
    if 'ylabel' not in opt_list:
        opt['ylabel'] = 'state'
    if 'title' not in opt_list:
        opt['title'] = []
    if 'title_fontsize' not in opt_list:
        opt['title_fontsize'] = 'x-large'
    if 'suptitle' not in opt_list:
        opt['suptitle'] = []
    if 'ticklabel_fontsize' not in opt_list:
        opt['ticklabel_fontsize'] = 'small'
    if 'tight_layout' not in opt_list:
        opt['tight_layout'] = False
    if 'display_labels' not in opt_list:
        opt['display_labels'] = 'all'
    
    # if title should be displayed but title fontsize is not specified, add it to the dictionary
    if opt['title'] and ('title_fontsize' not in opt_list or not opt['title_fontsize']):
        if 'fontsize' in opt_list:
            opt['title_fontsize'] = opt['fontsize']
        else:
            opt['title_fontsize'] = 'x-large'
    # if xlabel or ylabel should be displayed but their fontsize is not specified, add it to the dictionary
    if (opt['xlabel'] or opt['ylabel']) and ('label_fontsize' not in opt_list or not opt['label_fontsize']):
        if 'fontsize' in opt_list:
            opt['label_fontsize'] = opt['fontsize']
        else:
            opt['label_fontsize'] = 'x-large'
    # if xticks are not default but rotation and fontsize are not specified, add them to the dictionary
    if 'xticks' in opt_list:
        if 'xtick_rotation' not in opt_list:
            opt['xtick_rotation'] = 0
        if 'tick_fontsize' not in opt_list:
            opt['xtick_fontsize'] = 'large'
        else:
            opt['xtick_fontsize'] = opt['tick_fontsize']
    else:
        opt['xticks'] = []
    # if yticks are not default but rotation and fontsize are not specified, add them to the dictionary
    if 'yticks' in opt_list:
        if 'ytick_rotation' not in opt_list:
            opt['ytick_rotation'] = 0
        if 'tick_fontsize' not in opt_list:
            opt['ytick_fontsize'] = 'large'
        else:
            opt['ytick_fontsize'] = opt['tick_fontsize']
    else:
        opt['yticks'] = []

    return opt

def set_legend_options(leg_opt: dict, s_names: List[str]) -> dict:
    """
    Set all remaining legend options given the legend options already specified by the users "leg_opt", 
    and the names of the time series in the plot "s_names."
    
    The visualize library works with specific values for some legend options.
    if those are not specified by the users, this function assign them their default values.

    Parameters
    ----------
    leg_opt : dictionary of entries for the legend specified by the user. 
    s_names : list of strings, names of the time series in the current plot, whose names should appear in the legend.

    Returns
    -------
    leg_opt : dict
        dictionary of enetries for the legend.

    Example
    -------
    >>> s_names = ['x', 'v']
    >>> leg_opt = {'display': True, 'loc': 'best', 'fontsize': 14}
    >>> leg_opt = set_legend_options(leg_opt, s_names)
    >>> print(leg_opt['labels'])
    ['x', 'v']
    >>> print(leg_opt['fancybox'])
    False
    >>> print(leg_opt['facecolor'])
    w
    """
    try:
        leg_list = list(leg_opt.keys())  # Check whether a dictionary has been provided. If not, initialize the dictionary leg_opt as empty
    except Exception:
        leg_opt, leg_list = {}, []

    if 'display' not in leg_list:
        leg_opt['display'] = False
    if 'display_at_subplot' not in leg_list:
        leg_opt['display_at_subplot'] = len(s_names)
    if 'labels' not in leg_list:
        leg_opt['labels'] = s_names
    if 'loc' not in leg_list:
        leg_opt['loc'] = 'best'
    if 'bbox_to_anchor' not in leg_list:
        leg_opt['bbox_to_anchor'] = None
    if 'ncol' not in leg_list:
        leg_opt['ncol'] = 1
    if 'fontsize' not in leg_list:
        leg_opt['fontsize'] = 'x-large'
    if 'shadow' not in leg_list:
        leg_opt['shadow'] = False
    if 'fancybox' not in leg_list:
        leg_opt['fancybox'] = False
    if 'framealpha' not in leg_list:
        leg_opt['framealpha'] = 1.0
    if 'facecolor' not in leg_list:
        leg_opt['facecolor'] = 'w'
    if 'edgecolor' not in leg_list:
        leg_opt['edgecolor'] = 'w'
    if 'title' not in leg_list:
        leg_opt['title'] = None
    if 'title' in leg_list and 'title_fontsize' not in leg_list:                 
        leg_opt['title_fontsize'] = 'medium'

    return leg_opt

def set_savefig_options(sfo: dict) -> dict:
    """
    Set all remaining save figure options given the options already specified by the user "sfo".

    Parameters
    ----------
    sfo : dictionary of options to save the current figure
          if not provided, a figure will not be saved by default. Otherwise, sfo should provide the following dictionary entries:\n
           * 'save' (bool, whether to save the figure or not, True or False)
           * 'dpi' (int, resolution of the figure in dots per inch)
           * 'filename' (str, figure filename, including file type, e.g., .pdf or .png. Default is 'timeseries_plot.pdf')
    
    Returns
    -------
    sfo  : dict
        dict of default save figure options.

    Example
    -------
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot([0, 1], [3, 4]) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D ...>]
    >>> sfo = {'save': True}    # a figure has to be saved, but no dpi nor filename for the figure has been specified in sfo
    >>> sfo = set_savefig_options(sfo)
    >>> print(sfo)  # sfo = {'save': True, 'dpi': 300, 'filename': 'timeseries_plot.pdf'}
    {'save': True, 'dpi': 300, 'filename': 'timeseries_plot.pdf'}
    """
    try:
        sfo_list = list(sfo.keys())
    except Exception:
        sfo, sfo_list = {}, []
    if 'save' not in sfo_list:
        sfo['save'] = False
    if 'dpi' not in sfo_list:
        sfo['dpi'] = 300
    if 'save' in sfo_list and 'filename' not in sfo_list:
        sfo['filename'] = 'timeseries_plot.pdf'
    return sfo

def set_legend(ax: plt.axis, item: int, s_names: List[str], leg_opt: dict) -> plt.axis:
    """
    Set legend for axis 'ax' for the 'item-th' time series entry. All time series labels are defined in 's_names'.
    For a comprehensive explanation of all legend options, see the Matplotlib guide on their website.

    Parameters
    ----------
    ax      : matplotlib axis object
    item    : int, index of the time series to be displayed in the legend
    s_names : list of strings, names of all time series in the plot or subplot
    leg_opt : dictionary containing all legend options necessary to place and modify the legend. Dictionary entries can be:\n
               * 'bbox_to_anchor' (tuple, coordinates of the legend location),
               * 'ncol' (int, number of columns of the legend),
               * 'fontsize' (int, legend font size),
               * 'fancybox' (Boolean, Use a fancybox for the legend, True or False),
               * 'shadow' (Boolean, Whether the legend box should have a shadow, True or False)
               * 'facecolor' (string with color code, background color of the legend box),
               * 'edgecolor' (string with color code, edge color of the legend box),
               * 'title' (string, legend title)
    
    Returns
    -------
    ax : axis object
        axis object with legend

    Example
    -------
    >>> # doctest: +SKIP
    >>> s = {}
    >>> s_names = list(s.keys())
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot()
    >>> ax.plot(t, [list(s_i.values()) for s_i in s])
    >>> leg_opt = {}
    >>> leg_opt['labels'] = s_names
    >>> leg_opt['loc'] = 'best'
    >>> leg_opt['bbox_to_anchor'] = None
    >>> leg_opt['ncol'] = 1
    >>> leg_opt['fontsize'] = 'x-large'
    >>> leg_opt['shadow'] = False
    >>> leg_opt['fancybox'] = False
    >>> leg_opt['framealpha'] = 1.0
    >>> leg_opt['facecolor'] = 'w'
    >>> leg_opt['edgecolor'] = 'w'
    >>> leg_opt['title'] = None
    >>> ax.legend(series_names, bbox_to_anchor=legend_options['bbox_to_anchor'],
    >>>           ncol=legend_options['ncol'], fontsize=legend_options['fontsize'],
    >>>           fancybox=legend_options['fancybox'], shadow=legend_options['shadow'],
    >>>           framealpha=legend_options['framealpha'], facecolor=legend_options['facecolor'],
    >>>           edgecolor=legend_options['edgecolor'], title=legend_options['title'])
    """
    return ax.legend(s_names[item], bbox_to_anchor=leg_opt['bbox_to_anchor'], 
                     ncol=leg_opt['ncol'], fontsize=leg_opt['fontsize'],
                     fancybox=leg_opt['fancybox'], shadow=leg_opt['shadow'],
                     framealpha=leg_opt['framealpha'], facecolor=leg_opt['facecolor'],
                     edgecolor=leg_opt['edgecolor'], title=leg_opt['title'])

def display_labels(nrows: int, ncols: int, subplot_num: int, ax: plt.axis, opt: dict, series_names: List[str]) -> None:
    """
    Display label option for time series plot

    Parameters
    ----------
    nrows        : int
        number of subplot rows in plot
    ncols        : int
        number of subplot columns in plot
    subplot_num  : int
        subplot number
    ax           : matplotlib axis object
        current axis
    opt          : dict
        display options for the plot. Minimum options to be included are:\n
         1. 'display_labels' = 'minimal' or 'all'. if 'minimal', only the minimum number of axis ticks and labels are displayed according
            to the number of subplots. If 'all', then all x-ticks, x-labels, y-ticks, and y-labels will be displayed for each subplot.\n
         2. 'xlabel' and 'ylabel'. strings containing the xlabels and ylabels to display.
    series_names : list of strings
        names of the time series

    Example
    -------
    >>> # doctest: +SKIP
    >>> nrows, ncols = 2, 1
    >>> subplot_num = 0
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(nrows, ncols, subplot_num+1)
    >>> opt = {'xlabel': 'time', 'ylabel': 'state', 'display_labels': 'all'}
    >>> series_names= ['x', 'v']
    >>> display_labels(nrows, ncols, subplot_num, ax, opt, series_names)
    """
    if 'minimal' in opt['display_labels']:
        if subplot_num+1 == nrows*ncols and opt['xlabel']:
            set_labels(ax, opt, series_names, axis='x')
        if (subplot_num+1 == 1 or ncols==1) and opt['ylabel']:
            set_labels(ax, opt, series_names, axis='y')
        if subplot_num+1 == 1:
            ax.set_xticks([], minor=[])    # If 'display_labels' is minimal, kill xticks that are not needed according to subplots
    elif 'all' in opt['display_labels']:    
        set_labels(ax, opt, series_names)

def extract_option(opt: dict, idx: int, series_names: List[str]) -> str:
    """
    Extract option from either dictionary or list of plot options.
    The function takes the option "opt" and returns the option at index "idx" if opt is a list,
    the option corresponding to the series name at index idx, "series_names[idx]" if opt is a dictionary,
    or return the option "opt" if opt is neither a dictionary or a list.

    Parameters
    ----------
    opt             :   dictionary of strings
        dictionary of label name) corresponding to entries in series_names, list of strings corresponding to the series names (in order), or a simple string
    idx             :   int
        index of the option to extract in label options if opt is a string, or index of the corresponding series_names to extract if opt is a dictionary
    series_names    :   list of strings
        series names used if opt is a dictionary

    Returns
    -------
    option: string
        option corresponding to the index idx (if opt is a list of strings), corresponding to the series_names[idx] if opt is dictionary, or simply return opt if neither of those

    Examples
    --------
    | Example 1:
    >>> # doctest: +SKIP
    >>> ...........
    >>> opt = {'ylabel': {'x': 'position', 'v': 'velocity}}
    >>> series_names = ['x', 'v']
    >>> print(extract_option(opt['ylabel'], 0, series_names)) # 'position'
    >>> print(extract_option(opt['ylabel'], 1, series_names)) # 'velocity'
    | 
    | Example 2:
    >>> # doctest: +SKIP
    >>> opt = {'ylabel': ['state value 1', 'state value 2'] }
    >>> print(extract_option(opt['ylabel'], 0, series_names)) # 'state value 1', please note that series_names is ignored
    >>> print(extract_option(opt['ylabel'], 1, series_names)) # 'state value 2'
    | 
    | Example 3:
    >>> # doctest: +SKIP
    >>> opt = {'ylabel': 'state output'}
    >>> series_names = ['x', 'v']
    >>> print(extract_option(opt['ylabel'], 0, series_names)) # 'state output', please note that idx and series_names are ignored
    >>> print(extract_option(opt['ylabel'], 1, {})) # 'state output', please note that idx and series_names are ignored
    >>> print(extract_option(opt['ylabel'], np.inf, [])) # 'state output', please note that idx and series_names are ignored
    """
    if isinstance(opt, dict):
        return opt[series_names[idx]]
    if isinstance(opt, list) and (len(opt)>0):
        return opt[idx]
    return opt

def set_ax_options(ax: plt.axis, opts: dict) -> None:
    """
    Set label options for plot axis.

    The function set a number of labels defined in dictionary "opts" for the current axis "ax".\n
    At the moment, the options that can be set are:\n
        title, title_fontsize, xlabel, ylabel, xticks, xtick_rotation, xtick_fontsize, tticks, ytick_rotation, ytick_fontsize\n
    Input options are not mandatory, so only a subset of them can be defined.

    Parameters
    ----------
    ax : matplotlib axis object
        matplotlib axis to visualize in figure
    opts : dict
        display options for the axis / subplot. Available: title, title_fontsize, xlabel, ylabel, xticks, xtick_rotation, xtick_fontsize, tticks, ytick_rotation, ytick_fontsize

    Example
    -------
    >>> # doctest: +SKIP
    >>> opts = {'title': 'this is a test title',
    >>>         'title_fontsize': 16, 'xlabel': 'time', 'ylabel': 'state value',
    >>>         'xtick_rotation': 45, 'xtick_fontsize': 14, 'ytick_rotation': 0, 'ytick_fontsize': 14}
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> set_ax_options(ax, opts)
    """
    if opts['title']:
        ax.set_title(opts['title'], fontsize=opts['title_fontsize'])
    if opts['xlabel']:
        ax.set_xlabel(opts['xlabel'])
    if opts['ylabel']:
        ax.set_ylabel(opts['ylabel'])
    if opts['xticks']:
        ax.set_xticklabels(opts['xticks'], rotation=opts['xtick_rotation'], fontsize=opts['xtick_fontsize'])
    if opts['yticks']:
        ax.set_yticklabels(opts['yticks'], rotation=opts['ytick_rotation'], fontsize=opts['ytick_fontsize'])

def set_labels(ax: plt.axis, opt: dict, series_names: List[str], axis: str = 'all') -> None:
    """
    Set labels of axis "ax" according to figure options "opt" and the time series names "series_names."
    The function can set both x and y axis when input axis=='all' (default), or rather set only x or y axis (axis='x' or axis='y', respectively).

    Parameters
    ----------
    ax : matplotlib axis object
        axis to add labels and tick options to
    opt : dictionary of label options
        options must include labels and ticks for both x and y axes; 'xlabel', 'xticks', 'ylabel', 'yticks'
        if 'xticks' is not empty (or None), then options can include also 'xtick_rotation', to rotate the axis ticks w.r.t. the plot, as well as 'xtick_fontsize', to change tick fontsize.
        the same applies to 'yticks'.
    series_names : list of strings
        name of time series
    axis : string to decide which axis to display the options on.
        options are: 'x', 'y', or 'all' for both x and y

    Example
    -------
    >>> # doctest: +SKIP
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(221)
    >>> series_names = ['x', 'y']
    >>> opt = {'xlabel': 'time', 'ylabel': 'state value', 'xticks': ['-\pi', '0', '\pi'], 'xtick_fontsize', 12, 'xtick_rotation', 90,
    >>>        'yticks': ['-\pi', '0', '\pi'], 'ytick_fontsize', 12, 'ytick_rotation', -90}
    >>> set_labels(ax, opt, series_names, axis='all')
    """
    idx = ax._subplotspec.colspan[0] + ax._subplotspec.rowspan[0]   # Extract the index of the current subplot
    if axis=='all' or axis=='x':    # add properties to x-axis
        xlabel    = extract_option(opt['xlabel'], idx, series_names)
        xtick     = extract_option(opt['xticks'], idx, series_names)
        ax.set_xlabel(xlabel)
        if xtick:                   # if xtick options are passed, add them to the axis
            xtick_rot = extract_option(opt['xtick_rotation'], idx, series_names)
            xtick_fs  = extract_option(opt['xtick_fontsize'], idx, series_names)
            ax.set_xticklabels(xtick, rotation=xtick_rot, fontsize=xtick_fs)
    if axis=='all' or axis=='y':    # add properties to y-axis
        ylabel    = extract_option(opt['ylabel'], idx, series_names)
        ytick     = extract_option(opt['yticks'], idx, series_names)
        ax.set_ylabel(ylabel)
        if ytick:                   # if ytick options are passed, add them to the axis
            ytick_rot = extract_option(opt['ytick_rotation'], idx, series_names)
            ytick_fs  = extract_option(opt['ytick_fontsize'], idx, series_names)
            ax.set_yticklabels(ytick, rotation=ytick_rot, fontsize=ytick_fs)

def plot_timeseries(t: List[float], s: List[dict], legend: dict = None, options: dict = None) -> plt.figure:
    """
    Plot time series 's' parametrized by time 't'.
    The function plot time series (in a single plot or subplots) contained in the array of dictionary s, produced by a prognostic model.

    Input legend, and options are optional (default is None). If provided, they must be dictionaries with options for legend, and
    plot options, respectively. 
    
    The function is capable of plotting time series in a single plot (options['compact']=True), or in multiple subplots in the same figure (options['compact']=False).
    Legends can be displayed in each subplot or only one subplot, and names of time series, axis labels, plot title, legend titles and other are all customizable.
    Please read the help of the other functions suggested below for more info.

    Parameters
    ----------
    t : list of floats
        time vector
    s : array of dictionaries
        Each entry of the array is a dictionary with all time series, and one value per time series (corresponding to the time instant).\n
        Example: s = [ {'x': 0.0,  'v': 1.0},\n 
                       {'x': 1.0,  'v': 0.9},\n
                       {'x': 1.83, 'v': 0.75},...]
    legend : dict
        dict of legend options. See 'set_legend' function for more details
    options : dict
        dict of plot options. See 'set_plot_options' function and other functions therein for more details

    Returns
    -------
    fig : matplotlib figure object
        object corresponding to the generated figure.
     
    Example
    -------
    See new model example
    """

    series_names = options.get('keys', list(s[0].keys()))

    m = len(series_names)
    n = len(s)
    
    # Set up options
    fig_options    = set_plot_options(options)                 # Set up figure options
    legend_options = set_legend_options(legend, series_names)  # Set up legend options
    
    # Generate figure
    if fig_options['figsize'] is not None:
        fig = plt.figure(figsize=fig_options['figsize'])
    else:
        fig = plt.figure()

    if fig_options['compact']:  # Compact option: all time series in one plot
        # Add plot
        ax = fig.add_subplot()
        ax.plot(t, [[s_i[key] for key in series_names] for s_i in s])

        # Add options: plot options, title, labels, and legend
        set_ax_options(ax, fig_options)

        if fig_options['title']:
            ax.set_title(fig_options['title'], fontsize=fig_options['title_fontsize'])

        if fig_options['display_labels'] or fig_options['display_labels'] != 'no':
            display_labels(1, 1, 0, ax, fig_options, series_names)
        
        if legend_options['display']:
            ax.legend(series_names, bbox_to_anchor=legend_options['bbox_to_anchor'], 
                      ncol=legend_options['ncol'], fontsize=legend_options['fontsize'],
                      fancybox=legend_options['fancybox'], shadow=legend_options['shadow'],
                      framealpha=legend_options['framealpha'], facecolor=legend_options['facecolor'],
                      edgecolor=legend_options['edgecolor'], title=legend_options['title'])
    
    else:  # "Not compact" option: one subplot per time series
        nrows, ncols = get_subplot_dim(m)  # get the number of subplots
        
        # Iterate over all subplots to plot the time series
        for item in range(m):
            ax = fig.add_subplot(nrows, ncols, item+1)                  # add subplot
            series_ = [s[ii][series_names[item]] for ii in range(n)]    # extract time series data from array of dictionaries
            ax.plot(t, series_)                                         # add time series to subplot
            
            # Add options: display labels, title, legend
            if fig_options['display_labels'] or fig_options['display_labels'] != 'no':
                display_labels(nrows, ncols, item, ax, fig_options, series_names)
            
            if fig_options['title']:
                ax.set_title(series_names[item], fontsize=fig_options['title_fontsize'])
            
            if legend_options['display']:
                if legend_options['display_at_subplot'] == 'all':
                    set_legend(ax, item, series_names, legend_options)
                elif legend_options['display_at_subplot'] == item+1:
                    set_legend(ax, item, series_names, legend_options)
            
    # Other options
    if fig_options['suptitle']:
        fig.suptitle(fig_options['suptitle'], fontsize=fig_options['title_fontsize'])  # Add subtitle
    if fig_options['tight_layout']:
        plt.tight_layout()  # If tight-layout
        
    return fig

