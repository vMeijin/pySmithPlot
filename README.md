pySmithPlot
===========

Matplotlib extension for creating Smith charts with Python

Library for plotting a fully automatic Smith Chart with various customizable
parameters and well selected default values. It also provides the following 
modifications and features:

    - circle shaped drawing area with labels placed around 
    - plot() accepts complex numbers and numpy.ndarray's
    - lines can be automatically interpolated in evenly spaced steps 
    - linear connections can be transformed to arcs
    - start/end markers of lines can be modified and rotated
    - gridlines are arcs, which is more efficient
    - optional fancy grid with adaptive spacing
    - own tick locators for nice axis labels
    
For making a Smith Chart plot, it is sufficient to `import smithplot` and
create a new subplot with projection set to 'smith'. (Requires matplotlib 
version 1.2)

A short example can be found in the `example` directory and testet with:

    `python smith_test.py`
    
For more details, take a look into `smithplot/smithaxes.py`. 

`example/smith_example.py` creates a bunch of Smith charts with varying 
parameters for comparision. Here are the sample plots: 

![Gridstyles](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_grid.png)
[Gridstyles - PDF](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_grid.pdf)

![Fancy Threshold](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_fancy_threshold.png)
[Fancy Threshold - PDF](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_fancy_threshold.pdf)

![Grid Locators](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_grid_maxn.png)
[Grid Locators - PDF](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_grid_maxn.pdf)

![Marker Modification](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_marker.png)
[Marker Modification - PDF](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_marker.pdf)

![Interpolation](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_interp.png)
[Interpolation - PDF](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_interp.pdf)

![Scaling](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_scale.png)
[Scaling - PDF](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_scale.pdf)

![VSWR Circles](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_circle.png)
[VSWR Circles - PDF](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_circle.pdf)

![Misc](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_misc.png)
[Misc - PDF](https://github.com/vMeijin/pySmithPlot/wiki/images/examples/ex_misc.pdf)




