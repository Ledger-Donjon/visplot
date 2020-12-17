# Visplot

A side-channel trace visualizer based on [vispy](https://github.com/vispy/).

Where [matplotlib](https://matplotlib.org/) excels at producing article-grade outputs, it is cumbersome to use during result _analysis_ - when one needs to look around to find what's going on.

## Design

*Fast* display and responsive panzoom, thanks to vispy (hold right-click and move around to see it in action).

*No visual clogging*: traces are drawn with smooth colors until you select them.
Selection is done with a click. Multiple traces can be selected by holding _CTRL_

(the demo below runs at 60 FPS)

![demo.gif](demo.gif)