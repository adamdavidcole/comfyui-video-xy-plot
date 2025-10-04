"""
ComfyUI XY Video Plot Suite

A collection of nodes for generating parameter sweep video matrices.
Designed to work with comfyui-easy-use For Loop nodes.

Nodes:
- XYPlotSetup: Initialize plot parameters and calculate iterations
- XYPlotDirectorySetup: Generate organized directory paths (optional)
- XYPlotGetValues: Get current X/Y values (dynamic typing) 
- XYPlotCollectVideo: Collect video filepath with labels
- XYPlotVideoGrid: Assemble final video grid with styling
"""

from .xy_video_plot import (
    XYPlotSetup,
    XYPlotDirectorySetup,
    XYPlotGetValues,
    XYPlotCollectVideo,
    XYPlotVideoGrid
)

NODE_CLASS_MAPPINGS = {
    "XYPlotSetup": XYPlotSetup,
    "XYPlotDirectorySetup": XYPlotDirectorySetup,
    "XYPlotGetValues": XYPlotGetValues,
    "XYPlotCollectVideo": XYPlotCollectVideo,
    "XYPlotVideoGrid": XYPlotVideoGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XYPlotSetup": "XY Plot Setup",
    "XYPlotDirectorySetup": "XY Plot Directory Setup",
    "XYPlotGetValues": "XY Plot Get Values",
    "XYPlotCollectVideo": "XY Plot Collect Video",
    "XYPlotVideoGrid": "XY Plot Video Grid",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']