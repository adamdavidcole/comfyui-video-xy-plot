"""
ComfyUI XY Video Plot Suite

A collection of nodes for generating parameter sweep video matrices.
Designed to work with comfyui-easy-use For Loop nodes.

Nodes:
- XYPlotSetup: Initialize plot parameters and calculate iterations
- XYPlotGetValues: Get current X/Y values (dynamic typing) 
- XYPlotGetValuesTyped: Get current X/Y values (all types - fallback)
- XYPlotCollectVideo: Collect video filepath with labels
- XYPlotVideoGrid: Assemble final video grid with styling
"""

from .xy_video_plot import (
    XYPlotSetup,
    XYPlotGetValues,
    XYPlotGetValuesTyped,
    XYPlotCollectVideo,
    XYPlotVideoGrid
)

NODE_CLASS_MAPPINGS = {
    "XYPlotSetup": XYPlotSetup,
    "XYPlotGetValues": XYPlotGetValues,
    "XYPlotGetValuesTyped": XYPlotGetValuesTyped,
    "XYPlotCollectVideo": XYPlotCollectVideo,
    "XYPlotVideoGrid": XYPlotVideoGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XYPlotSetup": "XY Plot Setup",
    "XYPlotGetValues": "XY Plot Get Values",
    "XYPlotGetValuesTyped": "XY Plot Get Values (Typed)",
    "XYPlotCollectVideo": "XY Plot Collect Video",
    "XYPlotVideoGrid": "XY Plot Video Grid",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']