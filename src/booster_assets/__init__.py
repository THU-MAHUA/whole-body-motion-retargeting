"""booster_assets package
"""

from wbmr.paths import ROOT

BOOSTER_ASSETS_DIR = str(ROOT)

from . import motions

__all__ = ['BOOSTER_ASSETS_DIR', 'motions']
