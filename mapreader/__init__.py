from mapreader.load.images import MapImages
from mapreader.load.loader import loader
from mapreader.load.loader import load_patches

from mapreader.download.sheet_downloader import SheetDownloader
from mapreader.download.downloader import Downloader
from mapreader.download.downloader_utils import create_polygon_from_latlons, create_line_from_latlons

from mapreader.learn.load_annotations import AnnotationsLoader
from mapreader.learn.datasets import PatchDataset
from mapreader.learn.datasets import patchContextDataset
from mapreader.learn.classifier import ClassifierContainer
from mapreader.learn.classifier_context import classifierContext
from mapreader.learn import custom_models

from mapreader.process import process

from . import _version

__version__ = _version.get_versions()["version"]

from mapreader.load import geo_utils
