from __future__ import annotations

import pathlib
import pickle

try:
    import adet
except ImportError:
    raise ImportError(
        "[ERROR] Please install DeepSolo from the following link: https://github.com/rwood-97/DeepSolo"
    )

import geopandas as gpd
import pandas as pd
import torch
from adet.config import get_cfg

try:
    from detectron2.engine import DefaultPredictor
except ImportError:
    raise ImportError("[ERROR] Please install Detectron2")

from adet.utils.vitae_predictor import ViTAEPredictor

# first assert we are using the deep solo version of adet
if adet.__version__ != "0.2.0-maptextpipeline":
    raise ImportError(
        "[ERROR] Please install MapTextPipeline from the following link: https://github.com/rwood-97/MapTextPipeline"
    )

from .rec_runner_base import RecRunner


class MapTextRunner(RecRunner):
    def __init__(
        self,
        patch_df: pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path,
        parent_df: pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path | None = None,
        cfg_file: str
        | pathlib.Path = "./MapTextPipeline/configs/ViTAEv2_S/rumsey/final_rumsey.yaml",
        weights_file: str | pathlib.Path = "./rumsey-finetune.pth",
        device: str = "default",
        delimiter: str = ",",
    ) -> None:
        """Initialise the MapTextRunner.

        Parameters
        ----------
        patch_df : pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path
            The dataframe containing the patch information. If a string/path, it should be a path to a CSV/TSV/etc or geojson file.
        parent_df : pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path, optional
            The dataframe containing the parent information. If a string/path, it should be a path to a CSV/TSV/etc or geojson file.
        cfg_file : str | pathlib.Path, optional
            The path to the config file (yaml), by default "./MapTextPipeline/configs/ViTAEv2_S/rumsey/final_rumsey.yaml"
        weights_file : str | pathlib.Path, optional
            The path to the weights file (.pth), by default, by default "./rumsey-finetune.pth"
        device : str, optional
            The device to use for the model, by default "default". If default, the device will be set to cuda if available, otherwise cpu.
        delimiter : str, optional
            The delimiter to use if loading dataframes from CSV files, by default ",".
        """
        # setup the dataframes
        self._load_df(patch_df, parent_df, delimiter)

        # set up predictions as dictionaries
        self.patch_predictions = {}
        self.parent_predictions = {}
        self.geo_predictions = {}
        self.search_results = {}

        # setup the config
        cfg = get_cfg()  # get a fresh new config
        cfg.merge_from_file(cfg_file)
        cfg.MODEL.WEIGHTS = weights_file
        if device == "default":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.DEVICE = device

        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.use_customer_dictionary = cfg.MODEL.TRANSFORMER.CUSTOM_DICT
        if self.voc_size == 96:
            self.CTLABELS = [
                " ",
                "!",
                '"',
                "#",
                "$",
                "%",
                "&",
                "'",
                "(",
                ")",
                "*",
                "+",
                ",",
                "-",
                ".",
                "/",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                ":",
                ";",
                "<",
                "=",
                ">",
                "?",
                "@",
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
                "[",
                "\\",
                "]",
                "^",
                "_",
                "`",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                "{",
                "|",
                "}",
                "~",
            ]
        elif self.voc_size == 37:
            self.CTLABELS = [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            ]
        elif self.voc_size == 148:
            self.CTLABELS = [
                " ",
                "!",
                '"',
                "#",
                "$",
                "%",
                "&",
                "'",
                "(",
                ")",
                "+",
                ",",
                "-",
                ".",
                "/",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                ":",
                ";",
                "<",
                "=",
                ">",
                "?",
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
                "_",
                "`",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                "\x8d",
                "\xa0",
                "¡",
                "£",
                "¨",
                "©",
                "®",
                "¯",
                "°",
                "¹",
                "Á",
                "Â",
                "Ã",
                "Ä",
                "Å",
                "É",
                "Ê",
                "Ì",
                "Í",
                "Î",
                "Ó",
                "ß",
                "à",
                "á",
                "â",
                "ä",
                "è",
                "é",
                "ê",
                "ë",
                "í",
                "ï",
                "ñ",
                "ó",
                "ô",
                "õ",
                "ö",
                "ú",
                "û",
                "ü",
                "ÿ",
                "ā",
                "ė",
                "ī",
                "ő",
                "Œ",
                "ŵ",
                "ƙ",
                "ˆ",
                "ˈ",
                "̓",
                "Ї",
                "ї",
                "ḙ",
                "Ṃ",
                "ἀ",
                "‘",
                "’",
                "“",
                "”",
                "‰",
                "›",
            ]
        else:
            with open(self.use_customer_dictionary, "rb") as fp:
                self.CTLABELS = pickle.load(fp)
        # voc_size includes the unknown class, which is not in self.CTABLES
        assert int(self.voc_size - 1) == len(
            self.CTLABELS
        ), f"voc_size is not matched dictionary size, got {int(self.voc_size - 1)} and {len(self.CTLABELS)}."

        # setup the predictor
        if "vitae" in cfg.MODEL.BACKBONE.NAME.lower():
            self.predictor = ViTAEPredictor(cfg)
        self.predictor = DefaultPredictor(cfg)

    def _ctc_decode_recognition(self, rec):
        last_char = "###"
        s = ""
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if (
                        self.voc_size == 37
                        or self.voc_size == 96
                        or self.voc_size == 148
                    ):
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            else:
                last_char = "###"
        return s
