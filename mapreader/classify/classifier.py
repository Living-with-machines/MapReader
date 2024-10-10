#!/usr/bin/env python
from __future__ import annotations

import copy
import os
import random
import socket
import sys
import time
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch import optim
from torch.utils.data import DataLoader, Sampler
from torchinfo import summary
from torchvision import models

from .datasets import PatchDataset


class ClassifierContainer:
    """
    A class to store and train a PyTorch model.

    Parameters
    ----------
    model : str, nn.Module or None
        The PyTorch model to add to the object.

        - If passed as a string, will run ``_initialize_model(model, **kwargs)``. See https://pytorch.org/vision/0.8/models.html for options.
        - Must be ``None`` if ``load_path`` is specified as model will be loaded from file.

    labels_map: Dict or None
        A dictionary containing the mapping of each label index to its label, with indices as keys and labels as values (i.e. idx: label).
        Can only be ``None`` if ``load_path`` is specified as labels_map will be loaded from file.
    dataloaders: Dict or None
        A dictionary containing set names as keys and dataloaders as values (i.e. set_name: dataloader).
    device : str, optional
        The device to be used for training and storing models.
        Can be set to "default", "cpu", "cuda:0", etc. By default, "default".
    input_size : int, optional
        The expected input size of the model. Default is ``(224,224)``.
    is_inception : bool, optional
        Whether the model is an Inception-style model.
        Default is ``False``.
    load_path : str, optional
        The path to an ``.obj`` file containing a
    force_device : bool, optional
        Whether to force the use of a specific device.
        If set to ``True``, the default device is used.
        Defaults to ``False``.
    kwargs : Dict
        Keyword arguments to pass to the
        :meth:`~.classify.classifier.ClassifierContainer._initialize_model`
        method (if passing ``model`` as a string).

    Attributes
    ----------
    device : torch.device
        The device being used for training and storing models.
    dataloaders : dict
        A dictionary to store dataloaders for the model.
    labels_map : dict
        A dictionary mapping label indices to their labels.
    dataset_sizes : dict
        A dictionary to store sizes of datasets for the model.
    model : torch.nn.Module
        The model.
    input_size : Tuple of int
        The size of the input to the model.
    is_inception : bool
        A flag indicating if the model is an Inception model.
    optimizer : None or torch.optim.Optimizer
        The optimizer being used for training the model.
    scheduler : None or torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler being used for training the model.
    loss_fn : None or nn.modules.loss._Loss
        The loss function to use for training the model.
    metrics : dict
        A dictionary to store the metrics computed during training.
    last_epoch : int
        The last epoch number completed during training.
    best_loss : torch.Tensor
        The best validation loss achieved during training.
    best_epoch : int
        The epoch in which the best validation loss was achieved during
        training.
    tmp_save_filename : str
        A temporary file name to save checkpoints during training and
        validation.
    """

    def __init__(
        self,
        model: str | nn.Module | None = None,
        labels_map: dict[int, str] | None = None,
        dataloaders: dict[str, DataLoader] | None = None,
        device: str = "default",
        input_size: int = (224, 224),
        is_inception: bool = False,
        load_path: str | None = None,
        force_device: bool = False,
        **kwargs,
    ):
        # set up device
        if device == "default":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"[INFO] Device is set to {self.device}")

        # check if loading an pre-existing object
        if load_path:
            if model:
                print("[WARNING] Ignoring ``model`` as ``load_path`` is specified.")
            if labels_map:
                print(
                    "[WARNING] Ignoring ``labels_map`` as ``load_path`` is specified."
                )

            # load object
            self.load(load_path=load_path, force_device=force_device)

            # add any extra dataloaders
            if dataloaders:
                for set_name, dataloader in dataloaders.items():
                    self.dataloaders[set_name] = dataloader

        else:
            if model is None or labels_map is None:
                raise ValueError(
                    "[ERROR] ``model`` and ``labels_map`` must be defined."
                )

            self.labels_map = labels_map

            # set up model and move to device
            print("[INFO] Initializing model.")
            if isinstance(model, nn.Module):
                self.model = model.to(self.device)
                self.input_size = input_size
                self.is_inception = is_inception
            elif isinstance(model, str):
                self._initialize_model(model, **kwargs)

            self.optimizer = None
            self.scheduler = None
            self.loss_fn = None

            self.metrics = {}
            self.last_epoch = 0
            self.best_loss = torch.tensor(np.inf)
            self.best_epoch = 0

            # temp file to save checkpoints during training/validation
            if not os.path.exists("./tmp_checkpoints"):
                os.makedirs("./tmp_checkpoints")
            self.tmp_save_filename = (
                f"./tmp_checkpoints/tmp_{random.randint(0, int(1e10))}_checkpoint.pkl"
            )

            # add colors for printing/logging
            self._set_up_print_colors()

            # add dataloaders and labels_map
            self.dataloaders = dataloaders if dataloaders else {}

        for set_name, dataloader in self.dataloaders.items():
            print(f'[INFO] Loaded "{set_name}" with {len(dataloader.dataset)} items.')

    def _initialize_model(
        self,
        model_name: str,
        weights: str | None = "DEFAULT",
        last_layer_num_classes: str | int | None = "default",
    ) -> tuple[Any, int, bool]:
        """
        Initializes a PyTorch model with the option to change the number of
        classes in the last layer (``last_layer_num_classes``).

        Parameters
        ----------
        model_name : str
            Name of a PyTorch model. See https://pytorch.org/vision/0.8/models.html for options.
        weights : str, optional
            Weights to load into the model. If ``"DEFAULT"``, loads the default weights for the chosen model.
            By default, ``"DEFAULT"``.
        last_layer_num_classes : str or int, optional
            Number of elements in the last layer. If ``"default"``, sets it to
            the number of classes. By default, ``"default"``.

        Returns
        -------
        model : PyTorch model
            The initialized PyTorch model with the changed last layer.
        input_size : int
            Input size of the model.
        is_inception : bool
            True if the model is Inception v3.

        Raises
        ------
        ValueError
            If an invalid model name is passed.

        Notes
        -----
        Inception v3 requires the input size to be ``(299, 299)``, whereas all
        of the other models expect ``(224, 224)``.

        See https://pytorch.org/vision/0.8/models.html.
        """

        # Initialize these variables which will be set in this if statement.
        # Each of these variables is model specific.
        model_dw = models.get_model(model_name, weights=weights)
        input_size = (224, 224)
        is_inception = False

        if last_layer_num_classes in ["default"]:
            last_layer_num_classes = len(self.labels_map)
        else:
            last_layer_num_classes = int(last_layer_num_classes)

        if "resnet" in model_name:
            num_ftrs = model_dw.fc.in_features
            model_dw.fc = nn.Linear(num_ftrs, last_layer_num_classes)

        elif "alexnet" in model_name:
            num_ftrs = model_dw.classifier[6].in_features
            model_dw.classifier[6] = nn.Linear(num_ftrs, last_layer_num_classes)

        elif "vgg" in model_name:
            # vgg11_bn
            num_ftrs = model_dw.classifier[6].in_features
            model_dw.classifier[6] = nn.Linear(num_ftrs, last_layer_num_classes)

        elif "squeezenet" in model_name:
            model_dw.classifier[1] = nn.Conv2d(
                512, last_layer_num_classes, kernel_size=(1, 1), stride=(1, 1)
            )
            model_dw.num_classes = last_layer_num_classes

        elif "densenet" in model_name:
            num_ftrs = model_dw.classifier.in_features
            model_dw.classifier = nn.Linear(num_ftrs, last_layer_num_classes)

        elif "inception" in model_name:
            # Inception v3:
            # Be careful, expects (299,299) sized images + has auxiliary output

            # Handle the auxilary net
            num_ftrs = model_dw.AuxLogits.fc.in_features
            model_dw.AuxLogits.fc = nn.Linear(num_ftrs, last_layer_num_classes)
            # Handle the primary net
            num_ftrs = model_dw.fc.in_features
            model_dw.fc = nn.Linear(num_ftrs, last_layer_num_classes)
            is_inception = True
            input_size = 299

        else:
            raise NotImplementedError(
                "[ERROR] Invalid model name. Try loading your model directly and this as the `model` argument instead."
            )

        self.model = model_dw.to(self.device)
        self.input_size = input_size
        self.is_inception = is_inception

    def generate_layerwise_lrs(
        self,
        min_lr: float,
        max_lr: float,
        spacing: str | None = "linspace",
    ) -> list[dict]:
        """
        Calculates layer-wise learning rates for a given set of model
        parameters.

        Parameters
        ----------
        min_lr : float
            The minimum learning rate to be used.
        max_lr : float
            The maximum learning rate to be used.
        spacing : str, optional
            The type of sequence to use for spacing the specified interval
            learning rates. Can be either ``"linspace"`` or ``"geomspace"``,
            where `"linspace"` uses evenly spaced learning rates over a
            specified interval and `"geomspace"` uses learning rates spaced
            evenly on a log scale (a geometric progression). By default ``"linspace"``.

        Returns
        -------
        list of dicts
            A list of dictionaries containing the parameters and learning
            rates for each layer.
        """

        if spacing.lower() not in ["linspace", "geomspace"]:
            raise NotImplementedError(
                '[ERROR] ``spacing`` must be one of "linspace" or "geomspace"'
            )

        if spacing.lower() == "linspace":
            lrs = np.linspace(min_lr, max_lr, len(list(self.model.named_parameters())))
        elif spacing.lower() in ["log", "geomspace"]:
            lrs = np.geomspace(min_lr, max_lr, len(list(self.model.named_parameters())))
        params2optimize = [
            {"params": params, "learning rate": lr}
            for (_, params), lr in zip(self.model.named_parameters(), lrs)
        ]

        return params2optimize

    def initialize_optimizer(
        self,
        optim_type: str = "adam",
        params2optimize: str | Iterable = "default",
        optim_param_dict: dict | None = None,
    ):
        """
        Initializes an optimizer for the model and adds it to the classifier
        object.

        Parameters
        ----------
        optim_type : str, optional
            The type of optimizer to use. Can be set to ``"adam"`` (default),
            ``"adamw"``, or ``"sgd"``.
        params2optimize : str or iterable, optional
            The parameters to optimize. If set to ``"default"``, all model
            parameters that require gradients will be optimized.
            Default is ``"default"``.
        optim_param_dict : dict, optional
            The parameters to pass to the optimizer constructor as a
            dictionary, by default ``{"lr": 1e-3}``.

        Notes
        -----
        Note that the first argument of an optimizer is parameters to optimize,
        e.g. ``params2optimize = model_ft.parameters()``:

        - ``model_ft.parameters()``: all parameters are being optimized
        - ``model_ft.fc.parameters()``: only parameters of final layer are being optimized

        Here, we use:

        .. code-block:: python

            filter(lambda p: p.requires_grad, self.model.parameters())
        """
        if optim_param_dict is None:
            optim_param_dict = {"lr": 0.001}
        if params2optimize == "default":
            params2optimize = filter(lambda p: p.requires_grad, self.model.parameters())

        if optim_type.lower() in ["adam"]:
            optimizer = optim.Adam(params2optimize, **optim_param_dict)
        elif optim_type.lower() in ["adamw"]:
            optimizer = optim.AdamW(params2optimize, **optim_param_dict)
        elif optim_type.lower() in ["sgd"]:
            optimizer = optim.SGD(params2optimize, **optim_param_dict)
        else:
            raise NotImplementedError(
                '[ERROR] At present, only Adam ("adam"), AdamW ("adamw") and SGD ("sgd") are options for ``optim_type``.'
            )

        self.add_optimizer(optimizer)

    def add_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Add an optimizer to the classifier object.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to add to the classifier object.

        Returns
        -------
        None
        """
        self.optimizer = optimizer

    def initialize_scheduler(
        self,
        scheduler_type: str = "steplr",
        scheduler_param_dict: dict | None = None,
    ):
        """
        Initializes a learning rate scheduler for the optimizer and adds it to
        the classifier object.
        Only `StepLR` is implemented - otherwise use `torch.optim.lr_scheduler` directly and the `add_scheduler` method.

        Parameters
        ----------
        scheduler_type : str, optional
            The type of learning rate scheduler to use. Default is ``"steplr"``.
        scheduler_param_dict : dict, optional
            The parameters to pass to the scheduler constructor, by default
            ``{"step_size": 10, "gamma": 0.1}``.

        Raises
        ------
        ValueError
            If the specified ``scheduler_type`` is not implemented.
        """
        if self.optimizer is None:
            raise ValueError(
                "[ERROR] Optimizer is not yet defined. \n\n\
Use ``initialize_optimizer`` or ``add_optimizer`` to define one."  # noqa
            )

        if scheduler_type.lower() == "steplr":
            if scheduler_param_dict is None:
                scheduler_param_dict = {"step_size": 10, "gamma": 0.1}
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, **scheduler_param_dict
            )
        else:
            raise NotImplementedError(
                '[ERROR] At present, only StepLR ("steplr") is implemented. \n\n\
Use ``torch.optim.lr_scheduler`` directly and then the ``add_scheduler`` method for other schedulers.'  # noqa
            )

        self.add_scheduler(scheduler)

    def add_scheduler(self, scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        """
        Add a scheduler to the classifier object.
        Note that during training, `scheduler.step()` is called after each epoch - i.e. do not use schedulers that should be stepped after each batch!

        Parameters
        ----------
        scheduler : torch.optim.lr_scheduler._LRScheduler
            The scheduler to add to the classifier object.

        Raises
        ------
        ValueError
            If no optimizer has been set. Use ``initialize_optimizer`` or
            ``add_optimizer`` to set an optimizer first.

        Returns
        -------
        None
        """
        if self.optimizer is None:
            raise ValueError(
                "[ERROR] Optimizer is needed first. Use `initialize_optimizer` or `add_optimizer`"  # noqa
            )

        self.scheduler = scheduler

    def add_loss_fn(
        self, loss_fn: str | nn.modules.loss._Loss = "cross entropy"
    ) -> None:
        """
        Add a loss function to the classifier object.

        Parameters
        ----------
        loss_fn : str or torch.nn.modules.loss._Loss
            The loss function to add to the classifier object.
            Accepted string values are "cross entropy" or "ce" (cross-entropy), "bce" (binary cross-entropy) and "mse" (mean squared error).

        Returns
        -------
        None
            The function only modifies the ``loss_fn`` attribute of the
            classifier and does not return anything.
        """
        if isinstance(loss_fn, str):
            if loss_fn in ["cross entropy", "ce", "cross_entropy", "cross-entropy"]:
                loss_fn = nn.CrossEntropyLoss()
            elif loss_fn in [
                "bce",
                "binary_cross_entropy",
                "binary cross entropy",
                "binary cross-entropy",
            ]:
                loss_fn = nn.BCELoss()
            elif loss_fn in [
                "mse",
                "mean_square_error",
                "mean_squared_error",
                "mean squared error",
            ]:
                loss_fn = nn.MSELoss()
            else:
                raise NotImplementedError(
                    '[ERROR] At present, if passing ``loss_fn`` as a string, the loss function can only be "cross entropy" or "ce" (cross-entropy), "bce" (binary cross-entropy) or "mse" (mean squared error).'
                )

            print(f'[INFO] Using "{loss_fn}" as loss function.')

        elif not isinstance(loss_fn, nn.modules.loss._Loss):
            raise ValueError(
                '[ERROR] Please pass ``loss_fn`` as a string ("cross entropy", "bce" or "mse") or torch.nn loss function (see https://pytorch.org/docs/stable/nn.html).'
            )

        self.loss_fn = loss_fn

    def model_summary(
        self,
        input_size: tuple | list | None = None,
        trainable_col: bool = False,
        **kwargs,
    ) -> None:
        """
        Print a summary of the model.

        Parameters
        ----------
        input_size : tuple or list, optional
            The size of the input data.
            If None, input size is taken from "train" dataloader (``self.dataloaders["train"]``).
        trainable_col : bool, optional
            If ``True``, adds a column showing which parameters are trainable.
            Defaults to ``False``.
        **kwargs : Dict
            Keyword arguments to pass to ``torchinfo.summary()`` (see https://github.com/TylerYep/torchinfo).

        Notes
        -----
        Other ways to check params:

        .. code-block:: python

            sum(p.numel() for p in myclassifier.model.parameters())

        .. code-block:: python

            sum(p.numel() for p in myclassifier.model.parameters()
                if p.requires_grad)

        And:

        .. code-block:: python

            for name, param in self.model.named_parameters():
                n = name.split(".")[0].split("_")[0]
                print(name, param.requires_grad)
        """
        if not input_size:
            if "train" in self.dataloaders.keys():
                batch_size = self.dataloaders["train"].batch_size
                channels = len(self.dataloaders["train"].dataset.image_mode)
                input_size = (batch_size, channels, *self.input_size)
            else:
                raise ValueError("[ERROR] Please pass an input size.")

        if trainable_col:
            col_names = ["num_params", "output_size", "trainable"]
        else:
            col_names = ["output_size", "output_size", "num_params"]

        model_summary = summary(
            self.model, input_size=input_size, col_names=col_names, **kwargs
        )
        print(model_summary)

    def freeze_layers(self, layers_to_freeze: list[str]) -> None:
        """
        Freezes the specified layers in the neural network by setting
        ``requires_grad`` attribute to False for their parameters.

        Parameters
        ----------
        layers_to_freeze : list of str
            List of names of the layers to freeze.
            If a layer name ends with an asterisk (``"*"``), then all parameters whose name contains the layer name (excluding the asterisk) are frozen. Otherwise, only the parameters with an exact match to the layer name are frozen.

        Returns
        -------
        None
            The function only modifies the ``requires_grad`` attribute of the
            specified parameters and does not return anything.

        Notes
        -----
        e.g. ["layer1*", "layer2*"] will freeze all parameters whose name contains "layer1" and "layer2" (excluding the asterisk).
        e.g. ["layer1", "layer2"] will freeze all parameters with an exact match to "layer1" and "layer2".
        """
        if not isinstance(layers_to_freeze, list):
            raise ValueError(
                '[ERROR] ``layers_to_freeze`` must be a list of strings. E.g. ["layer1*", "layer2*"].'
            )

        for layer in layers_to_freeze:
            for name, param in self.model.named_parameters():
                if (layer.endswith("*")) and (
                    layer.strip("*") in name
                ):  # if using asterix wildcard
                    param.requires_grad = False
                elif (not layer.endswith("*")) and (
                    layer == name
                ):  # if using exact match
                    param.requires_grad = False

    def unfreeze_layers(self, layers_to_unfreeze: list[str]):
        """
        Unfreezes the specified layers in the neural network by setting
        ``requires_grad`` attribute to True for their parameters.

        Parameters
        ----------
        layers_to_unfreeze : list of str
            List of names of the layers to unfreeze.
            If a layer name ends with an asterisk (``"*"``), then all parameters whose name contains the layer name (excluding the asterisk) are unfrozen. Otherwise, only the parameters with an exact match to the layer name are unfrozen.

        Returns
        -------
        None
            The function only modifies the ``requires_grad`` attribute of the
            specified parameters and does not return anything.

        Notes
        -----
        e.g. ["layer1*", "layer2*"] will unfreeze all parameters whose name contains "layer1" and "layer2" (excluding the asterisk).
        e.g. ["layer1", "layer2"] will unfreeze all parameters with an exact match to "layer1" and "layer2".
        """

        if not isinstance(layers_to_unfreeze, list):
            raise ValueError(
                '[ERROR] ``layers_to_unfreeze`` must be a list of strings. E.g. ["layer1*", "layer2*"].'
            )

        for layer in layers_to_unfreeze:
            for name, param in self.model.named_parameters():
                if (layer.endswith("*")) and (layer.strip("*") in name):
                    param.requires_grad = True
                elif (not layer.endswith("*")) and (layer == name):
                    param.requires_grad = True

    def only_keep_layers(self, only_keep_layers_list: list[str]) -> None:
        """
        Only keep the specified layers (``only_keep_layers_list``) for
        gradient computation during the backpropagation.

        Parameters
        ----------
        only_keep_layers_list : list
            List of layer names to keep. All other layers will have their
            gradient computation turned off.

        Returns
        -------
        None
            The function only modifies the ``requires_grad`` attribute of the
            specified parameters and does not return anything.
        """
        if not isinstance(only_keep_layers_list, list):
            raise ValueError(
                '[ERROR] ``only_keep_layers_list`` must be a list of strings. E.g. ["layer1", "layer2"].'
            )

        for name, param in self.model.named_parameters():
            if name in only_keep_layers_list:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def inference(
        self,
        set_name: str = "infer",
        verbose: bool = False,
        print_info_batch_freq: int = 5,
    ):
        """
        Run inference on a specified dataset (``set_name``).

        Parameters
        ----------
        set_name : str, optional
            The name of the dataset to run inference on, by default
            ``"infer"``.
        verbose : bool, optional
            Whether to print verbose outputs, by default False.
        print_info_batch_freq : int, optional
            The frequency of printouts, by default ``5``.

        Returns
        -------
        None

        Notes
        -----
        This method calls the
        :meth:`~.train.classifier.classifier.train` method with the
        ``num_epochs`` set to ``1`` and all the other parameters specified in
        the function arguments.
        """
        self.train(
            phases=[set_name],
            num_epochs=1,
            save_model_dir=None,
            verbose=verbose,
            tensorboard_path=None,
            tmp_file_save_freq=2,
            remove_after_load=False,
            print_info_batch_freq=print_info_batch_freq,
        )

    def train_component_summary(self) -> None:
        """
        Print a summary of the optimizer, loss function, and trainable model
        components.

        Returns:
        --------
        None
        """
        divider = 20 * "="
        print(divider)
        print("* Optimizer:")
        print(str(self.optimizer))
        print(divider)
        print("* Loss function:")
        print(str(self.loss_fn))
        print(divider)
        print("* Model:")
        self.model_summary(trainable_col=True)

    def train(
        self,
        phases: list[str] | None = None,
        num_epochs: int = 25,
        save_model_dir: str | None = "models",
        verbose: bool = False,
        tensorboard_path: str | None = None,
        tmp_file_save_freq: int | None = 2,
        remove_after_load: bool = True,
        print_info_batch_freq: int | None = 5,
    ) -> None:
        """
        Train the model on the specified phases for a given number of epochs.

        Wrapper function for
        :meth:`~.train.classifier.classifier.train_core` method to
        capture exceptions (``KeyboardInterrupt`` is the only supported
        exception currently).

        Parameters
        ----------
        phases : list of str, optional
            The phases to run through during each training iteration. Default is
            ``["train", "val"]``.
        num_epochs : int, optional
            The number of epochs to train the model for. Default is ``25``.
        save_model_dir : str or None, optional
            The directory to save the model in. Default is ``"models"``. If
            set to ``None``, the model is not saved.
        verbose : int, optional
            Whether to print verbose outputs, by default ``False``.
        tensorboard_path : str or None, optional
            The path to the directory to save TensorBoard logs in. If set to
            ``None``, no TensorBoard logs are saved. Default is ``None``.
        tmp_file_save_freq : int, optional
            The frequency (in epochs) to save a temporary file of the model.
            Default is ``2``. If set to ``0`` or ``None``, no temporary file
            is saved.
        remove_after_load : bool, optional
            Whether to remove the temporary file after loading it. Default is
            ``True``.
        print_info_batch_freq : int, optional
            The frequency (in batches) to print training information. Default
            is ``5``. If set to ``0`` or ``None``, no training information is
            printed.

        Returns
        -------
        None
            The function saves the model to the ``save_model_dir`` directory,
            and optionally to a temporary file. If interrupted with a
            ``KeyboardInterrupt``, the function tries to load the temporary
            file. If no temporary file is found, it continues without loading.

        Notes
        -----
        Refer to the documentation of
        :meth:`~.train.classifier.classifier.train_core` for more
        information.
        """

        if phases is None:
            phases = ["train", "val"]

        try:
            self.train_core(
                phases,
                num_epochs,
                save_model_dir,
                verbose,
                tensorboard_path,
                tmp_file_save_freq,
                print_info_batch_freq=print_info_batch_freq,
            )
        except KeyboardInterrupt:
            print("[INFO] Exiting...")
            if os.path.isfile(self.tmp_save_filename):
                print(f'[INFO] Loading "{self.tmp_save_filename}" as model.')
                self.load(self.tmp_save_filename, remove_after_load=remove_after_load)
            else:
                print("[INFO] No checkpoint file found - model has not been updated.")

    def train_core(
        self,
        phases: list[str] | None = None,
        num_epochs: int | None = 25,
        save_model_dir: str | None | None = "models",
        verbose: bool = False,
        tensorboard_path: str | None | None = None,
        tmp_file_save_freq: int | None | None = 2,
        print_info_batch_freq: int | None | None = 5,
    ) -> None:
        """
        Trains/fine-tunes a classifier for the specified number of epochs on
        the given phases using the specified hyperparameters.

        Parameters
        ----------
        phases : list of str, optional
            The phases to run through during each training iteration. Default is
            ``["train", "val"]``.
        num_epochs : int, optional
            The number of epochs to train the model for. Default is ``25``.
        save_model_dir : str or None, optional
            The directory to save the model in. Default is ``"models"``. If
            set to ``None``, the model is not saved.
        verbose : bool, optional
            Whether to print verbose outputs, by default ``False``.
        tensorboard_path : str or None, optional
            The path to the directory to save TensorBoard logs in. If set to
            ``None``, no TensorBoard logs are saved. Default is ``None``.
        tmp_file_save_freq : int, optional
            The frequency (in epochs) to save a temporary file of the model.
            Default is ``2``. If set to ``0`` or ``None``, no temporary file
            is saved.
        print_info_batch_freq : int, optional
            The frequency (in batches) to print training information. Default
            is ``5``. If set to ``0`` or ``None``, no training information is
            printed.

        Raises
        ------
        ValueError
            If the loss function is not set. Use the
            :meth:`~.classify.classifier.ClassifierContainer.add_loss_fn`
            method to set the loss function.

            If the optimizer is not set and the phase is "train". Use the
            :meth:`~.classify.classifier.ClassifierContainer.initialize_optimizer`
            or :meth:`~.classify.classifier.ClassifierContainer.add_optimizer`
            method to set the optimizer.

        KeyError
            If the specified phase cannot be found in the keys of the object's
            :attr:`~.classify.classifier.ClassifierContainer.dataloaders`
            dictionary property.

        Returns
        -------
        None
        """

        if phases is None:
            phases = ["train", "val"]
        print(f"[INFO] Each step will pass: {phases}.")

        for phase in phases:
            if phase not in self.dataloaders.keys():
                raise KeyError(
                    f'[ERROR] "{phase}" dataloader cannot be found in dataloaders.\n\
    Valid options for ``phases`` argument are: {self.dataloaders.keys()}'  # noqa
                )

        if verbose:
            self.train_component_summary()

        since = time.time()

        # initialize variables
        train_phase_names = ["train", "training"]
        valid_phase_names = ["val", "validation", "eval", "evaluation"]
        best_model_wts = copy.deepcopy(self.model.state_dict())
        self.pred_conf = []
        self.pred_label_indices = []
        self.gt_label_indices = []
        if save_model_dir is not None:
            save_model_dir = os.path.abspath(save_model_dir)

        # Check if SummaryWriter (for tensorboard) can be imported
        tboard_writer = None
        if tensorboard_path is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tboard_writer = SummaryWriter(tensorboard_path)
            except ImportError:
                print(
                    "[WARNING] Could not import ``SummaryWriter`` from torch.utils.tensorboard"  # noqa
                )
                print("[WARNING] Continuing without tensorboard.")
                tensorboard_path = None

        start_epoch = self.last_epoch + 1
        end_epoch = self.last_epoch + num_epochs

        # --- Main train loop
        for epoch in range(start_epoch, end_epoch + 1):
            # --- loop, phases
            for phase in phases:
                if phase.lower() in train_phase_names:
                    self.model.train()

                    # check if optimizer, scheduler and loss_fn are defined
                    if self.optimizer is None:
                        raise ValueError(
                            "[ERROR] An optimizer should be defined for training the model."
                        )
                    if self.scheduler is None:
                        raise ValueError(
                            "[ERROR] A scheduler should be defined for training the model."
                        )
                    if self.loss_fn is None:
                        raise ValueError(
                            "[ERROR] A loss function should be defined for training the model."
                        )
                else:
                    self.model.eval()

                # initialize vars with one epoch lifetime
                running_loss = 0.0
                running_pred_conf = []
                running_pred_label_indices = []
                running_gt_label_indices = []

                phase_batch_size = self.dataloaders[phase].batch_size
                total_input_counts = len(self.dataloaders[phase].dataset)

                # --- loop, batches
                for batch_idx, (inputs, _, gt_label_indices) in enumerate(
                    self.dataloaders[phase]
                ):
                    inputs = tuple(input.to(self.device) for input in inputs)
                    gt_label_indices = gt_label_indices.to(self.device)

                    if self.optimizer:  # only if optimizer is defined
                        self.optimizer.zero_grad()

                    if phase.lower() in train_phase_names + valid_phase_names:
                        # forward, track history if only in train
                        with torch.set_grad_enabled(phase.lower() in train_phase_names):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training,
                            # it has an auxiliary output.
                            #     In train mode we calculate the loss by
                            #     summing the final output and the auxiliary
                            #     output but in testing we only consider the
                            #     final output.
                            if self.is_inception and (
                                phase.lower() in train_phase_names
                            ):
                                outputs, aux_outputs = self.model(*inputs)

                                if not isinstance(outputs, torch.Tensor):
                                    outputs = self._get_logits(outputs)
                                if not isinstance(aux_outputs, torch.Tensor):
                                    aux_outputs = self._get_logits(aux_outputs)

                                loss1 = self.loss_fn(outputs, gt_label_indices)
                                loss2 = self.loss_fn(aux_outputs, gt_label_indices)
                                # https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                loss = loss1 + 0.4 * loss2  # calculate loss

                            else:
                                outputs = self.model(*inputs)

                                if not isinstance(outputs, torch.Tensor):
                                    outputs = self._get_logits(outputs)

                                loss = self.loss_fn(
                                    outputs, gt_label_indices
                                )  # calculate loss

                            _, pred_label_indices = torch.max(outputs, dim=1)

                            # backward + optimize only if in training phase
                            if phase.lower() in train_phase_names:
                                loss.backward()
                                self.optimizer.step()

                        # XXX (why multiply?)
                        running_loss += loss.item() * inputs[0].size(0)

                    else:  # if not in train or valid phase
                        outputs = self.model(*inputs)

                        if not isinstance(outputs, torch.Tensor):
                            outputs = self._get_logits(outputs)

                        _, pred_label_indices = torch.max(outputs, dim=1)

                    running_pred_conf.extend(
                        torch.nn.functional.softmax(outputs, dim=1).cpu().tolist()
                    )
                    running_pred_label_indices.extend(pred_label_indices.cpu().tolist())
                    running_gt_label_indices.extend(gt_label_indices.cpu().tolist())

                    if batch_idx % print_info_batch_freq == 0:
                        current_input_counts = min(
                            total_input_counts,
                            (batch_idx + 1) * phase_batch_size,
                        )
                        progress = current_input_counts / total_input_counts * 100.0
                        progress_msg = f"{current_input_counts}/{total_input_counts} ({progress:5.1f}% )"  # noqa

                        epoch_msg = f"{phase: <8} -- {epoch}/{end_epoch} -- "
                        epoch_msg += f"{progress_msg: >20} -- "

                        if phase.lower() in valid_phase_names:
                            epoch_msg += f"Loss: {loss.data:.3f}"
                            self.cprint("[INFO]", "dred", epoch_msg)
                        elif phase.lower() in train_phase_names:
                            epoch_msg += f"Loss: {loss.data:.3f}"
                            self.cprint("[INFO]", "dgreen", epoch_msg)
                        else:
                            self.cprint("[INFO]", "dgreen", epoch_msg)
                    # --- END: one batch

                # scheduler, step after each epoch
                if phase.lower() in train_phase_names and (self.scheduler is not None):
                    self.scheduler.step()

                if phase.lower() in train_phase_names + valid_phase_names:
                    # --- collect statistics
                    epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                    self._add_metrics(phase, "loss", epoch_loss)

                    if tboard_writer is not None:
                        tboard_writer.add_scalar(
                            f"loss/{phase}",
                            epoch_loss,
                            epoch,
                        )

                    # other metrics (precision/recall/F1)
                    self.calculate_add_metrics(
                        running_gt_label_indices,
                        running_pred_label_indices,
                        running_pred_conf,
                        phase,
                        epoch,
                        tboard_writer,
                    )

                    epoch_msg = f"{phase: <8} -- {epoch}/{end_epoch} -- "
                    epoch_msg = self._gen_epoch_msg(phase, epoch_msg)

                    if phase.lower() in valid_phase_names:
                        self.cprint("[INFO]", "dred", epoch_msg + "\n")
                    else:
                        self.cprint("[INFO]", "dgreen", epoch_msg)

                # labels/confidence
                self.pred_conf.extend(running_pred_conf)
                self.pred_label_indices.extend(running_pred_label_indices)
                self.gt_label_indices.extend(running_gt_label_indices)

                # Update best_loss and _epoch?
                if phase.lower() in valid_phase_names and epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                if phase.lower() in valid_phase_names:
                    if epoch % tmp_file_save_freq == 0:
                        tmp_str = f'[INFO] Checkpoint file saved to "{self.tmp_save_filename}".'  # noqa
                        print(
                            self._print_colors["lgrey"]
                            + tmp_str
                            + self._print_colors["reset"]
                        )
                        self.last_epoch = epoch
                        self.save(self.tmp_save_filename, force=True)

        self.pred_label = [
            self.labels_map.get(i, None) for i in self.pred_label_indices
        ]

        time_elapsed = time.time() - since
        print(f"[INFO] Total time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        # --- SAVE model/object
        if phase.lower() in train_phase_names + valid_phase_names:
            self.last_epoch = epoch
            if save_model_dir is not None:
                save_filename = f"checkpoint_{self.best_epoch}.pkl"
                save_model_path = os.path.join(save_model_dir, save_filename)
                self.save(save_model_path, force=True)
                with open(os.path.join(save_model_dir, "info.txt"), "a+") as fio:
                    fio.writelines(f"{save_filename},{self.best_loss:.5f}\n")

                print(
                    f"[INFO] Model at epoch {self.best_epoch} has least valid loss ({self.best_loss:.4f}) so will be saved.\n\
[INFO] Path: {save_model_path}"
                )

    @staticmethod
    def _get_logits(out):
        try:
            out = out.logits
        except AttributeError as err:
            raise AttributeError(err.message)
        return out

    def _gen_epoch_msg(self, phase: str, epoch_msg: str) -> str:
        """
        Generates a log message for an epoch during training or validation.
        The message includes information about the loss, F-score, and recall
        for a given phase (training or validation).

        Parameters
        ----------
        phase : str
            The training phase, either ``"train"`` or ``"val"``.
        epoch_msg : str
            The message string to be modified with the epoch metrics.

        Returns
        -------
        epoch_msg : str
            The updated message string with the epoch metrics.
        """
        loss = self.metrics[phase]["loss"][-1]
        epoch_msg += f"Loss: {loss:.3f}; "

        fscore = self.metrics[phase]["fscore_macro"][-1]
        epoch_msg += f"F_macro: {fscore:.2f}; "

        recall = self.metrics[phase]["recall_macro"][-1]
        epoch_msg += f"R_macro: {recall:.2f}"

        return epoch_msg

    def calculate_add_metrics(
        self,
        y_true,
        y_pred,
        y_score,
        phase: str,
        epoch: int | None = -1,
        tboard_writer=None,
    ) -> None:
        """
        Calculate and add metrics to the classifier's metrics dictionary.

        Parameters
        ----------
        y_true : 1d array-like of shape (n_samples,)
            True binary labels or multiclass labels. Can be considered ground
            truth or (correct) target values.

        y_pred : 1d array-like of shape (n_samples,)
            Predicted binary labels or multiclass labels. The estimated
            targets as returned by a classifier.

        y_score : array-like of shape (n_samples, n_classes)
            Predicted probabilities for each class.

        phase : str
            Name of the current phase, typically ``"train"`` or ``"val"``. See
            ``train`` function.

        epoch : int, optional
            Current epoch number. Default is ``-1``.

        tboard_writer : object, optional
            TensorBoard SummaryWriter object to write the metrics. Default is
            ``None``.

        Returns
        -------
        None

        Notes
        -----
        This method uses both the
        :func:`sklearn.metrics.precision_recall_fscore_support` and
        :func:`sklearn.metrics.roc_auc_score` functions from ``scikit-learn``
        to calculate the metrics for each average type (``"micro"``,
        ``"macro"`` and ``"weighted"``). The results are then added to the
        ``metrics`` dictionary. It also writes the metrics to the TensorBoard
        SummaryWriter, if ``tboard_writer`` is not None.
        """
        # convert y_score to a numpy array:
        if not isinstance(y_score, np.ndarray):
            y_score = np.array(y_score)

        for average in [None, "micro", "macro", "weighted"]:
            precision, recall, fscore, support = precision_recall_fscore_support(
                y_true, y_pred, average=average
            )

            if average is None:
                for i in range(
                    y_score.shape[1]
                ):  # y_score.shape[1] represents the number of classes
                    self._add_metrics(phase, f"precision_{i}", precision[i])
                    self._add_metrics(phase, f"recall_{i}", recall[i])
                    self._add_metrics(phase, f"fscore_{i}", fscore[i])
                    self._add_metrics(phase, f"support_{i}", support[i])

                    if tboard_writer is not None:
                        tboard_writer.add_scalar(
                            f"Precision/{phase}/binary_{i}",
                            precision[i],
                            epoch,
                        )
                        tboard_writer.add_scalar(
                            f"Recall/{phase}/binary_{i}",
                            recall[i],
                            epoch,
                        )
                        tboard_writer.add_scalar(
                            f"Fscore/{phase}/binary_{i}",
                            fscore[i],
                            epoch,
                        )

            else:  # for micro, macro, weighted
                self._add_metrics(phase, f"precision_{average}", precision)
                self._add_metrics(phase, f"recall_{average}", recall)
                self._add_metrics(phase, f"fscore_{average}", fscore)
                self._add_metrics(phase, f"support_{average}", support)

                if tboard_writer is not None:
                    tboard_writer.add_scalar(
                        f"Precision/{phase}/{average}",
                        precision,
                        epoch,
                    )
                    tboard_writer.add_scalar(
                        f"Recall/{phase}/{average}",
                        recall,
                        epoch,
                    )
                    tboard_writer.add_scalar(
                        f"Fscore/{phase}/{average}",
                        fscore,
                        epoch,
                    )

                # --- compute ROC AUC
                if y_score.shape[1] == 2:
                    # ---- binary case
                    # From scikit-learn:
                    #     The probability estimates correspond to the probability
                    #     of the class with the greater label, i.e.
                    #     estimator.classes_[1] and thus
                    #     estimator.predict_proba(X, y)[:, 1]
                    roc_auc = roc_auc_score(y_true, y_score[:, 1], average=average)
                elif y_score.shape[1] > 2:
                    # ---- multiclass
                    # In the multiclass case, it corresponds to an array of shape
                    # (n_samples, n_classes)
                    # ovr = One-vs-rest (OvR) multiclass strategy
                    try:
                        roc_auc = roc_auc_score(
                            y_true, y_score, average=average, multi_class="ovr"
                        )
                    except:
                        continue
                else:
                    continue
                self._add_metrics(phase, f"rocauc_{average}", roc_auc)

    def _add_metrics(
        self, phase: str, metric: str, value: int | (float | (complex | np.number))
    ) -> None:
        """
        Adds a metric value to a dictionary of metrics tracked during training.

        Parameters
        ----------
        phase: str
            The phase of the training (e.g., "train", "val") to which the
            metric value corresponds.
        metric : str
            The name of the metric to add to the dictionary of metrics.
        value : numeric
            The metric value to add to the corresponding list of metric values.

        Returns
        -------
        None

        Notes
        -----
        If the key ``k`` does not exist in the dictionary of metrics, a new
        key-value pair is created with ``k`` as the key and a new list
        containing the value ``v`` as the value. If the key ``k`` already
        exists in the dictionary of metrics, the value `v` is appended to the
        list associated with the key ``k``.
        """
        if phase not in self.metrics.keys():
            self.metrics[phase] = {}
        if metric not in self.metrics[phase].keys():
            self.metrics[phase][metric] = []

        self.metrics[phase][metric].append(value)

    def plot_metric(
        self,
        metrics: str | list[str],
        phases: str | list[str] = "all",
        colors: list[str] | None = None,
        figsize: tuple[int, int] = (10, 5),
        plt_yrange: tuple[float, float] | None = None,
        plt_xrange: tuple[float, float] | None = None,
    ):
        """
        Plot the metrics of the classifier object.

        Parameters
        ----------
        metrics : str or list of str
            A string of list of strings containing metric names to be plotted on the y-axis.
        phases : str or list of str, optional
            The phases for which the metric is to be plotted. Defaults to ``"all"``.
        colors : list of str, optional
            Colors to be used for the lines of each metric. Length must be at least the length of the number of phases being plotted (``phases``). If None, will use the default matplotlib colors. Defaults to ``None``.
        figsize : tuple of int, optional
            The size of the figure in inches. Defaults to ``(10, 5)``.
        plt_yrange : tuple of float, optional
            The range of values for the y-axis. Defaults to ``None``.
        plt_xrange : tuple of float, optional
            The range of values for the x-axis. Defaults to ``None``.
        """
        plt.figure(figsize=figsize)

        if isinstance(metrics, str):
            metrics = [metrics]
        if not isinstance(metrics, list):
            raise TypeError("`metrics` must be a string or a list of strings.")

        # make list of colors iterable
        if colors:
            colors = iter(colors)

        for metric_name in metrics:
            if isinstance(phases, str):
                if phases == "all":
                    phases = [*self.metrics.keys()]
                else:
                    phases = [phases]
            if not isinstance(phases, list):
                raise TypeError("`phases` must be a string or a list of strings.")

            for phase in phases:
                if metric_name not in self.metrics[phase].keys():
                    raise KeyError(
                        f"Metric {metric_name} not found in {phase} metrics."
                    )

                # get color
                if colors:
                    color = next(colors)
                else:
                    color = plt.gca()._get_lines.get_next_color()

                plt.plot(
                    range(
                        1, len(self.metrics[phase][metric_name]) + 1
                    ),  # i.e. epochs, starting from 1
                    self.metrics[phase][metric_name],
                    label=f"{phase}_{metric_name}",
                    color=color,
                    linestyle="-",
                    marker="o",
                    linewidth=2,
                )

        # set labels and ticks
        plt.xlabel("Epochs", size=24)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel(" | ".join(metrics), size=24)
        plt.xticks(size=18)
        plt.yticks(size=18)

        # legend
        plt.legend(
            fontsize=18,
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            ncol=2,
            borderaxespad=0,
            loc="lower center",
        )

        # x/y range
        if plt_xrange is not None:
            plt.xlim(plt_xrange[0], plt_xrange[1])
        if plt_yrange is not None:
            plt.ylim(plt_yrange[0], plt_yrange[1])

        plt.grid()
        plt.show()

    def print_batch_info(self, set_name: str | None = "train") -> None:
        """
        Print information about a dataset's batches, samples, and batch-size.

        Parameters
        ----------
        set_name : str, optional
            Name of the dataset to display batch information for (default is
            ``"train"``).

        Returns
        -------
        None
        """
        if set_name not in self.dataloaders.keys():
            raise ValueError(
                f"[ERROR] ``set_name`` must be one of {list(self.dataloaders.keys())}."
            )

        batch_size = self.dataloaders[set_name].batch_size
        num_samples = len(self.dataloaders[set_name].dataset)
        num_batches = int(np.ceil(num_samples / batch_size))

        print(
            f"[INFO] dataset: {set_name}\n\
        - items:        {num_samples}\n\
        - batch size:   {batch_size}\n\
        - batches:      {num_batches}"
        )

    def show_inference_sample_results(
        self,
        label: str,
        num_samples: int | None = 6,
        set_name: str | None = "test",
        min_conf: None | float | None = None,
        max_conf: None | float | None = None,
        figsize: tuple[int, int] | None = (15, 15),
    ) -> None:
        """
        Shows a sample of the results of the inference with current model.

        Parameters
        ----------
        label : str, optional
            The label for which to display results.
        num_samples : int, optional
            The number of sample results to display. Defaults to ``6``.
        set_name : str, optional
            The name of the dataset split to use for inference. Defaults to
            ``"test"``.
        min_conf : float, optional
            The minimum confidence score for a sample result to be displayed.
            Samples with lower confidence scores will be skipped. Defaults to
            ``None``.
        max_conf : float, optional
            The maximum confidence score for a sample result to be displayed.
            Samples with higher confidence scores will be skipped. Defaults to
            ``None``.
        figsize : tuple[int, int], optional
            Figure size (width, height) in inches, displaying the sample
            results. Defaults to ``(15, 15)``.

        Returns
        -------
        None
        """

        # eval mode, keep track of the current mode
        was_training = self.model.training
        self.model.eval()

        if set_name not in self.dataloaders.keys():
            raise ValueError(
                f"[ERROR] ``set_name`` must be one of {list(self.dataloaders.keys())}."
            )

        counter = 0
        plt.figure(figsize=figsize)

        with torch.no_grad():
            for inputs, _, label_indices in iter(self.dataloaders[set_name]):
                inputs = tuple(input.to(self.device) for input in inputs)
                label_indices = label_indices.to(self.device)

                outputs = self.model(*inputs)

                if not isinstance(outputs, torch.Tensor):
                    outputs = self._get_logits(outputs)

                pred_conf = torch.nn.functional.softmax(outputs, dim=1) * 100.0
                _, preds = torch.max(outputs, 1)

                # reverse the labels_map dict
                label_index_dict = {v: k for k, v in self.labels_map.items()}

                # go through images in batch
                for i, pred in enumerate(preds):
                    predicted_index = int(pred)
                    if predicted_index != label_index_dict[label]:
                        continue
                    if (min_conf is not None) and (
                        pred_conf[i][predicted_index] < min_conf
                    ):
                        continue
                    if (max_conf is not None) and (
                        pred_conf[i][predicted_index] > max_conf
                    ):
                        continue

                    counter += 1

                    conf_score = pred_conf[i][predicted_index]
                    ax = plt.subplot(int(num_samples / 2.0), 3, counter)
                    ax.axis("off")
                    ax.set_title(f"{label} | {conf_score:.3f}")

                    inp = inputs[0].cpu().data[i].numpy().transpose((1, 2, 0))
                    inp = np.clip(inp, 0, 1)
                    plt.imshow(inp)

                    if counter == num_samples:
                        self.model.train(mode=was_training)
                        plt.show()
                        return

            self.model.train(mode=was_training)
            plt.show()

    def save(
        self,
        save_path: str | None = "default.obj",
        force: bool | None = False,
    ) -> None:
        """
        Save the object to a file.

        Parameters
        ----------
        save_path : str, optional
            The path to the file to write.
            If the file already exists and ``force`` is not ``True``, a ``FileExistsError`` is raised.
            Defaults to ``"default.obj"``.
        force : bool, optional
            Whether to overwrite the file if it already exists. Defaults to
            ``False``.

        Raises
        ------
        FileExistsError
            If the file already exists and ``force`` is not ``True``.

        Notes
        -----
        The object is saved in two parts. First, a serialized copy of the
        object's dictionary is written to the specified file using the
        ``joblib.dump`` function. The object's ``model`` attribute is excluded
        from this dictionary and saved separately using the ``torch.save``
        function, with a filename derived from the original ``save_path``.
        """
        if os.path.isfile(save_path):
            if force:
                os.remove(save_path)
            else:
                raise FileExistsError(f"[INFO] File already exists: {save_path}")

        # parent/base-names
        par_name = os.path.dirname(os.path.abspath(save_path))
        base_name = os.path.basename(os.path.abspath(save_path))

        # Extract model, write it separately using torch.save
        obj2write = copy.deepcopy(self.__dict__)
        mymodel = obj2write["model"]
        del obj2write["model"]

        os.makedirs(par_name, exist_ok=True)
        with open(save_path, "wb") as myfile:
            # pickle.dump(self.__dict__, myfile)
            joblib.dump(obj2write, myfile)

        torch.save(mymodel, os.path.join(par_name, f"model_{base_name}"))
        torch.save(
            mymodel.state_dict(),
            os.path.join(par_name, f"model_state_dict_{base_name}"),
        )

    def save_predictions(
        self,
        set_name: str,
        save_path: str | None = None,
        delimiter: str = ",",
    ):
        if set_name not in self.dataloaders.keys():
            raise ValueError(
                f"[ERROR] ``set_name`` must be one of {list(self.dataloaders.keys())}."
            )

        patch_df = self.dataloaders[set_name].dataset.patch_df
        patch_df["predicted_label"] = self.pred_label
        patch_df["pred"] = self.pred_label_indices
        patch_df["conf"] = np.array(self.pred_conf).max(axis=1)

        if save_path is None:
            save_path = f"{set_name}_predictions_patch_df.csv"
        patch_df.to_csv(save_path, sep=delimiter)
        print(f"[INFO] Saved predictions to {save_path}.")

    def load_dataset(
        self,
        dataset: PatchDataset,
        set_name: str,
        batch_size: int | None = 16,
        sampler: Sampler | None | None = None,
        shuffle: bool | None = False,
        num_workers: int | None = 0,
        **kwargs,
    ) -> None:
        """Creates a DataLoader from a PatchDataset and adds it to the ``dataloaders`` dictionary.

        Parameters
        ----------
        dataset : PatchDataset
            The dataset to add
        set_name : str
            The name to use for the dataset
        batch_size : Optional[int], optional
            The batch size to use when creating the DataLoader, by default 16
        sampler : Optional[Union[Sampler, None]], optional
            The sampler to use when creating the DataLoader, by default None
        shuffle : Optional[bool], optional
            Whether to shuffle the PatchDataset, by default False
        num_workers : Optional[int], optional
            The number of worker threads to use for loading data, by default 0.
        """
        if sampler and shuffle:
            print("[INFO] ``sampler`` is defined so train dataset will be unshuffled.")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

        self.dataloaders[set_name] = dataloader

    def load(
        self,
        load_path: str,
        force_device: bool | None = False,
    ) -> None:
        """
        This function loads the state of a class instance from a saved file
        using the joblib library. It also loads a PyTorch model from a
        separate file and maps it to the device used to load the class
        instance.

        Parameters
        ----------
        load_path : str
            Path to the saved file to load.
        force_device : bool or str, optional
            Whether to force the use of a specific device, or the name of the
            device to use. If set to ``True``, the default device is used.
            Defaults to ``False``.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.

        Returns
        -------
        None
        """

        load_path = os.path.abspath(load_path)
        mydevice = self.device

        if not os.path.isfile(load_path):
            raise FileNotFoundError(f'[ERROR] "{load_path}" cannot be found.')

        print(f'[INFO] Loading "{load_path}".')

        with open(load_path, "rb") as myfile:
            # objPickle = pickle.load(myfile)
            objPickle = joblib.load(myfile)

        self.__dict__ = objPickle

        if force_device:
            if not isinstance(force_device, str):
                force_device = str(force_device)
            os.environ["CUDA_VISIBLE_DEVICES"] = force_device

        par_name = os.path.dirname(load_path)
        base_name = os.path.basename(load_path)
        path2model = os.path.join(par_name, f"model_{base_name}")
        self.model = torch.load(path2model, map_location=mydevice)

        try:
            self.device = mydevice
            self.model = self.model.to(mydevice)
        except:
            pass

    def _set_up_print_colors(self):
        """Private function, setting color attributes on the object."""
        self._print_colors = {}

        # color
        self._print_colors["lgrey"] = "\033[1;90m"
        self._print_colors["grey"] = "\033[90m"  # boring information
        self._print_colors["yellow"] = "\033[93m"  # FYI
        self._print_colors["orange"] = "\033[0;33m"  # Warning

        self._print_colors["lred"] = "\033[1;31m"  # there is smoke
        self._print_colors["red"] = "\033[91m"  # fire!
        self._print_colors["dred"] = "\033[2;31m"  # Everything is on fire

        self._print_colors["lblue"] = "\033[1;34m"
        self._print_colors["blue"] = "\033[94m"
        self._print_colors["dblue"] = "\033[2;34m"

        self._print_colors["lgreen"] = "\033[1;32m"  # all is normal
        self._print_colors["green"] = "\033[92m"  # something else
        self._print_colors["dgreen"] = "\033[2;32m"  # even more interesting

        self._print_colors["lmagenta"] = "\033[1;35m"
        self._print_colors["magenta"] = "\033[95m"  # for title
        self._print_colors["dmagenta"] = "\033[2;35m"

        self._print_colors["cyan"] = "\033[96m"  # system time
        self._print_colors["white"] = "\033[97m"  # final time
        self._print_colors["black"] = "\033[0;30m"

        self._print_colors["reset"] = "\033[0m"
        self._print_colors["bold"] = "\033[1m"
        self._print_colors["under"] = "\033[4m"

    def _get_dtime(self) -> str:
        """
        Get the current date and time as a formatted string.

        Returns
        -------
        str
            A string representing the current date and time.
        """
        dtime = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        return dtime

    def cprint(self, type_info: str, bc_color: str, text: str) -> None:
        """
        Print colored text with additional information.

        Parameters
        ----------
        type_info : str
            The type of message to display.
        bc_color : str
            The color to use for the message text.
        text : str
            The text to display.

        Returns
        -------
        None
            The colored message is displayed on the standard output stream.
        """
        host_name = socket.gethostname().split(".")[0][:10]

        print(
            self._print_colors["green"]
            + self._get_dtime()
            + self._print_colors["reset"],
            self._print_colors["magenta"] + host_name + self._print_colors["reset"],
            self._print_colors["bold"]
            + self._print_colors["grey"]
            + type_info
            + self._print_colors["reset"],
            self._print_colors[bc_color] + text + self._print_colors["reset"],
        )

    def update_progress(
        self,
        progress: float | int,
        text: str | None = "",
        barLength: int | None = 30,
    ) -> None:
        """Update the progress bar.

        Parameters
        ----------
        progress : float or int
            The progress value to display, between ``0`` and ``1``.
            If an integer is provided, it will be converted to a float.
            If a value outside the range ``[0, 1]`` is provided, it will be
            clamped to the nearest valid value.
        text : str, optional
            Additional text to display after the progress bar, defaults to
            ``""``.
        barLength : int, optional
            The length of the progress bar in characters, defaults to ``30``.

        Raises
        ------
        TypeError
            If progress is not a floating point value or an integer.

        Returns
        -------
        None
            The progress bar is displayed on the standard output stream.
        """

        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress provided must be float or integer\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = f"\r[{'#'*block + '-'*(barLength-block)}] {progress*100:.1f}% {status} {text}"  # noqa
        sys.stdout.write(text)
        sys.stdout.flush()
