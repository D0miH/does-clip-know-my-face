from typing import Type, Optional, Dict, Any, Literal, List

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck, resnet50, WeightsEnum, ResNet50_Weights
from torchvision.models.resnet import ResNet as PyTorchResNet

from facescrub_training.pl_modules import Classifier


class ResNet(Classifier):
    def __init__(
        self,
        block: Literal['BasicBlock', 'Bottleneck'],
        num_blocks: List[int],
        lr: float = 1e-4,
        num_classes=10,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        lr_scheduler_cls: Optional[Any] = None,
        lr_scheduler_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            lr=lr,
            num_classes=num_classes,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_args=lr_scheduler_args
        )
        self.save_hyperparameters(
            'block',
            'num_blocks',
            'lr',
            'num_classes',
            'optimizer_cls',
            'optimizer_args',
            'lr_scheduler_cls',
            'lr_scheduler_args'
        )

        blocks = {"BasicBlock": BasicBlock, "Bottleneck": Bottleneck}

        self.model = PyTorchResNet(block=blocks[block], layers=num_blocks, num_classes=num_classes)


class ResNet50(ResNet):
    def __init__(
        self,
        lr: float = 1e-4,
        num_classes=10,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        lr_scheduler_cls: Optional[Any] = None,
        lr_scheduler_args: Optional[Dict[str, Any]] = None,
        not_pretrained: bool = False,
        pretrained_weights: Optional[WeightsEnum] = ResNet50_Weights.IMAGENET1K_V2
    ):
        super().__init__(
            block="Bottleneck",
            num_blocks=[3, 4, 6, 3],
            lr=lr,
            num_classes=num_classes,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_args=lr_scheduler_args
        )
        self.save_hyperparameters(ignore='pretrained_weights')

        self.model = resnet50(weights=None if not_pretrained else pretrained_weights)

        # exchange the last layer to match the number of classes
        if self.model.fc.out_features != num_classes:
            self.model.fc = nn.Linear(
                in_features=self.model.fc.in_features,
                out_features=num_classes,
                device=self.model.fc.weight.device,
                dtype=self.model.fc.weight.dtype
            )
