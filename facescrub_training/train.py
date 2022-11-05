import argparse
from typing import Optional

from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pytorch_lightning import seed_everything
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import MultiStepLR

import pl_models
import datasets

from pl_modules import Classifier
from pl_modules.datamodule import DataModule
from utils import get_class_from_module, LightningRtpt


def train_model(arg_parser):
    temp_args, _ = arg_parser.parse_known_args()

    # get the model
    model_cls: Classifier = get_class_from_module(pl_models, temp_args.arch)
    parser = model_cls.add_model_specific_args(arg_parser)

    args, _ = parser.parse_known_args()

    seed_everything(args.seed, workers=True)

    # define the training and eval transforms
    eval_transforms = T.Compose(
        [T.Resize(args.image_size), T.CenterCrop(args.image_size), T.ToTensor(), imagenet_normalization()]
    )
    train_transforms = T.Compose(
        [
            T.Resize(args.image_size),
            T.CenterCrop(args.image_size),
            T.RandomResizedCrop(size=(224, 224), scale=(0.85, 1), ratio=(1, 1)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            imagenet_normalization()
        ]
    )

    # get the datamodule
    dataset_cls = get_class_from_module(datasets, args.dataset)
    datamodule = DataModule(
        dataset_cls,
        dataset_args={
            'root': f'./data/{args.dataset}', 'download': True
        },
        batch_size=128,
        train_transforms=train_transforms,
        val_transforms=eval_transforms,
        test_transforms=eval_transforms
    )
    if args.num_classes is None:
        args.num_classes = datamodule.num_classes()

    # configure the lr scheduler
    setattr(args, "optimizer_args", {'weight_decay': args.weight_decay})
    setattr(args, "lr_scheduler_cls", None if args.no_lr_scheduling else MultiStepLR)
    setattr(
        args,
        "lr_scheduler_args",
        None if args.no_lr_scheduling else {
            'milestones': [int(args.epochs * 0.75), int(args.epochs * 0.9)], 'gamma': 0.1
        }
    )

    model = model_cls.from_argparse_args(args)

    model_checkpoint = ModelCheckpoint(save_last=True, verbose=True)
    rtpt = LightningRtpt(name_initials="DH", experiment_name=model.get_architecture_name(), max_iterations=args.epochs)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [rtpt, model_checkpoint, lr_monitor]

    # if not disabled use early stopping
    if args.early_stopping:
        early_stopper = EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            verbose=True
        )
        callbacks.append(early_stopper)

    wandb_logger = True
    if args.wandb:
        arch_name = model.get_architecture_name()
        wandb_logger = WandbLogger(
            name=f'{arch_name} {args.dataset}',
            project="clipping_privacy",
            entity="d0mih",
            log_model=not args.do_not_save_model
        )
        wandb_logger.watch(model)
        wandb_logger.log_hyperparams(args)

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        max_epochs=args.epochs,
        deterministic=True,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


def get_model_training_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # dataset arguments
    parser.add_argument('--dataset', default='FaceScrubCropped', help='The dataset to use for training')
    parser.add_argument(
        '--num_classes',
        default=None,
        help='Specify the number of classes. If `None` the number of classes of the dataset will be used.'
    )

    # model name
    parser.add_argument('--arch', type=str, default="ResNet50")
    parser.add_argument('--not_pretrained', action='store_true')

    # training specific arguments
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--do_not_save_model', action='store_true')

    # optimizer arguments
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=3e-3)

    # early stopping arguments
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0)

    # learning rate scheduling arguments
    parser.add_argument('--no_lr_scheduling', action='store_true', help='Whether to use lr scheduling')

    # other arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    model_args_parser = get_model_training_arg_parser()

    args, _ = parser.parse_known_args()

    train_model(arg_parser=model_args_parser)
