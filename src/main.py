import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.optim

from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from src.cnn import *
from src.eval.evaluate import eval_fn, accuracy
from src.training import train_fn, train_model
from src.data_augmentations import *
from src.ImageHolder import ImageHolder


def main(data_dir,
         torch_model,
         batch_size=128,
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer=torch.optim.Adam,
         save_model_str=None,
         exp_name='',
         continue_training=False,
         train_with_all_data=False):
    """
    Training loop for configurableNet.
    :param torch_model: model that we are training
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during training (torch.optim.Optimizer)
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :param save_model_str: path of saved models (str)
    :param use_all_data_to_train: indicator whether we use all the data for training (bool)
    :param exp_name: experiment name (str)
    :param continue_training: if the continue training the given model
    :return:
    """

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    """if data_augmentations is None:
        data_augmentations = transforms.ToTensor()
    elif isinstance(data_augmentations, list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError"""
    data_augmentations = data_augmentation_pipline_repeated
    base_transform = resize_to_128x128

    # Load the dataset
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=base_transform)
    val_data = ImageFolder(os.path.join(data_dir, 'val'), transform=base_transform)
    test_data = ImageFolder(os.path.join(data_dir, 'test'), transform=base_transform)

    channels, img_height, img_width = train_data[0][0].shape

    # image size
    input_shape = (channels, img_height, img_width)

    # instantiate training criterion
    train_criterion = train_criterion().to(device)

    model = torch_model(input_shape=input_shape,
                        num_classes=len(train_data.classes)).to(device)

    # Info about the model being trained
    # You can find the number of learnable parameters in the model here
    logging.info('Model being trained:')
    summary(model, input_shape,
            device='cuda' if torch.cuda.is_available() else 'cpu')
    train_data = [train_data, val_data]
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    if continue_training:
        logging.info('loading model state dict!!')
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models', 'pre-trained_model')))

    data_augmentations = [data_augmentation_pipline_repeated]
    augmentation_times = [5]

    augmentation_types = len(data_augmentations)
    for i in range(augmentation_types):
        data_augmentation = data_augmentations[i]
        augmentation_time = augmentation_times[i]
        train_data = [ImageFolder(os.path.join(data_dir, 'train'), transform=data_augmentation) for i in
                      range(augmentation_time)] + train_data
        train_data = [ImageFolder(os.path.join(data_dir, 'val'), transform=data_augmentation) for i in
                      range(augmentation_time)] + train_data
        if train_with_all_data:
            train_data = [ImageFolder(os.path.join(data_dir, 'test'), transform=data_augmentation) for i in
                          range(augmentation_time)] + train_data

    info = 'Training'
    score = []
    train_model(save_model_str, 21, model, model_optimizer, 0.01, ConcatDataset(train_data), test_loader, 7
                , batch_size, train_criterion, device, exp_name, score, info)


    logging.info('Accuracy at each epoch: ' + str(score))
    logging.info('Mean of accuracies across all epochs: ' + str(100 * np.mean(score)) + '%')
    logging.info('Accuracy of model at final epoch: ' + str(100 * score[-1]) + '%')


if __name__ == '__main__':
    """
    This is just an example of a training pipeline.

    Feel free to add or remove more arguments, change default values or hardcode parameters to use.
    """
    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss, "nll_loss": torch.nn.NLLLoss}  # Feel free to add more
    opti_dict = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adamW': torch.optim.AdamW}  # Feel free to add more

    cmdline_parser = argparse.ArgumentParser('DL WS24/25 Competition')

    cmdline_parser.add_argument('-m', '--model',
                                default='FastCNN',
                                help='Class name of model to train',
                                type=str)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=256,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'dataset'),
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-L', '--training_loss',
                                default='nll_loss',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='adamW',
                                help='Which optimizer to use during training',
                                choices=list(opti_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-p', '--model_path',
                                default='models',
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-n', '--exp_name',
                                default='fast',
                                help='Name of this experiment',
                                type=str)
    cmdline_parser.add_argument('-c', '--continue_training',
                                action='store_true',
                                help='continue training the existing model.')
    cmdline_parser.add_argument('-a', '--train_with_all_data',
                                action='store_true',
                                help='train with all data.')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    main(
        data_dir=args.data_dir,
        torch_model=eval(args.model),
        batch_size=args.batch_size,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=opti_dict[args.optimizer],
        save_model_str=args.model_path,
        exp_name=args.exp_name,
        continue_training=args.continue_training,
        train_with_all_data=args.train_with_all_data
    )
