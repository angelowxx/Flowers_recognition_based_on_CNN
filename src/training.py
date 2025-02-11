import logging
import math
import os

import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import Subset, DataLoader, ConcatDataset
from tqdm import tqdm
import time


from src.eval.evaluate import AverageMeter, accuracy, eval_fn


def train_model(save_model_str, num_epochs, model, model_optimizer, lr, train_data, test_loader, folds
                , batch_size, train_criterion, device, exp_name, test_scores, info):
    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        model_save_dir = os.path.join(os.getcwd(), save_model_str)

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        save_model_str = os.path.join(model_save_dir, exp_name + '_model')

    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    lrs = []
    val_scores = []
    divides = []
    e = 0
    factor = 1/lr

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):

        optimizer = model_optimizer(
            model.parameters(),
            lr=lr,
            weight_decay=0.02,
            # momentum=0.9,
            # nesterov=True,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=math.ceil(num_epochs / 7),
            T_mult=2,
        )

        lr *= 0.5

        train_subset = Subset(train_data, train_idx)
        val_subset = Subset(train_data, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Train the model
        min_val_loss = 5
        de_cnt = 0
        divides.append(e)
        for epoch in range(num_epochs):
            lrs.append(scheduler.get_last_lr()[0])
            logging.info('#' * 50)
            logging.info(info)
            logging.info('Fold [{}/{}]'.format(fold + 1, folds))
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

            train_score, train_loss = train_fn(model, optimizer, train_criterion, train_loader, device)
            scheduler.step()
            logging.info('Training accuracy: %f', train_score)

            val_score, val_loss = eval_model(model, train_criterion, val_loader, device, 'Validation')
            logging.info('Validation accuracy: %f', val_score)
            val_scores.append(val_score)

            test_score, test_loss = eval_model(model, train_criterion, test_loader, device, 'Test')
            logging.info('Test accuracy: %f', test_score)
            test_scores.append(test_score)

            e += 1

            if min_val_loss >= 5 or epoch == math.ceil(num_epochs/7) or epoch == 3*math.ceil(num_epochs/7):
                min_val_loss = val_loss * 2
                for module in model.modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        prune.l1_unstructured(module, name="weight", amount=0.5 / (fold + 1))
                        prune.remove(module, "weight")  # Make pruning permanent

            if min_val_loss >= val_loss:
                de_cnt = 0
                min_val_loss = val_loss * 0.3 + min_val_loss * 0.7

            else:
                de_cnt += 1
                # min_val_loss = val_loss * 0.2 + min_val_loss * 0.8

            if de_cnt > 3:
                logging.info('#' * 19 + 'early stop!!' + '#' * 19)
                break

        torch.save(model.state_dict(), save_model_str)

        plt.figure(0)

        plt.plot([lr * factor for lr in lrs], color='red', label='learning rate', linestyle='-')
        plt.plot(test_scores, color='green', label='test accuracy', linestyle='-')
        plt.plot(val_scores, color='blue', label='validation accuracy', linestyle='-')

        for pos in divides:
            plt.axvline(x=pos, color='black', linestyle='--')
            plt.text(pos, 0, str(pos), color='black', ha='center', va='bottom')  # Add label at the bottom

        plt.xlabel('epochs')
        plt.legend()

        save_fig_dir = os.path.join(os.getcwd(), 'figures')
        if not os.path.exists(save_fig_dir):
            os.mkdir(save_fig_dir)
        save_fig_dir = os.path.join(save_fig_dir, exp_name + '_fig' + ".png")
        plt.savefig(save_fig_dir)
        plt.close()


def train_fn(model, optimizer, criterion, train_loader, device):
    """
  Training method
  :param model: model to train
  :param optimizer: optimization algorithm
  :param criterion: loss function
  :param train_loader: data loader for either training or testing set
  :param device: torch device
  :return: (accuracy, loss) on the data
  """
    score = AverageMeter()
    losses = AverageMeter()
    model.train()

    t = tqdm(train_loader)
    for images, labels in t:
        images = torch.cat(images).to(device)
        labels = torch.cat(labels).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(logits, labels)
        n = images.size(0)
        losses.update(loss.item(), n)
        score.update(acc.item(), n)

        t.set_description('(=> Training) Loss: {:.4f}'.format(losses.avg))

    return score.avg, losses.avg


def eval_model(model, criterion, val_loader, device, info):
    model.eval()
    losses = AverageMeter()
    score = AverageMeter()
    t = tqdm(val_loader)
    with torch.no_grad():  # no gradient needed
        for images, labels in t:
            images = torch.cat(images).to(device)
            labels = torch.cat(labels).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            acc = accuracy(outputs, labels)
            n = images.size(0)
            losses.update(loss.item(), n)
            score.update(acc.item(), n)

            t.set_description('(=> {}) Loss: {:.4f}'.format(info, losses.avg))

    return score.avg, losses.avg
