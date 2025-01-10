import logging
import os

import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import time


from src.eval.evaluate import AverageMeter, accuracy, eval_fn


def train_model(save_model_str, num_epochs, model, model_optimizer, lr, train_data, test_loader, folds
                , batch_size, train_criterion, device, exp_name, score, info):
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

    optimizer = model_optimizer(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=5,
        T_mult=2,
    )

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):

        train_subset = Subset(train_data, train_idx)
        val_subset = Subset(train_data, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Train the model
        pre_val_score = 0
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
            logging.info('Train accuracy: %f', train_score)

            val_score = eval_fn(model, val_loader, device)
            logging.info('Validation accuracy: %f', val_score)
            val_scores.append(val_score)

            test_score = eval_fn(model, test_loader, device)
            logging.info('Test accuracy: %f', test_score)
            score.append(test_score)

            e += 1

            if pre_val_score < val_score:
                de_cnt = 0
            else:
                de_cnt += 1
            pre_val_score = val_score

            if de_cnt >= 3 or train_score > 0.98:
                logging.info('#' * 20 + 'early stop!' + '#' * 19)
                break
    plt.figure(0)

    plt.plot([lr * factor for lr in lrs], color='red', label='learning rate', linestyle='-')
    plt.plot(score, color='green', label='test score', linestyle='-')
    plt.plot(val_scores, color='blue', label='validation score', linestyle='-')

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

    torch.save(model.state_dict(), save_model_str)

def train_fn(model, optimizer, criterion, train_loader, device):
    """
  Training method
  :param model: model to train
  :param optimizer: optimization algorithm
  :param criterion: loss function
  :param loader: data loader for either training or testing set
  :param device: torch device
  :return: (accuracy, loss) on the data
  """
    time_begin = time.time()
    score = AverageMeter()
    losses = AverageMeter()
    model.train()
    time_train = 0

    t = tqdm(train_loader)
    for images, labels in t:
        images = images.to(device)
        labels = labels.to(device)

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

    time_train += time.time() - time_begin
    print('(=> Training) Loss: {:.4f}'.format(losses.avg))
    print('training time: ' + str(time_train))
    return score.avg, losses.avg
