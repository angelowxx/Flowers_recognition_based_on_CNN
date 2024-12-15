import logging
import os

import torch
from tqdm import tqdm
import time


from src.eval.evaluate import AverageMeter, accuracy, eval_fn


def train_model(save_model_str, num_epochs, model, scheduler, optimizer
                , train_criterion, train_loader, device
                , use_all_data_to_train, val_loader, exp_name, score, info):

    # Train the model
    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        model_save_dir = os.path.join(os.getcwd(), save_model_str)

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        save_model_str = os.path.join(model_save_dir, exp_name + '_model')
    min_loss = 100
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info(info)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        train_score, train_loss = train_fn(model, optimizer, train_criterion, train_loader, device)
        scheduler.step()
        logging.info('Train accuracy: %f', train_score)
        if min_loss > train_loss and save_model_str:
            min_loss = train_loss
            torch.save(model.state_dict(), save_model_str)

        if not use_all_data_to_train:
            test_score = eval_fn(model, val_loader, device)
            logging.info('Validation accuracy: %f', test_score)
            score.append(test_score)

    if save_model_str:
        model.load_state_dict(torch.load(save_model_str))

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
