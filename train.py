import torch
import numpy as np
from tqdm import tqdm
from utils import save_plt, save_PIL


def train(model, opt, criterion, trainloader, valloader, epochs, device):
    history = {'train loss by epoch': [],
               'valid loss by epoch': [],
               'train loss by batch': [],
               'valid loss by batch': []}

    log_template = "Epoch {ep:d}:\t mean train_loss: {t_loss:0.4f}\t mean val_loss {v_loss:0.4f}\n"

    model.train()

    for epoch in range(epochs):
        train_loss_per_epoch = []
        val_loss_per_epoch = []

        for x, y in tqdm(trainloader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            probas, y_hat = model(x)
            loss = criterion(probas, y)
            train_loss_per_epoch.append(loss.item())

            loss.backward()
            opt.step()

        history['train loss by epoch'].append(np.mean(train_loss_per_epoch))
        history['train loss by batch'].extend(train_loss_per_epoch)

        model.eval()
        with torch.no_grad():
            for x, y in valloader:
                x, y = x.to(device), y.to(device)
                probas, y_hat = model(x)
                loss = criterion(probas, y)
                val_loss_per_epoch.append(loss.item())

            history['valid loss by epoch'].append(np.mean(val_loss_per_epoch))
            history['valid loss by batch'].extend(val_loss_per_epoch)

        tqdm.write(log_template.format(ep=epoch + 1, t_loss=np.mean(train_loss_per_epoch),
                                       v_loss=np.mean(val_loss_per_epoch)))

        # show history
        save_plt(history, epoch, type='epoch')

        # visualization during training
        save_PIL(x.cpu(), fp='True image.png', nrow=8)
        save_PIL(y.cpu(), fp='Ground truth.png', nrow=8)
        save_PIL(y_hat.cpu(), fp='Model\'s output.png', nrow=8)

    return history
