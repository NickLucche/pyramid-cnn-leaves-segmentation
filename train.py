import torch.nn as nn
import torch
from utils import device
from utils import parse_args
from msu_leaves_dataset import MSUDenseLeavesDataset
from torch.utils.data import DataLoader
from pyramid_network import PyramidNet
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create dataloader
    dataset = MSUDenseLeavesDataset(args.dataset_filepath, args.predictions_number)
    dataloader = DataLoader(dataset, batch_size=6)
    # todo totally arbitrary weights
    model = PyramidNet(n_layers=5).to(device)#, loss_weights=torch.tensor([.2, .8])).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(0, args.epochs):
        # samples made of image-targets-masks
        for batch_no, samples in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            input_batch = samples[0].to(device)
            targets = samples[1]
            targets = [t.to(device) for t in targets]
            masks = samples[1]
            masks = [t.to(device) for t in masks]
            # print("Input shape:", input_batch.shape)
            predictions = model(input_batch)
            # print(len(outputs), 'Output:', [o.shape for o in outputs], '\nTargets:', len(targets), targets[0].shape)
            # heads_losses = model.compute_losses(outputs, targets)
            # loss = multiloss(heads_losses, task_weights)

            # for i in range(len(predictions)):
            #     print(predictions[i].shape, targets[i].shape, masks[i].shape)
            loss = model.compute_multiscale_loss(predictions, targets, masks)
            loss.backward()
            # print("Current Loss:", loss.item())
            optimizer.step()

            if batch_no % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_no * len(samples), len(dataset),
                           100. * batch_no / len(dataloader), loss.item()))
