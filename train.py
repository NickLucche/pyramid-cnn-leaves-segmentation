import torch.nn as nn
import torch
from utils import device
from utils import parse_args
from msu_leaves_dataset import MSUDenseLeavesDataset
from torch.utils.data import DataLoader
from pyramid_network import PyramidNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def evaluate(net, eval_dataset):
    net.eval()
    print('*'*50)
    with torch.no_grad():
        for batch_no, (image, targets, masks) in tqdm(enumerate(eval_dataset)):
            image = image.to(device)
            targets = [t.to(device) for t in targets]
            masks = [t.to(device) for t in masks]
            predictions = net(image)
            loss = net.compute_multiscale_loss(predictions, targets, masks)
            # for i in range(len(predictions)):
            #  print(predictions[i].shape, targets[i].shape, masks[i].shape)
            print('Eval Loss:', loss.item())
            # pixel-wise accuracy of multiscale predictions (edges-only)
            for p, t, m in zip(predictions, targets, masks):
                p = (p>0.).float()
                pixel_acc = (p * m) * t
                acc = pixel_acc.sum() / t.sum()
                print(f"Accuracy at scale ({p.shape[2]}x{p.shape[3]}) is {acc} ({pixel_acc.sum()}/{t.sum()} edge pixels)")
    print('*'*50)
    net.train()


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create dataloader
    dataset = MSUDenseLeavesDataset(args.dataset_filepath, args.predictions_number)
    dataloader = DataLoader(dataset, batch_size=24)

    eval_dataloader = DataLoader(MSUDenseLeavesDataset(args.dataset_filepath[:-1] + '_eval/', args.predictions_number),
                                 shuffle=True, batch_size=24)
    # todo totally arbitrary weights
    model = PyramidNet(n_layers=5, loss_weights=[torch.tensor([1.0])]*5)#, torch.tensor([1.9]), torch.tensor([3.9]),
                                                 # torch.tensor([8]), torch.tensor([10])])
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9)

    viz = args.viz_results
    for epoch in range(0, args.epochs):
        # samples made of image-targets-masks
        for batch_no, (input_batch, targets, masks) in enumerate(dataloader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            targets = [t.to(device) for t in targets]
            masks = [t.to(device) for t in masks]
            # print("Input shape:", input_batch.shape)
            predictions = model(input_batch)

            if batch_no % 10 == 0:
                print('\n',predictions[-1].max().item(), predictions[-1].min().item(), predictions[-1].sum().item())
                print('\n',torch.sigmoid(predictions[-1]).max().item(), torch.sigmoid(predictions[-1]).min().item(),
                      torch.sigmoid(predictions[-1]).sum().item())
            # print(targets[0].max().item(), targets[0].min().item(), targets[0].sum().item())
            # print(masks[0].max().item(), masks[0].min().item(), masks[0].sum().item())
            # for i in range(len(predictions)):
            #     print(predictions[i].shape, targets[i].shape, masks[i].shape)
            loss = model.compute_multiscale_loss(predictions, targets, masks)
            loss.backward()
            # print("Current Loss:", loss.item())
            optimizer.step()

            if batch_no % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_no*24, len(dataset),
                           100. * batch_no * 24 / len(dataloader), loss.item()))

                evaluate(model, eval_dataloader)

            torch.save(model.state_dict(), args.save_path+'pyramid_net.pt')
            # visualize result
            if viz:
                with torch.no_grad():
                    predictions = model(input_batch)
                    p = predictions[-1][10, :, :, :]
                    # p = (torch.nn.functional.sigmoid(p) > .5).float()
                    # avoid using sigmoid, it's the same thing
                    print(p.shape, p.max().item(), p.min().item(), p.sum().item())
                    p = (p > 0.).float()
                    p = p.squeeze().cpu().numpy().astype(np.float32)
                    print(p.shape, np.amax(p), np.sum(p), np.amin(p))

                    plt.imshow(p, cmap='Greys')
                    plt.show()

