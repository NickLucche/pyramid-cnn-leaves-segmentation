import torch
from utils import device
from utils import parse_args
from msu_leaves_dataset import MSUDenseLeavesDataset
from torch.utils.data import DataLoader
from pyramid_network import PyramidNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create dataloader
    eval_dataloader = DataLoader(MSUDenseLeavesDataset(args.dataset_filepath[:-1] + '_eval/', args.predictions_number),
                                 shuffle=False, batch_size=4)
    # todo totally arbitrary weights
    model = PyramidNet(n_layers=5, loss_weights=[torch.tensor([1.0])]*5)#, torch.tensor([1.9]), torch.tensor([3.9]),
                                                 # torch.tensor([8]), torch.tensor([10])])
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    else:
        print("Please provide a valid path to the pre-trained model to evaluate")
        exit(1)

    model = model.to(device)

    viz=args.viz_results
    # samples made of image-targets-masks
    model.eval()
    with torch.no_grad():
        for batch_no, (image, targets, masks) in tqdm(enumerate(eval_dataloader)):
            og_im = copy.copy(image[0, :, :, :].cpu().numpy())
            image = image.to(device)
            targets = [t.to(device) for t in targets]
            masks = [t.to(device) for t in masks]
            predictions = model(image)
            loss = model.compute_multiscale_loss(predictions, targets, masks)
            # for i in range(len(predictions)):
            #  print(predictions[i].shape, targets[i].shape, masks[i].shape)
            print('Eval Loss:', loss.item())
            # pixel-wise accuracy of multiscale predictions (edges-only)
            for p, t, m in zip(predictions, targets, masks):
                p = (p > 0.).float()
                pixel_acc = (p * m) * t
                acc = pixel_acc.sum() / t.sum()
                print(
                    f"Accuracy at scale ({p.shape[2]}x{p.shape[3]}) is {acc} ({pixel_acc.sum()}/{t.sum()} edge pixels)")


            # visualize result
            if viz:
                predictions = model(image)
                images = []
                for p in predictions:
                    p = p[0, :, :, :] # get first image of batch
                    # print(p.shape, p.max().item(), p.min().item(), p.sum().item())
                    p = (p > 0.).float()
                    p = p.squeeze().cpu().numpy().astype(np.float32)
                    # print(p.shape, np.amax(p), np.sum(p), np.amin(p))
                    images.append(p)

                fix, ax = plt.subplots(1, len(images)+2)
                for i in range(len(images)):
                    ax[i].imshow(images[i], cmap='Greys')
                image = og_im
                ax[-2].imshow(image.transpose(1, 2, 0).astype(np.float32))
                ax[-1].imshow(targets[-1][0, :, :, :].cpu().squeeze().numpy().astype(np.float32))
                plt.show()

            print('*' * 50)
