import os
import cv2
import torch
from utils import device
from utils import parse_args
from pyramid_network import PyramidNet
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = parse_args()

    # todo totally arbitrary weights
    model = PyramidNet(n_layers=5, loss_weights=[torch.tensor([1.0])]*5)#, torch.tensor([1.9]), torch.tensor([3.9]),
                                                 # torch.tensor([8]), torch.tensor([10])])
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    else:
        print("Please provide a valid path to the pre-trained model to evaluate")
        exit(1)

    model = model.to(device)

    og_image = cv2.imread(args.image)
    if og_image.shape[0] > 1024 and og_image.shape[1]>1024:
        og_image = og_image[(og_image.shape[0]-1024)//2:og_image.shape[0]-(og_image.shape[0]-1024)//2, (og_image.shape[1]-1024)//2:og_image.shape[1]-(og_image.shape[1]-1024)//2, :]
    # resize image or crop patches
    # image = cv2.resize(og_image, (128, 128))
    og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(og_image.transpose(2, 0, 1) / 255.).float()
    print(image.shape)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(image.unsqueeze(0))
        # get prediction at max resolution
        p = predictions[-1]
        # sigmoid + thresholding
        p = (p > 0.).float()
        p = p.squeeze().cpu().numpy().astype(np.float32)
        cv2.imwrite(os.path.join(args.save_path, 'custom_prediction.png'), (p * 255).astype(np.uint8))
        print(p.shape)
        fix, ax = plt.subplots(1, 2)

        ax[0].imshow(og_image)
        ax[1].imshow(p, cmap='Greys')
        plt.show()



