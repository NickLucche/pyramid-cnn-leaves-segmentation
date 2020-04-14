from argparse import ArgumentParser
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device == torch.device("cuda"):
    print("Using device:", torch.cuda.get_device_name(0))
else:
    print("Using device:", device)


def set_gpu_number(n_gpu):
    global device
    device = torch.device("cuda:{}".format(n_gpu)) if torch.cuda.is_available() else torch.device("cpu")

parser = ArgumentParser()
parser.add_argument('-e', '--epochs', help='Number of epochs the training will be run for', default=20)
parser.add_argument('--seed', type=int, default=7, help='random seed (default: 7)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('-d', '--dataset-filepath', help='Filepath of the dataset to load', required=True)
parser.add_argument('--predictions-number', help='Number of predictions the network will do at different scales', default=5)
parser.add_argument('-s', '--save-path', help='Where to save model checkpoints', required=True)
parser.add_argument('-l', '--load-model', help='Where to load checkpoint of model from')
# todo save model

def parse_args():
    return parser.parse_args()