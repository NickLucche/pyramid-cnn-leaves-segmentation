import torch
import torch.nn as nn


class PyramidNet(nn.Module):

    def __init__(self, n_layers, input_image_channels=3, output_channels=1, loss_weights=None):
        super(PyramidNet, self).__init__()
        # fixed number of channels throughout the network
        self.no_channels = 32
        self.no_rc_per_block = 4

        # first convolution sets the image to the 'correct' number of channels
        self.conv1 = nn.Conv2d(input_image_channels, self.no_channels, (3, 3), padding=1)

        # create network structure
        self.upsample_blocks = []
        self.downsample_blocks = []
        self.pre_loss_convs = []
        for i in range(n_layers):
            self.upsample_blocks.append(self._create_rc_block(self.no_rc_per_block))
            self.downsample_blocks.append(self._create_rc_block(self.no_rc_per_block))
            self.pre_loss_convs.append(nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(self.no_channels, output_channels, 3, padding=1)  # todo 1x1 conv instead?
            ))
        # add one more upsample block
        self.upsample_blocks.append(self._create_rc_block(self.no_rc_per_block))

        self.upsample_blocks = nn.ModuleList(self.upsample_blocks)
        self.downsample_blocks = nn.ModuleList(self.downsample_blocks)
        self.pre_loss_convs = nn.ModuleList(self.pre_loss_convs)

        self.loss = nn.CrossEntropyLoss(weight=loss_weights)  # softmax inside todo maybe BCEloss?

    def forward(self, x):
        x = self.conv1(x)

        # todo downsample blocks take as input prev step and upsample one as in unet
        downsampled = []
        for layer in self.downsample_blocks:
            x = layer(x)
            downsampled.append(x)  # todo graph dependency may blow up gpu?
            # max pooling must be done separately so that x can be re-used later
            x = nn.MaxPool2d(2, stride=2)(x)

        # keep track of multi-scale prediction of network
        multiscale_predictions = []
        # first upsample block has no summing of map from downsampled
        x = self.upsample_blocks[0](x)
        x = nn.Upsample(scale_factor=2.0, mode='nearest')(x)
        # [print(d.shape) for d in downsampled]
        for i, layer in enumerate(self.upsample_blocks[1:]):
            # sum map coming from correspondent downsample layer
            # print(x.shape, downsampled[-i-1].shape)
            x = x + downsampled[-i - 1]  # todo concat here?
            x = layer(x)
            # store current resolution prediction (apply RC block first)
            multiscale_predictions.append(self.pre_loss_convs[i](x))
            x = nn.Upsample(scale_factor=2.0, mode='nearest')(x)

        return multiscale_predictions

    # rc stands for ReLU followed by a 3x3 conv as depicted in paper
    def _create_rc_block(self, number_of_rc):
        block = []
        for i in range(number_of_rc):
            block.append(nn.ReLU(inplace=True))
            block.append(nn.Conv2d(self.no_channels, self.no_channels, (3, 3), padding=1))

        return nn.Sequential(*block)

    # def _create_upsample_block(self, number_of_rc):
    #     block = []
    #     for i in range(number_of_rc):
    #         block.append(nn.ReLU(inplace=True))
    #         block.append(nn.Conv2d(self.no_channels, self.no_channels, (3, 3), padding=1))
    #
    #     block.append(nn.Upsample(scale_factor=2.0, mode='nearest'))
    #     return nn.Sequential(*block)

    # Computes multi-scale loss given a list of predictions and a list
    # of matching size targets; loss at different scale is summed up.
    # A mask is applied to the loss so that unlabeled pixels are ignored
    def compute_multiscale_loss(self, multiscale_prediction, multiscale_targets, multiscale_masks):
        losses = [self.loss(x*mask, y) for x, y, mask in zip(multiscale_prediction, multiscale_targets, multiscale_masks)]
        # here sum will call overridden + operator
        return sum(losses)
# todo mask loss

if __name__ == '__main__':
    net = PyramidNet(5, loss_weights=torch.tensor([.2, .8]))
    print(net)

    x = torch.randn((2, 3, 128, 128), requires_grad=True)
    targets = [torch.ones((2, s, s), requires_grad=True).long() for s in [8, 16, 32, 64, 128]]
    masks = [torch.ones((2, s, s), requires_grad=True) for s in [8, 16, 32, 64, 128]]
    ys = net(x)
    [print(y.shape) for y in ys]
    [print(y.shape) for y in targets]

    loss = net.compute_multiscale_loss(ys, targets, masks)
    loss.backward()
