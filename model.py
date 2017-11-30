import numpy as np
from torch import nn
from constants import IMG_SIZE


def get_convnet_output_size(network, input_size=IMG_SIZE):
    input_size = input_size or IMG_SIZE

    if type(network) != list:
        network = [network]

    in_channels = network[0].conv1.in_channels

    output = Variable(torch.ones(1, in_channels, input_size, input_size))
    output.require_grad = False
    for conv in network:
        output = conv.forward(output)

    return np.asscalar(np.prod(output.data.shape)), output.data.size()[2]


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, max_pool_stride=2,
                 dropout_ratio=0.5):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=kernel_size)
        self.max_pool2d = nn.MaxPool2d(max_pool_stride)
        self.dropout = nn.Dropout2d(p=dropout_ratio)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return self.dropout(self.max_pool2d(x))


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.args = args

        self.conv1 = ConvLayer(3, 32, kernel_size=11)
        self.conv2 = ConvLayer(32, 64, kernel_size=9)
        self.conv3 = ConvLayer(64, 128, kernel_size=7)
        self.conv4 = ConvLayer(128, 256, kernel_size=5)
        inputs = [self.conv1, self.conv2, self.conv3, self.conv4]
        conv_output = get_convnet_output_size(inputs)
        self.fully_connected = nn.Linear(, self.args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return nn.functional.log_softmax(self.fully_connected(x))
