import argparse
import torch
from torch import optim
from torch.nn.functional import nll_loss
from torch.autograd import Variable
from torchvision import datasets, transforms
from model import Network

parser = argparse.ArgumentParser(description="CNN based Face Recognition")

parser.add_argument("--data", type=str, default="data",
                    metavar="D",
                    help="Folder containing test and validation data")
parser.add_argument("--epochs", type=int,
                    default=10, metavar="E",
                    help="Number of epochs for which" +
                    "the model is to be trained")
parser.add_argument("--batch_size", type=int,
                    default=1, metavar="E",
                    help="Mini Batch size for the trainer")
parser.add_argument("--lr", type=float, default=0.001, metavar="L",
                    help="Learning Rate")


args = parser.parse_args()


def main(args):
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(0.6, 0.6, 0.6, 0.2),
        RandomAffineTransform(rotation_range=(-0.2, 0.2),
                              translation_range=(-0.1, 0.1),
                              scale_range=(0.8, 1.2),
                              shear_range=(-0.1, 0.1)),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(args.data + "/train_images",
                                         transforms=train_transforms)
    val_dataset = datasets.ImageFolder(args.data + "/val_images",
                                       transforms=val_transforms)

    train_loader = torch.utils.DataLoader(train_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=1)

    val_loader = torch.utils.DataLoader(val_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=1)

    model = Network(args)
    optimizer = optim.AdamOptimizer(model.parameters(),
                                    lr=args.lr)
    train(model, optimizer, train_loader, val_loader)


def train(args, model, optimizer, train_loader, val_loader):
    for i in range(args.epochs):
        for images, labels in train_loader:
            images, labels = Variable(images), Variable(labels)
            optimizer.zero_grad()

            output = model(images)

            loss = nll_loss(output, labels)
            loss.backword()
            optimizer.step()

        print("Epoch: %d/%d" % (i, args.epochs))
        validation(model, train_loader)
        validation(model, val_loader)


def validation(model, loader):
    model.eval()
    validation_loss = 0
    correct = 0
    for images, labels in loader:
        images, labels = Variable(images, volatile=True), Variable(labels)

        output = model(images)
        validation_loss += F.nll_loss(output, labels,
                                      size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

    validation_loss /= len(loader.dataset)
    print('\n' + loader_type +
          ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(validation_loss, correct, len(loader.dataset),
                  100. * correct / len(loader.dataset)))

    return 100 * correct / len(loader.dataset)


if __name__ == '__main__':
    main()
