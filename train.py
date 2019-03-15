import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from numpy import max, sum

TRAIN_FOLDER = '/home/alfarihfz/data/Pulmonary/train/'
TEST_FOLDER = '/home/alfarihfz/data/Pulmonary/test/'
BATCH_SIZE=8
EPOCH = 10000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def image_loader(path):
    return Image.open(path).convert('RGB')

transform = transforms.Compose(
    [transforms.Resize((640, 480)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainData = datasets.DatasetFolder(root=TRAIN_FOLDER,
                                    loader=image_loader,
                                    extensions=['jpeg', 'png'],
                                    transform=transform)

testData = datasets.DatasetFolder(root=TEST_FOLDER,
                                    loader=image_loader,
                                    extensions=['jpeg', 'png'],
                                    transform=transform)
train_test = {
    'train': DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True),
    'test': DataLoader(testData, batch_size=BATCH_SIZE, shuffle=True)
}

data_size = {
    'train': len(trainData),
    'test': len(testData)
}

def train():
    model = models.resnet152(pretrained=False, num_classes=3).to(DEVICE)

    nOfFeatures = model.fc.in_features
    model.fc = nn.Linear(nOfFeatures, 3)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    best_acc = 0
    best_model = model.state_dict()

    for epoch in range(EPOCH):
        for mode in ['train', 'test']:
            batch_loss = 0.0
            nOfCorrect = 0
            if mode == 'train':
                model.train(True)
            else:
                model.train(False)
            for i, data in enumerate(train_test[mode]):
                    img, label = data
                    if use_gpu: img, label = img.to(DEVICE), label.to(DEVICE)
                    else: img, label = Variable(img), Variable(label)

                    optimizer.zero_grad()

                    predicts = model(img)
                    predict = max(predicts.data)
                    loss = criterion(predict, label)
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                    batch_loss += loss.data[0]
                    nOfCorrect += sum(predict == label.data)
            epoch_loss = batch_loss/data_size[mode]
            epoch_accuracy = nOfCorrect/data_size[mode]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                mode, epoch_loss, epoch_accuracy))

            if mode == 'test' and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                best_model = model.state_dict()

        if epoch%1000 == 0 and epoch != 0:
            torch.save(best_model, 'models/%d_%d.pth'%(epoch, best_acc))

if __name__ == '__main__':
    train()
