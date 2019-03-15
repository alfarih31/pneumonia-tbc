import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

TRAIN_FOLDER = '/home/alfarihfz/data/Pulmonary/train/'
TEST_FOLDER = '/home/alfarihfz/data/Pulmonary/test/'
BATCH_SIZE=64
EPOCH = 500
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
def image_loader(path):
    return Image.open(path).convert('RGB')

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainData = datasets.DatasetFolder(root=TRAIN_FOLDER,
                                    loader=image_loader,
                                    extensions=['jpeg', 'png'],
                                    transform=transform)
print(trainData.classes)
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
print(data_size['train'], data_size['test'])
def train():
    model = models.resnet34(pretrained=False)
    print(model.fc.in_features)
    print(model.fc.out_features)
    nOfFeatures = model.fc.in_features
    model.fc = nn.Linear(nOfFeatures, 3)
    print(model.fc.out_features)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    best_acc = 0
    best_model = model.state_dict()
    best_optimizer = optimizer.state_dict()

    for epoch in range(EPOCH):
        for mode in ['train', 'test']:
            batch_loss = 0.0
            nOfCorrect = 0.0
            if mode == 'train':
                model.train(True)
            else:
                model.train(False)
            for i, data in enumerate(train_test[mode]):
                    img, label = data
                    img, label = img.to(DEVICE), label.to(DEVICE)

                    optimizer.zero_grad()

                    predicts = model(img)
                    _, predict = torch.max(predicts.data, 1)
                    loss = criterion(predicts, label)
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                    batch_loss += loss.item()
                    nOfCorrect += torch.sum(predict == label.data).item()

                    if i%16 == 0 and i != 0:
                        print('Batch Loss: {:.4f} Acc: {:.4f}'.format(
                                batch_loss, nOfCorrect))

            epoch_loss = batch_loss/data_size[mode]
            epoch_accuracy = nOfCorrect/data_size[mode]

            print('Epoch {} {} Loss: {:.4f} Acc: {:.4f}'.format(
                epoch, mode, epoch_loss, epoch_accuracy))

            if mode == 'test' and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                best_model = model.state_dict()

        if epoch%50 == 0:
            print('SAVING MODEL')
            torch.save({'optimizer':best_optimizer, 'best_state':best_model}, 'models/%d_%d.pth'%(epoch, best_acc))

if __name__ == '__main__':
    train()
