# Training script, simplified from the original, cutting out all non-essentials (like CLI access)
# Mainly for use as quick reference while building other projects
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import losses
from retinanet import model
from retinanet.dataloader import JSONDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

save_prefix = 'dice' # saves nnet as {save_prefix}_retinanet_{iter/final} 
csv_classes = './classes.csv' # Path to file containing class list
json_train = './annotation.json' # Path to file containing json training annotations
img_path = './images' # Path to where the images are located
depth = 50 # Resnet depth, must be one of 18, 34, 50, 101, 152
epochs = 1 # Number of epochs
start_epoch = 0

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):

    dataset_train = JSONDataset(train_file=json_train, class_file=csv_classes, img_path=img_path, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    # Create the model
    if depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, loss=losses.DiceLossCombined())
    elif depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    # Optimizer
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    start_epoch = 0
    if start_epoch != 0:
        state_path = '/content/drive/MyDrive/GMM/dice_retinanet_1.pt'
        state = torch.load(state_path)
        retinanet.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch'] + 1
        retinanet.training = True
        retinanet.train()
        retinanet.module.freeze_bn()

    for epoch_num in range(start_epoch, start_epoch + epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))

        state = {
            'epoch': epoch_num,
            'state_dict': retinanet.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, f'./{save_prefix}_retinanet_{epoch_num}.pt')

    retinanet.eval()

    torch.save(retinanet, f'./{save_prefix}_retinanet_final.pt')
    torch.save(retinanet.state_dict(), f'./{save_prefix}_retinanet_final_sd.pt')


if __name__ == '__main__':
    main()
