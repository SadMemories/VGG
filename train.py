import torch
import os
import json
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from shutil import rmtree
from model import VGG
from tqdm import tqdm


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('using {} device for training...'.format(device))

    # writ_path = 'runs/flower_experiment'
    # if os.path.exists(writ_path):
    #    rmtree(writ_path)  # 删除文件夹以及里面的文件
    # os.makedirs(writ_path)  # 递归创建文件夹
    writer = SummaryWriter()

    data_root = '/home/wwj/Image_classification/AlexNet/data'
    assert os.path.exists(data_root), '{} path is not exists...'.format(data_root)

    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    }

    train_path = data_root + os.sep + 'train'
    assert os.path.exists(train_path), '{} path is not exists...'.format(train_path)
    train_dataset = datasets.ImageFolder(root=train_path,
                                         transform=data_transform['train'])
    train_num = len(train_dataset)

    class_index = dict((val, key) for key, val in train_dataset.class_to_idx.items())
    json_val = json.dumps(class_index, indent=4)
    json_path = './class_index.json'
    if os.path.exists(json_path):
        os.remove(json_path)
    with open(json_path, 'w') as f:
        f.write(json_val)

    val_path = data_root + os.sep + 'val'
    assert os.path.exists(val_path), '{} path is not exists...'.format(val_path)
    val_dataset = datasets.ImageFolder(root=val_path,
                                       transform=data_transform['val'])
    val_num = len(val_dataset)
    print("{} train samples, {} val samples are used...".format(train_num, val_num))

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 0 else 0, 8])
    print("using {} num workers...".format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw)

    model = VGG(num_classes=5, init_weight=True)  # 这里必须要初始化权重 不初始化权重准确率无法提升
    model.to(device)
    input = torch.zeros([32, 3, 224, 224], device=device)
    writer.add_graph(model, input)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 50
    iteration_loss = 0.0
    best_acc = 0.0  # 表示准确率
    model_save_path = './Vgg.pth'
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t1 = time.perf_counter()

        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            optimizer.zero_grad()
            img, label = data
            output = model(img.to(device))
            loss = loss_function(output, label.to(device))
            iteration_loss += loss.item()
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            train_bar.desc = 'trian loss: {}'.format(loss.item())

            if step % 10 == 9:
                writer.add_scalar("iteration loss", iteration_loss / 10, epoch * len(train_loader) + step)
                iteration_loss = 0.0

        t2 = time.perf_counter() - t1
        writer.add_scalar("epoch loss", epoch_loss, epoch)
        print("time consuming: {}".format(t2))

        model.eval()  # 将模型切换至测试模式
        val_loss = 0.0
        right_num = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='val process...')
            for val_step, val_data in enumerate(val_bar):
                val_img, val_label = val_data
                output = model(val_img.to(device))
                valLoss = loss_function(output, val_label.to(device))
                val_loss += valLoss.item()

                # torch.max(output, dim=1)返回两个值 分别是 (value, indice) 而我们只需要使用indice
                predict = torch.max(output, dim=1)[1]
                right_num += (predict == val_label.to(device)).sum().item()

            precision = right_num / val_num
            writer.add_scalar("val precision", precision, epoch)
            writer.add_scalar("val loss", val_loss, epoch)
            if precision > best_acc:
                best_acc = precision
                # model.state_dict()返回的是一个Orderdict 存储了网络结构的名字和对应的参数
                torch.save(model.state_dict(), model_save_path)
            print("traing epoch: {}/{}  test accuracy: {}".format(epoch+1, epochs, precision))

    print("Finish Training!!")


if __name__ == '__main__':
    main()