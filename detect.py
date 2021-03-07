import torch
import os
import json
from torchvision import transforms
from model import VGG
from PIL import Image


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img_path = './tulips.jpg'
    assert os.path.exists(img_path), '{} img file is not exists...'.format(img_path)
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0).to(device)

    json_path = './class_index.json'
    assert os.path.exists(json_path), '{} json file is not exists...'.format(json_path)
    cls_index = json.load(open(json_path, 'r'))

    weight_path = './VGG.pth'
    assert os.path.exists(weight_path), '{} weight path is not exits...'.format(weight_path)
    model = VGG(num_classes=5).to(device)
    model.load_state_dict(torch.load(weight_path))  # 加载模型权重
    model.eval()

    with torch.no_grad():
        predict = model(img)
        output = torch.squeeze(predict)
        pre_val = torch.softmax(output, dim=0).cpu()
        pre_cls = torch.argmax(pre_val).numpy()

    print("class:{}, prob:{}".format(cls_index[str(pre_cls)], pre_val[pre_cls].numpy()))


if __name__ == '__main__':
    main()