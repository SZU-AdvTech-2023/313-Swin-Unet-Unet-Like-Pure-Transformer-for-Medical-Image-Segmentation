import os

import cv2
import torch
from tqdm import tqdm

from dataset import Dataset
from transforms import get_transform


def main():
    print("=> creating model")
    model = torch.load("/home/tjc/pycharmprojects/swin-unet/model1.pth")
    model.eval()

    val_dataset = Dataset("/home/tjc/pycharmprojects/newdata/dataset1/train_val_seg/val", get_transform(train=False))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False,
                                             collate_fn=val_dataset.collate_fn, drop_last=False)
    val_names = val_dataset.names
    count = 0

    os.makedirs(os.path.join('outputs', 'mask_pred'), exist_ok=True)

    with torch.no_grad():
        for input, target in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()

            output = model(input)

            mask = output.clone()
            mask = torch.sigmoid(mask).cpu().numpy() > 0.5

            for i in range(len(mask)):
                cv2.imwrite(os.path.join('outputs', 'mask_pred', val_names[count].split('.')[0] + '.png'),
                            (mask[i, 0] * 255).astype('uint8'))
                count = count + 1

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
