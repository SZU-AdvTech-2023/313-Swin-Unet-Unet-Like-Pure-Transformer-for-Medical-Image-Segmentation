import os

import torch
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm

from dataset import Dataset
from networks.vision_transformer import SwinUnet as ViT_seg
from transforms import get_transform
from utils.config import parse_args
from utils.losses import BCEDiceLoss
from utils.metrics import iou_score, dice_coef
from utils.utils import AverageMeter


def main():
    os.environ["WANDB_API_KEY"] = 'cdc9d021d94adc1de2796c1c3be4f798060945cf'
    os.environ["WANDB_MODE"] = "offline"

    config = vars(parse_args())

    accelerator = Accelerator(mixed_precision='no', log_with='wandb')
    if accelerator.is_main_process:
        accelerator.init_trackers('swin-unet', config=config, init_kwargs={'wandb': {'name': 'data7 adamw'}})

    # prepare the data
    train_root_path = os.path.join(config['root_dir'], 'train')
    val_root_paht = os.path.join(config['root_dir'], 'val')

    train_dataset = Dataset(train_root_path, get_transform(train=True))
    val_dataset = Dataset(val_root_paht, get_transform(train=False))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=val_dataset.collate_fn)

    model = ViT_seg()
    model.load_from()
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    best_iou = 0.

    for epoch in range(config['epochs']):
        accelerator.print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))
        avg_meters = {'train_loss': AverageMeter(), 'val_iou': AverageMeter(), 'val_dice': AverageMeter()}

        # train
        model.train()
        for image, mask in tqdm(train_loader):
            output = model(image).squeeze()
            loss = criterion(output, mask)
            avg_meters['train_loss'].update(loss.item(), image.size(0))

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        accelerator.log({'train_loss': avg_meters['train_loss'].avg})
        print('Training Loss : {:.4f}'.format(avg_meters['train_loss'].avg))

        model.eval()
        for image, mask in tqdm(val_loader):
            with torch.no_grad():
                pred = model(image).squeeze()

            pred, mask = accelerator.gather_for_metrics((pred, mask))
            iou = iou_score(pred, mask)
            dice = dice_coef(pred, mask)

            avg_meters['val_iou'].update(iou, image.size(0))
            avg_meters['val_dice'].update(dice, image.size(0))

        accelerator.log({'val_iou': avg_meters['val_iou'].avg, 'val_dice': avg_meters['val_dice'].avg})
        accelerator.print('val_iou %.4f - val_dice %.4f' % (avg_meters['val_iou'].avg, avg_meters['val_dice'].avg))

        accelerator.wait_for_everyone()
        if avg_meters['val_iou'].avg > best_iou:
            best_iou = avg_meters['val_iou'].avg
            model = accelerator.unwrap_model(model)
            accelerator.save(model, "model1.pth")

    print(best_iou)


if __name__ == '__main__':
    main()
