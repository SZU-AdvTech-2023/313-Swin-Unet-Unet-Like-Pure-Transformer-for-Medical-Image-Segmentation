import os

from PIL import Image
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transforms = transform

        self.names = [i for i in os.listdir(os.path.join(self.root_dir, 'image')) if i.endswith('.jpg')]
        self.image_list = [os.path.join(self.root_dir, 'image', i) for i in self.names]
        self.mask_list = [os.path.join(self.root_dir, 'mask', i.replace('.jpg', '.png')) for i in self.names]

        # check files
        for i in self.image_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        for i in self.mask_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert('RGB')
        mask = Image.open(self.mask_list[idx]).convert('L')

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask / 255

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
