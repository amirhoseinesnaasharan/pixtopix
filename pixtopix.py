import glob
from PIL import Image
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.classification import Accuracy
from torchsummary import summary
import segmentation_models_pytorch as smp

print(f'Pytorch: {torch.__version__}')
print(f'Pytorch Vision: {torchvision.__version__}')
print(f'Pytorch Lightning: {pl.__version__}')

# Hyperparameters
DATASET_TRAIN_PATH = '/content/drive/MyDrive/train'  
TRAIN_BATCH_SIZE = 8
TRAIN_IMAGE_SIZE = 256
VAL_BATCH_SIZE = 4
VAL_IMAGE_SIZE = 256
LAMBDA = 100
NUM_EPOCH = 100

# Determine the number of available GPUs
AVAILABLE_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Dataset and Dataloader
def horizontal_split(image, ratio=0.5):
    w, h = image.size
    idx = int(w * ratio)
    left = TF.crop(image, top=0, left=0, height=h, width=idx)
    right = TF.crop(image, top=0, left=idx, height=h, width=idx)
    return left, right

class SketchDataset(Dataset):
    def __init__(self, filenames, split='train', transform=None):
        self.filenames = filenames
        self.split = split

        if not transform:
            if self.split == 'train':
                self.transform = transforms.Compose([
                    transforms.Lambda(lambda img: horizontal_split(img, 0.5)),
                    transforms.Lambda(lambda images: torch.stack([transforms.ToTensor()(item) for item in images])),
                    transforms.CenterCrop((TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE)),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(45, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=1),
                ])
            elif self.split == 'val':
                self.transform = transforms.Compose([
                    transforms.Lambda(lambda img: horizontal_split(img, 0.5)),
                    transforms.Lambda(lambda images: torch.stack([transforms.ToTensor()(item) for item in images])),
                    transforms.CenterCrop((VAL_IMAGE_SIZE, VAL_IMAGE_SIZE)),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Lambda(lambda img: horizontal_split(img, 0.5)),
                    transforms.Lambda(lambda images: torch.stack([transforms.ToTensor()(item) for item in images])),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        image = self.transform(image)
        image, target = torch.split(image, 1)
        image = torch.squeeze(image)
        image = TF.rgb_to_grayscale(image)
        target = torch.squeeze(target)
        return image, target

# All filenames
all_filenames = sorted(glob.glob(f'{DATASET_TRAIN_PATH}/*.jpg'))

# Split filenames into train and validation sets
train_filenames, val_filenames = train_test_split(all_filenames, test_size=0.2, random_state=42)

# Training dataset and dataloader
train_dataset = SketchDataset(train_filenames, split='train')
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=2, shuffle=True, drop_last=True)

# Validation dataset and dataloader
val_dataset = SketchDataset(val_filenames, split='val')
val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, num_workers=1, shuffle=False)

# Models
class Generator(nn.Module):
    def __init__(self, dropout_p=0.4):
        super(Generator, self).__init__()
        self.dropout_p = dropout_p
        self.unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                             in_channels=1, classes=3, activation=None)
        for idx in range(1, 3):
            self.unet.decoder.blocks[idx].conv1.add_module('3', nn.Dropout2d(p=self.dropout_p))
        for module in self.unet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def forward(self, x):
        x = self.unet(x)
        x = F.relu(x)
        return x

generator = Generator()
summary(generator, input_size=(1, 256, 256), device='cpu')

class Discriminator(nn.Module):
    def __init__(self, dropout_p=0.4):
        super(Discriminator, self).__init__()
        self.dropout_p = dropout_p
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(4, 128, 3, stride=2, padding=2)),
            ('bn1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(128, 256, 3, stride=2, padding=2)),
            ('bn2', nn.BatchNorm2d(256)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(256, 512, 3)),
            ('dropout3', nn.Dropout2d(p=self.dropout_p)),
            ('bn3', nn.BatchNorm2d(512)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(512, 1024, 3)),
            ('dropout4', nn.Dropout2d(p=self.dropout_p)),
            ('bn4', nn.BatchNorm2d(1024)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(1024, 512, 3, stride=2, padding=2)),
            ('dropout5', nn.Dropout2d(p=self.dropout_p)),
            ('bn5', nn.BatchNorm2d(512)),
            ('relu5', nn.ReLU()),
            ('conv6', nn.Conv2d(512, 256, 3, stride=2, padding=2)),
            ('dropout6', nn.Dropout2d(p=self.dropout_p)),
            ('bn6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU()),
            ('conv7', nn.Conv2d(256, 128, 3, stride=2, padding=2)),
            ('dropout7', nn.Dropout2d(p=self.dropout_p)),
            ('bn7', nn.BatchNorm2d(128)),
            ('relu7', nn.ReLU()),
            ('conv8', nn.Conv2d(128, 1, 3)),
        ]))

    def forward(self, x, target):
        x = torch.cat((x, target), 1)
        x = self.model(x)
        return x

discriminator = Discriminator()
summary(discriminator, input_size=[(1, 256, 256), (3, 256, 256)], device='cpu')

# Losses
adversarial_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()

# PyTorch Lightning Module
class SketchModel(pl.LightningModule):
    def __init__(self, generator, discriminator, lambda_reconstr=100):
        super(SketchModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_reconstr = lambda_reconstr

        self.criterion_GAN = nn.MSELoss()
        self.criterion_pixelwise = nn.L1Loss()

        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.val_acc = Accuracy(task='binary')

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        return self.criterion_GAN(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch

        valid = torch.ones(real_A.size(0), 1, 1, 1, requires_grad=False).type_as(real_A)
        fake = torch.zeros(real_A.size(0), 1, 1, 1, requires_grad=False).type_as(real_A)

        if optimizer_idx == 0:
            fake_B = self.generator(real_A)
            pred_fake = self.discriminator(fake_B, real_A)
            loss_GAN = self.adversarial_loss(pred_fake, valid)

            loss_pixel = self.criterion_pixelwise(fake_B, real_B)
            loss_G = loss_GAN + self.lambda_reconstr * loss_pixel

            self.log("train_loss_G", loss_G, prog_bar=True)
            return loss_G

        if optimizer_idx == 1:
            fake_B = self.generator(real_A)
            pred_real = self.discriminator(real_B, real_A)
            loss_real = self.adversarial_loss(pred_real, valid)

            pred_fake = self.discriminator(fake_B.detach(), real_A)
            loss_fake = self.adversarial_loss(pred_fake, fake)

            loss_D = 0.5 * (loss_real + loss_fake)

            self.log("train_loss_D", loss_D, prog_bar=True)
            return loss_D

    def validation_step(self, batch, batch_idx):
        real_A, real_B = batch
        fake_B = self.generator(real_A)

        loss_pixel = self.criterion_pixelwise(fake_B, real_B)

        pred_fake = self.discriminator(fake_B, real_A)
        valid = torch.ones(real_A.size(0), 1, 1, 1, requires_grad=False).type_as(real_A)
        loss_GAN = self.adversarial_loss(pred_fake, valid)

        loss_G = loss_GAN + self.lambda_reconstr * loss_pixel
        self.log("val_loss_G", loss_G, prog_bar=True)

        val_psnr = self.psnr(fake_B, real_B)
        val_ssim = self.ssim(fake_B, real_B)
        self.val_acc.update(val_psnr)
        self.val_acc.update(val_ssim)

        self.log("val_psnr", val_psnr, prog_bar=True)
        self.log("val_ssim", val_ssim, prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)

        return loss_G

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0.999
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

# Callbacks
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='checkpoints',
    filename='best-checkpoint',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

# Training
model = SketchModel(generator, discriminator, lambda_reconstr=LAMBDA)

trainer = pl.Trainer(
    max_epochs=NUM_EPOCH,
    accelerator='gpu',  # یا 'ddp' اگر از چند GPU استفاده می‌کنید
    enable_progress_bar=True,
    callbacks=[checkpoint_callback]
)
#Evaluation
#Load model
checkpoint = sorted(glob.glob('./pix2pix/checkpoints/*.ckpt'), key=os.path.getmtime)[0]
checkpoint = torch.load(checkpoint)
model = Pix2Pix()
model.load_state_dict(checkpoint['state_dict'], strict=True)
data = next(iter(test_dataloader))
image, _ = data
with torch.no_grad():
    reconstruction = model(image)
    reconstruction = torch.clip(reconstruction, 0, 1)

image = torch.stack([image for _ in range(3)], dim=1)
image = torch.squeeze(image)
grid_image = torchvision.utils.make_grid(torch.cat([image, reconstruction]), nrow=20)
plt.figure(figsize=(24, 24))
plt.imshow(grid_image.permute(1, 2, 0))
