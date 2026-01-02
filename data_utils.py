import torch
from  torchvision import datasets , transforms
from torch.utils.data import DataLoader , SubsetRandomSampler
import matplotlib.pyplot as plt 


def load_data(batch_size , num_workers =0 ,train_transform=None ,test_transform=None):
    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()


    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=train_transform
    )

    valid_data = datasets.MNIST(
        root ='data',
        train=False,
        download=True,
        transform=test_transform
    )

    test_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=test_transform
    )
    
    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
        )
    
    valid_data_loader = DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
        )
    
    test_data_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
        )
    

    return train_data_loader ,valid_data_loader,test_data_loader


def visualize_reconstruction(model,dataloader,device,num_images=16):
    model.eval()
    images,_ = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        recon = model(images)

    images = images.cpu()
    recon = recon.cpu()

    fig ,axes = plt.subplot(2 ,num_images, figsize=(num_images,3))

    for i in range(num_images):
        axes[0 ,i].imshow(images[i][0],cmap="grey")
        axes[0, i].axis("off")

        axes[1,i].imshow(recon[i][0],cmap="grey")
        axes[1,i].axis("off")

    axes[0,0].set_ylabel("orginal" ,fontsize=12)
    axes[1,0].set_ylabel("Recon",fontsize=12)


    plt.tight_layout()
    plt.show 



def log_4_reconstruction_tb(model,dataloader,writer,epoch,device,num_images=4):
    model.eval()
    images ,_ = next(iter(dataloader))
    images = images[:num_images].to(device)

    with torch.no_grad():
        recon = model(images)
    
    combined = torch.cat([images ,recon],dim=3)
    writer.add_images("Orginal vs Recon" ,combined ,epoch)

