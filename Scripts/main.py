from trainer import train
from architecture import *
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from config import *


def trainer(config):

    # Move the model to the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'currently using : {device}')
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    train_set = MNIST(root='./MNIST',train=True,download=True,transform=transforms.ToTensor())
    test_set = MNIST(root='./MNIST',train=False,download=True,transform=transforms.ToTensor())

    train_loader = DataLoader(train_set,batch_size=config["batch_size"],shuffle=True,
                            pin_memory=config["pin_memory"],num_workers=config["num_workers"])
    
    test_loader = DataLoader(test_set,config["batch_size"],shuffle=False,
                            pin_memory=config["pin_memory"],num_workers=config["num_workers"])

    # Initialize the model
    vit = ViT(num_classes=config["num_classes"],
            in_channels=config["in_channels"],
            patch_size=config["patch_size"],
            img_size=config["img_size"])
    
    vit.to(device)
    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vit.parameters(), lr=config["lr"],fused=True)

    results = train(model=vit,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=criterion,
        epochs=config["epochs"],
        device=device)
    return results

if __name__ == "__main__":
    config = get_config()
    trainer(config)