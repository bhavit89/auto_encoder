import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data_utils import load_data, log_4_reconstruction_tb
from model import AutoEncoder
from asci import print_asci

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = AutoEncoder().to(device)
    dummy_image = torch.rand(1, 1, 28, 28).to(device)
    # print(summary(autoencoder, input_size=(1, 28, 28)))
    print(print_asci())

    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 1e-3

    CRITERION = nn.MSELoss()
    OPTIMIZER = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter("runs/auto_encoder_mnist")

    train_loader, valid_loader, test_loader = load_data(batch_size=BATCH_SIZE, num_workers=0)

    for image, _ in train_loader:
        print("[------IMAGE DIMENSIONS---------]", image.shape)
        break

    for epoch in range(1, EPOCHS + 1):
        autoencoder.train()
        train_loss = 0.0

        for images, _ in tqdm(train_loader, desc=f"EPOCH[{epoch}/{EPOCHS}]"):
            images = images.to(device)
            recon = autoencoder(images)
            loss = CRITERION(recon, images)
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in valid_loader:
                images = images.to(device)
                recon = autoencoder(images)
                val_loss += CRITERION(recon, images).item()

        val_loss /= len(valid_loader)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        log_4_reconstruction_tb(autoencoder, test_loader, writer, epoch, device, num_images=4)
        writer.flush()

        if epoch % 5 == 0:
            print(f"EPOCH {epoch:03d} ------ Train Loss {train_loss:.4f} ----- Valid Loss {val_loss:.4f}")

    writer.close()

if __name__ == "__main__":
    main()
