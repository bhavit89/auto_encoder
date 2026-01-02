from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter("runs/autoencoder_mnist")

for i in range(5):
    writer.add_scalar("test/loss", 1.0 / (i + 1), i)
    time.sleep(0.1)

writer.close()

print("TensorBoard logs written!")
