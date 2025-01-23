import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def init_process(rank, world_size, backend="nccl"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, use_libuv=False)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

def main(rank, world_size):
    init_process(rank, world_size)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=False, transform=transform_train)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=2, sampler=train_sampler)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(num_classes=100).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, train_loader, criterion, optimizer, device, num_epochs=50)

    if rank == 0:
        torch.save(model.state_dict(), "weights.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['OMP_NUM_THREADS'] = "8"
    os.environ['KMP_AFFINITY'] = "granularity=fine,compact,1,0"
    os.environ['NCCL_DEBUG'] = "trace"

    rank = 0#int(os.environ['OMPI_COMM_WORLD_RANK'])
    world_size = 1#int(os.environ['OMPI_COMM_WORLD_SIZE'])

    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
