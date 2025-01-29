import torch
import os, socket, csv, time
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def init_process(rank, world_size, backend="nccl"):
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "12355")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50, rank=0, config={}):
    start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Rank {dist.get_rank()} - Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    total_time = time.time() - start_time

    data = {
        "hostname": socket.gethostname(),
        "rank": rank,
        "num_epochs": num_epochs,
        "batch_size": config.get("batch_size", 128),
        "learning_rate": config.get("learning_rate", 0.0001),
        "num_machines": config.get("num_machines", 1),
        "gpus_per_machine": config.get("gpus_per_machine", 1),
        "model": config.get("model", "resnet18"),
        "final_loss": avg_loss,
        "training_time_sec": round(total_time, 2)
    }

    if rank == 0:
        csv_filename = "training_results.csv"
        file_exists = os.path.isfile(csv_filename)

        with open(csv_filename, mode="a", newline="") as csv_file:
            fieldnames = data.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(data)

    print(f"Resultados salvos (Rank {rank})")

def main():
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))

    init_process(rank, world_size)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=False, transform=transform_train)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=2, sampler=train_sampler)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(num_classes=100).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    config = {
        "batch_size": 128,
        "learning_rate": 0.0001,
        "num_machines": world_size,
        "gpus_per_machine": torch.cuda.device_count(),
        "model": "resnet18"
    }

    train_model(model, train_loader, criterion, optimizer, device, num_epochs=50, rank=rank, config=config)

    if rank == 0:
        torch.save(model.state_dict(), "weights.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = "8"
    os.environ['KMP_AFFINITY'] = "granularity=fine,compact,1,0"
    os.environ['NCCL_DEBUG'] = "trace"

    main()
