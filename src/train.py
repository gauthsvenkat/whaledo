from pytorch_metric_learning import losses
from src.dataloader import WhaleDoDataset
from src.models import WhaleDoModel
from torch.utils.data import DataLoader
import torch

config = {
    'csv_path': 'data/metadata.csv',
    'backbone_model': 'resnet18',
    'input_dim': (3, 443, 291),
    'rep_dim': 512,
    'pretrained': True,
    'device': 'cpu',

    'projector' : {
        'hidden_dim': 1024,
        'output_dim': 2
    }
}

loss_func = losses.TripletMarginLoss()
dataset = WhaleDoDataset(csv_path=config['csv_path'])
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = WhaleDoModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i, (x_batch, y_batch) in enumerate(train_loader):
    optimizer.zero_grad()
    embeddings = model(x_batch)
    loss = loss_func(embeddings, y_batch)
    loss.backward()
    optimizer.step()

    print(i)