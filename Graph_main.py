import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn import preprocessing
from data import get_data,GNN_data
import random
label_encoder = preprocessing.LabelEncoder() 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from torch_geometric.loader import DataLoader
from GNN_models import train,test,GCN,GNN

random.seed(0)

if __name__ == '__main__': 
    path=r"E:\DSS\DSS_data.xlsx" 
    _,_, data, _=get_data(path=path)
    train_dataset, test_dataset=GNN_data(data)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = GNN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()



    for epoch in range(1, 200):
        train(model,train_loader,optimizer,criterion)
        train_acc, train_precision, train_recall, train_f1 = test(model,train_loader)
        val_acc, val_precision, val_recall, val_f1 = test(model,test_loader)  # Assuming you have a separate validation loader.
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Prec: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')