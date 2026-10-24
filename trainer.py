import torch.nn as nn
import torch

class Trainer:

    def __init__(self, model, data_loader, device:str, a: dict, resnet_lr: float, 
                 head_lr: float, batch_size: int=32):
        
        self.data_loader = data_loader
        self.a = a
        self.model = model
        self.batch_size = batch_size
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        params_list = [{'params': self.model.resnet.parameters(), 'lr': resnet_lr}]
        for i in range(len(self.model.heads)):
            params_list.append({'params': self.model.heads[i].parameters(), 'lr': head_lr})

        self.optimizer = torch.optim.AdamW(params_list)

    def start(self, epochs:int = 160, display: bool = False, display_in: bool | int = False):

        epoch_losses = []    

        for epoch in range(epochs):
            if epoch > epochs:
                break

            losses = []

            for data, targets in self.data_loader:

                data = data.to(self.device)
                Ps = self.model(data)

                los = []

                for i in range(len(Ps)):

                    true = targets[f'task_{i+1}'].to(self.device)
                    loss = self.criterion(Ps[i], true) * self.a[f"task_{i+1}"]
                    los.append(loss)

                total_loss = torch.sum(torch.stack(los))
                
                total_loss.backward()
                self.optimizer.step()

                losses.append(total_loss.item())
            mean_loss = torch.mean(torch.tensor(losses)).item()
            epoch_losses.append(mean_loss)

            if display:
                if epoch % display_in == 0:
                    print(f"\nTrainer summary for the {epoch}/{epochs} epoch:") 
                    print(f"Loss: {mean_loss:.2f}")
                    print(f"Computing device: {self.device}")
                    print(f"Criterion: {type(self.criterion)}")

        return self.model, epoch_losses           

    def end(self):
        torch.save(self.model.state_dict(), "trained_ap.pt")
        print("successfull trained!")


