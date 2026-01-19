import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, recall_score
class EpiDeNet(nn.Module):
    def __init__(self, output_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1,4), stride = (1,1), padding = 'same')
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,8), stride=(1,8))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(1,16), stride = (1,1), padding = 'same')
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,8), stride = (1,1), padding = 'same')
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        # Adjusted kernel size for pool4 to avoid reducing the height
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(16,1), stride=(1,1), padding='same')
        self.bn4 = torch.nn.BatchNorm2d(16)
        # Adjusted pooling to maintain the height
        self.pool4 = nn.MaxPool2d(kernel_size=(1,1), stride=(4,1))
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(8,1), stride=(1,1), padding='same')
        self.bn5 = torch.nn.BatchNorm2d(16)
        self.pool6 = nn.AdaptiveAvgPool2d((1,1))    
        self.flatten = nn.Flatten()
        self.fcn = nn.Linear(16, output_classes)

    
    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x= self.pool1(x)
        x=F.relu(self.bn2(self.conv2(x))) 
        x= self.pool2(x)
        x=F.relu(self.bn3(self.conv3(x)))
        x= self.pool3(x)
        x=F.relu(self.bn4(self.conv4(x)))
        x= self.pool4(x)
        x=F.relu(self.bn5(self.conv5(x)))
        x= self.pool6(x)
        x = self.flatten(x)
        x= self.fcn(x)
        return x
class LitEpiDeNet(pl.LightningModule):
    def __init__(self, config, output_classes=6, learning_rate=1e-3,steps_per_epoch=None):
        super().__init__()
        self.learning_rate = learning_rate

        self.accuracy = Accuracy(task='multiclass', num_classes=output_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.steps_per_epoch = steps_per_epoch
        self.model = EpiDeNet(output_classes)

    def forward(self, x: torch.Tensor):
        """
        Forward pass adjusting the input tensor shape to [B, 1, C, T]
        where B is the batch size, C is the number of channels, and T is the number of time samples.
        """
        # Adjusting the input shape to match the expected format: [B, 1, C, T]
        x = x.permute(0, 2, 1).unsqueeze(1)  # Now shape is [B, 1, 8, 2000]
        y = self.model(x)
        
        return y

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_preds = self(X)
        loss = self.criterion(y_preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_preds = self(X)
        loss = self.criterion(y_preds, y)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        
        preds = torch.argmax(y_preds, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_acc_multiclass', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        output = {'val_loss': loss, 'preds': preds, 'labels': y}
        # Store predictions and labels for further analysis in on_validation_epoch_end
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        scheduler = {
            'scheduler': OneCycleLR(optimizer, max_lr=1e-3, epochs=self.config.EPOCHS, steps_per_epoch=self.steps_per_epoch,pct_start=0.1, anneal_strategy='cos', final_div_factor=100),
            'interval': 'step',
        }
        return [optimizer], [scheduler]
