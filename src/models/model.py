import lightning as L

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch

import math
class HateDetection_LM_CNN(L.LightningModule):
    def __init__(self,
                  language_model_name_or_path: str,
                  batch_size: int = 4,
                    max_length: int = 36, 
                    freeze_lm: bool = True,
                    number_of_classes: int = 3):
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.language_model_name_or_path = language_model_name_or_path  
        self.language_model =  AutoModelForSequenceClassification.from_pretrained(language_model_name_or_path) 
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name_or_path)
        self.config = self.language_model.config
        if freeze_lm:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        self.num_in_channels = self.language_model.config.num_hidden_layers + 1
        self.num_out_channels = self.language_model.config.num_hidden_layers + 1
        
        self.conv = nn.Conv2d(
            in_channels=self.num_in_channels,
            out_channels=self.num_out_channels,
            kernel_size=(number_of_classes, self.config.hidden_size),
            padding=(1, 1))
        
        H_out_size = math.floor((self.max_length + (2 * self.conv.padding[0]) - (self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)) - 1) / self.conv.stride[0] + 1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.1)
        self.linear_input_size =  (H_out_size - self.pool.kernel_size + 1) * self.num_out_channels
        self.linear = nn.Linear(self.linear_input_size, number_of_classes)
        self.flat = nn.Flatten()
        
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.language_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        all_layers = torch.stack(outputs.hidden_states, dim=0)
        x = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in all_layers]), 0), 0, 1)
        torch.cuda.empty_cache()
        x = self.pool(self.dropout(self.relu(self.conv(self.dropout(x)))))
        x = self.linear(self.dropout(self.flat(self.dropout(x))))
        return self.softmax(x)

    def training_step(self, batch):
        input_ids, attention_mask, labels = batch
        output = self.forward(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(output, labels)
        self.log(f'train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        input_ids, attention_mask, labels = batch
        output = self.forward(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(output, labels)
        self.log(f'val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)

