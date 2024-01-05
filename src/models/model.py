import matplotlib.pyplot as plt 
import numpy as np
import json
# from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
from pathlib import Path
import lightning as L
from torchmetrics import F1Score
from torchmetrics.classification import ConfusionMatrix, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch
from src.preprocessing.constants import BASE_PATH

import math
class HateDetection_LM_CNN(L.LightningModule):
    def __init__(self,
                  language_model_name_or_path: str,
                  batch_size: int = 4,
                  max_length: int = 36, 
                  freeze_lm: bool = True,
                  number_of_classes: int = 3,
                  lr: float = 2e-5,
                  class_weights=None):
        super().__init__()
        self.number_of_classes = number_of_classes

        self.batch_size = batch_size
        self.max_length = max_length
        self.current_device = "cuda" if torch.cuda.is_available() else "cpu"
        if class_weights is not None:
            self.class_weights = class_weights.to(self.current_device)
        else:
            self.class_weights = None
        self.language_model_name_or_path = language_model_name_or_path
        self.language_model =  AutoModelForSequenceClassification.from_pretrained(language_model_name_or_path) 
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name_or_path, use_fast=False)
        self.config = self.language_model.config
        self.lr = lr
        if freeze_lm:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        self.num_in_channels = self.language_model.config.num_hidden_layers + 1
        self.num_out_channels = self.num_in_channels 
        
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
        self.test_step_outputs = []
        self.test_labels = []

        self.accuracy = MulticlassAccuracy(num_classes=self.number_of_classes).to(self.current_device)
        self.f1_score = F1Score(num_classes=self.number_of_classes, average='weighted', task="multiclass").to(self.current_device)
        self.precision_weighted = MulticlassPrecision(num_classes=self.number_of_classes, average='weighted').to(self.current_device)
        self.recall_weighted = MulticlassRecall(num_classes=self.number_of_classes, average='weighted').to(self.current_device)
        self.confusion_matrix = ConfusionMatrix(num_classes=self.number_of_classes, task='multiclass').to(self.current_device)

    def get_logger_path(self):
        """Constructs and returns the logger path"""
        class_weight_flag = "class_weights" if self.class_weights is not None else "no_class_weights"
        base_logs_path = Path(BASE_PATH) / "logs" / self.language_model_name_or_path
        logger_path = base_logs_path / class_weight_flag
        logger_path.mkdir(parents=True, exist_ok=True)
        return logger_path
    
    
    def forward(self, input_ids, attention_mask):
        outputs = self.language_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        all_layers = torch.stack(outputs.hidden_states, dim=0)
        x = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in all_layers]), 0), 0, 1)
        x = self.pool(self.dropout(self.relu(self.conv(self.dropout(x)))))
        x = self.linear(self.dropout(self.flat(self.dropout(x))))
        return self.softmax(x)

    def training_step(self, batch):
        input_ids, attention_mask, labels = batch
        output = self.forward(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(output, labels)
        self.log(f'train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        input_ids, attention_mask, labels = batch
        output = self.forward(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(output, labels)
        self.log(f'val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch):
        input_ids, attention_mask, labels = batch
        output = self.forward(input_ids, attention_mask)
        predictions = torch.argmax(output, dim=1)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(output, labels)

        #calculate metrics
        accuracy = self.accuracy(predictions, labels)
        self.test_labels.append(labels)
        self.test_step_outputs.append(predictions)
        self.log(f'test_loss', loss, prog_bar=True)
        self.log(f'test_acc', accuracy, prog_bar=True)

        return loss
    
    def on_test_epoch_end(self):
        f1 = self.f1_score(torch.cat(self.test_step_outputs), torch.cat(self.test_labels))
        confusion_matrix = self.confusion_matrix(torch.cat(self.test_step_outputs),
                                                  torch.cat(self.test_labels))
        accuracy = self.accuracy(torch.concat(self.test_step_outputs), torch.concat(self.test_labels))
        precision_test = self.precision_weighted(torch.cat(self.test_step_outputs), torch.cat(self.test_labels))
        recall_test = self.recall_weighted(torch.cat(self.test_step_outputs), torch.cat(self.test_labels))

        # Construct the logger path
        logger_path = self.get_logger_path()

        # Writing results to file
        result_path = logger_path / f"test_results.json"
        test_results = {"f1": f1, "accuracy": accuracy, "precision": precision_test, "recall": recall_test}
        with open(result_path, "w") as f:
            json.dump(test_results, f, indent=2)


        # Plot and log confusion matrix
        self.plot_confusion_matrix(confusion_matrix, path=logger_path)
        self.log('test_f1', f1)
        return f1

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def plot_confusion_matrix(self, confusion_matrix, path, save_fig: bool = True, show_fig: bool = False)):

        # Assuming confusion_matrix is a PyTorch tensor
        confusion_matrix_np = confusion_matrix.cpu().numpy()  # Convert to a NumPy array


        plt.figure(figsize=(10, 10))
        sns.heatmap(confusion_matrix_np, annot=True, fmt='g', cmap='Blues',
                    xticklabels=["hate_speech", "offensive_language", "neither"],
                    yticklabels=["hate_speech", "offensive_language", "neither"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title('Confusion Matrix Heatmap')
        
        class_weight_flag = "class_weights" if self.class_weights is not None else "no_class_weights"

        if save_fig:
            plt.savefig(path / f"confusion_matrix_class_weights_{class_weight_flag}_not_normalized.png")
        
        if show_fig:
            plt.show()
        
        plt.close()

