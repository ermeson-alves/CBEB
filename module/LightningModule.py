import lightning as L
import torch
from torch import nn, optim
import torchmetrics 
import matplotlib.pyplot as plt
from .setup import lr, CHECKPOINTS_DIR, metrics, num_classes


class ClassificationModule(L.LightningModule):
    def __init__(self, model_pretrained, loss_function, optimizer, figname=None):
        super().__init__()
        self.figname = figname # for confusion matrix and roc curve plots
        self.model = model_pretrained
        self.optimizer = optimizer
        self.loss_module = loss_function
        self.test_step_outputs = []
        self.training_step_outputs = []
        self.metrics = metrics

    
    def shared_step(self, batch, stage):
        imgs, labels = batch['img'], batch['retinopathy_grade']
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
                
        outputs = {
            "loss": loss,
            "labels": labels,
            "preds": preds
        }
        if stage == 'train':
            self.training_step_outputs.append(outputs)

        return outputs

    
    def shared_epoch_end(self, outputs, stage):
        # labels = torch.cat([x["labels"] for x in outputs]).cpu() # flaten
        # preds = torch.cat([x["preds"] for x in outputs]).cpu().argmax(dim=-1) # flaten
        # acc = (preds == labels).float().mean()
        
        labels = torch.cat([x["labels"] for x in outputs])
        preds = torch.cat([x["preds"] for x in outputs]).argmax(dim=-1)
        
        self.loss_ = self.loss_module(preds.float(), labels.float())
        if stage == 'test':
            
            # confusion matrix
            # disp_cm = ConfusionMatrixDisplay.from_predictions(labels, preds,
            #                 display_labels=['No DR', 'DR'] if num_classes==2 else ['Mild', 'Moderate', 'Severe', 'Proliferative'],
            #                 cmap='Blues')
            
            # roc curve
            # disp_roc = RocCurveDisplay.from_predictions(labels, preds,
            #                                 name=self.model._get_name())
            
            # disp_cm.figure_.savefig(f'{CHECKPOINTS_DIR}/figs/FOLD_{self.figname}_cm.png')
            # disp_roc.figure_.savefig(f'{CHECKPOINTS_DIR}/figs/FOLD_{self.figname}_RocCurve.png')



            
        metrics = self.metrics(preds, labels)
         }
        
        self.log_dict(metrics, prog_bar=True)
        
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def test_step(self, batch, batch_idx, dataloader_idx):
        self.test_step_outputs.append(self.shared_step(batch, "test"))
        return self.shared_step(batch, "test") 

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs.copy()
        self.training_step_outputs.clear()
        return self.shared_epoch_end(outputs,"train")

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.test_step_outputs, 'test')

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=lr)