import lightning as L
import torch
from torch import nn, optim
import torchmetrics 
import numpy as np
import matplotlib.pyplot as plt
from .setup import lr, CHECKPOINTS_DIR, metrics, multiclass


class ClassificationModule(L.LightningModule):
    def __init__(self, pretrained_model, loss_function, optimizer, figname=None):
        super().__init__()
        self.model = pretrained_model
        self.figname = figname # for confusion matrix and roc curve plots
        self.optimizer = optimizer
        self.loss_module = loss_function
        self.test_step_outputs = []
        self.training_step_outputs = []
        self.metrics_train = metrics.clone(prefix='train-')
        self.metrics_test = metrics.clone(prefix='test-')
    
    def shared_step(self, batch, stage):
        imgs, labels = batch['img'], batch['retinopathy_grade']
        dataset_names = batch['dataset_name']
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
                
        outputs = {
            "loss": loss,
            "labels": labels,
            "preds": preds,
            "dataset_names": dataset_names,
        }
        
        if stage == 'train':
            self.training_step_outputs.append(outputs)

        return outputs

    def shared_epoch_end(self, outputs, stage):
        labels = torch.cat([x["labels"] for x in outputs])
        preds_prob = torch.cat([x["preds"] for x in outputs])
        dataset_name = np.unique(np.array([set(x["dataset_names"]) for x in outputs]))
        
        assert len(dataset_name) == 1 # test/train in a unique loader
        dataset_name = list(dataset_name[0])[0]
        
        self.loss_ = self.loss_module(preds_prob, labels)
            
        if not multiclass:
            preds_prob = preds_prob.max(dim=-1).values
            
        if stage == 'test':
            # Isso deve salvar apenas uma imagem. Garantir q as metricas representam todo o treino!
            metrics = self.metrics_test(preds_prob, labels)
            # confusion matrix # AJEITAR ISSO
            fig_, ax_ = self.metrics_test['cm_to_plot'].plot(metrics['test-cm_to_plot'])
            fig_.savefig(f'{CHECKPOINTS_DIR}/figs/FOLD_{self.figname}-{dataset_name}_cm.png')
            
            # roc curve
            fig_, ax_ = self.metrics_test['roccurve_to_plot'].plot(metrics['test-roccurve_to_plot'])  
            fig_.savefig(f'{CHECKPOINTS_DIR}/figs/FOLD_{self.figname}-{dataset_name}_ROC.png')
        else:
            metrics = self.metrics_train(preds_prob, labels)

        metrics = {dataset_name+'-'+key: value for key, value in metrics.items() if not '_' in key}
        self.log_dict(metrics, prog_bar=True)
        if multiclass:
            metrics_per_class = {dataset_name+'-'+key: value for key, value in metrics.items() \
                                 if '_' in key and not '_to' in key}
            self.log_dict(metrics_per_class, prog_bar=True)
            

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs.copy()
        self.training_step_outputs.clear()
        return self.shared_epoch_end(outputs,"train") 
        

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self.shared_step(batch, "test"))
        return self.shared_step(batch, "test")  

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.test_step_outputs, "test")

    def on_test_end(self):
        self.test_step_outputs.clear() # POSSO FAZER iSSO?
        
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=lr)

