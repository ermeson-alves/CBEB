import lightning as L
import torch
from torch import nn, optim
import torchmetrics 
import numpy as np
import matplotlib.pyplot as plt
from .setup import lr, CHECKPOINTS_DIR, metrics, multiclass, roccurve, aucroc, ticklabels
import seaborn as sns
import time

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
        self.train_time = 0
        self.test_time = 0
        
    
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
        preds = torch.max(preds_prob, dim=1).indices

        self.loss_ = self.loss_module(preds_prob, labels)
        
        dataset_name = np.unique(np.array([set(x["dataset_names"]) for x in outputs]))
        assert len(dataset_name) == 1 # test/train in a unique loader
        dataset_name = list(dataset_name[0])[0]

        
        if stage == 'test':
            tensorboard = self.logger.experiment
            metrics = self.metrics_test(preds, labels)
            metrics['test-loss'] = self.loss_
            
            # confusion matrix
            fig, ax = plt.subplots()
            cm = metrics['test-cm-to-plot']
            print(cm.cpu().numpy())
            sns.heatmap(cm.cpu().numpy(), annot=True, cmap=sns.color_palette("mako", as_cmap=True).reversed(), xticklabels=ticklabels,
                        yticklabels=ticklabels, ax=ax, fmt='d')
            plt.ylabel('True Labels')
            plt.xlabel('Predict')
            plt.savefig(f'{CHECKPOINTS_DIR}/figs/FOLD_{self.figname}-{dataset_name}_cm.png')
            tensorboard.add_histogram(f'Confusion matrix ({dataset_name})', cm)
            tensorboard.add_figure(f'plot-Confusion matrix ({dataset_name})', fig)
            
            
            # roc curve
            metrics['test-aucroc'] = aucroc(preds_prob[:, 1], labels)
            metrics['test-roccurve-to-plot'] = roccurve(preds_prob[:, 1], labels)
            fig_, ax_ = roccurve.plot(metrics['test-roccurve-to-plot'], score=True)  
            fig_.savefig(f'{CHECKPOINTS_DIR}/figs/FOLD_{self.figname}-{dataset_name}_ROC.png')
            tensorboard.add_histogram(f'fpr ({dataset_name})', metrics['test-roccurve-to-plot'][0])
            tensorboard.add_histogram(f'tpr ({dataset_name})', metrics['test-roccurve-to-plot'][1])
            tensorboard.add_histogram(f'thresholds ({dataset_name})', metrics['test-roccurve-to-plot'][2])
            tensorboard.add_figure(f'plot-ROC curve ({dataset_name})', fig_)
        else:
            metrics = self.metrics_train(preds, labels)
            metrics['train-loss'] = self.loss_
            metrics['train-aucroc'] = aucroc(preds_prob[:, 1], labels)
            metrics['train-roccurve-to-plot'] = roccurve(preds_prob[:, 1], labels)

        metrics_g = {dataset_name+'-'+key: value for key, value in metrics.items() if not '_' in key and not 'to' in key}
        self.log_dict(metrics_g, prog_bar=True)
        if multiclass:
            metrics_per_class = {dataset_name+'-'+key: value for key, value in metrics.items() \
                                 if '_' in key and not 'to' in key}
            self.log_dict(metrics_per_class, prog_bar=True)
    # Treino
    # def on_train_star(self):
    #     self.train_time = time.time()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs.copy()
        self.training_step_outputs.clear()
        return self.shared_epoch_end(outputs,"train") 

    # def on_train_end(self):
    #     self.train_time = time.time() - self.train_time
    #     self.logger.experiment.add_scalar('train_time (s)', self.train_time)
    #     self.train_time = 0

    # Teste
    # def on_test_star(self):
    #     self.test_time = time.time()
        
    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self.shared_step(batch, "test"))
        return self.shared_step(batch, "test")  

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.test_step_outputs, "test")

    def on_test_end(self):
        # self.test_time = time.time() - self.test_time
        # self.logger.experiment.add_scalar('test_time (s)', self.test_time)
        self.test_step_outputs.clear() # POSSO FAZER iSSO?
        # self.test_time = 0
        
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=lr)
