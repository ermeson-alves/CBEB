import lightning as L
from torch import nn
from torch import optim
import torchmetrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error


class ClassificationModule(L.LightningModule):
    def __init__(self, model_pretrained, loss_function, optimizer):
        super().__init__()
        self.model = model_pretrained
        self.optimizer = optimizer
        self.loss_module = loss_function
        self.test_step_outputs = []
        self.training_step_outputs = []

    
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
        labels = torch.cat([x["labels"] for x in outputs]).cpu()
        preds = torch.cat([x["preds"] for x in outputs]).cpu().argmax(dim=-1)
        
        acc = (preds == labels).float().mean()
        self.loss_ = self.loss_module(preds.float(), labels.float())
        if stage == 'test':

            cm = confusion_matrix(labels, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  #display_labels=['Normal', 'Pneumonia', 'Covid'])
                                  display_labels=['0', '1'])
            disp.plot(cmap='Blues')

            plt.show()
            print(classification_report(labels, preds))
            print(accuracy_score(labels, preds))
            print(f1_score(labels, preds))
            print(precision_score(labels, preds))
            print(recall_score(labels, preds))
            print(recall_score(labels, preds, pos_label=0))
        
            
        metrics = {
             f"{stage}_acc": acc,
             f"{stage}_f1": f1_score(labels, preds),
             f"{stage}_precision": precision_score(labels, preds),
             f"{stage}_recall": recall_score(labels, preds),
             
         }
        
        self.log_dict(metrics, prog_bar=True)
        
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def test_step(self, batch, batch_idx):
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