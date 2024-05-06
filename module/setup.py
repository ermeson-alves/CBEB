from torch import nn
from torchvision.transforms import v2
import torch
import numpy as np
import cv2
from torchmetrics import MetricCollection 
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score, Specificity, ConfusionMatrix, ROC, AUROC

num_classes = 2
random_state = 42
k_folds = 5
num_epochs = 25
batch_size = 32
loss_function = nn.CrossEntropyLoss()
lr=1e-3
CHECKPOINTS_DIR = './checkpoints_kfold'
multiclass = True if num_classes > 2 else False
global metrics, roccurve, aucroc, ticklabels

class CLAHETransform(nn.Module):
    def forward(self, img):
        channels = img.split()
        clahe = cv2.createCLAHE(clipLimit = 5)
        img = cv2.merge(list(map(clahe.apply, [np.array(c) for c in channels])))
        return img
        

data_transforms = {
    'train': v2.Compose([
        # CLAHETransform(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(10),
        v2.RandomEqualize(0.05),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': v2.Compose([
        # CLAHETransform(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

binary_metrics = MetricCollection({'acc': Accuracy('binary'),
                                   'recall': Recall('binary'),
                                   'precision': Precision('binary'),
                                   'f1': F1Score('binary'),
                                   'specificity': Specificity('binary'),
                                   # 'auroc': AUROC('binary'),
                                   # 'roccurve_to_plot': ROC('binary'),
                                   'cm-to-plot': ConfusionMatrix('binary')
                                  })

multiclass_metrics = MetricCollection({'acc': Accuracy('multiclass', num_classes=num_classes),
                            'acc_per_class': Accuracy('multiclass', num_classes=num_classes, average=None),
                            'recall': Recall('multiclass', num_classes=num_classes),
                            'recall_per_class': Recall('multiclass', num_classes=num_classes, average=None),
                            'precision': Precision('multiclass', num_classes=num_classes),
                            'precision_per_class': Precision('multiclass', num_classes=num_classes, average=None),
                            'f1': F1Score('multiclass', num_classes=num_classes),
                            'f1_per_class': F1Score('multiclass', num_classes=num_classes, average=None),
                            'specificity': Specificity('multiclass', num_classes=num_classes),
                            'specificity_per_class': Specificity('multiclass', num_classes=num_classes, average=None),
                            # 'auroc': AUROC('multiclass', num_classes=num_classes),
                            # 'auroc_per_class': AUROC('multiclass', num_classes=num_classes, average=None),
                            # 'roccurve_to_plot': ROC('multiclass', num_classes=num_classes),
                            'cm-to-plot': ConfusionMatrix('multiclass', num_classes=num_classes)
                            })
if num_classes==2:
    metrics = binary_metrics
    roccurve = ROC('binary')
    aucroc = AUROC('binary')
    ticklabels = ['NO DR', 'DR']

else:
    metrics = multiclass_metrics
    roccurve = ROC('multiclass', num_classes=num_classes)
    aucroc = AUROC('multiclass', num_classes=num_classes)
    ticklabels = ['MILD', 'MODERATE', 'SEVERE', 'PROLIFERATIVE']
    