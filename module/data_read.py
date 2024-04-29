from pathlib import Path
import torch
import torchvision
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class ClassificationDataset(Dataset):
    def __init__(self, images_dir: Path, annotations_file: Path, transform=None, transform_target=None):
        self.images_dir = images_dir
        try:
            self.annotations = pd.read_excel(annotations_file)
        except:
            self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        self.transform_target = transform_target


    def __len__(self):
        return len(self.annotations)


    @staticmethod
    def load_and_preprocess_img(path):
        img = Image.open(path)
        return img



class MessidorDataset(ClassificationDataset):
    def __init__(self, images_dir: Path, annotations_file: Path, transform=None, transform_target=None):
        super().__init__(images_dir, annotations_file, transform, transform_target)


    def __getitem__(self, idx):
        img_path = self.images_dir / self.annotations.iloc[idx, 0]
        image = self.load_and_preprocess_img(img_path)
        retinopathy_grade = self.annotations.iloc[idx, 2]
        risk_macular_edema = self.annotations.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        if self.transform_target:
             retinopathy_grade = self.transform_target(retinopathy_grade)
             risk_macular_edema = self.transform_target(risk_macular_edema)

        return {'img': image,
                'retinopathy_grade': retinopathy_grade,
                'risk_macular_edema': risk_macular_edema,
                'img_name': img_path.name
               }


class IDRIDDataset(ClassificationDataset):
    def __init__(self, images_dir: Path, annotations_file: Path, transform=None, transform_target=None, convert_to_binary=False):
        super().__init__(images_dir, annotations_file, transform, transform_target)
        self.convert_to_binary = convert_to_binary
        if not convert_to_binary: # se não realizar classificação binaria, pega as imagens com grau de RD entre 1 e 4
            self.annotations = self.annotations[(self.annotations['Retinopathy grade'] < 5) \
                                                & (self.annotations['Retinopathy grade'] > 0)]
        

    def __getitem__(self, idx):
        img_path = self.images_dir / f'{self.annotations.iloc[idx, 0]}.jpg'
        image = self.load_and_preprocess_img(img_path)
        retinopathy_grade = self.annotations.iloc[idx, 1]

        # Convertendo para binário se necessário
        if self.convert_to_binary:
            retinopathy_grade = 1 if retinopathy_grade != 0 else 0

        # Aplicando transformações, se fornecidas
        if self.transform:
            image = self.transform(image)

        if self.transform_target:
            retinopathy_grade = self.transform_target(retinopathy_grade)

        return {'img': image,
                'retinopathy_grade': retinopathy_grade
              }


class DDRDataset(ClassificationDataset):
    def __init__(self, images_dir: Path, annotations_file: Path, transform=None, transform_target=None, convert_to_binary=False):
        super().__init__(images_dir, annotations_file, transform, transform_target)
        self.annotations = pd.read_csv(annotations_file, header=None, sep=' ')

        if convert_to_binary: # filtro de imagens com qualidade ruim (classe 5)
            self.annotations = self.annotations[self.annotations[1] < 5]
        else: # se não realizar classificação binaria, pega as imagens com grau de RD entre 1 e 4
            self.annotations = self.annotations[(self.annotations[1] < 5) & (self.annotations[1] > 0)]
        self.convert_to_binary = convert_to_binary

    def __getitem__(self, idx):
        img_path = self.images_dir / f'{self.annotations.iloc[idx, 0]}'
        image = self.load_and_preprocess_img(img_path)
        retinopathy_grade = self.annotations.iloc[idx, 1]

        # Convertendo para binário se necessário
        if self.convert_to_binary:
            retinopathy_grade = 1 if retinopathy_grade != 0 else 0
        # Aplicando transformações nos rótulos, se fornecidas
        elif self.transform_target:
            retinopathy_grade = self.transform_target(retinopathy_grade)

        # Aplicando transformações nas imagens, se fornecidas
        if self.transform:
            image = self.transform(image)

        return {'img': image,
                'retinopathy_grade': retinopathy_grade
               }


class FGADRDataset(ClassificationDataset):
    def __init__(self, images_dir: Path, annotations_file: Path, transform=None, transform_target=None):
        super().__init__(images_dir, annotations_file, transform, transform_target)


    def __getitem__(self, idx):
        img_path = self.images_dir / f'{self.annotations.iloc[idx, 0]}'
        image = self.load_and_preprocess_img(img_path)
        retinopathy_grade = self.annotations.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.transform_target:
             retinopathy_grade = self.transform_target(retinopathy_grade)

        return {'img': image,
                'retinopathy_grade': retinopathy_grade,
                'img_name': img_path.name}