import torch.nn as nn
import torch
import torchvision
from torchvision.transforms import v2
import utils

from PIL import Image

class AppearanceDescriber(nn.Module):
    def __init__(self, signs: list):

        """
        signs: количество вариантов ответа для признаков внешности.
        Каждый sign задаёт их в каждой голове. Например, signs[0] 
        задаёт количество вариантов (out_features) ответа в heads[0]
        """
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        last_input = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.heads = []
        
        for i in range(len(signs)):
            head = nn.Linear(last_input, signs[i])
            self.heads.append(head)
        self.heads = nn.ModuleList(self.heads)



    def forward(self, normalize_picture):

        '''
        Принимает тензор normalize_picture и возвращает 
        5 тензоров размером (signs,), которые были
        переданы в __init__ для каждой головы
        '''
        
        signs = self.resnet(normalize_picture)
        answers = []

        for head in self.heads:
            answer_n = head(signs)
            answers.append(answer_n)        

        return answers

    def __str__(self):

        print("Apperance Describer based on ResNet18\n\nNumber of outputs in heads:")

        for i in range(len(self.heads)):
            value = self.heads[i].out_features
            print(f"Head {i}: {value}")

        return ""

    def from_file(self, path: str, H: int=224, W: int=224):

        '''
        Функция позволяет читать картинку из path,
        обрабатывать и проводить описание.
        Принимает:
        1) path - путь к файлу с картинкой
        2) H - высота картинки, до которой она будет сжата
        3) W - ширина картинки, до которой она будет сжата

        **Возвращает кортеж из 5 тензоров**
        '''

        image = Image.open(path)

        transforms = v2.Compose([

            v2.Resize((H, W), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ])
        normalize_picture = transforms(image)
        normalize_picture = normalize_picture.unsqueeze(0)
        
        return self(normalize_picture)

    def directly(self, normalize_picture):

        '''
        Принимает нормализованую картинку и
        выполняет forward.

        **Возвращает кортеж из 5 тензоров**
        '''

        return self.forward(normalize_picture)
