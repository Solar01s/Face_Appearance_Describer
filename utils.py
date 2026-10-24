import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2

def get_default_tasks():
    tasks = {
    "task_1": ["Мужской", "Женский"],
    "task_2": ["Русые", "Рыжие", "Чёрные", "Каштановые", "Седые", "Отсутствуют"],
    "task_3": ["Карие", "Серые", "Чёрные", "Зелёные", "Голубые", "Двуцветные"],
    "task_4": ["Страх", "Радость", "Нейтральная", "Гнев", "Удивление"],
    "task_5": ["Ребёнок (до 14 лет)", "Подросток (14-18 лет)", "Взрослый (18-50) лет", "Пожилой (50+)"]
    }
    return tasks, (2, 6, 6, 5, 4)

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_default_a():
    a = {
        "task_1": 0.9,
        "task_2": 0.6,
        "task_3": 0.6,
        "task_4": 0.9,
        "task_5": 0.8,
    }
    return a

def get_default_transforms():
    transforms = v2.Compose([

            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ])
    return transforms

def load_model(path:str, model):
    state_dict = torch.load(path, map_location=get_default_device())
    model.load_state_dict(state_dict)
    return model

def choices(logits):
    win_idxs = [torch.argmax(logit, dim=-1).item() for logit in logits]
    return win_idxs

class FaceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        
        image = Image.open(row['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        targets = {
            'task_1': torch.tensor(row['task_1'], dtype=torch.long),
            'task_2': torch.tensor(row['task_2'], dtype=torch.long),
            'task_3': torch.tensor(row['task_3'], dtype=torch.long),
            'task_4': torch.tensor(row['task_4'], dtype=torch.long),
            'task_5': torch.tensor(row['task_5'], dtype=torch.long)
        }
        
        return image, targets
