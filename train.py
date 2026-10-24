print("Импорт модулей...")

import utils
import trainer
import model
from torch.utils.data import DataLoader

print("Модули импортированы!\n")

print("Конфигурация элементов...")

batch_size = 24
heads_lr = 1e-3
resnet_lr = 1e-4
epochs = 30

transforms = utils.get_default_transforms()
a = utils.get_default_a()
tasks, sizes = utils.get_default_tasks()
device = utils.get_default_device()

print("Конфигурация завершена!\n")

print("Подготовка модели и тренера...")

# подготовка даталоадера
df = utils.FaceDataset("train_table2.csv", transforms)
dl = DataLoader(batch_size=batch_size, dataset=df, shuffle=True, num_workers=0)

# Подготовка модели и обучателя
ap = model.ApperanceDescriber(list(sizes)).to(device)
train = trainer.Trainer(ap, dl, device, a, resnet_lr, heads_lr, batch_size)

print("Подготовка полностью завершена!\nПереходим к обучению...")

# обучение
train.start(epochs=epochs, display=True, display_in=1)
train.end()
