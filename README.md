# Useful extract

## Characteristics
- Appearance Describer is an artifical intellegence model built on the **ResNet18 (CNN) architecture and linear layers**.
- The accuracy on the training data was **98%**. On the new data, as expected, it was **70%**.
- The model was trained on a **GPU T4 for 5 minutes**, which amounted to **280 epochs**.
- The model's loss decreased by **0.32, i.e., 0.06 per head**.

## Used Python programming language and his libraries
- PyTorch(torch), torchvision: the main libraries for creating and training a model.
- Pillow: It allows you to work with images, convert objects this library into tensors, and pass them to the model.
- Pandas: a library that allows you to work with tables conveniently via code; it was used to create a teble of correct answers(targets) for training.
- Customtkinter: a library for creating interfaces (graphical windows). Is used to test the model in a convenient format.

## Modules
- utils.py: auxiliary tools for training and parameters templates.
- model.py: model architecture, ApperanceDescriber class.
- trainer.py: The Trainer class, which trains th model according to the specifed parameters.
- train.py: The training code for the original Apperance Describer, an example of using the training modules.
- test.py: The code for testing the model's knowledge with an interface via customtkinter. Example of using pre-trained weights.

## Instructions for use
- If you want to train a model for your task using this architecture, use the following modules: **utils, trainer, train(optional), model**. **Analyze the train file for normal work your model!**. Also note that utils contains ready-made templates for all necessary paramters.
- If you want to test and use the pre-trained model, the **test** module and the model weights on Hugging Face are sufficient. The link to the weights is provided in the about section.

# Functionality

## Format
- a-importance coefficients, correct anwers are pressented in the form of a dictonary.

## utils.py
### function: get_default_device()
Check the availability of CUDA. Returns the available device: cpu or cuda
### function: get_default_a()
Returns a dictonary of importance coefficietns for loss on 5 tasks of the following type: {"task_n": x.x, "task_n+1": y.y}.
### function: get_default_transforms()
Returns a torchvision.transforms.v2.Compose object with transformations for a Pillow.Image object.
### function: load_model(path, model)
Sets the state dict of the passed model at the specifed path.
### function: choices(logits)
In each logit, the logits make a choice via argmax and return a list of selected indices.

### class: FaceDataset
**WARNING: here targets are only for 5 task.** You can write your own or rewrite this one if you want a different number of characteristics. In addition, your task mav be different, or your data nay not be stored in a CSV table format like mine! I recommend rewriting it.
Regarding the work, it inherits from the PyTorch Dataset class. In the __getitem__ function, it retrieves the data of a single row (a single image) from a Pandas object by index and returns a dictonary of tensors of correct answers(targets) for each of the 5 tasks of the following form: {"task_1": torch.tensor(4), "task_2": ...}

## trainer.py
### The only thing that's needed here is the **Trainer** class for training the Appearance Describer. Class functions:
| **Class function** | **Accepts** | **Dependencies** | **Process** | **Returns** |
| :--- | :--- | :--- | :--- | :--- |
| __init__ | model, data_loader(PyTorch DataLoader), device, a(importance coefficients for loss), resnet_lr, head_lr, batch_size | - | Sets the configuration of the Trainer class | configurated class Trainer |
| start | epochs, display, display_in | torch | Starts the model training cycle for "epoch" epochs. If "display" is True, an intermediate training summary is displayed in "in_display" epochs | trained model, list of losses |
| end | - | torch | Save model to path "trained_ap.pt" and display message to concole | None |

## model.py
### The AppearenceDescriber class funcions:
| **Class function** | **Accepts** | **Dependencies** | **Process** | **Returns** |
| :--- | :--- | :--- | :--- | :--- |
| __init__ | signs(a list of answer options for each characteristics) | torch, torchvision | Creates a ResNet18, len(signs) linear layers via ModuleList | configurated AppearanceDescriber class |
| forward, directly | normalize_picture: A tensor of size (3, 224, 224) | torch | It receives from ResNet18, then receives logits from each of the n heads | tuple of logits from n heads|
| from_file | path (to image), H (height to crop image), W(weight to crp image) | torchvision, torch, Pillow, forward | Using Pillow, it opens the image, transforms it via torchvision.transforms.v2, and uses the forward function to obtian a response from the nueral network. | tuple of logits from n heads |
