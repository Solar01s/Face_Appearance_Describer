# Useful extract

## Characteristics
- Appearance Describer is an artifical intellegence model built on the **ResNet18 (CNN) architecture and linear layers**.
- The accuracy on the training data was **98%**. On the new data, as expected, it was **80%**.
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

## trainer.py
### The only thing that's needed here is the **Trainer** class for training the Appearance Describer. Class functions:
| **Class function** | Keep | **Dependencies** | Process |
| :--- | :--- | :--- | :--- |
| __init__ | model, data_loader(PyTorch DataLoader), device, a(importance coefficients for loss), resnet_lr, head_lr, batch_size | - | Sets the configuration of the Trainer class |
| start | epochs, display, display_in | torch | Starts the model training cycle for "epoch" epochs. If "display" is True, an intermediate training summary is displayed in "in_display" epochs |
| end | - | torch | Save model to path "trained_ap.pt" and display message to concole |

## model.py
