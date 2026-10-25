# Useful extract

# Characteristics
- Apperance Describer is an artifical intellegence model built on the **ResNet18 architecture and linear layers**.
- The accuracy on the training data was **98%**. On the new data, as expected, it was **80%**.
- The model was trained on a **GPU T4 for 5 minutes**, which amounted to **280 epochs**.
- The model's loss decreased by **0.32, i.e., 0.06 per head**.

# What is used?
## Python programming language and his libraries:
- PyTorch(torch), torchvision: the main libraries for creating and training a model.
- Pandas: a library that allows you yo work with tables conveniently via code; it was used to create a teble of correct answers(targets) for training.
- Customtkinter: a library for creating interfaces (graphical windows). Is used to test the model in a convenient format.

# Modules
- If you want to train a model for your task using this architecture, use the following modiles: **utils, trainer, train(optional), model**. **Analyze the train file for normal work your model!**. Also note that utils contains ready-made templates for all necessary paramters.
- If you want to test and use the pre-trained model, the **test** module and the model weights on Hugging Face are sufficient. The link to the weights is provided in the about section.
