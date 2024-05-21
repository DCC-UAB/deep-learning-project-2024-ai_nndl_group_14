[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jPcQNmHU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14965960&assignment_repo_type=AssignmentRepo)
# Handwriting Recognition
In this project, we are going to train a neural network in order to recognize words from handwritten data. The input dataset is composed by labelled images (that one can find [here](https://www.kaggle.com/code/samfc10/handwriting-recognition-using-crnn-in-keras/input)). The main goal of the project is to obtain a final model which is functional. We don't expect it to be well accurate, but only see that the model have learnt interesting features (of course, we will try to optimize as much as possible our model).

First, we have cleaned the datasets : we deleted the NA values, but also the images labelled 'UNREADABLE', and we transformed all the labels into upper case letters.

In the preprocessing part, we defined a standardized size of 64x256 for the images, and we transformed the labels into vectors of size 24, with numbers corresponding to the characters of the word (and 0 values at the end if the length is shorter than 24).

Considering the model, we have implemented a CRNN model (Convolutional and Recurrent Neural Network). In this network, we first apply 3 convolutional layers with 3x3 filters, ReLU activation, same padding, and batch normalization. Then, we apply a linear layer, with ReLU activation, in order to obtain a final activation map of 64x64. This map will then pass through 2 LSTM bi-directional recurrent layers, of 512 neurons. The final activation map is then of size 64x1024. The output is obtained after a linear transformation, and a Log-Softmax, with a shape 64x30.

The final map is finally compared to the label using the CTC Loss. Here, the 64x30 image obtained corresponds to 64 probability distributions (over the 30 possible characters). Using those probability distributions, we can obtain a predicted word. For example, with the label 'DIMOS', considering the maximum probability character for each pixel, one could obtain 'DDDDDDDDD_IIIII_MMMMMMMMMM_OOOO_SSSS______' (vector of size 64, '_' being the blank symbol), which leads to 'DIMOS' after the collapse operation.



## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```



## Contributors
Write here the name and UAB mail of the group members

Xarxes Neuronals i Aprenentatge Profund
Grau de __Write here the name of your estudies (Artificial Intelligence, Data Engineering or Computational Mathematics & Data analyitics)__, 
UAB, 2023
