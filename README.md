[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jPcQNmHU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14965960&assignment_repo_type=AssignmentRepo)
# Handwriting Recognition
## 1 - Context
In this project, we are going to train a neural network in order to recognize words from handwritten data. The input dataset is composed by labelled images (that one can find in the "Inputs" folder, or via this [link](https://www.kaggle.com/code/samfc10/handwriting-recognition-using-crnn-in-keras/input)). The data is composed by 2 types of files : .zip files containing the images, and .csv files linking the images to their label. The folder "Inputs" is then composed of the 3 .csv files, as well as 3 sub-folders, which correspond to the extracted .zip files. 

## 2 - Preprocessing
The data provided is not cleaned yet. In fact, one can find remaining NA values, but also some images labelled 'UNREADABLE', which can't be used for our training. As a consequence, the first step of our work was to preprocess the 3 datasets provided (Training, Validation and Test) : delete the NA and 'UNREADABLE' labels, and turn all of them into upper case. For the images, we defined a standardized size of 64x256, obtained by cropping the current images.

After this cleaning operation, we converted the labels into vectors. To do that, we first looked at the sizes of the labels, and found out that the maximum length in the dataset is 34. But, this value is really far from the rest of the labels. In fact, all the others are of size < 24.
We decided to put this special value apart, and consider a maximum length of 24. Now, each label can be converted into a vector of size 24, with numbers corresponding to the characters of the word (and with a padding value of -1 for shorter words). For this convertion, we used 30 different characters : " ABCDEFGHIJKLMNOPQRSTUVWXYZ-'" (+1 for the CTC pseudo blank).

## 3 - Models
Considering the model, we have implemented 3 models : 
  -  CRNN model using LSTM
  -  CRNN model using GRU
  -  ...

For the Convolutional and Recurrent Neural Networks, we first apply 3 convolutional layers with 3x3 filters, ReLU activation, same padding, and batch normalization, followed by a linear layer, with ReLU activation, in order to obtain a final activation map of 64x64. This map will then pass through 2 LSTM bi-directional recurrent layers, of 512 neurons. 
The final activation map obtained has a size 64x1024, which leads to an output of shape 64x30, by applying a linear transformation, and a Log-Softmax.

The final map is finally compared to the label using the CTC Loss. Here, the 64x30 output corresponds to 64 probability distributions (over the 30 possible characters). Using those probability distributions, we can obtain a predicted word. For example, with the label 'DIMOS', considering the maximum probability character for each line, one could obtain 'DDDDDDDDD_IIIII_MMMMMMMMMM_OOOO_SSSS______' (vector of size 64, '_' being the blank symbol), which leads to 'DIMOS' after the collapse operation.

For the training of those models, we used those parameters :
- Size of the training sammple : 64 000 images
- Size of the validation sample : 6 400 images
- Batch size of 128
- Adam optimizer
- Drop rate of 0.3 on the 2 last convolutional layers, and the recurrent ones
- Learning rate of 0.001
- 10 epochs

Afterwards, we wanted to see if by training a bigger number of samples in the training and validation images. So for the final model we took 300800 as the train size and 30080 as the validation size. This allowed us to ... 

For the training of the final model, we used these parameters :
- Size of the training sammple : 300 800 images
- Size of the validation sample : 30 080 images
- 10 epochs


## 4 - Test of the models

In this part, we have applied the models implemented on the test dataset. We obtain acceptable results since the Letter Accuracy is around 80%, and the Word Accuracy, around 60%.  

In that part, we also did some particular tests, such as ploting at some predictions with its image, looking at mispredicted images, to indentify any recurrent aspect of those images.
We also looked at the top-3 most mispredicted letters, and their top mispredicted letters. In this part, we observe that with LSTM neurons, 'L' is often predicted as 'E' or 'I', and the same goes with 'N' ann 'E'. For the model with GRU neurons, we find again the 'L' often predicted as 'E', 'N' as 'E', and 'E' as 'R'.
What we can see here is that, in most of the cases, the model mispredict a letter for a letter having some shape similarities.

At the end, we also applied the models on own written images. To do so, we have written our names on a paper, and applied the model on them. We encountered some problems regarding the format of the images and their sizes. This is because in the dataset our model was trained on, all images are of a similar shape(around 256,80) and colour (binarized images). To fix this changes and make our predictions good, we used the functions inside the cv2 library, which allow us to binarize them and resize them in a way that the predictions could be done. This happened because we saw poor performance for images that had very large sizes and also that where taken in pour light and not preprocessed as they should be. After improving this, we were able to get very good predictions, taking into account the necessities of the function, eventhough in some cases there are still errors, since the accuracy is not 100%, so it can fail for some letters but mostly if they are cleary written, it will perform fairly good. 

We tried this with an early model with smaller sizes for training and validation, around 30000 and 3000, where the predictions are good. Some letter predictions missed, still the accuracy was fairly good. Afterwards, we tried it with the final model, where we took in all images for training, and 


## Code structure
The code is composed by different .py files :
- Data_Preprocessing.py : containing all the preprocessing functions
- [models.py](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/blob/main/models/models.py) : containing the definition of the models
- train.py : containing training/validation and some ploting functions
- test.py : containing test and ploting functions
- main.py : managing all the previous files

To run the code, first, we have to run the Data_Preprocessing.py file. This will extract the .zip files if it's not already done. If it's done, it will not do anything else. The only thing that has to be changed is the paths "path_zip" and "destination", which correspond, respectively, to the location of the .zip files and the place where to put the extracted folder.

Then, simply running the main.py file will carry out the pre-processing operations, create the models, train and test them. 
This run will print some information in the terminal, such as the number of parameters of each model, the results of the training/test; as well as some plots (that will be stored in a folder "Plots"). At the end, we also print a comparative table of both models.
Again, some paths may need changes : "path_csv", which has to be the location of the 3 .csv files / "path_images", which has to be the location of the extracted folders (="destination") / and "save_plots", which corresponds to the place where we will save the plots during the training.


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
