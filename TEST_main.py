import os
import random
#import wandb

import cv2
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from IPython import display
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid


from TEST_train import *
from TEST_test import *
from TEST_Data_Preprocessing import *
from TEST_models import *
#from utils.utils import *
#from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
assert torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":   
    print("Using device:", device)
    ############################# DATA PROCESSING #############################
    print("Pre-processing...")
    # Definition of the paths
    path_csv = '/home/xnmaster/github-classroom/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/Inputs/'
    path_images = '/home/xnmaster/github-classroom/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/Inputs/'
    # Sizes of the datasets
    train_size = 30080
    valid_size = 3200
    test_size = valid_size
    batch_size = 128
    
    # Used alphabet
    alphabet = u" ABCDEFGHIJKLMNOPQRSTUVWXYZ-'"
    # Maximum length of input labels
    max_str_len = 24 
    # Number of characters (+1 for ctc pseudo blank)
    num_of_characters = len(alphabet) + 1 
    # Maximum length of predicted labels
    num_of_timestamps = 64
    
    train, valid, test, train_loader, valid_loader, test_loader = data_preprocessing(path_csv,path_images,train_size,valid_size,test_size,batch_size,max_str_len,alphabet)
    
    train_label_len, train_input_len = ctc_inputs(train, train_size, num_of_timestamps)
    valid_label_len, valid_input_len = ctc_inputs(valid, valid_size, num_of_timestamps)
    test_label_len, test_input_len = ctc_inputs(test, test_size, num_of_timestamps)
    
    print("Pre-processing done successfully.")

    ###########################################################################
    ############################### CRNN MODEL ################################
    ###########################################################################
    print("Building CRNN Model...")
    # Parameters of the model
    ## Size of the RNN inputs
    rnn_input_dim = 64
    ## Number of neurons of the RNN layers
    rnn_hidden_dim = 512
    ## Number of RNN layers
    n_rnn_layers = 2
    ## Dimension of the output
    output_dim = num_of_characters
    ## Drop rate
    drop_prob = 0.3

    # Loss function
    criterion = torch.nn.CTCLoss()
    # Initialisation of the model
    model = CRNN(rnn_input_dim, rnn_hidden_dim, n_rnn_layers, output_dim, drop_prob).to(device)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    print("Model successfully created.")
    print(f"--> Number of parameters : {get_n_params(model)}")
    model.to(device)
    
    ###########################################################################
    ############################# CRNN TRAINING ###############################
    ###########################################################################
    print("CRNN Training...")
    num_epochs = 1
    
    train_loss, valid_loss, words_acc_val, letters_acc_val = train_CRNN(train_loader, model, batch_size, 
                                                                        criterion, optimizer, num_epochs, valid_loader, 
                                                                        train_label_len, train_input_len, valid_label_len, 
                                                                        valid_input_len, max_str_len, device)
                                                                       
    visualize_results(train_loss,valid_loss,words_acc_val,letters_acc_val)
    print("Training successfully completed.")
    ###########################################################################
    ############################### CRNN TEST #################################
    ###########################################################################
    print("CRNN Test...")
    test_loss, test_accuracy_words, test_accuracy_letters, n_letters, mispred_prop_letters, mispred_images, mispred_pred,mispred_target = test_CRNN(criterion, model, test_loader, batch_size, test_label_len, test_input_len, max_str_len, device)
    print("Test successfully applied.")
    print(f"--> Accuracy of the model on the {test_size} test images: {test_accuracy_words:%}")
    print(f"--> Accuracy of the model on the {n_letters} test letters: {test_accuracy_letters:%}")
    print(f"--> Average word's proportion well predicted on mispredicted words : {mispred_prop_letters:%}")
    plot_misclassified(mispred_images, mispred_pred, mispred_target, alphabet)
    
    ###########################################################################
    ############################## OWN IMAGES #################################
    ###########################################################################
    path = '/home/xnmaster/github-classroom/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/IMAGES_EXTRA/'
    pred_andreu = test_own_image(model,path+'name_trial_andreu.jpg',alphabet,max_str_len,device)
    pred_mathias = test_own_image(model,path+'name_trial_mathias.jpg',alphabet,max_str_len,device)
    pred_pere = test_own_image(model,path+'name_trial_pere.jpg',alphabet,max_str_len,device)
    
    print("ANDREU : predicted as {pred_andreu}")
    print("MATHIAS : predicted as {pred_mathias}")
    print("PERE : predicted as {pred_pere}")