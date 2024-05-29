import torch
import numpy as np
import matplotlib.pyplot as plt

from TEST_train import decode
from TEST_Data_Preprocessing import num_to_label, preprocess_image
import cv2
import os


@torch.no_grad()  # prevent this function from computing gradients
def test_CRNN(criterion, model, loader, batch_size, test_label_len, test_input_len, max_str_len, device):
    """
    Applies the given model on the test set

    Parameters
    ----------
    criterion : torch.nn - Loss function used
    model : CRNN - The model applied
    loader : Dataloader - Training values
    batch_size : Int - Size of a batch
    test_label_len : torch.tensor - Real lengths of the labels
    test_input_len : torch.tensor - Lengths of the outputs of the model
    max_str_len : Int - maximum label length
    device : torch.device - GPU or CPU

    Returns
    -------
    test_loss : Float - Value of the loss 
    accuracy_words : Float - Number of well predicted words / Total number of words
    accuracy_letters : Float - Number of well predicted letters / Total number of letters
    n_letters : Int - The total number of letters over the whole dataset
    mispred_prop_letters : Float - Proportion of well predicted letters on a failed word prediction
    mispred_images : array - Misclassified images
    mispred_pred : array - Their predicted labels 
    mispred_target : array - Their true labels 
    """
    
    # Initialisations
    test_loss = 0
    correct_words = 0
    correct_letters = 0
    n_letters = 0
    
    mispred_prop_letters = 0
    mispred_nb_letters = 0

    # Gradients are not needed
    model.eval()
    
    for batch, (data, target) in enumerate(loader):
        # Initialisation of the hidden states of the RNN part of the model
        h_state, c_state = model.init_hidden(batch_size)

        # We put the variables to the GPU's memory
        h_state = h_state.to(device)
        if c_state is not None:
            c_state = c_state.to(device)

        data, target = data.to(device), target.to(device)

        # Application of the model
        output, h_state, c_state = model(data, h_state, c_state)

        # Inputs of the CTC Loss
        target_lengths = test_label_len[(batch*batch_size):((batch+1)*batch_size)]
        input_lengths = test_input_len[(batch*batch_size):((batch+1)*batch_size)]
        # Application of the loss function
        loss = criterion(output.transpose(0, 1), target, input_lengths, target_lengths)
        # Upgrade the loss value
        test_loss += loss.item()

        # Computation of the accuracies
        _, pred = torch.max(output.data,dim=2)
        pred = decode(pred,batch_size,max_str_len)

        target = target.cpu().numpy()

        correct_words += np.sum(np.sum((abs(target-pred)),axis=1)==0)
        correct_letters += np.sum(abs(target-pred)==0, where=(target!=-1))
        
        # We keep in memory the misclassified images
        mispred_index = np.sum((abs(target-pred)),axis=1)!=0
        mispred_images = data[mispred_index,:,:,:]
        
        mispred_pred = pred[mispred_index]
        mispred_target = target[mispred_index]
        
        # Proportion of well predicted letters in a wrong word prediction
        mispred_prop_letters += np.sum(abs(mispred_target-mispred_pred)==0, where=(mispred_target!=-1))
        mispred_nb_letters += np.sum(mispred_target!=-1)
        
        n_letters += np.sum(target!=-1)
        
    # Average loss over each batch (25 batches in the test set) 
    test_loss /= 25
    # Average accuracies over each batch
    accuracy_words = correct_words / len(loader.dataset)
    accuracy_letters = correct_letters / n_letters
    mispred_prop_letters = mispred_prop_letters/mispred_nb_letters
    

    return test_loss, accuracy_words, accuracy_letters, n_letters, mispred_prop_letters, mispred_images, mispred_pred,mispred_target


def plot_misclassified(mispred_images, mispred_pred,mispred_target,alphabet,save_path,name):
    """
    Plots 6 mispredicted images, with the true and predicted label.
    
    Parameters
    ----------
    mispred_images : array - Some mispredicted images
    mispred_pred : array - Their predicted label
    mispred_target : array - Their true label
    alphabet : String - Alphabet used for decoding
    save_path : String - The path where to save the plot
    name : String - Name of the plot file
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
     
    plt.figure(figsize=(15, 10))

    for i in range(6):
        ax = plt.subplot(2, 3, i+1)
        image = mispred_images[i]
        image = image.squeeze(0)
        image = image.cpu().numpy()
        pred = num_to_label(mispred_pred[i],alphabet)
        target = num_to_label(mispred_target[i],alphabet)

        plt.imshow(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap = 'gray')
        plt.title(target+"/"+pred, fontsize=12)
        plt.axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=-0.8)
    plt.savefig(os.path.join(save_path,name))
    plt.clf()
    
def test_some_images(model, batch_size, loader, max_str_len, alphabet, device, save_path, name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
     
    data, targets = next(iter(loader))
    data, targets = data.to(device), targets.to(device)
    
    h, c = model.init_hidden(128)
    h = h.to(device)
    if c is not None:
        c = c.to(device)
        
    output, h, c = model(data,h,c)

    _,predictions = torch.max(output.data,dim=2)
    predictions = decode(predictions,batch_size,max_str_len)
    targets = targets.cpu().numpy()
      
    plt.figure(figsize=(16, 6))

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        image = data[i].cpu().numpy().reshape((256,64))
        pred = predictions[i]
        target = targets[i]
        
        pred = num_to_label(pred,alphabet)
        target = num_to_label(target,alphabet)
            
        plt.imshow(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap = 'gray')
        plt.title(target+"/"+pred, fontsize=12)
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0.2, hspace=-0.8)
    plt.savefig(os.path.join(save_path,name))
    plt.clf()
        
def test_own_image(model,dir,names,targets,alphabet,max_str_len,device):
    """
    Apply the model on 3 images made by ourselves.
    
    Parameters 
    ----------
    model : CRNN - Model to apply
    dir : String - Path of the images
    names : List - Names of the files
    alphabet : String - Alphabet used for decoding
    max_str_len : Int - Maximum label length
    device : torch.device - GPU or CPU
    """
    k=0
    for name in names:
        path = os.path.join(dir,name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = preprocess_image(image)/255.
        image = image.astype(np.float32)
        
        h, c = model.init_hidden(1)
        h = h.to(device)
        if c is not None:
            c = c.to(device)
            
        input = torch.tensor(image)
        input = input.reshape((1, 1, input.shape[0], input.shape[1]))
        input = input.to(device)
            
        pred, h, c = model(input,h,c)

        _, pred = torch.max(pred,dim=2)
        pred = decode(pred,1,max_str_len)
        pred = num_to_label(pred[0],alphabet)
        
        plt.subplot(1, 3, k+1)
        plt.imshow(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap = 'gray')
        plt.title(targets[k]+"/"+pred, fontsize=12)
        plt.axis('off')
        k+=1
    
    plt.subplots_adjust(wspace=0.2, hspace=-0.8)
    plt.savefig(os.path.join(save_path,name))
    plt.clf()
    

    



