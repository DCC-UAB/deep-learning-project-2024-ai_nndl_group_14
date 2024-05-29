def test_own_image(model,dir,max_str_len,device,save_path, print_pred:bool = False):
    """
    Apply the model on 3 images made by ourselves.
    
    Parameters 
    ----------
    model : CRNN - Model to apply
    dir : list of paths to images
    max_str_len : Int - Maximum label length
    device : torch.device - GPU or CPU
    save_path : where the resulting predictions shoudl be stored.
    """
    k=0
    for image_path in dir:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (256, 71))
        ret, otsu_thresholded = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = preprocess(otsu_thresholded)/255.
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
        pred = num_to_label(pred[0])
        
        plt.subplot(1, 3, k+1)
        plt.imshow(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap = 'gray')
        plt.title(pred, fontsize=12)
        plt.axis('off')
        k+=1
        if print_pred == True:
            print(f'For the image in {image_path} the prediction is {pred}.')
            print('--------------------------------------------------------')
    
    plt.subplots_adjust(wspace=0.2, hspace=-0.8)
    plt.savefig(save_path)
    plt.clf()