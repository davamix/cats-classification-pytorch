import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

'''
Create a file with the values of the hyperparameters
exp_path: experiment folder path
hyperparameters: dictionary. {"lr":"0.0.1", ...}
'''
def save_hyperparameters(exp_path, hyperparameters):
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    params_path = os.path.join(exp_path, "parameters.txt")
    f = open(params_path, "w")
    
    for hp in hyperparameters:
        f.write(f"{hp}: {hyperparameters[hp]}\n")
    
    f.close()

# Save image metrics
def save_metrics(exp_path, t_losses, v_losses, t_accs, v_accs):
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    fig = plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(t_losses, color = 'blue', label = 'Train')
    plt.plot(v_losses, color = 'red', label = 'Validation')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(t_accs, color = 'blue', label = 'Train')
    plt.plot(v_accs, color = 'red', label = 'Validation')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    metrics_path = os.path.join(exp_path, "metrics.png")
    fig.savefig(metrics_path)

'''
Save a sample with the images predicted.
exp_path: experiment folder path
predictions: Tuple with (image data, class predicted, probability, class expected)
'''
def save_predictions(exp_path, predictions):
    predictions_path = os.path.join(exp_path, "predictions")
    
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    for i, (image, pred, prob, expected) in enumerate(predictions):
        fig = plt.figure()
    
        plt.imshow(image)
        plt.title(f"Predicted:{pred} ({prob:.2f}%)\nExpected: {expected}")
        
        image_path = os.path.join(predictions_path, f"pred_{i}.png")
        fig.savefig(image_path)

'''
Fix the color map of the image applying the mean and std used in transformations
image: Image data resulting from dataloader
std: array with the std applied on transformations
mena: array with the mean values applied on transformations
'''
def fix_colormap(image, mean, std):
    image = image.numpy().transpose(1,2,0)
    mean = np.array([mean])
    std = np.array([std])

    image = std * image + mean
    image = np.clip(image, 0, 1)

    return image