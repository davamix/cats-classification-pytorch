import os
import datetime
import matplotlib.pyplot as plt

'''
Create a file with the values of the hyperparameters
exp_path: experiment folder path
hyperparameters: dictionary. {"lr":"0.0.1", ...}
'''
def save_experiment(exp_path, hyperparameters):
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