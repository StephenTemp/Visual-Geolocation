# SUMMARY: 

# IMPORTS
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# sklearn
from sklearn.metrics import f1_score
# utils
from utils.torch_utils import Flatten
# END IMPORTS

# CONSTANTS
USE_GPU = True
dtype = torch.float32
# END CONSTANTS

class CNN_Regression():

    # init method or constructor   
    def __init__(self, output_dims):  
        self.model = self.build_model(output_dims=output_dims)
        self.device = None
        self.loader_train = None
        self.loader_val = None

    # returns the CNN model based on dimension criteria
    def build_model(self, output_dims=2430):
        # QUICK model creation based more or less off inception
        model = nn.Sequential(
                    # INPUT: (3, 100, 100) 
                    # CNN LAYER: 30 5x5 channels
                    nn.Conv2d(in_channels=3, out_channels=30, kernel_size=5),

                    # INPUT: (3, 96, 96) 
                    # CNN LAYER: 30 5x5 channels
                    nn.Conv2d(in_channels=30, out_channels=20, kernel_size=5, padding=2),

                    # INPUT: (30, 96, 96)
                    # MAX-POOL: 3x3 filter
                    nn.MaxPool2d(kernel_size=4, stride=2), 
            
                    # INPUT: (30, 47, 47)
                    # CNN LAYER: 20 3x3 channels
                    nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=2),
                    
                    # INPUT: (20, 23, 23)
                    # CNN LAYER: 20 3x3 channels
                    nn.Conv2d(in_channels=20, out_channels=30, kernel_size=4, stride=1),
            
                    # INPUT: (30, 20, 20)
                    # CNN LAYER: 20 3x3 channels
                    nn.MaxPool2d(kernel_size=4, stride=2),
                    
                    # INPUT: (20, 9, 9)
                    nn.Flatten(),
                    nn.Linear(in_features=output_dims, out_features=2, bias=True),
                )

        return model

    def set_up_train(self, loader_train, loader_val):
        if USE_GPU and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.loader_train = loader_train
        self.loader_val = loader_val
        
    
    def check_accuracy_part34(self, loader, model):
        device = self.device

        print('Checking accuracy on test set')   
        num_samples = 0
        total_preds = torch.Tensor([])
        total_ground = torch.Tensor([])
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.float32)
                scores = model(x)

                num_samples += scores.size(0)
                total_preds = torch.cat([total_preds, scores])
                total_ground = torch.cat([total_ground, y])

            dist = (nn.L1Loss()(total_preds, total_ground)).mean()
            print("Average L1 Distance", dist)
        return dist
    
    def train_model(self, optimizer, epochs=10, lr=0.0001, print_every=10):
        model = self.model
        device = self.device
        model = model.to(device=self.device)

        loader_train = self.loader_train
        loader_val = self.loader_val
        
        evals = {"l1" : [], "loss" : []}
        for e in range(epochs):
            for t, (x, y) in enumerate(loader_train):
                model.train()  # put model to training mode
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.float32)
                scores = model(x)
                
                loss = torch.nn.L1Loss()(scores, y)

                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                optimizer.zero_grad()

                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                optimizer.step()

                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    l1 = self.check_accuracy_part34(loader_val, model)
                    evals["loss"].append(loss.item())
                    evals["l1"].append(l1)
                    print()
        return evals