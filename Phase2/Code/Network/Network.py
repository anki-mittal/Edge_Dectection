"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch 

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))



class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
        """
        Inputs: 
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        self.Base = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Output size (32 x 32 x 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), # Output size (32 x 32 x 32)
            nn.ReLU(),
            nn.MaxPool2d(2,2), # Output (16 x 16 x 32)
            nn.Conv2d(32, 64, 3, padding=1), # output(16 x 16 x 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), # output (16 x 16 x 128)
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output (8 x 8 x 128)
            nn.Flatten(),
            nn.Linear(8*8*128,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )

      
  def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        out = self.Base(xb)
        return out

class CIFAR10ModelNorm(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      """
      Inputs: 
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      super().__init__()
      self.basenorm = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Output size (64 x 64 x 16)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), # Output size (64 x 64 x 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # Output (32 x 32 x 32)
            nn.Conv2d(32, 64, 3, padding=1), # output(32 x 32 x 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), # output (32 x 32 x 128)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output (16 x 16 x 128)
            
            nn.Conv2d(128, 256, 3, padding=1), # output(16 x 16 x 256)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1), # output (16 x 16 x 512)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output (8 x 8 x 512)

            nn.Flatten(),
            nn.Linear(8*8*512,100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100,10)
      )

      
  def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """

        out = self.basenorm(xb)

        return out
  
class ResNet(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(2))

        self.res_block_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU())

        self.conv_block_2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.MaxPool2d(2))

        self.res_block_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(512),
                                nn.ReLU())
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv_block_1(xb)
        # print(out.shape)
        out = self.res_block_1(out) + out
        # print(out.shape)
        out = self.conv_block_2(out)
        # print(out.shape)
        out = self.res_block_2(out) + out
        # print(out.shape)
        out = self.classifier(out)
        return out
