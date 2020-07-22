import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image

def training_model(dataloaders,dataset_sizes,model,criterion,optimizer,scheduler,num_epochs=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    acc_dict  = {"train":[],"val":[]}
    loss_dict = {"train":[],"val":[]}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1,num_epochs))
        print("-"*10)

        for phase in ["train","val"]:
            print("---{}---".format(phase))
            sum_img = 0
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.

            for inputs, labels,_ in dataloaders[phase]:
                sum_img += inputs.size(0)
                print("{:6}/{:6}".format(sum_img,dataset_sizes[phase]),end="\r")

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):
                    preds = model(inputs)
                    labels = labels.view_as(preds)
                    loss = criterion(preds,labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects  += torch.sum( (preds>0.5) == labels ).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects / dataset_sizes[phase]   
            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)

            print('{} Loss: {:.4f} ,ACC:{:.4f}'.format(phase, epoch_loss,epoch_acc))
    return model,loss_dict,acc_dict

def test_model(dataloaders,dataset_sizes,model,criterion):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sum_img = 0
    model.eval()
    
    all_labels = []
    all_preds  = []
    all_clses  = []
    
    running_loss = 0.0
    running_corrects = 0.
    
    phase="val"
    for inputs, labels, cls in dataloaders[phase]:
        
        sum_img += inputs.size(0)
        print("{:6}/{:6}".format(sum_img,dataset_sizes[phase]),end="\r")

        inputs = inputs.to(device)
        labels = labels.to(device)

        preds = model(inputs)
        labels = labels.view_as(preds)
        loss = criterion(preds,labels)
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects  += torch.sum( (preds>0.5) == labels ).item()
        
        all_labels += list(labels.to("cpu").numpy().reshape(-1))
        all_preds  += list(preds.detach().to("cpu").numpy().reshape(-1))
        all_clses  += cls
        
    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc  = running_corrects / dataset_sizes[phase]
    print('Loss: {:.4f} ,ACC:{:.4f}'.format(epoch_loss,epoch_acc))
    return epoch_loss,epoch_acc,np.array(all_labels),np.array(all_preds),all_clses

def data_transformer_torch_train(): #rgb
    data_transforms = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])
    return data_transforms

def data_transformer_torch_test(): #rgb
    data_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])
    return data_transforms

def convert_im_torch(img): # rgb
    mean= np.array( [0.485, 0.456, 0.406] ).reshape(-1,1,1)
    std = np.array( [0.229, 0.224, 0.225] ).reshape(-1,1,1)

    img = img.detach().cpu().numpy()
    img = img*std + mean
    img *= 255
    img = img.transpose(1,2,0)
    img[img>255] = 255
    img[img<0] = 0
    return img.astype(np.uint8)

class Img_Dataset(data.Dataset):
    def __init__(self, file_list, transform,labels,class_labels):
        self.file_list = file_list
        self.transform = transform
        self.labels    = labels
        self.class_labels = class_labels
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path).convert("RGB")
        img_transformed = self.transform(img)
        label = torch.tensor( self.labels[index] ,dtype=torch.float)
        class_label = self.class_labels[index]
        
        return [img_transformed,label,class_label]

def get_paths(target_classes):
    nonperiodic_paths = []
    periodic_paths    = []
    labels = []
    class_labels = []

    for cls in target_classes:
        filepaths = glob.glob("../Images/nonperiodic/all/{}/*".format(cls))
        nonperiodic_paths+=filepaths
        class_labels += [cls]*len(filepaths)

    for cls in target_classes:
        filepaths = glob.glob("../Images/periodic/all/{}/*".format(cls))
        periodic_paths+=filepaths
        class_labels += [cls]*len(filepaths)

    labels += [0]*len(nonperiodic_paths)
    labels += [1]*len(periodic_paths)
    image_paths = nonperiodic_paths + periodic_paths
    return image_paths,labels,class_labels,target_classes
