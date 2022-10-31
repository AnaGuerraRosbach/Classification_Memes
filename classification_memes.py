# -*- coding: utf-8 -*-
"""Classification_memes_final_MANA_24.03.2022.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z4t5m1MGAoJwUNq8qjaERN63uPthGsZ5

# Project: classification of memes



---
In this project we will do task 5: *Multimedia Automatic Misogyny Identification (MAMI)*, subtask A.

The purpose of this task is to identify if a meme should be considered misogynous or nor misogynous.

To be able to acces the data it is necessary to put all the files (training, test, trial) from the MAMI data set on a file in google drive named 'data_memes'.

Open this [formular](https://docs.google.com/forms/d/e/1FAIpQLSe3yJ6ggV0WlPpupN8Hy51F4zmulq_HgdC8rHU1ptZMnqUhjA/viewform) to request the data from the SemEval2022 organizators.
"""

# to be able to acces the data it is necessary to put all the files (training, test, trial)
# from the MAMI data set on a file in google drive named 'data_memes'
from google.colab import drive
drive.mount('/content/drive')

"""**Preprossesing steps**

Here we will explore the data and make some changes when necessary.

We noticed that we have 10000 memes and 5000 were labeled as misogynous and 5000 as not misogynous on the train data set. That means that the data is balanced. And that we already have the transcription of what is written in the images on the 'Text Transcription' columm. 

Also relevant to notice is the fact that the test set does not have the column with the labels.

Since the test set is no balanced, we decided to split our train data in train and test set to analyse the performance of our model.
"""

import pandas as pd
import re
import torchvision
import random
import matplotlib.pyplot as plt
import numpy as np
import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
nltk.download("wordnet")
from nltk import pos_tag, word_tokenize
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer


from PIL import Image
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# paths for the dada -> we did not change how it was structured on the MANI data set
training_path = ('/content/drive/My Drive/data_memes/training/TRAINING/training.csv')
test_path = ('/content/drive/My Drive/data_memes/test/test/Test.csv')
trial_path = ('/content/drive/My Drive/data_memes/trial/Users/fersiniel/Desktop/MAMI - TO LABEL/TRIAL DATASET/trial.csv')

print('-----------Train data-----------')
df_train = pd.read_csv(training_path, delimiter="\t")
print(df_train['misogynous'].value_counts())

# print('-----------Test data-----------')
df_test = pd.read_csv(test_path, delimiter="\t")
# print(df_test.value_counts()) -> does not work, since we do not have labels for
# test set

print('-----------Trial data-----------')
df_trial = pd.read_csv(trial_path, delimiter="\t")
print(df_trial['misogynous'].value_counts())

"""**Data vizualization: texts**

We want to vizualize the data to check what can be cleaned from the 'Text transcripions' to reduce noise.

For this model, we tought about to getting rid of the capital letters, as we consider them to have an important meaning. But since the model was performing badly, we decided afterwars to lemmatize and lower case everything. Here some examples:

*    'When Feminists realize that most of them "MENstruate"'

*    'BREAKING NEWS: Russia releases photo of DONALD TRUMP with hooker in Russian hotel.....wait....sorry....wrong file....never mind.'

Similarly, repeated letters or repeated punctuation marks also meaning to the sentence. We decided to leave repetions until the maximum of three letters or punctuation marks.

So this sentence:

*   ME DEAAAATH! MY SISTER DEAAAAAAAAATH!

has been changed to:

*   ME DEAAAH! MY SISTER DEAAATH!

On the other hand, we will remove the websites from which the memes were extract, since it does not add information to the data. Some examples:

* "ROSES ARE RED, VIOLETS ARE BLUE IF YOU DON'T SAY YES, I'LL JUST RAPE YOU **quickmeme.com**",
"""

# observe the characteristics of our text transcriptions 
list(df_train["Text Transcription"][:25])

"""**Data sets transformations**

To make it easier to work with the image files, we will add the image path to the data set. That will be done for the training set, test set and trial set. 

Since we are only doing task 5.a. we will drop the columns that are not necessary for our task.

We also created a method to print images, given the index.
"""

class preprocessing():

  '''
  This class contains customized funtions to preprocess the data.
  '''

  def __init__(self, data, image_path):
    self.data = data
    self.image_path = image_path
   
  def add_image_path(self):
    '''
    This will change the data a bit:
    Image_path will be added as a new column.
    For the train and trial data sets:
    drop colums 'shaming', 'stereotype', 'objectification', and 
    'violence' that are not relevant for task 5.a
    '''
    self.data['image_path'] = self.image_path + self.data['file_name'] 
    #df = df.sort_values(by=['id'])
    if 'misogynous' in self.data:
      self.data = self.data.drop(['shaming', 'stereotype', 'objectification', 'violence' ], axis = 1)
    return self.data

  def print_images_and_labels(self, index):
    '''
    To help us to visualize the images, this method prints a meme and its label.
    @input: 
      index: index of the memes to be printed
    @output:
      printed memes with label
    '''  
    # create figure
    fig = plt.figure(figsize=(15, (6)))
  
    # set font style
    font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 22}

    image = Image.open(self.data.loc[index,'image_path'])

    # add label to image
    if (self.data.loc[index, 'misogynous']).astype(int) == 1:
        plt.title('Label = misogynous',  fontdict=font)
    else:
        plt.title('Label = not misogynous',  fontdict=font)
  
    plt.imshow(image)     
    return None
  
  def lemmatize(self):
    '''
    We noticed that the linear regression model is not generalizing well.
    We decided to add this method to try and see what happens if we train the 
    model after lower casing and lemmatizing. This will reduce the indexed
    vocabulary considerably
    '''
    df = self.data["Text Transcription"]
    self.data["Text Transcription"] = df.apply(self.lemmatize_text)
    return self.data
  
  def lemmatize_text(self, text):
    '''
    Here the lemmatizing operations are done.
    '''
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    wordnet_lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = [wordnet_lemmatizer.lemmatize(w, pos = "v") for w in w_tokenizer.tokenize(text)]
    final_text = ' '.join(map(str, text)) 
    return final_text

image_path_train = '/content/drive/My Drive/data_memes/training/TRAINING/'
image_path_test = '/content/drive/My Drive/data_memes/test/test/'
image_path_trial = '/content/drive/My Drive/data_memes/trial/Users/fersiniel/Desktop/MAMI - TO LABEL/TRIAL DATASET/' 

# preprocess train, trial and test data:
pp_train = preprocessing(df_train, image_path_train)
df_train = pp_train.add_image_path()
df_train = pp_train.lemmatize()

pp_trial = preprocessing(df_trial, image_path_trial)
df_trial = pp_trial.add_image_path()
df_trial = pp_trial.lemmatize()

pp_test = preprocessing(df_test, image_path_test)
df_test = pp_test.add_image_path()
df_test = pp_test.lemmatize()

# print some memes:
pp_train.print_images_and_labels(55)
pp_train.print_images_and_labels(1165)

# check if image path has been added to data frame and if unnecessary columns were droped
df_train

"""
**Clean text data and create indexed vocabulary**

Some cleaning has been necessary on the provided text data to transform it into an indexed vocabulary.

This indexed vocabulary will be used in the class custom_Dataset to construct a bag of words tensor. """

class index_vocab():

  '''
  Create indexed vocab with the text transcriptions.
  Passing the indexed_vocab as an argument to the custom data set
  increases efficiency.
  '''

  def __init__(self, data):
    self.data = data

  def word_extraction(self, sentence):
    '''
    This method takes one sentence and divide words and punctuation into a list 
    of words. 
    '''
    ignore_lower = ['a', "the", "is", "and", "other", "from", "to", "of", "at", "in", "by", "be", "for", "that", "with", "was", "were", "will", "is", "it's"]
    ignore_upper = [each.upper() for each in ignore_lower]
    ignore = ignore_lower + ignore_upper 
    delete = ["(15251 ()", "Meme", "Mickmeme.com1", "memecenter.com Meme Center.c", 
              "meme", "Posted By: Jorge Kerby", 
              "434 points - 6 comments NSFW-1d Cookie did Wankie f Facebook 279 points - 4 comments Pinterest Porn actress after Bukkakke scene:", 
              "ple*", "V=1 krÂ². 3", "Meme Barbie - Www.sham.store", 
              "memegenarator.aut'", "memes", "'Ü\x90Ü\x90Ü\x90'"]
    cleaned_text = []
    
    # delete the elements in the list given
    for x in delete: 
      sentence = sentence.replace(x, '')
    
    
    # delete every punctuation signed that is repeated more than 3 times 
    new_str = re.sub(r'(\W)\1\1\1+', r'\1\1\1', sentence)
    # clean all characters that are not letters or special puntuation signs (?) 
    new_str = re.sub(r'[^a-zA-Z.!?*]',' ', new_str)
    #create a new list with the avobe elements already cleaned 

    # split all sentences into words or puntuation signs
    words = re.sub(r'([a-zA-Z])([,.!?/+*:])', r'\1 \2', 
                   re.sub( r'([,.!?/+*:])([a-zA-Z])', 
                          r'\1 \2', new_str)).split()
    #check all words to make sure we have none repeated 
    cleaned_text = [w for w in words if w not in ignore]
      
    return cleaned_text 
  
  def tokenize(self, sentences):
    '''
    This method  creates a dictionary with each unique word and the index number 
    in all words in all data.
    '''
    unique_words = list(set(sentences))
    index_words = {word: i for i, word in enumerate(unique_words)}
    return index_words

  def create_index_vocab(self):
    all_words = []
    for sen in self.data['Text Transcription']:
      all_words.extend(self.word_extraction(sen))
    indexed_words = self.tokenize(all_words)
    return indexed_words

"""**Custom dataset**

This custom dataset has been created to be able to use the Dataloader from the PyTorch library. In this class, we transform each image and each text to a tensor. When loaded with the PyTorch Dataloader, a sample containing the text tensor, image tensor, label and index will be returned for every batch.
"""

class custom_Dataset(Dataset):
  '''
  This class will create a custom Dataset for our data.
  We inherit the PyTorch Dataset class. 
  '''

  def __init__(self, data, index_vocab):
    self.data = data

    # method to transform images to tensors and to resize all of to (224, 224)
    self.image_transform = torchvision.transforms.Compose([
          torchvision.transforms.Resize(size=(224, 224)),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
          # Normalizes tensor with imagenet standars 
      ])

    self.data = self.data.reset_index(drop=True)
    self.index_vocab = index_vocab
    
  def __len__(self):
    '''
    This method returns the number of samples in our dataset.
    '''
    return len(self.data)

  def __getitem__(self, idx):  
    '''
    This method loads ans returns a sample from the dataset, given an index.
    @input:
      idx: index number of the data point 
    @output: 
      dict_data: a dictionary for each data point, with id, 
      text embeddings, image embeddings and label.
    '''
    
    if torch.is_tensor(idx):
            idx = idx.tolist()

    # load image, if in grey scale convert to color
    image = Image.open(self.data.loc[idx,'image_path'])
    image = image.convert('RGB')
    
    # transform it to a tensor (dimension: (3, 224, 224))
    tensor_img = self.image_transform(image)

    # get the tensor from the text transcription
    # txt = self.data.loc[idx,'Text Transcription']
    # transfrom our bag of word to a tensor 
    tensor_txt = torch.from_numpy(self.bow_matrix(idx)).float() 
      
    # only the trial and the train sets have labels so for the test set the 
    # dictionary will not have labels
    if 'misogynous' in self.data:
      # get the labels
      label = self.data.loc[idx, 'misogynous']
      sample = {'text': tensor_txt,
                'image': tensor_img,
                'label': label,
                'file_name': self.data.loc[idx,'file_name']
                }
      
    # here the same for the test set, without the labels line
    else:
      label = None
      sample = {'text': tensor_txt,
                'image': tensor_img,
                'file_name': self.data.loc[idx,'file_name']
                }
    return sample
  
  # some extra methods to create a bag of words:
  def word_extraction(self, sentence):
    '''
    This method takes one sentence and divide words and punctuation into a list 
    of words. 
    '''
    ignore_lower = ['a', "the", "is", "and", "other", "from", "to", "of", "at", 
                    "in", "by", "be", "for", "that", "with", "was", "were", 
                    "will", "is", "it's"]
    ignore_upper = [each.upper() for each in ignore_lower]
    ignore = ignore_lower + ignore_upper 
    delete = ["(15251 ()", "Meme", "Mickmeme.com1", "memecenter.com Meme Center.c", 
              "meme", "Posted By: Jorge Kerby", 
              "434 points - 6 comments NSFW-1d Cookie did Wankie f Facebook 279 points - 4 comments Pinterest Porn actress after Bukkakke scene:", 
              "ple*", "V=1 krÂ². 3", "Meme Barbie - Www.sham.store", 
              "memegenarator.aut'", "memes", "'Ü\x90Ü\x90Ü\x90'"]
    cleaned_text = []
    
    # delete the elements in the list given
    for x in delete: 
      sentence = sentence.replace(x, '')
    # delete every punctuation signed that is repeated more than 3 times 
    new_str = re.sub(r'(\W)\1\1\1+', r'\1\1\1', sentence)
    # clean all characters that are not letters or special puntuation signs (?) 
    new_str = re.sub(r'[^a-zA-Z.!?*]',' ',new_str)
    #create a new list with the avobe elements already cleaned 

    # split all sentences into words or puntuation signs
    words = re.sub(r'([a-zA-Z])([,.!?/+*:])', r'\1 \2', 
                   re.sub( r'([,.!?/+*:])([a-zA-Z])', 
                          r'\1 \2', new_str)).split()
    #check all words to make sure we have none repeated 
    cleaned_text = [w for w in words if w not in ignore]
    return cleaned_text 

  def create_matrix(self, index, index_words):
    '''
    This method creates a vector matrix with the len of all words in the 
    sentence.
    '''
    data = self.word_extraction(self.data.loc[index, 'Text Transcription'])
    matrix = np.empty((len(index_words)))

    for word, word_idx in index_words.items():
      matrix[word_idx] = 1 if word in data else 0
    return matrix

  def bow_matrix(self,index):
    '''
    @inputs: data and the index point of the text transcription
    @output: a bag of words matrix for each given sentence 
    ''' 
    bow_matrix = self.create_matrix(index, self.index_vocab)
    return bow_matrix

"""**Make a train, validation and test split**

We will split the train data into train, validation and test set. This will be done, so we will be able to evaluate performance better while training and to calculate the accuracy of our best model on unseen data. 

The test split will be used, instead of the given test set, to analyse the performance of our model, since we don't have the labels from the original test set.
"""

# in this split, 10% of the train data we set aside to test the performance of the model
x_train, X_test = train_test_split(df_train,test_size=0.1,shuffle = True, random_state=123)

# we will also make a validation set, with 9% of the x_train data
X_train, X_val = train_test_split(x_train, test_size=0.1, random_state= 10)

# since the data was balanced, we don't need to worry about balanced split according to labels, 
# as you can see, the data is still quite balanced after the random split
print('Checking size of our new train data and if data is still balanced:')
print("X_train shape: {}".format(X_train.shape))
print(X_train['misogynous'].value_counts())

print('Checking size of our validation data is also balanced:')
print("X_val shape: {}".format(X_val.shape))
print(X_val['misogynous'].value_counts())

print('Checking size of our test data is also balanced:')
print("X_test shape: {}".format(X_test.shape))
print(X_test['misogynous'].value_counts())

"""**Check dimensions and prepare data for training**

First we will prepare our data for training with tha DataLoader from pytorch
Our custom_Dataset() class retrives a sample from out data, with image tensor, text tensor, labels and index.

To train it with minibatches we will use the pytorch DataLoader. Here we can see the dimensions of the data set once we use the Dataloader. Once we use the Dataloader the chosen minibatching dimension is added.
"""

# create indexed vocabulary
iv = index_vocab(X_train)
index_vocab_xtrain = iv.create_index_vocab()
len_index_vocab_xtrain = len(index_vocab_xtrain)
print('Lenght of indexed vocabulary is of X_train:', len_index_vocab_xtrain)

# crete custom datasets:
train_dataset = custom_Dataset(X_train, index_vocab_xtrain)
val_dataset = custom_Dataset(X_val, index_vocab_xtrain)
test_dataset = custom_Dataset(X_test, index_vocab_xtrain)

# Create a loader for the training set which will read the data within 
# batch size and put into memory.
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# here the dimensions of the first 3 samples on the train data set
print('------------------ Data set dimensions: -------------------')
for i in range(len(train_dataset)):
    sample = train_dataset[i]
    print(i, sample['text'].size(), sample['image'].size(), sample['label'])
    if i == 2:
        break

# just to compare, when we use the Dataloader, the bath dimension is added:
print('----------------- Dataloader dimensions: ------------------')
count = 0
for i_batch, sample in enumerate(train_dataloader, 0):
  print(i_batch, sample['text'].size(), sample['image'].size(), len(sample['label']))
  if i_batch == 2:
    break

"""**Create model**

Here we created a flexible model that can be used with or without image data.

For the image data we use the pretrained resnet50 model. 

The language model is a liner model.

On the multimodal model, we concatenate language and images.

On the linear model we just use the language features.
"""

class mm_model(nn.Module):
  
  def __init__(self, len_index_vocab):

    super(mm_model, self).__init__()    

    # len of index vocab is the embedding dimension of text features
    self.embedding_dim = len_index_vocab 
    self.txt_feature_dim = 500
    self.img_feature_dim = 1000
    self.concat_output_size = 256
    self.num_classes = 1
    self.dropout_p = 0.1

    # pass text through Language Model to extract the text features
    # here we choose a Linear model
    self.language_module = nn.Linear(
        in_features=self.embedding_dim,
        out_features=self.txt_feature_dim)

    # pass images through Image Model resnet50 pre-trained on ImageNet
    # to get the image features
    self.vision_module = torchvision.models.resnet50(
            pretrained=True)

    # pass combined text and image features trough a Linear model
    self.fusion = nn.Linear(
        in_features=(self.img_feature_dim + self.txt_feature_dim), 
        out_features=self.concat_output_size)

    # check https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    self.dropout = nn.Dropout(self.dropout_p)

    # output layer:
    self.fc = nn.Linear(
        in_features=self.concat_output_size, 
        out_features=self.num_classes)
    
    # for the model without images - liner regression model
    self.text_lr = nn.Linear(
        in_features= self.embedding_dim, 
        out_features=self.num_classes
        ) 
        
  def forward(self, text, image):
    '''
    This method allows two diffenrent models to be trained, one with text images, 
    and one other without.
    If image is not available, the model will train a simple linear model.
    Otherwise, the model will concatenate image and text features.
    @ input:
      text: input text
      image: input image
      label: label
    '''
  
    if image is not None: 
      text_features = F.relu(self.language_module(text))
      image_features = F.relu(self.vision_module(image))
      # concatenate text and image features and treat it as a new imput vector
      concat = torch.cat((text_features, image_features), dim=1)
      # pass the combined features trough a linear model
      concat = self.dropout(F.relu(self.fusion(concat)))
      pred = self.fc(concat).float() 
    else:
      # pass linear regression model to get predictions
      pred = self.text_lr(text)
    
    return pred

"""**Train model**

Here are some methods to train the model, calculate and plot loss and accuracy.

Since the data is balanced, accuracy is a good enough for measuring performance.
"""

class train_model():

  def __init__(self, train_data, val_data, num_epochs, train_dataloader, 
               val_dataloader, start_time, lr, model, path, train_image = True):
    
    self.data = train_data
    self.val_data = val_data
    self.num_epochs = num_epochs
    self.start_time = start_time
    self.dataloader = train_dataloader
    self.val_dataloader = val_dataloader
    self.model =  model
    self.train_image = train_image
    self.learning_rate = lr
    self.loss_function = nn.BCEWithLogitsLoss()
    self.optimizer = Adam(model.parameters(), self.learning_rate)
    self.path = path
  
  def time_since(self, since):
    '''
    This method will help to calculate the time to train the model.
    It is the same used in on the NLP assignments.
    '''
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

  def correct(self, pred, label):
    '''
    This function sum the correct labels for each batch.
    '''
    pred_y = torch.round(torch.sigmoid(pred))
    correct_predictions= (pred_y == label).sum().float()
    return correct_predictions
  
  def fit(self):
    '''
    This method loops over our data iterator, feed the inputs to the network and optimize 
    weights.
    @ outputs: 
    train_loss: average train loss for each train iteration.
    train_accuracy: accuracy of the model for each train iterarion.
    '''

    train_running_loss = 0.0
    train_running_correct = 0
      
    for i, sample in enumerate(self.dataloader):
            
      # get the inputs
      texts = sample['text'].to(device)
      labels = sample['label'].float().to(device)
      if self.train_image == True:
        images = sample['image'].to(device)
      else:
        images = None
          
      # zero the parameter gradients
      self.optimizer.zero_grad()
      # predict classes using texts and images from the training set
      outputs = model(texts, images).squeeze()
      # compute the loss based on model output and real labels
      loss = self.loss_function(outputs, labels)
      # backpropagate the loss
      loss.backward()
      # calculate correct predictions per batch
      correct_preditions = self.correct(outputs, labels)
      # adjust parameters based on the calculated gradients
      self.optimizer.step()
      # sum all losses
      train_running_loss += loss.item()
      # sum all correct preedictions
      train_running_correct += correct_preditions.item()
        
    train_loss = train_running_loss/len(self.data)
    train_accuracy = 100. * train_running_correct/len(self.data)      
    return train_loss, train_accuracy
  
  def validate(self):
    '''
    This method loops over our data iterator, feed the inputs to the network and
    calculate loss and accuracy without optimizing (parameters are not ajusted, 
    loss is not backpropagated). 
    @ outputs: 
    train_loss: average train loss for each train iteration.
    train_accuracy: accuracy of the model for each train iterarion.
    '''

    val_running_loss = 0.0
    val_running_correct = 0

    for i, sample in enumerate(self.val_dataloader):
            
      # get the inputs
      texts = sample['text'].to(device)
      labels = sample['label'].float().to(device)
      if self.train_image == True:
        images = sample['image'].to(device)
      else:
        images = None

      # predict classes using texts and images from the training set
      outputs = model(texts, images).squeeze()
      # compute the loss based on model output and real labels
      loss = self.loss_function(outputs, labels)    
      # sum all losses
      val_running_loss += loss.item()
      # calculate correct predictions per batch
      correct_preditions = self.correct(outputs, labels)
      # sum all correct preedictions:
      val_running_correct += correct_preditions.item()
        
    val_loss = val_running_loss/len(self.val_data)
    val_accuracy = 100. * val_running_correct/len(self.val_data)      
    return val_loss, val_accuracy
  
  def train_and_validade(self):

    print('Trainning...')
    best_accuracy = 0   
    self.train_loss , self.train_accuracy = [], []
    self.val_loss , self.val_accuracy = [], []

    for epoch in range(self.num_epochs):
      train_epoch_loss, train_epoch_accuracy = self.fit()
      self.train_loss.append(train_epoch_loss)
      self.train_accuracy.append(train_epoch_accuracy)

      val_epoch_loss, val_epoch_accuracy = self.validate()
      self.val_loss.append(val_epoch_loss)
      self.val_accuracy.append(val_epoch_accuracy)
      
      print(f'Epoch {epoch+1} of {self.num_epochs} '
      f'| Train loss: {train_epoch_loss:.5f} '
      f'| Train acc: {train_epoch_accuracy:.2f} '
      f'| Val loss: {val_epoch_loss:.5f} '
      f'| Val acc: {val_epoch_accuracy:.2f} '
      f'| Time: {self.time_since(self.start_time)}' )  

      # save model with validation best acc:
      if val_epoch_accuracy > best_accuracy: 
        print(f'Validation accuracy increased ({best_accuracy:.2f}--->{val_epoch_accuracy:.2f}) \t Saving The Model')
        torch.save({'epoch': epoch,
                   'model_state_dict': model.state_dict()},
                   self.path)
        best_accuracy = val_epoch_accuracy 
     
  def plot_acc(self):
    '''
    Method to plot the loss and accuracy per epoch.
    '''
    print('')
    # plot validation and train accuracies
    plt.figure(figsize=(10, 7))
    plt.plot(self.train_accuracy, color='green', label='train accuracy')
    plt.plot(self.val_accuracy, color='blue', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plot validation and train losses
    plt.figure(figsize=(10, 7))
    plt.plot(self.train_loss, color='green', label='train loss')
    plt.plot(self.val_loss, color='blue', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# check if we are training on cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""**Training the model**

WARNING:

**If you dont want to train the model again, which will take some time, don't run the 2 cells below. Use the saved checkpoints on cell (Test the model).**

In the following cells we will train the model on the train data. Tho models will be trained:

1. linear model on text data (about 40  to 100 min to train on cuda)
2. multimodal model on train data (about 50 to 100 min to train on cuda)

"""

# set model parameters to train model without images: 
model = mm_model(len_index_vocab_xtrain).to(device) # instantiate model 
lr = 0.00003 # learning rate
start_time = time.time() # start time again
num_epochs = 20 # number of epochs
data_loader = train_dataloader # dataloader to load the X_train data
val_loader = val_dataloader # dataloader to load the X_val data
path = "/content/drive/My Drive/data_memes/best_model_train_lr" # path to save the best model

print('- 1 - Trainnig linear model on train data... \n')
# instantiate class to train the model. 
# Set train_image to false, so we don't train with images
train_lr = train_model(X_train, X_val, num_epochs, data_loader, val_loader, 
                       start_time, lr, model, path, train_image = False)

# train:
train_lr.train_and_validade()
# plot loss and accuracy of train and validation data:
train_lr.plot_acc()

# train multimodal model
# set parameters to train with images:
model = mm_model(len_index_vocab_xtrain).to(device) # instantiate model with 80% of the the train data set data
lr=0.000001 # learning rate
start_time = time.time() # start time again
num_epochs = 20 # number of epochs
# here I will decrease batch size because I need more memory to load images
data_loader = train_dataloader # dataloader to load the X_train data
val_loader = val_dataloader # dataloader to load the X_val data

path = "/content/drive/My Drive/data_memes/best_model_train_mm" 

# instantiate train object and train multimodal model
print('- 2 - Trainnig multimodal model on train data... \n')
train_mm = train_model(X_train, X_val, num_epochs, data_loader, val_loader, 
                       start_time, lr, model, path)
# train:
train_mm.train_and_validade()
# plot loss and accuracy of train and validation data:
train_mm.plot_acc()

"""**Evaluation of the model**

To evaluate or model, it is important to test it on unseen data. We already did a split on the data, reserving 10% of the train data. Now we will run some tests, to see how the models perform.
"""

def test(path, test_dataloader, x_test, test_image):
  # Load the model that we saved at the end of the training loop and use it to
  # make predictions
  model = mm_model(len_index_vocab_xtrain).to(device)
  # load the best model checkpoint
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict']) 
  # set model to evaluation mode
  model.eval()
  print('Epoch of model saved:', checkpoint['epoch']+1)
  pred_classes = {}
  correct_pred=0
 
  with torch.no_grad(): 
    for i, sample in enumerate(test_dataloader):
      # get the inputs
      texts = sample['text'].to(device) 
      if test_image == True:
        images = sample['image'].to(device) 
      else:
        images = None
      file_name = sample['file_name']
      labels = sample['label'].float().to(device) 

      # forward pass
      outputs = model(texts, images).squeeze()
      
      # calculate accuracy
      pred_class = torch.round(torch.sigmoid(outputs)).to(device) 
      correct_preditions = (pred_class == labels).sum().float()
      correct_pred += correct_preditions.item()
      pred_classes[file_name[0]] = pred_class.item()

    # calculate accuracy of predictions
    acc = (correct_pred/len(x_test))*100
    return acc, pred_classes

def print_labels_pred(data, init_index, pred_dict):
    '''
    To help us to visualize the images, labels and predictions. It prints 3 
    images, labels and predictions, starting at a certain index.
    @input: 
      data = data frame 
      init_index = index of first image to be printed
      pred_dict = predictions dictionary extracted while testing the model
    @output:
      printed memes with label and prediction, starting at a certain index.
    '''  
    files = X_test['file_name'].tolist()

    list_images=[]
    for i, fn in enumerate(files):
      if i == init_index:
        list_images.append(fn)
      if i == init_index + 1:
        list_images.append(fn)
      if i == init_index + 2:
        list_images.append(fn)
  
    # create figure
    fig = plt.figure(figsize=(20, (15)))
  
    # setting values to rows and column variables
    cols = 3 
    rows = 3

    # set font style
    font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 22}

    # read images
    images = []
    for i, file_name in enumerate(list_images):

      fn = "file_name == '", file_name,"'"
      fn = " ".join(str(x) for x in fn)
      whitespace = r"\s+"
      fn = re.sub(whitespace, "", fn)
      idx = X_test.query(fn).index.item()
      image = Image.open(data.loc[idx,'image_path'])

      # Adds a subplot for each position
      fig.add_subplot(rows, cols, i + 1)

      # add label to image
      if (data.loc[idx, 'misogynous']).astype(int) == 1:
        plt.title('Label = misogynous',  fontdict=font)
      else:
        plt.title('Label = not misogynous',  fontdict=font)

      # add prediction to image
      if (pred_dict[list_images[i]]) == 1.0:
        plt.xlabel('Pred = misogynous',  fontdict=font)
      else:
        plt.xlabel('Pred = not misogynous',  fontdict=font)

      plt.imshow(image)     
    return None     
    return None

# make predictions on unseen data, using the liner regression model 
path = "/content/drive/My Drive/data_memes/best_model_train_lr"
acc_lr, pred_lr = test(path, test_dataloader, X_test, test_image = False)
print('Accuracy of linear regression model on unseen data:', acc_lr, '%')
# print some predictions, labels and images
print_labels_pred(X_test, 2, pred_lr)
print_labels_pred(X_test, 267, pred_lr)

# make predictions on unseen data, using the multimodal model
path = "/content/drive/My Drive/data_memes/best_model_train_mm" 
acc_mm, pred_mm = test(path, test_dataloader, X_test, test_image = True)
print('Accuracy of multimodal model on unseen data:', acc_mm, '%')
# print some predictions, labels and images
print_labels_pred(X_test, 15, pred_mm)
print_labels_pred(X_test, 150, pred_mm)

# here multimodal model perfoms better:
print('-------------------------------------------> Predictions Linear Regression Model <---------------------------------------------')
print_labels_pred(X_test, 800, pred_lr)

print('-----------------------------------------------> Predictions Multimodal Model <--------------------------------------------------')
print_labels_pred(X_test, 800, pred_mm)

# here linear regression performs better:
print('-------------------------------------------> Predictions Linear Regression Model <---------------------------------------------')
print_labels_pred(X_test, 310, pred_lr)

print('-----------------------------------------------> Predictions Multimodal Model <--------------------------------------------------')
print_labels_pred(X_test, 310, pred_mm)

# here both models get the same predictions:
print('-------------------------------------------> Predictions Linear Regression Model <---------------------------------------------')
print_labels_pred(X_test, 940, pred_lr)

print('-----------------------------------------------> Predictions Multimodal Model <--------------------------------------------------')
print_labels_pred(X_test, 940, pred_mm)