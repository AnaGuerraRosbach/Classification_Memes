import pandas as pd
import re
import torchvision
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time, math


# paths for the dada -> we did not change how it was structured on the MANI data set
training_path = ('data_memes/training/TRAINING/training.csv')
test_path = ('data_memes/test/test/Test.csv')
image_path_train = 'data_memes/training/TRAINING/'
image_path_test = 'data_memes/test/test/'


print('-----------Train data-----------')
df_train = pd.read_csv(training_path, delimiter="\t")
print(df_train['misogynous'].value_counts())
print('--------------------------------')

# print('-----------Test data-----------')
df_test = pd.read_csv(test_path, delimiter="\t")
# print(df_test.value_counts()) -> does not work, since we do not have labels for
# test set

# observe the characteristics of our text transcriptions
list(df_train["Text Transcription"][:20])
# clean the strings by removing the web page from which it was extracted, to avoid noise in the classification.
df_train["Text Transcription"] = df_train["Text Transcription"].str.replace("(?i)([a-z]+\.(com|net|c|org))", "")
list(df_train["Text Transcription"][:20])

#we are going to use the time since function to calculate how long we need to train the model
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def add_image_path_and_index(image_path, df):
    '''
    This will change the data a bit:
    Image_path will be added as a new column.
    Id column will be added.


    For the train and test set:
    drop colums 'shaming', 'stereotype', 'objectification', and
    'violence' that are not relevant for task 5.a
    '''
    df['image_path'] = image_path + df['file_name']
    df['id'] = None
    df['id'] = df['file_name']
    df['id'] = df['id'].str.strip('.jpg')
    df['id'] = df['id'].astype(int)
    # df = df.sort_values(by=['id'])
    if 'misogynous' in df:
        df = df.drop(['shaming', 'stereotype', 'objectification', 'violence'], axis=1)
    return df


df_train = add_image_path_and_index(image_path_train, df_train)
df_test = add_image_path_and_index(image_path_test, df_test)

print('-----------Image path and Id added to data-----------')
print('-----------Columns shaming, stereotype, objectification, and violence removed from data-----------')


def random_print(num_memes_to_print, data):
    '''
    To help us to visualize the images, this method prints a certain number of
    memes and its correspondent label, randomly.
    Here we can see that images have different dimensions and we should change
    this.
    @input:
      num_memes_to_print: number of memes to be printed
    @output:
      printed memes
    '''

    file_len = len(data)
    start_index = random.randint(0, file_len - 1)

    # create figure
    fig = plt.figure(figsize=(20, (num_memes_to_print * 3)))

    # setting values to rows and column variables
    cols = 3
    if num_memes_to_print % 3 == 0:
        rows = num_memes_to_print / 3
    elif (num_memes_to_print + 1) % 3 == 0:
        rows = (num_memes_to_print + 1) / 3
    else:
        rows = (num_memes_to_print + 2) / 3

    # set font style
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 22,
            }

    # read images
    images = []
    for i in range(num_memes_to_print):
        # generate random number index to be printed
        # attention: image_file number and index are not the same in most cases
        random_n = random.randint(0, file_len - 1)
        image = Image.open(data.loc[random_n, 'image_path'])

        # Adds a subplot for each position
        fig.add_subplot(rows, cols, i + 1)

        if (data.loc[random_n, 'misogynous']).astype(int) == 1:
            plt.title('Label = misogynous', fontdict=font)
        else:
            plt.title('Label = not misogynous', fontdict=font)
        plt.imshow(image)
        plt.axis('off')
    return None


#random_print(12, df_train)


class custom_Dataset(Dataset):
    '''
    This class will create a custom Dataset for our data.
    We inherit the PyTorch Dataset class.
    '''

    def __init__(self, data):
        self.data = data

        # method to transform images to tensors and to resize all of
        # to (224, 224)
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)), # resize to 224x224 input
            torchvision.transforms.ToTensor(), # Turn PIL Image to torch.Tensor
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),   # Normalizes tensor imagenet standars
        ])

        # here we could also load the image in color with pil loader:
        # check: https://stackoverflow.com/questions/59218671/runtimeerror-output-with-shape-1-224-224-doesnt-match-the-broadcast-shape

        self.data = self.data.reset_index(drop=True)

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

        # load image (if it is in grey scale convert to colors)
        image = Image.open(self.data.loc[idx, 'image_path'])
        image = image.convert('RGB')

        # transform it to a tensor
        # size = (3, 224, 224)
        tensor_img = self.image_transform(image)

        # get the tensor from the text transcription
        # txt = self.data.loc[idx,'Text Transcription']
        # transform out bag of words to a tensor
        tensor_txt = torch.from_numpy(self.bow_matrix(idx)).float()

        # only the trial and the train sets have labels so for the test set the
        # dictionary will not have labels

        if 'misogynous' in self.data:
            # get the labels
            label = self.data.loc[idx, 'misogynous']
            sample = {'text': tensor_txt,
                      'image': tensor_img,
                      'label': label
                      }

        # here the same for the test set, without the labels line
        else:
            label = None
            sample = {'text': tensor_txt,
                      'image': tensor_img,
                      'label': label
                      }

        return sample

    # some extra methods to create a bag of words:
    def word_extraction(self, sentence):
        '''
        This method takes one sentence and divide words and punctuation into a list
        of words.
        '''
        ignore = ['a', "the", "is", "a"]
        # might need to check afterwards whether we can get rid of puntuation or not
        words = re.sub(r'([a-zA-Z])([,.!?])', r'\1 \2', re.sub(r'([,.!?])([a-zA-Z])', r'\1 \2', sentence)).split()
        cleaned_text = [w for w in words if w not in ignore]
        return cleaned_text

    def tokenize(self, sentences):
        '''
        This method  creates a dictionary with each unique word and the index number
        in all words in all data.
        '''
        unique_words = list(set(sentences))
        index_words = {word: i for i, word in enumerate(unique_words)}  # create word: index pairs

        return index_words

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

    def bow_matrix(self, index):
        '''
        @inputs: data and the index point of the text transcription
        @output: a bag of words matrix for each given sentence
        '''
        all_words = []
        for sen in self.data['Text Transcription']:
            all_words.extend(self.word_extraction(sen))
        indexed_words = self.tokenize(all_words)
        # indexed_words = tokenize(chain.from_iterable(word_extraction(data)))
        bow_matrix = self.create_matrix(index, indexed_words)
        return bow_matrix


# Preparing your data for training with tha DataLoader from pytorch
# Our custom_Dataset() class retrives a sample from out data,
# with image tensor, text tensor and label
# to train it with minibatches we will use  the pytorch DataLoader

test_dataset = custom_Dataset(df_test)
train_dataset = custom_Dataset(df_train)

print('Custom dataset created.')

# Create a loader for the training set which will read the data within
# batch size and put into memory.
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)



print('Data loader created with batch size 64.')


class multimodal_model(nn.Module):
    def __init__(self, data):
        super(multimodal_model, self).__init__()

        self.data = data
        # I hard coded the dimensions, but we will change this later
        self.num_classes = 1
        # dimension training set:
        self.embedding_dim = 32319

        self.txt_feature_dim = 500
        self.img_feature_dim = 500
        self.concat_output_size = 256
        self.dropout_p = 0.1

        # pass text through Language Model to extract the text features
        # here we choose a Linear model
        self.language_module = nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.txt_feature_dim
        )

        # pass images through Image Model resnet152 pre-trained on ImageNet
        # to get the image features
        self.vision_module = torchvision.models.resnet152(
            pretrained=True)

        # pass combined text and image features trough a Linear model
        self.fusion = nn.Linear(
            in_features=(self.img_feature_dim + self.txt_feature_dim),
            out_features=self.concat_output_size
        )

        # check https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        # we can choose latter if we use this or not
        self.dropout = nn.Dropout(self.dropout_p)

        # output layer:
        self.fc = nn.Linear(
            in_features=self.concat_output_size,
            out_features=self.num_classes)

    def forward(self, text, image):
        '''
        @ input:
          text: input text
          image: input image
          label: label
        '''
        text_features = F.relu(self.language_module(text))

        image_features = F.relu(self.vision_module(image))

        # concatenate text and image features and treat it as a new imput vector
        concat = torch.cat([image_features], dim=1)

        # pass the combined features trough a linear model
        concat = self.dropout(F.relu(self.fusion(concat)))

        pred = self.fc(concat).float()

        return pred


# instantiate model
model = multimodal_model(df_train)
print('Model created.')
# Define the loss function with Classification Binary Cross Entropy and an
# optimizer with Adam optimizer
# Binary Cross Entropy
loss_function = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.000001)

import torch
print('Trainning started... Time starts running now.')
start = time.time()

def correct(pred, label):
    '''
    This function sum the correct labels for each batch.
    '''

    pred_y = torch.round(torch.sigmoid(pred))
    correct_predictions = (pred_y == label).sum().float()

    return correct_predictions


def train(num_epochs):
    '''
    Loop over our data iterator and feed the inputs to the network and optimize.
    '''

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        epoch_loss = 0
        epoch_correct = 0

        for i, sample in enumerate(train_dataloader):
            # get the inputs
            texts = sample['text']
            images = sample['image']
            labels = sample['label'].float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # predict classes using images from the training set
            outputs = model(texts, images).squeeze()

            # compute the loss based on model output and real labels
            loss = loss_function(outputs, labels)

            # backpropagate the loss
            loss.backward()

            # calculate correct predictions per batc
            correct_preditions = correct(outputs, labels)

            # adjust parameters based on the calculated gradients
            optimizer.step()

            epoch_loss += loss.item()

            # sum all corrrect preedictions
            epoch_correct += correct_preditions.item()

        print(
            f'Epoch {epoch + 0:02}: '
            f'| Loss: {epoch_loss / len(df_train):.5f} '
            f'| Acc: {(epoch_correct / len(df_train))*100:.2f} '
            f'| Time: {time_since(start)}')

train(10)
