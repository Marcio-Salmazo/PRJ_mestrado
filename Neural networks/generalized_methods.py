'''        
    Importação das bibliotecas necessárias para a contrução e configuração das redes neurais convolucionais:

    1 - TensorFlow: amplamente usada para criar e treinar modelos de aprendizado profundo
    2 - models: fornece classes e funções para construir, configurar e treinar modelos de redes neurais
    3 - layers: contém uma variedade de classes que representam diferentes tipos de camadas em redes neurais. 
    4 - TensorBoard:  ferramenta de visualização do TensorFlow que ajuda a monitorar e visualizar o comportamento dos modelos
    5 - train_test_split:  usada para dividir conjuntos de dados em subconjuntos de treino e teste.
    
    Importação das bibliotecas necessárias para a manipulção dos dados e arrays:
    
    1 - numpy: fornece suporte para arrays e matrizes de grandes dimensões
    2 - pandas: biblioteca para manipulação e análise de dados em Python
    3 - pyplot: usada para criar gráficos e visualizações em Python
    4 - cv2: biblioteca de visão computacional e aprendizado de máquina.
    5 - os: biblioteca padrão do Python que fornece uma maneira de interagir 
        com o sistema operacional. Ela inclui funções para manipulação de arquivos, 
        diretórios, e processos.
    6 - glob: biblioteca padrão do Python que facilita a busca por arquivos em diretórios com base em padrões especificados, 
        usando caracteres curinga como * e ?.
'''
import tensorflow as tf
from keras.models import Sequential  
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
import os 
import glob

# FUNÇÕES RELACIONADAS A CONSTRUÇÃO DAS REDES NEURAIS

def AlexNet(input_shape = (512, 512, 3), 
            dataTrain = None, 
            classTrain = None, 
            batch_size = 32, 
            epochs = 5, 
            valData = None,
            loss='binary_crossentropy', 
            optimizer='adam',
            metrics=['accuracy']):
        

    model = Sequential([

        Conv2D(96, (11, 11), strides=4, activation='relu', input_shape = input_shape, padding='valid'),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        Conv2D(256, (5, 5), strides=1, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(3, 3), strides=2),
                
        Conv2D(384, (3, 3), strides=1, activation='relu', padding='same'),    
        Conv2D(384, (3, 3), strides=1, activation='relu', padding='same'),
        Conv2D(256, (3, 3), strides=1, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(3, 3), strides=2),
                
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(units = 1, activation = 'sigmoid')
    ])

    board = TensorBoard(log_dir='./logsAlex')
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    model.fit(dataTrain, classTrain, batch_size = batch_size, epochs = epochs, validation_data = valData, callbacks=[board])
    
# ------------------------------------------------------------------------------------------------------------------------------------------------

def ShallowNet(input_shape = (512, 512, 3), 
               dataTrain = None, 
               classTrain = None, 
               batch_size = 32, 
               epochs = 5, 
               valData = None,
               loss='binary_crossentropy', 
               optimizer='adam', 
               metrics=['accuracy']):
        
    model = Sequential([

        Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size = (2,2)),
        Conv2D(32, (3,3), activation = 'relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size = (2,2)),
            
        Flatten(),
        Dense(units = 128, activation = 'relu'),
        Dropout(0.2),
        Dense(units = 128, activation = 'relu'),
        Dropout(0.2),
        Dense(units = 1, activation = 'sigmoid')
    ])

    board = TensorBoard(log_dir='./logsShallow')
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    model.fit(dataTrain, classTrain, batch_size = batch_size, epochs = epochs, validation_data = valData, callbacks=[board])

# ------------------------------------------------------------------------------------------------------------------------------------------------

def ShallowMLP(dataTrain = None, 
               classTrain = None, 
               batch_size = 32, 
               epochs = 5, 
               valData = None,
               loss='binary_crossentropy', 
               optimizer='adam', 
               metrics=['accuracy']):
        
    model = Sequential([

        Dense(64, input_dim=dataTrain.shape[1], activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    board = TensorBoard(log_dir='./logsShallowMLP')
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    model.fit(dataTrain, classTrain, batch_size = batch_size, epochs = epochs, validation_data = valData, callbacks=[board])

# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------

# FUNÇÕES RELACIONADAS AO TRATAMENTO DOS DADOS

def getImages(ImagePath, csvPath):

    # Carregamento das imagens:
    # img_dir = "C:/Users/marci/Desktop/Projeto mestrado/CNN Egg application/Egg Dataset"
    # img_dir = "C:/Users/marci/Desktop/Arquivos/PRJ_mestrado/CNN Egg application/Egg Dataset"
    data_path = os.path.join(ImagePath,'*g') 

    folder = glob.glob(data_path) 
    data = [] 
    for files in folder: 
        img = cv2.imread(files) 
        data.append(img) 

    # Carregamento do .csv
    eggClass = pd.read_csv(csvPath)
    cList = eggClass.to_numpy()

    return data, cList

# ------------------------------------------------------------------------------------------------------------------------------------------------

def dataSplit(data, cList, perc, norm = 0):

    # Divisão do dataset entre subsets para treino e teste
    # A divisão é feita levando em consideração os mesmos índices
    ind = np.arange(len(data))
    train, test = train_test_split(ind, test_size=perc, random_state=42)

    dataTrain = []
    dataTest = []
    classTrain = []
    classTest = []

    for i in range(len(train)):

        dataTrain.append(data[train[i]])
        classTrain.append(cList[train[i]])

    for j in range(len(test)):

        dataTest.append(data[test[j]])
        classTest.append(cList[test[j]])

    dataTrain = np.array(dataTrain)
    dataTest = np.array(dataTest)
    classTrain = np.array(classTrain)
    classTest = np.array(classTest)

    # Processo de normalização de imagens
    if norm == 1:

        dataTrain = dataTrain.astype('float32')
        dataTest = dataTest.astype('float32')

        dataTrain /= 255
        dataTest /= 255

        return dataTrain, dataTest, classTrain, classTest
        
    elif norm == 2:
        dataTrain = dataTrain.astype('float32')
        dataTest = dataTest.astype('float32')

        scalar = MinMaxScaler()
        dataTrain = scalar.fit_transform(dataTrain)
        dataTest = scalar.fit_transform(dataTest)

    else:

        return dataTrain, dataTest, classTrain, classTest