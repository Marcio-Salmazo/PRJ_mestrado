import glob
from keras.models import Sequential  # Modelo sequencial
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import cv2
import os

# ----------------------------------------------------------------------------------------

''' 
    Carregando a pasta de imagens (Dataset dos ovos)
    * A função glob encontra todos os nomes de caminhos que correspondem a um padrão 
      especificado de acordo com as regras usadas pelo shell Unix

    * Para cada um dos arquivos encontrados é feita a leitura
      do arquivo .jpg (imagem), as imagens lidas são armazenadas
      na lista 'data'
'''

img_dir = "C:/Users/marci/Desktop/Projeto mestrado/CNN Egg application/Egg Dataset"
data_path = os.path.join(img_dir,'*g')

folder = glob.glob(data_path)
data = []
for files in folder:
    img = cv2.imread(files)
    data.append(img)

eggData = np.array(data)
eggClass = pd.read_csv("RealData.csv")

dataTrain, dataTest = train_test_split(eggData, test_size=0.2)
classTrain, classTest = train_test_split(eggClass, test_size=0.2)

print('Shape dataTrain:', dataTrain.shape)
print('Shape dataTest:', dataTest.shape)
print('Shape classTrain:', classTrain.shape)
print('Shape classTest:', classTest.shape)

# ----------------------------------------------------------------------------------------

'''
    Alterando o tipo dos dados para float32 a fim de aplicar 
    a normalização futuramente
'''

dataTrain = dataTrain.astype('float32')
dataTest = dataTest.astype('float32')

'''
    Realizando a normalização (min/max normalization) a fim de que os valores dos pixels estejam
    entre 0 e 1, tornando o processamento mais eficiente
    obs: 255 é o valor máximo do pixel
'''

dataTrain /= 255
dataTest /= 255

# ----------------------------------------------------------------------------------------

'''
   Definição da rede neural convolucional

    * Criação da cnn no modelo sequencial (sequencia de layers)
    * Criação de duas camadas de convolução com função de ativação Relu, 
      seguidas pelos processos de normalização dos mapas de características e max Pooling.
      O processo de flattening é adicionado ao final das camadas. 
'''

classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape = (512, 512, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(32, (3,3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

'''
    * Adição das hidden Layers
    * Aplicação da operação de dropout às saídas 
        Busca zerar uma determinada quantidade de entradas 
        a fim de otimizar o sistema e reduzir o overfitting
    * Adição da camada de saída, utilizando a sigmóide como função de ativação
        A sigmóide é utilzada em classificações binárias
        OBS -> units = 1 indica que há apenas uma unidade de saída para
        a classificação binária
'''
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

'''
    * Compilação da rede neural
        OBS: Para a classificação binária, a função de perda será definida 
        por 'binary_crossentropy'. Para a classificação em múltiplas classes
        é necessário utilizar o 'categorical_crossentropy'
'''

classifier.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# ----------------------------------------------------------------------------------------

''' 
    Etapa de treinamento da rede neural. É importante que a função 'fit_generator' está sendo
    utilizada ao invés da função 'fit' uma vez que ela suporta o processo de augmentation, contudo
    tal função está em processo de depreciação, uma vez que em versões mais atuais, a função 'fit' 
    também suporta.

    Explicação dos parâmetros:

    * trainDatabase -> Dados para treino (após a augmentation)
    * steps_per_epoch -> Número total de etapas (lotes de amostras) a serem produzidas 
                         pelo gerador antes de declarar uma época concluída e iniciar a próxima época.
                         É importante citar que o valor ideal para este parâmetro se dá pela quantidade 
                         total de imagens para treinamento (caso haja um alto poder de processamento) ou
                         pelo total de amostras dividido pel valor do batch_size (caso haja um baixo 
                         poder de processamento)
    * epochs -> Epocas de treinamento da rede
    * validation_data -> Dados para a validação (após a augmentation)
    * validation_steps -> Possui o mesmo princípio do 'steps_per_epoch', porém levando em 
                          consideração a etapa de validação. O valor ideal para este parâmetro 
                          se dá pelo total de amostras dividido pel valor do batch_size
'''

classifier.fit(dataTrain, classTrain, batch_size = 128, epochs = 5, validation_data = (dataTest, classTest))