Para abrir os arquivos de análise sobre o treinamento é necessário
utilizar o Tensorboard. No código, é necessário aplicar os seguintes comandos:

* import tensorflow as tf
* from tensorflow.keras.callbacks import TensorBoard
*
* board = TensorBoard(log_dir='./logs')
* model.fit(dataTrain, classTrain, batch_size = 16, epochs = 3, validation_data = (dataTest, classTest),     callbacks=[board])

Ao final da execução será criada uma pasta logs na raíz, contendo arquivos no formato .v2.
Para abrir o arquivo de log gerado pelo TensorBoard é necessário iniciar o servidor 
TensorBoard e apontar para o diretório onde o arquivo de log está armazenado. 

1 - Inserir o comando 'tensorboard --logdir=logs' no cmd Windows, já na pasta raiz do 
    projeto (onde foi gerada a pasta logs)
2 - Acessar o tensorboard no navegador. Após a execução do comando, será fornecido uma URL no terminal, 
    algo como http://localhost:6006/.
