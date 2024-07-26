import generalized_methods as gm

imgPath = 'C:/Users/marci/Desktop/Projeto mestrado/Images Dataset/Batch 06.04 Images'
csvPath = 'C:/Users/marci/Desktop/Projeto mestrado/Datasets/Batch 06.04/RealData.csv'

imgData, csvData = gm.getImages(imgPath, csvPath)
dataTrain, dataTest, classTrain, classTest = gm.dataSplit(imgData, csvData, perc = 0.2, norm = 1)

# trecho responsável pelo treinamento das redes
alexNet = gm.AlexNet(dataTrain = dataTrain,
                         classTrain= classTrain,
                         batch_size = 32,
                         epochs = 2,
                         valData = (dataTest, classTest))

# Salvar os pesos de um modelo, comentar caso não seja necessário
alexNet.save_weights('alex.weights.h5')
