from tkinter import * #Permite desenvolver interfaces gráficas
from tkinter import filedialog as dlg, messagebox as mbox #Exibe caixas de diálogo de manipulação de arquivos
import urllib.request  #Usado para abrir e ler URLs

import cv2
import numpy as np

import sys #Funções voltadas principalmente para trabalhar com as "configurações" obtidas na execução atual de um script
import time #Fornece várias funcionalidades relacionadas ao tempo
import os #Funcionalidades que permitem ao programador acessar e manipular arquivos, pastas, executar comandos do sistema
import Processamento #Importa o script 'processamento'

# Variáveis globais
# Lists to store the Lines coordinators
elementLines = []
calibrationLine = []
tempLines = []
clicked = 0
totalLines = 0
totalColumns = 0
pixFactor = 0.247525
dFactor = 1
auxFactor = 1
videoMode = -1


def retrieve_input(textBox):
    global pixFactor
    inputValue = textBox.get("1.0", "end-1c")
    textBox.quit()
    pixFactor = int(inputValue)


def capture_distance():
    root = Tk()
    root.title('Informe referência em cm')
    root.geometry("300x80")
    textBox = Text(root, height=2, width=10)
    textBox.pack()
    buttonCommit = Button(root, height=1, width=10, text="confirma", command=lambda: retrieve_input(textBox))
    # command=lambda: retrieve_input() >>> just means do this when i press the button
    buttonCommit.pack()
    mainloop()
    root.destroy()


# function which will be called on mouse input
def mouseActions(action, x, y, flags, *userdata):
    # Referencing global variables
    global elementLines, tempLines, originalImage, clicked, totalLines, totalColumns, calibrationLine, pixFactor, dFactor, auxFactor
    # Mark the top left corner when left mouse button is pressed

    if action == cv2.EVENT_LBUTTONDOWN:
        if clicked == 3:
            calibrationLine.append((x, y, clicked))
            calibrationLine.append((x, y, clicked))
        else:
            clicked = 1
            tempLines = [(y, clicked)]
            # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        if clicked == 3:
            calibrationLine[1] = (x, y, clicked)
            xi, yi, c = calibrationLine[0];
            xf, yf, c = calibrationLine[1];
            calibrationLine = []
            clicked = 0
            try:
                dX = xf * (1 / auxFactor) - xi * (1 / auxFactor);
                dY = yf * (1 / auxFactor) - yi * (1 / auxFactor);
                dFactor = (dX ** 2 + dY ** 2) ** 0.5
                print(f'Medida em Pixel => {dFactor}')
                capture_distance()
                pixFactor = pixFactor / dFactor
            except:
                pixFactor = 1
        else:
            elementLines.append((y, clicked))
        clicked = 0
        # Draw the rectangle
    elif action == cv2.EVENT_RBUTTONDOWN:
        clicked = 2
        tempLines = [(x, clicked)]
    elif action == cv2.EVENT_RBUTTONUP:
        elementLines.append((x, clicked))
        clicked = 0
    elif action == cv2.EVENT_MOUSEMOVE:
        if clicked != 0:
            tempLines = []
            if clicked == 3 and len(calibrationLine) > 1:
                calibrationLine[1] = (x, y, clicked)
            if clicked == 1:
                tempLines = [(y, clicked)]
            if clicked == 2:
                tempLines = [(x, clicked)]


def captureImage(source):
    fileName = ''
    if (source == 1):
        fileName = dlg.askopenfilename()  # .asksaveasfilename(confirmoverwrite=False)
        if fileName != '':
            print(fileName)
            fullImage = cv2.imread(fileName)
    elif (source == 0):
        with urlopen('http://10.14.38.133:8080/shot.jpg') as url:
            imgResp = url.read()
        imgNp = np.array(bytearray(imgResp), dtype=np.uint8)  # Numpy to convert into a array
        fullImage = cv2.imdecode(imgNp, -1)  # Finally decode the array to OpenCV usable format ;)
    elif (source == 2):
        fullImage = cv2.imread('egg_measure.jpeg')

    # fullImage = cv2.rotate(fullImage, cv2.ROTATE_90_CLOCKWISE)
    totalLines, totalColumns, rFactor = Processamento.adjustImageDimension(fullImage)  # Adjust full image to fit on screem
    down_points = (totalColumns, totalLines)
    originalImage = cv2.resize(fullImage, down_points, interpolation=cv2.INTER_LINEAR)

    return (fullImage, originalImage, totalLines, totalColumns, rFactor, fileName)


# Programa Principal
if sys.version_info[0] == 3: #Verifica se a versão do python é a 3
    from urllib.request import urlopen
else:
    from urllib.request import urlopen

cv2.namedWindow("Window")  #Cria uma nova janeja

#Highgui function called when mouse events occur
#Permite a interação do Mouse na janela criada
cv2.setMouseCallback("Window", mouseActions)

#Lê a imagem e atribui valores aos devidos parâmetros
fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(2)
auxFactor = rFactor

#Loop constante
while True:
    # Display the image
    if videoMode == 1:
        fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(0)

    image = originalImage.copy() #Cópia da imagem lida na etapa anterior



    cv2.imshow("Window", image) #Display da imagem
    k = cv2.waitKey(1) #Recebe um comando do teclado

    if (k == 113):  # 'q' pressed to quit the system
        break

    if (k == 100):  # 'd' pressed to delete the last builded line
        if len(elementLines) > 0:
            del elementLines[-1]
        tempLines = []
        clicked = 0

    if (k == 112):  # 'p' pressed to activate the image processing
        pixFactor = 0.25041736227045075125208681135225
        print("Nome Arquivo antes da chamada de processamento ==> ", nomeArquivo)
        Processamento.subImagens(elementLines, fullImage, totalColumns, totalLines, rFactor, pixFactor, dFactor, nomeArquivo)

        fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(1)

    if (k == 99):  # 'c' pressed to capture new image
        videoMode = -1
        fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(0)

    if (k == 102):  # 'f' to read image from file
        fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(1)

    if (k == 120):  # 'x' pressed to capture the calibration of pixFactor
        clicked = 3
        calibrationLine = []
    if (k == 118):  # 'v' video mode pressed
        videoMode = videoMode * -1

    if (k == 115):  # 's' Save the image into a file
        fileNameSave = dlg.asksaveasfilename(confirmoverwrite=False)
        if fileNameSave != '':
            cv2.imwrite(fileNameSave, fullImage)

cv2.destroyAllWindows()
