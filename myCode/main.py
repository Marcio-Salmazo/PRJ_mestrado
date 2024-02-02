from tkinter import *
from tkinter import filedialog as dlg, messagebox as mbox
import cv2
import Processamento

# globals
elementLines = []
totalLines = 0
totalColumns = 0
pixFactor = 0.247525
dFactor = 1
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
    buttonCommit.pack()
    mainloop()
    root.destroy()


def captureImage(source):
    fileName = ''
    if source == 1:
        fileName = dlg.askopenfilename()
        if fileName != '':
            print(fileName)
            fullImage = cv2.imread(fileName)
    elif source == 2:
        fullImage = cv2.imread('egg_measure.jpeg')

    # Adjust full image to fit on screem
    totalLines, totalColumns, rFactor = Processamento.adjustImageDimension(fullImage)
    down_points = (totalColumns, totalLines)
    originalImage = cv2.resize(fullImage, down_points, interpolation=cv2.INTER_LINEAR)

    return fullImage, originalImage, totalLines, totalColumns, rFactor, fileName


# -----------------------------------------------------------------------------------------------------------------------


fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(2)

while True:
    image = originalImage.copy()
    cv2.imshow("Window", image)
    k = cv2.waitKey(1)

    if k == 113:  # 'q' pressed to quit the system
        break

    if k == 112:  # 'p' pressed to activate the image processing
        pixFactor = 0.25041736227045075125208681135225
        print("Nome Arquivo antes da chamada de processamento ==> ", nomeArquivo)

        Processamento.subImagens(fullImage, totalColumns, totalLines, pixFactor, nomeArquivo)
        fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(1)

    if k == 102:  # 'f' to read image from file
        fullImage, originalImage, totalLines, totalColumns, rFactor, nomeArquivo = captureImage(1)

cv2.destroyAllWindows()
