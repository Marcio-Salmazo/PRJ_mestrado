from tkinter import *
from tkinter import filedialog as dlg, messagebox as mbox
import cv2
import Processamento

# globals
elementLines = []
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
    global fullImage
    fileName = ''
    if source == 1:
        fileName = dlg.askopenfilename()
        if fileName != '':
            print(fileName)
            fullImage = cv2.imread(fileName)
    elif source == 2:
        fullImage = cv2.imread('startMenu.png')

    return fullImage, fileName


# ----------------------------------------------------------------------------------------------------------------------
#                                                INICIO DO PROGRAMA
# ----------------------------------------------------------------------------------------------------------------------

fullImage, nomeArquivo = captureImage(2)

while True:
    image = fullImage.copy()
    cv2.imshow("Startup Menu", image)
    k = cv2.waitKey(1)

    if k == 113 or k == 81:  # 'q' pressed to quit the system
        break

    if k == 112 or k == 80:  # 'p' pressed to activate the image processing

        pixFactor = 0.25041736227045075125208681135225
        print("Nome Arquivo antes da chamada de processamento ==> ", nomeArquivo)
        Processamento.subImagens(fullImage, nomeArquivo)
        fullImage, nomeArquivo = captureImage(1)

    if k == 102 or k == 70:  # 'f' to read image from file
        fullImage, nomeArquivo = captureImage(1)

cv2.destroyAllWindows()
