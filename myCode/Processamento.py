import os
from pathlib import Path
from tkinter import filedialog as dlg
import numpy as np
import cv2


def check_and_create_directory_if_not_exist(path_directory):
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)
        print(f"The path {path_directory} is created!")


# ----------------------------------------------------------------------------------------------------------------------

def findeggs(originalImg):
    ovos = []
    (alt, larg, ch) = originalImg.shape
    AreaTotal = alt * larg

    # preprocess the image (GRAY, BLUR, OTSU, C.COMPONENTS)

    gray_img = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    analysis = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # Loop through each component

    for i in range(1, totalLabels):

        area = values[i, cv2.CC_STAT_AREA]
        percArea = (area * 100) / AreaTotal
        aspectRatio = float(int(values[i, cv2.CC_STAT_HEIGHT]) / int(values[i, cv2.CC_STAT_WIDTH]))

        if (percArea > 0.4) and (percArea < 1) and (aspectRatio > 1.1) and (aspectRatio < 1.6):
            (col, lin) = centroid[i]
            x = values[i, cv2.CC_STAT_LEFT]
            y = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]
            elem = [0, lin, x, y, w, h]
            ovos.append(elem)

    posVet = 0
    grupo = 1

    # Arranges the position of eggs according to the grid

    for i in range(posVet, len(ovos)):
        if ovos[i][0] == 0:
            ovos[i][0] = grupo
            for k in range(i + 1, len(ovos)):
                if ((abs(ovos[k][1] - ovos[i][1]) * 100) / ovos[i][1]) < 20:
                    ovos[k][0] = grupo
            grupo = grupo + 1

    # ordena pela linha
    for i in range(0, len(ovos) - 1):
        for j in range(i + 1, len(ovos)):
            if ovos[j][0] < ovos[i][0]:
                troca = ovos[j]
                ovos[j] = ovos[i]
                ovos[i] = troca

    # ordena pela coluna
    controle = 1
    for i in range(0, len(ovos) - 1):
        for j in range(i + 1, len(ovos)):
            if (ovos[j][0] == ovos[i][0]) and (ovos[j][2] < ovos[i][2]):
                troca = ovos[j]
                ovos[j] = ovos[i]
                ovos[i] = troca
    return ovos


# ----------------------------------------------------------------------------------------------------------------------

def subImagens(img, nomeArquivo):
    imgProcess = img.copy()
    proceed = False
    global __path_file_name
    global egg_num
    global egg_folder_fit_plot_path

    results_folder_path = Path(os.getcwd(), 'results')  # Retorna se há o diretório necessário
    check_and_create_directory_if_not_exist(results_folder_path)  # Cria um novo diretório caso não exista

    while proceed == False:

        if nomeArquivo == '':
            # Pede ao usuário um nome de arquivo para ser salvo, e retorna o caminho para ele
            new_file_name = dlg.asksaveasfilename(confirmoverwrite=False, initialdir=results_folder_path)
            # Atribui o caminho
            __path_file_name = Path(new_file_name)
        else:
            __path_file_name = Path(nomeArquivo)

        # Check whether the specified path exists or not
        print('Check whether the specified path exists or not')
        file_path_results = Path(results_folder_path, __path_file_name.stem)
        check_and_create_directory_if_not_exist(file_path_results)

        # processed_path = Path(__path_file_name.parent, 'processed')
        processed_path = Path(file_path_results, 'processed')
        check_and_create_directory_if_not_exist(processed_path)

        proceed = True

    posEggs = findeggs(img)
    nFile = 1
    lMin = cMin = +9999
    lMax = cMax = -9999

    for egg in posEggs:

        (gr, ll, col, lin, larg, alt) = egg

        lIni = int(lin - round(alt * 0.075)) if int(lin - round(alt * 0.075)) > 0 else 0
        lFin = int(lin + round(alt * 1.075))
        cIni = int(col - round(larg * 0.075)) if int(col - round(larg * 0.075)) > 0 else 0
        cFin = int(col + round(larg * 1.075))

        if lIni < lMin: lMin = lIni
        if cIni < cMin: cMin = cIni
        if lFin > lMax: lMax = lFin
        if cFin > cMax: cMax = cFin

        recImage = imgProcess[lIni:lFin, cIni:cFin]
        standardResult = stdImage(recImage)

        egg_num = str(nFile)
        processedImageName = Path(processed_path, f'{nFile}.png')
        nFile += 1

        cv2.imwrite(str(processedImageName), standardResult)


# ----------------------------------------------------------------------------------------------------------------------

def stdImage(image):
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imArray = np.zeros((512, 512), dtype=int)

    # Search for image dimensions
    hImg, wImg, ch = image.shape
    hArr, wArr = imArray.shape

    # Compute xoff and yoff for placement of upper left corner of resized image
    yoff = round((hArr - hImg) / 2)
    xoff = round((wArr - wImg) / 2)

    # Use numpy indexing to place the resized image in the center of background image
    imArray[yoff:yoff + hImg, xoff:xoff + wImg] = imgGray

    imArray = imArray.astype("uint8")
    for i in range(0, wArr):
        for j in range(0, hArr):
            if imArray[i][j] <= 130:
                imArray[i][j] = 0

    final = cv2.medianBlur(imArray, 3)

    # ret, th2 = cv2.threshold(imArray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return final
