import os
from math import atan, degrees
from pathlib import Path
from tkinter import filedialog as dlg
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2hsv
from skimage.filters import median
from skimage.measure import find_contours
from skimage.measure import label, regionprops_table
from skimage.morphology import disk
from skimage.transform import resize


def adjustImageDimension(im):
    tLines, tColumns, c = im.shape

    if tLines > tColumns:
        rFactor = 1800 / tLines
        tLines = 1800
        tColumns = int(tColumns * rFactor)
    else:
        rFactor = 1800 / tColumns
        tColumns = 1800
        tLines = int(tLines * rFactor)

    return tLines, tColumns, rFactor


def check_and_create_directory_if_not_exist(path_directory):
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)
        print(f"The path {path_directory} is created!")


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


# -----------------------------------------------------------------------------------------------------------------------

def subImagens(img, tColumns, tLines, pixFactor, nomeArquivo):
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

        if (lIni < lMin): lMin = lIni
        if (cIni < cMin): cMin = cIni
        if (lFin > lMax): lMax = lFin
        if (cFin > cMax): cMax = cFin

        recImage = imgProcess[lIni:lFin, cIni:cFin]

        # Daqui em diante eu acho que dá para ser paralelizado.....
        egg_num = str(nFile)
        resultado = process(recImage, 1, pixFactor)

        imgProc, a, b, c, d, v, area, pAi, pAf, pBi, pBf = resultado
        # imgProc = resultado

        processedImageName = Path(processed_path, f'{nFile}.png')
        nFile += 1

        cv2.imwrite(str(processedImageName), imgProc)
        imgProcess[lIni:lFin, cIni:cFin] = imgProc

        down_points = (tColumns, tLines)
        dispImage = cv2.resize(imgProcess, down_points, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Window", dispImage)
        cv2.waitKey(1)


def process(frame, factor, pixFactor):
    # Measures that will be returned
    A = 0
    B = 0
    C = 0
    D = 0

    # Create a copy to save the image at the original resolution.
    original = frame.copy()
    # original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Reduce the image according to the factor to optimize the processing time
    [lin, col, ch] = original.shape
    frame = resize(frame, [int(lin / factor), int(col / factor)])
    frame = np.uint8(frame * 255)

    data = rgb2hsv(frame)

    # % Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.000
    channel1Max = 1.000

    # % Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.000
    channel2Max = 1.000

    # % Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.578
    channel3Max = 1.000

    # Creating the mask for segmentation
    print('Creating the mask for segmentation')
    data = np.bitwise_and(np.bitwise_and(np.bitwise_and(data[:, :, 0] >= channel1Min, data[:, :, 0] <= channel1Max),
                                         np.bitwise_and(data[:, :, 1] >= channel2Min, data[:, :, 1] <= channel2Max)),
                          np.bitwise_and(data[:, :, 2] >= channel3Min, data[:, :, 2] <= channel3Max))
    data[data == True] = 1
    data[data == False] = 0

    data = median(data, disk(3))
    data[data > 0] = 1

    # Process to identify the bounding box that contains the egg and subsequently improve segmentation
    print('Process to identify the bounding box that contains the egg and subsequently improve segmentation')
    [linoriginal, coloriginal] = data.shape
    labels = label(data)
    props = regionprops_table(labels, properties=('bbox', 'major_axis_length', 'minor_axis_length'))
    df = pd.DataFrame(props)

    fl_find_bb = False
    for index, row in df.iterrows():
        if 0.30 * linoriginal <= row['major_axis_length'] <= 1 * linoriginal and \
                0.15 * coloriginal <= row['minor_axis_length'] <= 1 * coloriginal:
            data = data.astype('uint8')
            data[int(row['bbox-0']):int(row['bbox-2']), int(row['bbox-1']):int(row['bbox-3'])] = \
                data[int(row['bbox-0']):int(row['bbox-2']), int(row['bbox-1']):int(row['bbox-3'])] + 1
            data[data == 1] = 0
            data[data == 2] = 255
            fl_find_bb = True
            break

    if not fl_find_bb:
        return [original, -1, -1, -1, -1]

    # Identifying the two points that form the longest straight line
    print('Identifying the two points that form the longest straight line')
    border_points = np.array(np.vstack(find_contours(data, 0.1)))
    index_size, _ = border_points.shape

    pt1 = pt2 = []
    max_distance = 0
    for i in range(1, index_size):
        for j in range(i + 1, index_size):
            distance = np.linalg.norm(border_points[i] * factor - border_points[j] * factor)
            if distance > max_distance:
                pt1 = border_points[i] * factor
                pt2 = border_points[j] * factor
                max_distance = distance

    # Finds the angle of the line formed by the previous points and, later, finds the longest straight line
    print('Finds the angle of the line formed by the previous points and, later, finds the longest straight line')
    msRmaior = (pt1[1] * factor - pt2[1] * factor) / (pt1[0] * factor - pt2[0] * factor)
    ms = -1 / msRmaior
    ms = degrees(atan(ms))
    max_distance = 0
    pt3 = pt4 = []
    dicSlicesNoFit = {}
    xdata = []
    ydata = []

    for i in range(1, index_size):
        for j in range(1, index_size):
            if border_points[j][0] != border_points[i][0]:
                mdRmenor = (border_points[i][1] * factor - border_points[j][1] * factor) / (
                        border_points[i][0] * factor - border_points[j][0] * factor)
                bRmenor = border_points[i][1] - mdRmenor * border_points[i][0]  # defined constant of straight equation

                md = degrees(atan(mdRmenor))
                if 0 <= abs(md - ms) <= 0.3:
                    distance = np.linalg.norm(border_points[i] * factor - border_points[j] * factor)
                    distP1 = abs(-mdRmenor * pt1[0] + pt1[1] - bRmenor) / ((mdRmenor ** 2 + 1) ** 0.5)

                    original = cv2.line(original, (int(border_points[i][1]), int(border_points[i][0])),
                                        (int(border_points[j][1]), int(border_points[j][0])), (30, 105, 210),
                                        thickness=1)

                    dicSlicesNoFit[distP1] = distance
                    # !AQUI - Criar lista de X e Y para fazer o ajuste polinomial
                    xdata.append(distP1)
                    ydata.append(distance)

                    if distance > max_distance:
                        pt3 = border_points[i] * factor
                        pt4 = border_points[j] * factor

                        max_distance = distance

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    # ----------------------------
    chosen_volume = 0.0
    chosen_area = 0.0
    # ----------------------------

    # Finds the intersection of the lines and then draws the two sub-lines (Above and Below) of the perpendicular line
    inter = lineLineIntersection(pt1, pt2, pt3, pt4)
    if inter:
        distance_pt1_inter = np.linalg.norm(pt1 - inter)
        distance_pt2_inter = np.linalg.norm(pt2 - inter)

    A *= pixFactor
    B *= pixFactor
    C *= pixFactor
    D *= pixFactor

    return [original, A, B, C, D, chosen_volume, chosen_area, pt1, pt2, pt3, pt4]
    # return [original]


# Function to return the intersection between two lines
def lineLineIntersection(A, B, C, D):
    # Line 01
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1 * (A[0]) + b1 * (A[1])

    # Line 02
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2 * (C[0]) + b2 * (C[1])

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return False
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return [int(x), int(y)]


