from math import atan, degrees
from tkinter import filedialog as dlg, messagebox as mbox
from tkinter import simpledialog
import tkinter as tk
import cv2
import numpy as np
import pandas as pd
import os
import math
import imutils
from openpyxl import Workbook  # pip install openpyxl
from openpyxl import load_workbook
from skimage.measure import label, regionprops_table
from skimage.transform import resize
from skimage.measure import find_contours
from skimage.color import rgb2ycbcr, rgb2hsv
from skimage.morphology import disk
from skimage.filters import median
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path


# ! Fix for NSInvalidArgumentException in NSApplication macOSVersion
# ! No need to use _root for anything at all
# __root = Tk()

# Função destinada ao ajuste da imagem na tela (redimensionador)
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
    return (tLines, tColumns, rFactor)


def subImagens(elementos, img, tColumns, tLines, factor, pixFactor, dFactor, nomeArquivo):

    imgProcess = img.copy()
    proceed = False
    global __path_file_name
    global egg_num
    global egg_folder_fit_plot_path


    results_folder_path = Path(os.getcwd(), 'results') #Retorna se há o diretório necessário
    check_and_create_directory_if_not_exist(results_folder_path) #Cria um novo diretório caso não exista


    while proceed == False:

        if nomeArquivo == '':
            #Pede ao usuário um nome de arquivo para ser salvo, e retorna o caminho para ele
            new_file_name = dlg.asksaveasfilename(confirmoverwrite=False, initialdir=results_folder_path)
            #Atribui o caminho
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

    linhas = []
    colunas = []

    posEggs=findeggs(img)
    nFile = 1
    lMin = cMin = +9999
    lMax = cMax = -9999

    for egg in posEggs:
        (gr, ll, col, lin, larg, alt)=egg

        lIni = int(lin-round(alt*0.075)) if int(lin-round(alt*0.075)) > 0 else 0
        lFin = int(lin+round(alt*1.075))
        cIni = int(col-round(larg*0.075)) if int(col-round(larg*0.075)) > 0 else 0
        cFin = int(col+round(larg * 1.075))

        if (lIni < lMin) : lMin = lIni
        if (cIni < cMin) : cMin = cIni
        if (lFin > lMax) : lMax = lFin
        if (cFin > cMax) : cMax = cFin

        recImage = imgProcess[lIni:lFin, cIni:cFin]


        # Daqui em diante eu acho que dá para ser paralelizado.....
        egg_num = str(nFile)
        resultado = process(recImage, 1, pixFactor)


        imgProc, a, b, c, d, v, area, pAi, pAf, pBi, pBf = resultado


        processedImageName = Path(processed_path, f'{nFile}.png')
        #rotateProcessedImageName = Path(processed_path, f'rotate_{nFile}.png')
        nFile += 1

        cv2.imwrite(str(processedImageName), imgProc)
        #cv2.imwrite(str(rotateProcessedImageName), rotateImage)

        #imgProcess[linhas[x]:linhas[x + 1], colunas[y]:colunas[y + 1]] = imgProc
        imgProcess[lIni:lFin, cIni:cFin] = imgProc

        #entendo que aqui se encerra o trecho paralelizável....

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

    # List of polynomial curv functions to use
    print('List of polynomial curv functions to use')
    poly_degree_functions = [polynomial_curv_3, polynomial_curv_5, polynomial_curv_7, polynomial_curv_9,
                             polynomial_curv_11]

    # Calculate disc slices with curve fit and its errors
    dic_slices_fit, curve_fit_errors = disc_slices_curve_fit(xdata, ydata, poly_degree_functions)


    volume_results = {}

    # Measure egg volume based on trapezoidal rule
    print('Measure egg volume based on trapezoidal rule')
    (oldVolume, oldEggArea) = calcVolume(dicSlicesNoFit, pixFactor)
    volume_results["Antigo"] = (oldVolume, oldEggArea)

    # volume chosen to calculate the pixels at the final of this function
    chosen_volume = 0.0
    chosen_area = 0.0

    for poly_function in dic_slices_fit:
        print(f'Calculando e Escrevendo volume e área da função polinomial : {poly_function.__name__}')
        (poly_volume, poly_eggArea) = calcVolume(dic_slices_fit[poly_function], pixFactor)
        volume_results[poly_function] = (poly_volume, poly_eggArea)
        if poly_function == polynomial_curv_11:
            chosen_volume = poly_volume
            chosen_area = poly_eggArea



    # Finds the intersection of the lines and then draws the two sub-lines (Above and Below) of the perpendicular line
    inter = lineLineIntersection(pt1, pt2, pt3, pt4)
    if inter:
        distance_pt1_inter = np.linalg.norm(pt1 - inter)
        distance_pt2_inter = np.linalg.norm(pt2 - inter)
        if distance_pt1_inter > distance_pt2_inter:
            original = cv2.line(original, (int(pt1[1]), int(pt1[0])), (int(inter[1]), int(inter[0])), (0, 0, 255),
                                thickness=2)
            original = cv2.line(original, (int(pt2[1]), int(pt2[0])), (int(inter[1]), int(inter[0])), (255, 255, 255),
                                thickness=2)
            C = distance_pt1_inter
            D = distance_pt2_inter
        else:
            original = cv2.line(original, (int(pt2[1]), int(pt2[0])), (int(inter[1]), int(inter[0])), (0, 0, 255),
                                thickness=2)
            original = cv2.line(original, (int(pt1[1]), int(pt1[0])), (int(inter[1]), int(inter[0])), (255, 255, 255),
                                thickness=2)
            C = distance_pt2_inter
            D = distance_pt1_inter

    A *= pixFactor
    B *= pixFactor
    C *= pixFactor
    D *= pixFactor

    return [original, A, B, C, D, chosen_volume, chosen_area, pt1, pt2, pt3, pt4]


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

def calcVolume(slices, pixFactor):
    dicSlices = sorted(slices.items())
    volume = 0
    areaLateral = 0
    # print('Inicio Ovo')
    for t in range(1, len(dicSlices)):
        elem1 = dicSlices[t]
        elem0 = dicSlices[t - 1]
        h2, rMaior = elem1
        h1, rMenor = elem0

        # print ("(",h2,",",rMaior,")")

        h = abs(h2 - h1) * pixFactor
        rMaior = (rMaior / 2) * pixFactor
        rMenor = (rMenor / 2) * pixFactor
        g = math.sqrt(h ** 2 + (rMaior - rMenor) ** 2)
        volume += ((math.pi * h) / 3) * (rMaior ** 2 + rMenor ** 2 + rMaior * rMenor)
        areaLateral += math.pi * g * (rMaior + rMenor)
    volume = volume / 100
    return [volume, areaLateral]


def disc_slices_curve_fit(x_data, y_data, polynomial_fit_degree_functions):
    # last_function = polynomial_fit_degree_functions[-1]
    resultDiscSlices = {}
    # resultCurveError = []
    resultCurveError = {}
    for poly_fit_function in polynomial_fit_degree_functions:
        # Criar a curva de ajuste polinomial do grau escolhido na lista
        popt, pcov = curve_fit(poly_fit_function, x_data, y_data)

        # Using 'lm' method for Levenberg-Marquardt algorithm or 'trf' method for Trust Region Reflective algorithm.
        # p0 = [10, 0.1, 1, 10, 0.1, 1, 1]
        # popt, pcov = curve_fit(poly_fit_function, x_data, y_data, p0=p0, method='lm')

        y_data_fit = poly_fit_function(x_data, *popt)

        # plotando o gráfico do ajuste com os os valores de coeficiente otimizados
        plt.cla()
        plt.plot(x_data, y_data, 'b-', label='Sem ajuste (antigo)')
        plt.plot(x_data, y_data_fit, 'r-', label=f'Com ajuste: {poly_fit_function.__name__}')
        plt.suptitle(f'{__path_file_name.name} - Ovo {egg_num}', fontweight='bold')
        plt.title(f'Comparação: Sem ajuste X C/ ajuste ({poly_fit_function.__name__})')
        plt.xlabel("Distância X")
        plt.ylabel("Pontos de distância Y")
        plt.legend()
        # plt.show()



        # erro do ajuste polinomial
        perr = np.sqrt(np.diag(pcov))
        print(f'Error of the curve fit - {poly_fit_function.__name__}: {perr}\n')

        # Convert lists (x_data, y_data_fit) to dictionary and append it to the result list as tuple
        resultDiscSlices[poly_fit_function] = dict(zip(x_data, y_data_fit))
        # resultCurveError.append((poly_fit_function, perr))
        resultCurveError[poly_fit_function] = perr

        # if(poly_fit_function == last_function):
        # using dict() and zip() to convert lists to dictionary
        # return dict(zip(x_data, y_data_fit))
    # print(f'FINAL RESULT : {resultDiscSlices}')
    # return [np.array(resultDiscSlices), resultCurveError]
    return [resultDiscSlices, resultCurveError]


def findeggs(originalImg):
    ovos=[]
    linhas=[]
    # Loading the image
    (alt, larg, ch)=originalImg.shape;
    AreaTotal = alt*larg

    # preprocess the image
    gray_img = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
    # Applying 7x7 Gaussian Blur
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0) #convolui um filtro gaussiano de 7x7
    # Applying threshold
    threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # Loop through each component
    for i in range(1, totalLabels):
        # Area of the component
        area = values[i, cv2.CC_STAT_AREA]
        percArea = (area*100)/AreaTotal
        aspectRatio = float(int(values[i, cv2.CC_STAT_HEIGHT]) / int(values[i, cv2.CC_STAT_WIDTH]))

        if (percArea > 0.4) and (percArea < 1) and (aspectRatio > 1.1) and (aspectRatio < 1.6):
            (col, lin) = centroid[i]
            x = values[i, cv2.CC_STAT_LEFT]
            y = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]
            elem=[0,lin,x,y,w,h]
            ovos.append(elem);

    posVet=0
    grupo = 1;
    #categoriza pela posição na linha, são classificados aquelas regiões cuja posição não linha variam abaixo de 10%
    for i in range(posVet, len(ovos)):
        if (ovos[i][0]==0):
            ovos[i][0]=grupo
            for k in range(i+1, len(ovos)):
                if ((abs(ovos[k][1]-ovos[i][1])*100)/ovos[i][1])<20:
                    ovos[k][0]=grupo
            grupo=grupo+1

    #ordena pela linha
    for i in range(0, len(ovos)-1):
        for j in range(i+1, len(ovos)):
            if ovos[j][0]<ovos[i][0]:
                troca=ovos[j]
                ovos[j]=ovos[i]
                ovos[i]=troca

    #ordena pela coluna
    controle = 1
    for i in range(0, len(ovos)-1):
        for j in range(i+1, len(ovos)):
            if ((ovos[j][0]==ovos[i][0]) and (ovos[j][2] < ovos[i][2])):
                troca = ovos[j]
                ovos[j] = ovos[i]
                ovos[i] = troca

    return ovos


def polynomial_curv_3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def polynomial_curv_5(x, a, b, c, d, e, f):
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f


def polynomial_curv_7(x, a, b, c, d, e, f, g, h):
    return a * x ** 7 + b * x ** 6 + c * x ** 5 + d * x ** 4 + e * x ** 3 + f * x ** 2 + g * x + h


def polynomial_curv_9(x, a, b, c, d, e, f, g, h, i, j):
    return a * x ** 9 + b * x ** 8 + c * x ** 7 + d * x ** 6 + e * x ** 5 + f * x ** 4 + g * x ** 3 + h * x ** 2 + i * x + j


def polynomial_curv_11(x, a, b, c, d, e, f, g, h, i, j, k, l):
    return a * x ** 11 + b * x ** 10 + c * x ** 9 + d * x ** 8 + e * x ** 7 + f * x ** 6 + g * x ** 5 + h * x ** 4 + i * x ** 3 + j * x ** 2 + k * x + l


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x <= x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])


# def sigmoid(x, a, b, c, d):
# y = a / (1 + np.exp(-b*(x-c))) + d
# return y

# Não funcionou : Linha indo para a direita, sem seguir os pontos
def sigmoid(x, a, b, c, d):
    z = b * (x - c)
    z = np.clip(z, -500, 500)  # limit the values of the exponential term
    return a / (1 + np.exp(-z)) + d


# Resultado: Semelhante a regressão polinomial de grau 3
def gaussian(x, a, b, c, d):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d


def get_polynomial_curv_portuguese_name(polynomial_func):
    if polynomial_func == polynomial_curv_3:
        return "Ajuste Polinomial Grau 3"
    elif polynomial_func == polynomial_curv_5:
        return "Ajuste Polinomial Grau 5"
    elif polynomial_func == polynomial_curv_7:
        return "Ajuste Polinomial Grau 7"
    elif polynomial_func == polynomial_curv_9:
        return "Ajuste Polinomial Grau 9"
    elif polynomial_func == polynomial_curv_11:
        return "Ajuste Polinomial Grau 11"
    else:
        return "Ajuste desconhecido"


def exponential_combo(x, a1, b1, c1, a2, b2, c2, d):
    y1 = a1 * np.exp(-b1 * x) + c1
    y2 = a2 * np.exp(-b2 * (x - c2) ** 2) + d
    return y1 + y2


# Função que cria um novo diretório caso não exista préviamente
def check_and_create_directory_if_not_exist(path_directory):
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)
        print(f"The path {path_directory} is created!")