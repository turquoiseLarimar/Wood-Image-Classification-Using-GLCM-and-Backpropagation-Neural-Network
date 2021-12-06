import numpy as np
import cv2 as cv
import glob
import pandas as pd

#---------------pre-processing------------------#
def pre_pro(dt_citra):
    img = cv.imread(dt_citra)
    # konversi ukuran citra menjadi 512x512 px
    pjg = 128
    lbr = 128
    resz = cv.resize(img,(pjg,lbr))
    # konversi citra ke grayscale
    gray = cv.cvtColor(resz, cv.COLOR_BGR2GRAY)
    return gray

def glcm (img, degree):
    img = pre_pro(img)
    arr = np.array(img)
    co_oc = np.zeros((256, 256), dtype = float)
    width, height = arr.shape
    # print (" ukuran : ", height," ", width)
    if degree == 0:
        for i in range (height):
            for j in range (width-1):
                co_oc[arr[i,j], arr[i,j+1]] +=1
    elif degree == 45:
        for i in range (height-1):
            for j in range (width-1):
                co_oc[arr[i+1,j], arr[i,j+1]] +=1
    elif degree == 90:
        for i in range (height-1):
            for j in range (width):
                co_oc[arr[i,j], arr[i+1,j]] +=1
    elif degree == 135:
        for i in range (height-1):
            for j in range (width-1):
                co_oc[arr[i+1,j+1], arr[i,j]] +=1
    
    tr_co = co_oc.transpose()
    simetris = co_oc + tr_co
    jm_sim = simetris.sum()
    normali = np.zeros((256, 256), dtype = float)
    w, h = normali.shape
    for i in range (h):
        for j in range (w):
            normali[i,j] = simetris[i,j]/jm_sim
    return normali
# 1. perhitungan tekstur contrast
def contrast(matrix):
    width, height = matrix.shape
    res = 0
    for i in range (width):
        for j in range (height):
            res += matrix[i][j]*np.power(i-j,2)
    return res

# 3. perhitungan tekstur homogeneity
def homogeneity(matrix):
    width, height = matrix.shape
    res = 0
    for i in range (width):
        for j in range (height):
            res += matrix [i][j]/(1+np.power(i-j,2))
    return res

# 4. perhitungan tekstur energy
def energy(matrix):
    width, height = matrix.shape
    res = 0
    for i in range (width):
        for j in range (height):
            res += np.power(matrix[i][j],2)
    res = np.sqrt(res)
    return res

# 5. perhitungan tekstur entropy
def entropy(matrix):
    width, height = matrix.shape
    res = 0
    for i in range (width):
        for j in range (height):
            res += (-matrix[i][j])*(np.log1p(matrix[i][j])) 
    return res


def ekstraksi (citra):
    m_sudut_0 = glcm(citra, 0)
    m_sudut_45 = glcm(citra, 45)
    m_sudut_90 = glcm(citra, 90)
    m_sudut_135 = glcm(citra, 135)
    
    # #menghitung rata-rata tiap fitur
    # kontras = np.average([contrast(m_sudut_0), contrast(m_sudut_45), contrast(m_sudut_90), contrast(m_sudut_135)])
    # homogen = np.average([homogeneity(m_sudut_0), homogeneity(m_sudut_45), homogeneity(m_sudut_90), homogeneity(m_sudut_135)])
    # energi = np.average([energy(m_sudut_0), energy(m_sudut_45), energy(m_sudut_90), energy(m_sudut_135)])
    # entropi = np.average([entropy(m_sudut_0), entropy(m_sudut_45), entropy(m_sudut_90), entropy(m_sudut_135)])
    
    # #membulatkan nilai fitur
    # kontras = round(kontras, 4)
    # homogen = round(homogen,4)
    # energi = round(energi,4)
    # entropi = round(entropi,4)
    
    fitur = np.array([contrast(m_sudut_0), contrast(m_sudut_45), contrast(m_sudut_90), contrast(m_sudut_135), 
                      homogeneity(m_sudut_0), homogeneity(m_sudut_45), homogeneity(m_sudut_90), homogeneity(m_sudut_135),
                      energy(m_sudut_0), energy(m_sudut_45), energy(m_sudut_90), energy(m_sudut_135),
                      entropy(m_sudut_0), entropy(m_sudut_45), entropy(m_sudut_90), entropy(m_sudut_135),
                      '6'])
    return fitur

def training(datasets):
    fitur_datasets = []
    i = 0
    for data in datasets:
        fitur= ekstraksi(data)
        fitur_datasets.append(fitur)
        print("data : ",data," selesai")        
        i+=1
    X = np.vstack(fitur_datasets)
    return X

datasets = glob.glob("../kayu/dataset/data_train/mahoni/*.jpg")
coba= training(datasets)

# print ("manual : ", coba)

df = pd.DataFrame(coba, columns= ['kontras 0', 'kontras 45', 'kontras 90', 'kontras 135', 
                                  'homogen 0', 'homogen 45', 'homogen 90','homogen 135',
                                  'energi 0','energi 45','energi 90','energi 135',
                                  'entropi 0', 'entropi 45', 'entropi 90', 'entropi 135','kelas'])
df.to_csv (r'ma.csv', index = False, header=True)
# print ("Hasil :", df)
