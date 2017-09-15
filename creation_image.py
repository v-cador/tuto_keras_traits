import tensorflow as tf
import pandas as pd
import numpy as np
import pylab as plt
import math
import cv2





# creation des images
def creation_images(taille, debut_min):
    mat = np.zeros((taille, taille))
    x, y = np.random.randint(debut_min, taille - debut_min, 2)
    sens = int(np.random.randint(0, 2, 1))
    hv = int(np.random.randint(0, 2, 1))
    if hv == 1:  # vertical
        vertical, horizontal = 1, 0
        longueur = np.random.randint(3, math.floor((debut_min + 1) / 2))
        largeur = np.random.randint(1, longueur)
        longueur = longueur * 2
    elif hv == 0:  # horizontal
        vertical, horizontal = 0, 1
        largeur = np.random.randint(3, math.floor((debut_min + 1) / 2))
        longueur = np.random.randint(1, largeur)
        largeur = largeur * 2
    if sens == 1:  # vers droite ou bas
        mat[x:(x + longueur), y:(y + largeur)] = 1
        # mat[x:(x + longueur), y:(y + largeur)] = 1
    elif sens == 0:  # vers gauche ou haut
        mat[(x - longueur):x, (y - largeur):y] = 1
    return {"image" : mat, "hv" : (horizontal, vertical)}



# TRAIN
nb_img = 1000
for i in range(nb_img):
    Z = creation_images(50, 9)
    img = Z["image"]
    hv = Z['hv']
    if hv==(1, 0):
        filename = chemin_dossier + "train/horizontal/image_" + str(i) +'.png'
        cv2.imwrite(filename, img*255)
    elif hv==(0, 1):
        filename = chemin_dossier + "train/vertical/image_" + str(i) + '.png'
        cv2.imwrite(filename, img * 255)

# TEST
nb_img = 500
for i in range(nb_img):
    Z = creation_images(50, 9)
    img = Z["image"]
    hv = Z['hv']
    if hv==(1, 0):
        filename = chemin_dossier + "test/horizontal/image_" + str(i) +'.png'
        cv2.imwrite(filename, img*255)
    elif hv==(0, 1):
        filename = chemin_dossier + "test/vertical/image_" + str(i) + '.png'
        cv2.imwrite(filename, img * 255)

# VALID
nb_img = 5000
for i in range(nb_img):
    Z = creation_images(50, 9)
    img = Z["image"]
    hv = Z['hv']
    if hv==(1, 0):
        filename = chemin_dossier + "valid/horizontal/image_" + str(i) +'.png'
        cv2.imwrite(filename, img*255)
    elif hv==(0, 1):
        filename = chemin_dossier + "valid/vertical/image_" + str(i) + '.png'
        cv2.imwrite(filename, img * 255)




