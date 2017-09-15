
import os
import cv2
import numpy as np
import random
import pickle
import tensorflow as tf

## RECUPERATION DES IMAGES ET TRANSFORMATION ##



# Fonction de scan des repertoires
# Find a list of pictures jpg in a directory
def list_pictures(directory, ext='.jpg'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if f.endswith(ext)]





# Création des objets X et y pour le train, test et valid :
def create_train_test_valid(dir_global, type_base):

    if type_base not in ["train","test","valid"]:
        raise Exception("Le type de base demandé n'est pas valide, renseignez train, test ou valid")

    # Création des objets et sauvegarde
    print("   >> Create "+type_base+" dataset")

    # Load pictures which have horizontal lines
    X_Horizontal = []
    liste_images_horizontal = list_pictures(dir_global + '/' + type_base + '/horizontal/', ext='png')
    for picture in liste_images_horizontal:
        img = cv2.imread(picture, 0)
        X_Horizontal.append(img)
    X_Horizontal = np.asarray(X_Horizontal)
    print("     --> Found Horizontal pictures : ", len(X_Horizontal))

    # Load pictures which have vertical lines
    X_Vertical = []
    for picture in list_pictures(dir_global + '/' + type_base + '/vertical/', ext='png'):
        img = cv2.imread(picture, 0)
        X_Vertical.append(img)
    X_Vertical = np.asarray(X_Vertical)
    print("     --> Found true pictures : ", len(X_Vertical))

    # Create target feature :  1/0 = Horizontal/Vertical
    y_Horizontal = np.zeros(len(X_Horizontal))
    y_Vertical = np.array([1 for j in range(len(X_Vertical))])

    X_full = np.concatenate((X_Vertical, X_Horizontal), axis=0)
    y_full = np.concatenate((y_Vertical, y_Horizontal), axis=0)

    lindices = range(len(X_full))
    lrand_indices = random.sample(lindices, len(lindices))
    X_full = X_full[lrand_indices]
    y_full = y_full[lrand_indices]

    print("     --> X_full shape:", X_full.shape, "yfull len:", len(y_full))

    # Save datasets
    pickle.dump(X_full, open(dir_global + "X_"+type_base+".obj", "wb"), protocol=2)

    # Save targets
    pickle.dump(y_full, open(dir_global + "y_"+type_base+".obj", "wb"), protocol=2)

    # Normalize inputs from 0-255 to 0.0-1.0
    X_full = X_full.astype('float32')
    X_full = X_full / 255.0

    # Convert target to categorical for Keras
    #y_full = tf.one_hot(y_full)

    # return X_full, y_full
    return X_full, y_full


def load_train_test(dir_obj,type_base):
    print("   >> Load Train/Test/valid dataset")
    # load datasets
    X_full = pickle.load(open(dir_obj + "X_"+type_base+".obj", "rb"))

    # load targets
    y_full = pickle.load(open(dir_obj + "y_"+type_base+".obj", "rb"))

    return X_full, y_full


