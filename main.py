#----------------------------------------------------------------------------------------------------------------------#
#                                                             MAIN                                                     #
#----------------------------------------------------------------------------------------------------------------------#

# Pour executer ce projet, vous devez créer l'environnement "env35" dont le fichier .yml se trouve sur le répertoire
# Pour cela, écrivez dans une console anaconda prompt : "conda env create -n env35 --file Documents\env35.yml"
# Puis, activez le dans votre fenêtre anaconda, ou alors reliez l'environnement à votre projet sous pycharm par exemple

# Chemin de votre répertoire, à modifier :
chemin_dossier = "C:/Users/.../Documents/keras_trait/"

# Executer les scripts dans l'ordre suivant :
# - creation_image.py : vous pouvez aller modifier le nombre d'images qui seront générées aléatoirement
#                       par défaut, il y aura 1000 images pour le train, 500 pour le test et 5000 pour la validation
# - creation_base.py : permet de créer les objets numpy array à partir des images png
# - creation_modele.py : crée le modèle (avec les différentes couches du réseau de neurones)



# 1 - Création des dataset :

# Creation au bon format des donnees d apprentissage
type_base='train'
X_train, Y_train = create_train_test_valid(chemin_dossier, type_base)

# Creation au bon format des donnees de test
type_base='test'
X_test, Y_test = create_train_test_valid(chemin_dossier, type_base)

type_base='valid'
X_val, Y_val = create_train_test_valid(chemin_dossier, type_base)




# Chargement des dataset (si créé auparavant) :

# Creation au bon format des donnees d apprentissage
type_base='train'
X_train, Y_train = load_train_test(chemin_dossier, type_base)

# Creation au bon format des donnees de test
type_base='test'
X_test, Y_test = load_train_test(chemin_dossier, type_base)

# Creation au bon format des donnees de valid
type_base='valid'
X_val, Y_val = load_train_test(chemin_dossier, type_base)


# Modification des dimensions des objets (car lorsqu'il n'y a qu'une couleur, la dernière dimension doit être "1"
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],X_train.shape[2], 1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],X_test.shape[2], 1))
X_val = np.reshape(X_val, (X_val.shape[0],X_val.shape[1],X_val.shape[2], 1))



# Compilation et execution du modèle :
model = compile_sequential_model(num_classes=2, epochs=1)
for epoch in range(10):
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1, batch_size=25, shuffle=False, verbose=1)


# Predict sur la base de validation :
prevision = model.predict(X_val)
prevision = prevision>0.5

# Affichage de la matrice de confusion
print(confusion_matrix(prevision, Y_val))
#array([[2496,   26],
#       [  37, 2441]])


# Calcul du pourcentage d'erreur :
print(str(round(100 * (1-accuracy_score(Y_val, prevision)),2)) + "% d'erreur")
# 1.26% d'erreur

