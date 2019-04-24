
import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, metrics, model_selection

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature



dataset_idx_st=0
dataset_idx_end = 250
box_dim = [-32, 32, -32, 32]
np.random.seed(63986)
plotting=False

def check_overlap(box1, box2):
    if (box1[0] > box2[1]) or (box2[0] > box1[1]) or (box1[2] > box2[3]) or (box2[2] > box1[3]):
        return False
    else:
        return True


def process():

    X = np.array([])
    y = []
    # For each image

    for i in range(dataset_idx_st, dataset_idx_end + 1):
        image_string = '{:08d}'.format(i)
        image_file = "Vehicules1024/" + image_string + "_ir.png"
        annotation_file = "Annotation/" + image_string + ".txt"
        # Make sure both the image file and its annotations exist
        if os.path.isfile(image_file) and os.path.isfile(annotation_file):
            # print("Processing ", image_string)
            # Read the annotations file and eliminate any vehicles too close to the edge
            selected_v = np.array([])
            with open(annotation_file) as f:
                readcsv = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
                for r in readcsv:
                    corners = np.round(np.column_stack((r[0], r[0], r[1], r[1])) + box_dim)
                    if corners.min() >= 0 and corners.max() <= 1023:
                        # print("Object in bounds ", corners)
                        if selected_v.shape[0] == 0:
                            # print("First object")
                            selected_v = corners
                        else:
                            selected_v = np.append(selected_v, corners, axis=0)
                    # else:
                    # print("Object out of bounds - ", corners)
            # print("Found ", selected_v.shape[0], " in-bounds vehicles")

            # for each in-bounds vehicle, grab a background image
            bg_chip = np.array([])
            for v in range(0, 2 * selected_v.shape[0]):
                clean_bg = False
                while (not clean_bg):
                    corner = np.random.uniform(65, 900, 2)
                    corners = np.round(np.column_stack((corner[0], corner[0], corner[1], corner[1])) + box_dim)
                    is_overlap = False
                    for vv in range(0, selected_v.shape[0]):
                        if check_overlap(selected_v[vv, :], corners[0, :]):
                            is_overlap = True
                    if (not is_overlap):
                        clean_bg = True
                if bg_chip.shape[0] == 0:
                    bg_chip = corners
                else:
                    bg_chip = np.append(bg_chip, corners, axis=0)
            # print("Created ", bg_chip.shape[0], " background patches")

            # Eliminate any overlapping vehicles
            overlap_idx = []
            if selected_v.shape[0] > 1:
                for v in range(0, selected_v.shape[0]):
                    other_v = np.delete(selected_v, v, axis=0)  # remove this one
                    v_overlap = False
                    for vo in range(0, other_v.shape[0]):
                        if check_overlap(selected_v[v, :], other_v[vo, :]):
                            v_overlap = True
                    if v_overlap:
                        overlap_idx.append(v)
            selected_v_post_overlap = np.delete(selected_v, overlap_idx, axis=0)
            # print("Found ", selected_v_post_overlap.shape[0], " non-overlapping vehicles")

            # Read the image
            img = cv2.imread(image_file,0)
            #img = img[:, :, 0];  # only a single channel for IR
            if plotting:
                plt.imshow(img, cmap='gray');
                plt.title(image_string)
                plt.show()

            # Display each of the selected vehicles
            if selected_v_post_overlap.shape[0] > 0:
                for v in range(0, selected_v_post_overlap.shape[0]):
                    these_corners = selected_v_post_overlap[v, :].astype(int)
                    patch = img[these_corners[2]:these_corners[3], these_corners[0]:these_corners[1]];

                    if X.shape[0] == 0:
                        X = np.zeros((64, 64, 1))
                        X[:, :, 0] = patch
                    else:
                        X = np.dstack((X, patch))
                    y.append(1)

                    for k in range(0, 2):
                        these_corners = bg_chip[(k * selected_v_post_overlap.shape[0] + v), :].astype(int)
                        bg_patch = img[these_corners[2]:these_corners[3], these_corners[0]:these_corners[1]];

                        X = np.dstack((X, bg_patch))
                        y.append(0)

                    if plotting:
                        plt.subplot(2, selected_v_post_overlap.shape[0], v + 1)
                        plt.imshow(patch, cmap='gray');
                        plt.axis('off')
                        plt.subplot(2, selected_v_post_overlap.shape[0], selected_v_post_overlap.shape[0] + v + 1)
                        plt.imshow(patch, cmap='gray');
                        plt.axis('off')

                if plotting:
                    plt.show()
            # else:
            # print("No selected vehicles")
            # print("\n")
        else:
            print("Image ", image_file, " and/or annotation ", annotation_file, "does not exist")

    #Format the resulting data and write it to a file
    print("X shape=",X.shape)
    X = np.transpose(X,(2, 0, 1))
    y = np.asarray(y)
    print("X shape=",X.shape," y shape=",y.shape)

    X_flat = X.reshape((len(X), -1))


    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_flat, y, test_size=0.2, random_state=23)
    #print(X.reshape(X.shape[0], X.shape[1], X.shape[2], 1))
    x_train1, x_test1, y_train1, y_test1 = model_selection.train_test_split(X.reshape(X.shape[0], X.shape[1], X.shape[2], 1), y, test_size=0.2, random_state=23)
    return  [[X_train, X_test, y_train, y_test],[x_train1, x_test1, y_train1, y_test1]]



#c1, c2 = process()[0],process()[1]
def SVM(X_train, X_test, y_train, y_test):

    clf = svm.SVC(kernel='rbf', gamma=0.001)
    clf.fit(X_train,y_train)

    #Check performance
    y_pred = clf.predict(X_test)
    #print(y_pred)
    average_precision = average_precision_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)


    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))

    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))

    plt.show()

    '''plt.scatter(y_test, y_pred, label="SVM")
plt.legend()
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted ")
plt.show()
#print(len(X_test), len(y_test), len(y_pred))
plt.scatter(range(len(y_test)),y_test, color='yellow', label='Actual')  # plotting the initial datapoints
plt.scatter(range(len(y_test)),y_pred, color='red', label='Predicted ')  # plotting the line made by linear regression
#plt.scatter(y_test, y_pred,label = "SVM")
plt.legend()
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted ")
plt.show()'''


    print("SVM Accuracy score:\n", metrics.accuracy_score(y_test,y_pred))




# Same thing, with a grid-searching cross validation
#parameters = [
#    {'kernel':['rbf'], 'gamma':np.logspace(-6, -3, 20)}
#]
def GridSearch(X_train, X_test, y_train, y_test):
    parameters = [
        {'kernel':['rbf'], 'gamma':np.logspace(-8, -4, 20)}
        #{'kernel':['linear'], 'C':np.logspace(-5, 4, 20)}
    ]
    clf=svm.SVC()
    cv_clf=model_selection.GridSearchCV(estimator=clf, param_grid=parameters, cv=5, n_jobs=-1);
    cv_clf.fit(X_train,y_train)
    print("Best estimator:\n", cv_clf.best_estimator_)

    #Check performance
    y_pred = cv_clf.predict(X_test)

    print("Grid Search Accuracy score:\n", metrics.accuracy_score(y_test,y_pred))
    '''plt.scatter(y_test, y_pred, label="Grid Search")
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted ")'''
    average_precision = average_precision_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)


    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve'.format(
              average_precision))

    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))


    plt.show()



def seq(x_train, x_test, y_train, y_test):


    #input_shape = (X_train.shape[0], X_train.shape[1], 1)
    #print(x_train.shape[0])
    #print(x_train.shape[1])
    #print(x_train.shape[2])
    #print(X_train.shape)

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])


    model = Sequential()
    model.add(Conv2D(32, kernel_size=(4,4), input_shape=(64,64,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #print('x_train shape:', x_train.shape)
    #print('y_train shape:', y_train.shape)
    model.fit(x=x_train,y=y_train, epochs=10)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Sequential Classifier Test loss:', score[0])
    print('Sequential Classifier Test accuracy:', score[1])

    y_hat = model.predict(x_test)
    #print('y hat shape:', y_hat.shape)
    #print('y test shape:', y_test.shape)
    print(y_hat)
    print(y_test)

    y_hat_binary = [np.argmax(i) for i in y_hat]

    average_precision = average_precision_score(y_test, y_hat_binary)
    precision, recall, _ = precision_recall_curve(y_test, y_hat_binary)


    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve'.format(
              average_precision))

    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))


    '''for i in range(0, len(y_test)):
        if np.argmax(y_hat[i,:]) != y_test[i]:
            print("Guessed ", np.argmax(y_hat[i,:]), " was ", y_test[i], " probs", y_hat[i,:])
            plt.imshow(np.squeeze(x_test[i,:,:,:]), cmap='gray');
            plt.axis('off')
            plt.show()'''

#seq(x_train, x_test, y_train, y_test)