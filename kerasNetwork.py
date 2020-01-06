
import filefragmenter
import os
# import pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from datetime import datetime
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
# This sets keras to a lower level of logging, to stop the continual messages
# regarding environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def runNetwork(fragmentSize, classes, trainFragments, trainLabels,
               testFragments, testLabels):
    """
    This is used to set up, train and test the neural network, as well as
    producing the output images. It returns the confusion matrix produced, to
    be used in the external function.
    """
    # First lets one-hot encode our labels
    oneHotTrainLabels = to_categorical(trainLabels)
    oneHotTestLabels = to_categorical(testLabels)

    # Clear any backend sessions to try to save memory
    keras.backend.clear_session()

    # Lets build our Keras model, which is the actual network. We're using a
    # Keras Sequential model
    model = keras.Sequential()

    # Now we add some layers to our network.
    # We only need to add input shape for the first layer, Keras works it out
    # for every other layer. The reshaping layer is to present the data to the
    # convolution layer as expected
    model.add(keras.layers.Reshape((fragmentSize, 1),
                                   input_shape=(fragmentSize,)))

    # First convolutional layer. It will output fragmentsize/8 nodes,
    # effectively running that many filters over the data at once, using a
    # filter of size 8, and stepping 8 at a time. 'Same' padding is used to
    # preserve the edges of data
    model.add(keras.layers.Conv1D(int(fragmentSize / 8), 8, strides=8,
                                  padding='same', activation=tf.nn.relu))
    # A dense layer (fully connected), to give further interpretation over the
    # data output by the first convolution. This pattern will be repeated for
    # the next layers
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))

    # Now our number of filters halves, as does our filter size, but we are no
    # stepping, so the filters will take in all granular information
    model.add(keras.layers.Conv1D(64, 4, padding='same',
                                  activation=tf.nn.relu))
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))

    # As above, shrink the outputs, shrink the filter size
    model.add(keras.layers.Conv1D(32, 2, padding='same',
                                  activation=tf.nn.relu))
    model.add(keras.layers.Dense(32, activation=tf.nn.relu))

    # Our final layer is the output, and uses softMax to "choose" a single
    # value as its classification of the fragment. The flatten layer is to
    # return to single depth data for classification
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(len(classes), activation='softmax'))

    # Compile our model, setting opimisation, loss function and metrics
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # This prints the model to consol, so that we can see what's going on.
    print(model.summary())

    # This runs training against the network
    model.fit(trainFragments, oneHotTrainLabels, batch_size=256,
              epochs=15, validation_split=0.1)

    # This tests it against unseen data for final accuracy
    testLoss, testAcc = model.evaluate(testFragments, oneHotTestLabels)
    print('Test accuracy:', testAcc)

    # This generates the model's "feeling" about the testing data, giving more
    # information regarding its conclusions
    predictions = model.predict(testFragments)

    # SKLearn's confusion matrix over this data will allow us to calculate our
    # graphs
    cm = confusion_matrix(testLabels, predictions.argmax(axis=1))
    print(cm)

    # Set up our plot
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    fig.colorbar(im)

    # Build the labels on x and y axes
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    # Tilt the labels on the bottom, for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # This prints the data the image has been built from on top of the grid
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="w")
    fig.tight_layout()
    # Save the image to disk for later use and profit.
    plt.savefig("10k" + str(fragmentSize) + ".png")
    # plt.show()

    # Again, try to clear our model from memory, to avoid errors
    del model
    # Return the confusion matrix, for processing and profit.
    return cm


def buildDataSet(classes, fragmentSize, path, filesPerClass, writeFile):

    # this is the directory where our files are stored
    print("Path:", path)
    # fragments holds the data for presentation to the network, classes
    # holds the array of class tags. Both are striped by class type, such that
    # slices will still behave well.
    fragments, labels = filefragmenter.readData(path, classes, fragmentSize,
                                                filesPerClass, writeFile)
    return fragments, labels


def classifier():
    """
    This runs our neural network, building a dataset if necessary. FragmentSize
    defaults to the largest used, to allow the built data to be used for
    smaller runs, saving time.
    """
    print("Current Tensorflow version:", tf.__version__)
    defaultFragmentSize = 2048
    # This is an array of class extensions, which will be used to define what
    # the network is classifying
    classes = [".pdf", ".csv", ".doc", ".gif", ".html",
               ".jpg", ".ppt", ".txt", ".xls", ".xml"]
    print("Classes are:", classes)

    print("Some input is required to set things up. If you wish to use "+
          "default settings, please press enter at each prompt")
    build = bool(input("Would you like to build Train/Test datasets?"))
    if build:
        fragmentSizeIn = input("What size of fragment are we taking?\n")
        if not fragmentSizeIn:
            fragmentSize = defaultFragmentSize
        else:
            fragmentSize = int(fragmentSizeIn)
        # This is the number of fragments sought per class to be included in
        # the training and testing sets
        trainFilesPerClassIn = input("How many training fragments/class?\n")
        if not trainFilesPerClassIn:
            trainFilesPerClass = 150000
        else:
            trainFilesPerClass = int(trainFilesPerClassIn)
        testFilesPerClassIn = input("How many testing fragments/class?\n")
        if not testFilesPerClassIn:
            testFilesPerClass = 1000
        else:
            testFilesPerClass = int(testFilesPerClassIn)
        # These variables hold the location of training and testing data
        trainingPath = input("Please enter the training corpus path\n")
        if not trainingPath:
            trainingPath = "..\\Corpus\\Train\\"
        testingPath = input("Please enter the testing corpus path\n")
        if not testingPath:
            testingPath = "..\\Corpus\\Test\\"

        # This will be the location our datasets will be saved to. Files are
        # organised as two arrays, fragments and labels.
        trainOutputFile = input("Please enter a training file output name\n")
        if not trainOutputFile:
            trainOutputFile = "TrainData.npz"
        testOutputFile = input("Please enter a testing file output name\n")
        if not testOutputFile:
            testOutputFile = "TestData.npz"

        print("\nCreating datasets of size:", fragmentSize,
              "\nTaking", trainFilesPerClass, "training fragments per class",
              "\nTaking", testFilesPerClass, "testing fragments per class",
              "\nTraining corpus path:", trainingPath,
              "\nTesting corpus path:", testingPath,
              "\nSaving training dataset to:", trainOutputFile,
              "\nSaving testing dataset to:", testOutputFile)

        # Now that's out of the way, we'll actually gather our data.
        print("\nCreating Training Data at:",
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        trainFragments, trainLabels = buildDataSet(classes,
                                                   fragmentSize,
                                                   trainingPath,
                                                   trainFilesPerClass,
                                                   trainOutputFile)
        print("\nFinished creating Training Data at:",
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        print("\nCreating Testing Data at:",
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        testFragments, testLabels = buildDataSet(classes,
                                                 fragmentSize,
                                                 testingPath,
                                                 testFilesPerClass,
                                                 testOutputFile)
        print("\nFinished creating Testing Data at:",
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    else:
        trainInputFile = input("Please enter training dataset filename\n")
        if not trainInputFile:
            trainInputFile = "maxTrainData.npz"
        testInputFile = input("Please enter a testing dataset filename\n")
        if not testInputFile:
            testInputFile = "maxTestData.npz"
        print("Started loading training data at",
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'), )
        loadedTrainData = np.load(trainInputFile)
        trainFragments = loadedTrainData['fragments']
        trainLabels = loadedTrainData['labels']
        loadedTrainData.close()
        print("Finished loading training data at",
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        print("Started loading testing data at",
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        loadedTestData = np.load(testInputFile)
        testFragments = loadedTestData['fragments']
        testLabels = loadedTestData['labels']
        loadedTestData.close()
        print("Finished loading testing data at",
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    print("------------------------------------------------" +
          "\nWARNING: FOLLOWING INPUTS MUST BE DIVISIBLE BY 8" +
          "\n------------------------------------------------\n")
    maxSizeIn = input("\nWhat is the maximum fragment size to test?")
    if not maxSizeIn:
        maxSize = len(trainFragments[0])
    else:
        maxSize = int(maxSizeIn)
    minSizeIn = input("\nWhat is the minimum fragment size to test?")
    if not minSizeIn:
        minSize = 32
    else:
        minSize = int(minSizeIn)
    stepIn = input("\nWhat is the step in size for each run?")
    if not stepIn:
        step = 32
    else:
        step = int(stepIn)
    print("\nThis test will begin at size:", minSize, "bytes",
          "\nIncrementing by:", step, "bytes",
          "\nFinishing at:", maxSize, "bytes")
    # Everything before this was our setup. Now lets run a neural net.

    # This will store the output of multiple runs of the network
    confusionMatrixdict = {}
    # For one full sweep of our fragment sizes

    for i in range(int(maxSize/step)):
        # Set this test's size
        fragmentSize = minSize + (i * step)
        print("\nFragment size:", fragmentSize)
        print("\nStarting test at:",
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # Pass everything we set up to the network
        matrix = runNetwork(fragmentSize, classes,
                            trainFragments[:, :fragmentSize],
                            trainLabels,
                            testFragments[:, :fragmentSize], testLabels)
        # Save our data for this test
        confusionMatrixdict[fragmentSize] = matrix
        print("\nFinished test at:",
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # Save the dictionary we've created, alter the number to split our data

    resultsFileName = input("\nPlease enter output filename for these results")
    if not resultsFileName:
        resultsFileName = str(minSize)+"-"+str(maxSize)+" "+str(step)+".npy"

    print(resultsFileName)
    np.save(resultsFileName, confusionMatrixdict)

# Run the network upon script load.
classifier()
