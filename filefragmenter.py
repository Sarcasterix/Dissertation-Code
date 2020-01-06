import os
import numpy as np
from datetime import datetime
from keras.utils import generic_utils


def findFiles(dirPath, fileExt):
    """
    Returns a list of all files of given extension within a directory path.
    This function only works with an input directory containing at least one
    sub-directory which contains the files. (i.e. inputDir -> dir1, dir2...
    dir1 -> file1, file2, file3...)
    """
    # Search our given directory
    directoryList = os.listdir(dirPath)
    # Our list of filepaths needs initialised
    filePathList = []
    # For each directory we find
    for subdir in directoryList:
        # Grab all the filenames
        fileList = os.listdir(dirPath+subdir)
        # For each filename
        for fileName in fileList:
            # If it's of the type we're looking for
            if fileName.endswith(fileExt):
                # Add it to our list of paths
                filePathList.append(dirPath+subdir+"\\"+fileName)
    # We're done, return all our filepaths
    return filePathList


def countFragmentsFile(filePath, fragmentSize):
    '''
    Returns the number of fragments of given size the file given will create,
    minus first and last. This function is a prettified piece of code, to make
    things more readable. filepath/fragmentsize gives us a decimal, so we round
    up (to count the last partial fragment), then subtract 2 (to remove the
    first and last fragments)
    '''
    total = int(np.ceil(os.path.getsize(filePath)/fragmentSize)) - 2
    return total


def buildList(path, classes, fragmentSize, fragmentsPerClass):
    # This list is going to hold all our possible file fragments, in the form
    # of: [class index][file path, offset of fragment]
    fragOffsetList = []
    # Lets get working.
    # For index, filetype within our class list
    print("\nLocating files at",
          datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for i, c in enumerate(classes):
        # Add a blank row to our offsets list
        fragOffsetList.append([])
        # Our holder variable now has a list of all files of the current type
        files = findFiles(path, c)
        # For each of these files
        for f in files:
            # For every acceptable (not first or last) fragment in that file
            for j in range(countFragmentsFile(f, fragmentSize)):
                # Add to our list of acceptable offsets, at index depth (see
                # why we added that blank row earlier?), the filepath we're
                # looking at, as well as the offset of the acceptable fragment
                fragOffsetList[i].append([f, (fragmentSize+(fragmentSize*j))])
        # We've got all possible fragments of the filetype, so shuffle the list
        np.random.shuffle(fragOffsetList[i])
        # Print the total number of acceptable fragments, for dataset creation
        print("Number of", c, "fragments found:", len(fragOffsetList[i]))
        # Throw away everything after what we need.
        del(fragOffsetList[i][fragmentsPerClass:])
    return fragOffsetList


def getFragment(filePath, fragmentSize, offset):
    '''
    Returns a fragment of size fragmentSize, starting offset bytes into file
    found at filePath location.
    '''
    # Open up our supplied file.
    file = open(filePath, "rb")
    # Jump in to our offset
    file.seek(offset)
    # Read our fragment, cast it to a numpy array, and scale to decimal
    bytes = np.frombuffer(file.read(fragmentSize),
                          dtype=np.uint8) * np.float32(1/256)
    # Close the file
    file.close()
    # Return our data
    return bytes


def readData(path, classes, fragmentSize, fragmentsPerClass, outputFileName):
    '''
    This method does our gruntwork, tying together all previous methods.
    Returns two arrays of equal length, one two-dimensional of form
    [fragment][bytes of fragment], the other one-dimensional of form
    [fragment class label]
    '''
    # This is how many classes we have
    numClasses = len(classes)
    # This is the total number of fragments we will return
    totalDataPoints = numClasses * fragmentsPerClass

    # This array will hold our final fragments, eventually to be returned. It
    # needs initialised, because it's a big boy array.
    fragmentsArray = np.zeros([totalDataPoints, fragmentSize],
                              dtype=np.float32)
    # This will hold the one-hot encoding for out fragments
    labelsArray = np.zeros([totalDataPoints], dtype=int)

    # Call the buildlist method to create the listing of fragments to read
    fragOffsetList = buildList(path, classes, fragmentSize, fragmentsPerClass)

    print("\nStarting fragment reads at ",
          datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # Keras's progress bar gives terminal output
    progbar = generic_utils.Progbar(totalDataPoints)
    # For each fragment per class
    for i in range(fragmentsPerClass):
        # For the index of each class, which will be used as its tag
        for j in range(numClasses):
            # This variable is our offset within the dataset
            point = j+(i*numClasses)
            # Set our label data
            labelsArray[point] = j
            # Set our fragment, by pulling from the file at specified offset
            fragmentsArray[point] = getFragment(fragOffsetList[j][i][0],
                                                fragmentSize,
                                                fragOffsetList[j][i][1])
            # Update progress bar
            progbar.update(point)
    # Final update to finish the bar
    progbar.update(totalDataPoints)
    # print out our finishing time
    print("\nFinished fragment reads at ",
          datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # Print both of the arrays, for observation.
    print(fragmentsArray)
    print(labelsArray)
    # Save the arrays for usage later.
    np.savez(outputFileName, fragments=fragmentsArray, labels=labelsArray)
    # Return the arrays.
    return (fragmentsArray, labelsArray)


def testRead():
    '''
    This function is for testing of the file fragmentation script. It can also
    be used to create a single dataset, separately to an external call.
    '''
    print("\nStarting at ",
          datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # This is where we're searching
    directory = "D:\\Uni\\Dissertation\\Corpus\\Test\\"
    # This is how big our fragments will be
    fragmentSize = 2048
    # This is how many fragments we want to get
    fragmentGoal = 1000
    # These are the filetypes we're searching for
    classes = [".pdf", ".csv", ".doc", ".gif",
               ".html", ".jpg", ".ppt", ".txt", ".xls", ".xml"]
    # This file will hold our output
    outputFile = "maxTestData.npz"

    print("Searching directory", directory)
    print("For files within the classes", classes)
    print("Taking fragments of size", fragmentSize)
    print("Getting", fragmentGoal, "fragments per class")
    print("Saving dataset to", outputFile)

    #Call to read, returning data
    xArray, yArray = readData(directory, classes, fragmentSize,
                              fragmentGoal, outputFile)

    print(xArray)
    print(yArray)
    print("\nFinished at ",
          datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# Run testRead upon script load
# testRead()
