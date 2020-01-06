import numpy as np
import matplotlib.pyplot as plt
import os


def getStats(matrix):
    #print("F1 Method")
    TP = np.diag(matrix)
    FP = np.sum(matrix, axis=0) - TP
    FN = np.sum(matrix, axis=1) - TP
    Precision = np.nan_to_num(np.divide(TP, (TP+FP)))
    Recall =  np.nan_to_num(np.divide(TP,(TP+FN)))
    F1 = np.nan_to_num(2 * ((Precision*Recall)/(Precision+Recall)))
    #print("TP", TP, "\nFP", FP, "\nFN", FN, "\nPrecision", Precision, "\nRecall", Recall)
    #print("F1",F1)
    return TP, FP, FN, Precision, Recall, F1


def loadFile():
    classes = [".pdf", ".csv", ".doc", ".gif", ".html",
               ".jpg", ".ppt", ".txt", ".xls", ".xml"]
    directory = "Small\\"
    runsList = []

    filesList = os.listdir(directory)

    for i in filesList:
        runsList.append(np.load(directory+i).item())

    TPList = np.zeros([len(runsList),
                       len(classes),
                       len(runsList[0].keys())])

    FPList = np.zeros([len(runsList),
                       len(classes),
                       len(runsList[0].keys())])

    FNList = np.zeros([len(runsList),
                       len(classes),
                       len(runsList[0].keys())])

    PrecisionList = np.zeros([len(runsList),
                              len(classes),
                              len(runsList[0].keys())])

    RecallList = np.zeros([len(runsList),
                           len(classes),
                           len(runsList[0].keys())])

    F1List = np.zeros([len(runsList),
                       len(classes),
                       len(runsList[0].keys())])

    totalCounts = np.zeros([len(runsList),
                            len(runsList[0].keys()),
                            len(classes),
                            len(classes)])

    misClasCounts = np.zeros([len(runsList), len(runsList)])

    for runCounter, run in enumerate(runsList):
        for byteCounter, size in enumerate(sorted(run.keys())):
            TP, FP, FN, Prec, Rec, F1 = getStats(run[size])
            totalCounts[runCounter][byteCounter] = run[size]
            for c in classes:
                index = classes.index(c)
                TPList[runCounter][index][byteCounter] = TP[index]
                FPList[runCounter][index][byteCounter] = FP[index]
                FNList[runCounter][index][byteCounter] = FN[index]
                PrecisionList[runCounter][index][byteCounter] = Prec[index]
                RecallList[runCounter][index][byteCounter] = Rec[index]
                F1List[runCounter][index][byteCounter] = F1[index]

    print(totalCounts[7][30].astype('int'),"\n\n")
    for i in range(len(classes)):
        for run in totalCounts:
            for mat in run:#, axis=0:
                clas = np.argsort(mat[i])
                if clas[-2] == i:
                    misClasCounts[i][clas[-1]] += 1
                else:
                    misClasCounts[i][np.argsort(mat[i])[-2]] += 1

    print(misClasCounts)
    for c, i in enumerate(misClasCounts):
        print(classes[c], "misclassified as", classes[np.argmax(i)])



    xValues = sorted(runsList[0].keys())

    fig, ax = plt.subplots(figsize=(16,9))#figsize=(2560/96, 1440/96))

    #f1 Total
    #print(xValues[np.argmax(np.average(F1List, axis=(0,1)))])
    '''
    plt.title("Network F1 Score: 8-256")
    ax.plot(xValues, np.average(F1List, axis=(0,1)))
    ax.set_ylim(bottom=0, top=1)
    '''

    #f1 class
    '''
    plt.title("F1 Score by Class: 8-256")
    for c in np.average(F1List, axis=0):
        ax.plot(xValues, c)
    ax.legend(classes)
    '''
    '''
    plt.title("Precision by Class: 8-256")
    for c in np.average(PrecisionList, axis=0):
        ax.plot(xValues, c)
    ax.legend(classes)
    '''
    '''
    plt.title("Recall by Class")#: 8-256")
    for c in np.average(RecallList, axis=0):
        ax.plot(xValues, c)
    ax.legend(classes)
    '''
    '''
    plt.title("False Positives by Class")
    #for c in np.average(FPList, axis=0):
    for i in [0,2,5]:
        ax.plot(xValues, np.average(FPList, axis=0)[i], label=classes[i])
    ax.legend(loc='upper right', fontsize="large")
    '''
    '''
    plt.title("False Negatives by Class: 8-256")
    for c in np.average(FNList, axis=0):
        ax.plot(xValues, c)
    ax.legend(classes)
    '''

    #Correct Predictions
    #By Class
    #for i, c in enumerate(np.average(TPList, axis=0)):

    for i in [1,3,4,7,8,9]:
        ax.plot(xValues, np.average(TPList, axis=0)[i]/50 , label=classes[i])
        #data_to_plot = np.take(TPList, axis=1, indices=i) / 50
        #ax.boxplot(data_to_plot, positions = xValues, showfliers=False, widths = 3, whis='range')
    ax.legend(loc='lower right')
    plt.xlabel("Fragment Size (bytes)")
    plt.ylabel("Classification Accuracy")
    plt.title("CSV, GIF, HTML, TXT, XLS, XML Classification Accuracies")

    #ax.legend(map(classes.__getitem__, ))
    '''
    plt.title("Accuracy by Class: 8-256")
    for c in np.average(TPList, axis=0):
        ax.plot(xValues, c/50)
    ax.legend(classes)
    '''
    #By Run

    '''
    plt.title("Network Accuracy by Run: 8-256")
    for r in np.average(TPList, axis=1):
       ax.plot(xValues, r/50)
    '''

    #Plot by Class
    '''
    for i, c in enumerate(classes):
        fig, ax = plt.subplots(figsize=(2560/96, 1440/96))
        avg = np.average(TPList, axis=0) / 50
        data_to_plot = np.take(TPList, axis=1, indices=i) / 50
        max = np.amax(np.average(TPList, axis=0), axis=1)/50
        min = np.amin(np.average(TPList, axis=0), axis=1)/50
        ax.boxplot(data_to_plot, positions = xValues, showfliers=False, widths = 3, whis='range')
        title = c+" Classification Accuracy: 8-256"
        plt.title(title)
        ax.set_ylim(bottom=0, top=100)
        ax.set_xlim(left=0, right=262)
        ax.set_xticks(xValues)
        ax.set_xticklabels(xValues)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        fig.tight_layout()
        plt.savefig(c+" Classification Accuracy 8 256.png")
        #plt.show()
        #ax.plot(xValues, avg)

    '''
    #Total Network Accuracy Boxplot
    #print("Total Accuracy", np.mean(TPList, axis=(0, 1))[30]/50)
    '''
    plt.title("Total Network Accuracy: 8-256")
    avg = np.mean(TPList, axis=(0,1))/50
    data_to_plot = np.average(TPList, axis=1) / 50
    max = np.amax(np.average(TPList, axis=1), axis=0)/50
    min = np.amin(np.average(TPList, axis=1), axis=0)/50
    ax.boxplot(data_to_plot, positions = xValues, showfliers=False, widths = 15, whis='range')
    ax.plot(xValues, avg)
    '''
    ax.set_ylim(bottom=0, top=100)

    ax.set_xlim(left=0, right=262)#2080)
    ax.set_xticks(xValues)
    ax.set_xticklabels(xValues)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fig.tight_layout()
    #plt.savefig("Total Network Accuracy Unscaled .png")
    plt.show()


loadFile()
