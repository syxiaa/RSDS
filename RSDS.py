import numpy as np
from collections import Counter
import time
import warnings

warnings.filterwarnings("ignore")

minNumSample = 10


class BinaryTree:
    """An Special BinaryTree.

        Construct a special binary tree, store the data in the nodes of the tree,
        node labels, left and right subtree positions


    """

    def __init__(self, labels=np.array([]), datas=np.array([])):
        self.label = labels
        self.data = datas
        self.leftChild = None
        self.rightChild = None

    def set_rightChild(self, rightObj):
        self.rightChild = rightObj

    def set_leftChild(self, leftObj):
        self.leftChild = leftObj

    def get_rightChild(self):
        return self.rightChild

    def get_leftChild(self):
        return self.leftChild

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label


def RSDS_fun(train_data, tree_num=10):
    """Handling data noise using completely random forest judgment.

        Establish a tree_num completely random tree. The data label in each leaf node
        of the tree is compared with the parent node label to obtain the noise judgment
        label of each data in the case of a tree, and all the completely random tree noise
        judgment labels are combined to vote to determine the noise data. Denoised data
        set after processingEstablish a tree_num completely random tree. The data label
        in each leaf node of the tree is compared with the parent node label to obtain
        the noise judgment label of each data in the case of a tree, and all the completely
        random tree noise judgment labels are combined to vote to determine the noise data.
        Denoised data set after processing

        Parameters
        ----------
        train_data :Numpy type data set.

        tree_num :Total number of random trees.

    """

    m, n = train_data.shape
    forest = np.array([])
    for i in range(tree_num):
        tree = CRT(train_data)
        visiTree = visitCRT(tree)
        visiTree = visiTree[:, np.argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = np.hstack((forest, visiTree.reshape(m, 1)))
    noiseForest = np.sum(forest, axis=1)
    nn = 0.5 * tree_num
    noiseForest = np.array(list(map(lambda x: 1 if x >= nn or x == 0 else 0, noiseForest)))
    denoiseTraindata = deleteNoiseData(train_data, noiseForest)
    return denoiseTraindata


def CRT(data):
    """Build A Completely Random Tree.

        Add a column at the end of the data, store the initial sequence
        number of each piece of data, call the function ‘generateTree’
        spanning tree

         Parameters
         ----------
         data :Numpy type data set

     """
    numberSample = data.shape[0]
    orderAttribute = np.arange(numberSample).reshape(numberSample, 1)  # (862, 1)
    data = np.hstack((data, orderAttribute))
    completeRandomTree = generateTree(data)
    return completeRandomTree


def generateTree(data, uplabels=[]):
    """Iteratively Generating A Completely Random Tree.

         Complete random tree by random partitioning of random attributes

         Parameters
         ----------
         data :Numpy type data set

         uplabels :rootlabel

     """
    try:
        numberSample, numberAttribute = data.shape
    except ValueError:
        numberSample = 1
        numberAttribute = data.size

    if numberAttribute == 0:
        return None

    numberAttribute = numberAttribute - 2  # Subtract the added serial number and label

    # The category of the current data, also called the node category
    labelNumKey = []  # todo
    if numberSample == 1:  # Only one sample left
        labelvalue = data[0][0]
        rootdata = data[0][numberAttribute + 1]
    else:
        labelNum = Counter(data[:, 0])
        labelNumKey = list(labelNum.keys())  # Key (label)
        labelNumValue = list(labelNum.values())  # Value (quantity)
        labelvalue = labelNumKey[labelNumValue.index(max(labelNumValue))]  # Vote to find the label
        rootdata = data[:, numberAttribute + 1]
    rootlabel = np.hstack((labelvalue, uplabels))  # todo

    # Call the class 'BinaryTree', passing in tags and data
    CRTree = BinaryTree(rootlabel, rootdata)
    '''
    The 'rootlabel' and 'rootdata' are obtained above, the 'rootlabel' is a label (derived by voting), 
    the 'rootdata' is a series of serial numbers, and finally the class BinaryTree is called.
    '''
    # There are at least two conditions for the tree to stop growing:
    # 1 the number of samples is limited;
    # 2 the first column is all equal
    if numberSample < minNumSample or len(labelNumKey) < 2:
        # minNumSample defaults to 10 or only 1 of the label types are left.
        return CRTree
    else:
        maxCycles = 1.5 * numberAttribute  # Maximum number of cycles
        # maxCycles = 2
        i = 0
        while True:
            # Once a data exception occurs: except for the above two exceptions that
            # stop the tree growth condition, that is, the error data, the loop here will not stop.
            i += 1
            splitAttribute = np.random.randint(1, numberAttribute)  # Randomly select a list of attributes
            if splitAttribute > 0 and splitAttribute < numberAttribute + 1:
                dataSplit = data[:, splitAttribute]
                uniquedata = list(set(dataSplit))
                if len(uniquedata) > 1:
                    break
            if i > maxCycles:  # Tree caused by data anomaly stops growing
                return CRTree
        sv1 = np.random.choice(uniquedata)
        i = 0
        while True:
            i += 1
            sv2 = np.random.choice(uniquedata)
            if sv2 != sv1:
                break
            if i > maxCycles:
                return CRTree
        splitValue = np.mean([sv1, sv2])
        '''
        The above randomly selected rows and columns are obtained, and the final 'splitValue' is an average
        '''

        # Call split function
        leftdata, rightdata = splitData(data, splitAttribute, splitValue)

        # Set the left subtree, the right subtree
        CRTree.set_leftChild(generateTree(leftdata, rootlabel))
        CRTree.set_rightChild(generateTree(rightdata, rootlabel))
        return CRTree


'''
returns a matrix of two rows and N columns, the first row is the index of the sample, 
and the second row is the threshold of the label noise.
e.g.
[[ 36. 499. 547. 557. 563. 587.]
 [  0.   0.   0.   0.   0.   0.]]
'''


def visitCRT(tree):
    """
    Traversing the tree to get the relationship between the data and the node label.

         The traversal tree stores the data number and node label stored in each node of the
         completely random tree.

         Parameters
         ----------
         tree :Root node of the tree.


    """
    if not tree.get_leftChild() and not tree.get_rightChild():  # If the left and right subtrees are empty
        data = tree.get_data()  # data is the serial number of the sample
        labels = checkLabelSequence(tree.get_label())  # Existing tag sequence
        try:
            labels = np.zeros(len(data)) + labels
        except TypeError:
            pass
        result = np.vstack((data, labels))
        return result
    else:
        resultLeft = visitCRT(tree.get_leftChild())
        resultRight = visitCRT(tree.get_rightChild())
        result = np.hstack((resultLeft, resultRight))
        return result


def deleteNoiseData(data, noiseOrder):
    """Delete noise points in the training set.

         Delete the noise points in the training set according to the noise
         judgment result of each data in noiseOrder.

         Parameters
         ----------
         data :Numpy type data set.

         noiseOrder :Determine if each piece of data is a list of noise.

     """
    m, n = data.shape
    data = np.hstack((data, noiseOrder.reshape(m, 1)))
    redata = np.array(list(filter(lambda x: x[n] == 0, data[:, ])))
    redata = np.delete(redata, n, axis=1)
    return redata


"""check whether the label of the parent node and the leaf node are consistent."""


def checkLabelSequence(labels):
    """Check label sequence.

         Check if the leaf node is the same as the parent node.

         Parameters
         ----------
         labels :label sequence.

     """
    return 1 if labels[0] != labels[1] else 0


def splitData(data, splitAttribute, splitValue):
    """Dividing data sets.

         Divide the data into two parts, leftData and rightData, based on the splitValue
         of the split attribute column element.

         Parameters
         ----------
         data:Numpy type data set.

         splitAttribute:Randomly selected attributes when dividing.

         splitValue:Dividing the value obtained by dividing the selected attribute.
     """
    rightData = np.array(list(filter(lambda x: x[splitAttribute] > splitValue, data[:, ])))
    leftData = np.array(list(filter(lambda x: x[splitAttribute] <= splitValue, data[:, ])))
    return leftData, rightData
