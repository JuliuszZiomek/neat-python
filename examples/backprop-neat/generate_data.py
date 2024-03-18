import numpy as np

class Mat:
    def __init__(self, rows, cols):
        self.w = np.zeros((rows, cols))

    def set(self, row, col, value):
        self.w[row][col] = value

class DataPoint:
    def __init__(self, x, y, l):
        self.x = x
        self.y = y
        self.l = l

trainList = []
testList = []

nSize = 200  # training size
nTestSize = 200  # testing size
noiseLevel = 0.5

nBatch = 10  # minibatch size
dataBatch = Mat(nBatch, 2)
labelBatch = Mat(nBatch, 1)

trainData = None
trainLabel = None
testData = None
testLabel = None

def shuffleDataList(pList):
    np.random.shuffle(pList)

def generateXORData(numPoints=None, noise=None):
    if numPoints is None:
        numPoints = nSize
    if noise is None:
        noise = noiseLevel

    particleList = []
    N = numPoints
    for i in range(N):
        x = np.random.uniform(-5.0, 5.0) + np.random.randn() * noise
        y = np.random.uniform(-5.0, 5.0) + np.random.randn() * noise
        l = 0
        if x > 0 and y > 0:
            l = 1
        if x < 0 and y < 0:
            l = 1
        particleList.append(DataPoint(x, y, l))
    return particleList

def generateSpiralData(numPoints=None, noise=None):
    if numPoints is None:
        numPoints = nSize
    if noise is None:
        noise = noiseLevel

    particleList = []
    N = numPoints

    def genSpiral(deltaT, l):
        n = N // 2
        for i in range(n):
            r = i / n * 6.0
            t = 1.75 * i / n * 2 * np.pi + deltaT
            x = r * np.sin(t) + np.random.uniform(-1, 1) * noise
            y = r * np.cos(t) + np.random.uniform(-1, 1) * noise
            particleList.append(DataPoint(x, y, l))

    flip = 0
    backside = 1 - flip
    genSpiral(0, flip)  # Positive examples.
    genSpiral(np.pi, backside)  # Negative examples.
    return particleList

def generateGaussianData(numPoints=None, noise=None):
    if numPoints is None:
        numPoints = nSize
    if noise is None:
        noise = noiseLevel

    particleList = []
    N = numPoints

    def genGaussian(xc, yc, l):
        n = N // 2
        for i in range(n):
            x = np.random.normal(xc, noise * 1.0 + 1.0)
            y = np.random.normal(yc, noise * 1.0 + 1.0)
            particleList.append(DataPoint(x, y, l))

    genGaussian(2 * 1, 2 * 1, 1)  # Positive examples.
    genGaussian(-2 * 1, -2 * 1, 0)  # Negative examples.
    return particleList

def generateCircleData(numPoints=None, noise=None):
    if numPoints is None:
        numPoints = nSize
    if noise is None:
        noise = noiseLevel

    particleList = []
    N = numPoints
    n = N // 2
    radius = 5.0

    def getCircleLabel(x, y):
        return 1 if x ** 2 + y ** 2 < (radius * 0.5) ** 2 else 0

    for _ in range(n):
        r = np.random.uniform(0, radius * 0.5)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noiseX = np.random.uniform(-radius, radius) * noise / 3
        noiseY = np.random.uniform(-radius, radius) * noise / 3
        l = getCircleLabel(x, y)
        particleList.append(DataPoint(x + noiseX, y + noiseY, l))

    for _ in range(n):
        r = np.random.uniform(radius * 0.75, radius)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noiseX = np.random.uniform(-radius, radius) * noise / 3
        noiseY = np.random.uniform(-radius, radius) * noise / 3
        l = getCircleLabel(x, y)
        particleList.append(DataPoint(x + noiseX, y + noiseY, l))

    return particleList

def convertData():
    global trainData, trainLabel, testData, testLabel
    testData = Mat(nTestSize, 2)
    testLabel = Mat(nTestSize, 1)
    trainData = Mat(nSize, 2)
    trainLabel = Mat(nSize, 1)

    for i, p in enumerate(testList):
        testData.set(i, 0, p.x)
        testData.set(i, 1, p.y)
        testLabel.w[i] = p.l

    for i, p in enumerate(trainList):
        trainData.set(i, 0, p.x)
        trainData.set(i, 1, p.y)
        trainLabel.w[i] = p.l

def generateMiniBatch():
    for i in range(nBatch):
        randomIndex = np.random.randint(0, len(trainList))
        p = trainList[randomIndex]
        dataBatch.set(i, 0, p.x)
        dataBatch.set(i, 1, p.y)
        labelBatch.w[i] = p.l

def generateRandomData(choice_=None):
    global trainList, testList
    choice = np.random.randint(0, 4) if choice_ is None else choice_
    if choice == 0:
        trainList = generateCircleData(nSize, noiseLevel)
        testList = generateCircleData(nTestSize, noiseLevel)
    elif choice == 1:
        trainList = generateXORData(nSize, noiseLevel)
        testList = generateXORData(nTestSize, noiseLevel)
    elif choice == 2:
        trainList = generateGaussianData(nSize, noiseLevel)
        testList = generateGaussianData(nTestSize, noiseLevel)
    else:
        trainList = generateSpiralData(nSize, noiseLevel)
        testList = generateSpiralData(nTestSize, noiseLevel)

    shuffleDataList(trainList)
    shuffleDataList(testList)
    convertData()

def getTrainData():
    return trainData

def getTrainLabel():
    return trainLabel

def getTestData():
    return testData

def getTestLabel():
    return testLabel

def getBatchData():
    return dataBatch

def getBatchLabel():
    return labelBatch

def getTrainLength():
    return nSize

def getTestLength():
    return nTestSize

def getBatchLength():
    return nBatch

generateRandomData()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = generateCircleData(100, 0.1)

    plt.scatter([d.x for d in data if d.l==1], [d.y for d in data if d.l==1], label="positive")
    plt.scatter([d.x for d in data if d.l==0], [d.y for d in data if d.l==0], label="negative")
    plt.savefig("test_circle_data.png")
    plt.clf()

    data = generateSpiralData(100, 0.1)

    plt.scatter([d.x for d in data if d.l==1], [d.y for d in data if d.l==1], label="positive")
    plt.scatter([d.x for d in data if d.l==0], [d.y for d in data if d.l==0], label="negative")
    plt.savefig("test_spiral_data.png")
    plt.clf()

    data = generateGaussianData(100, 0.1)

    plt.scatter([d.x for d in data if d.l==1], [d.y for d in data if d.l==1], label="positive")
    plt.scatter([d.x for d in data if d.l==0], [d.y for d in data if d.l==0], label="negative")
    plt.savefig("test_gaussian_data.png")
    plt.clf()

    data = generateXORData(100, 0.1)

    plt.scatter([d.x for d in data if d.l==1], [d.y for d in data if d.l==1], label="positive")
    plt.scatter([d.x for d in data if d.l==0], [d.y for d in data if d.l==0], label="negative")
    plt.savefig("test_xor_data.png")

    