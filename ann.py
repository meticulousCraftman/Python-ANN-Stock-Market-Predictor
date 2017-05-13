# The main execution begins from line  which contains a statement ----> if __name__ == "__main__":
# THe actual thing that you see starts after that line of code which will be probably situated after 300 lines.

import urllib2, time      # urllib2 is used for fetching the data from the kibot api, time is used for measuring the time taken by the program
from datetime import datetime   
from time import mktime
import sys
import math, random, string


random.seed(0)      # used for geenrating random numbers. The method seed() sets the integer starting value used in generating random numbers.

## ================================================================

# calculate a random number a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

def makeMatrix(I, J, fill = 0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def sigmoid(x):
    # tanh is a little nicer than the standard 1/(1+e^-x)
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

## ================================================================

class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        # number of input, hidden, and output nodes
        self.inputNodes = inputNodes + 1 # +1 for bias node
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        # activations for nodes
        self.inputActivation = [1.0]*self.inputNodes
        self.hiddenActivation = [1.0]*self.hiddenNodes
        self.outputActivation = [1.0]*self.outputNodes
        
        # create weights
        self.inputWeight = makeMatrix(self.inputNodes, self.hiddenNodes)
        self.outputWeight = makeMatrix(self.hiddenNodes, self.outputNodes)

        # set them to random vaules
        for i in range(self.inputNodes):
            for j in range(self.hiddenNodes):
                self.inputWeight[i][j] = rand(-0.2, 0.2)
        for j in range(self.hiddenNodes):
            for k in range(self.outputNodes):
                self.outputWeight[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        # Another technique that can help the 
        # network out of local minima is the use of a momentum term. 
        # This is probably the most popular extension of the backprop algorithm; it is hard to find cases where this is not used.
        #
        # When the gradient keeps changing direction, momentum will smooth out the variations. 
        # This is particularly useful when the network is not well-conditioned. In such cases 
        # the error surface has substantially different curvature along different directions, 
        # leading to the formation of long narrow valleys. For most points on the surface, the 
        # gradient does not point towards the minimum, and successive steps of gradient descent
        #  can oscillate from one side to the other, progressing only very slowly to the minimum
        #   (Fig. 2a). Fig. 2b shows how the addition of momentum helps to speed up convergence to the minimum by damping these oscillations.
        #
        #
        # https://www.willamette.edu/~gorr/classes/cs449/momrate.html

        self.ci = makeMatrix(self.inputNodes, self.hiddenNodes)
        self.co = makeMatrix(self.hiddenNodes, self.outputNodes)




    def update(self, inputs):
        if len(inputs) != self.inputNodes-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.inputNodes-1):
            self.inputActivation[i] = inputs[i]

        # hidden activations
        for j in range(self.hiddenNodes):
            sum = 0.0
            for i in range(self.inputNodes):
                sum = sum + self.inputActivation[i] * self.inputWeight[i][j]
            self.hiddenActivation[j] = sigmoid(sum)

        # output activations
        for k in range(self.outputNodes):
            sum = 0.0
            for j in range(self.hiddenNodes):
                sum = sum + self.hiddenActivation[j] * self.outputWeight[j][k]
            self.outputActivation[k] = sigmoid(sum)

        return self.outputActivation[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.outputNodes:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.outputNodes
        for k in range(self.outputNodes):
            error = targets[k]-self.outputActivation[k]
            output_deltas[k] = dsigmoid(self.outputActivation[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.hiddenNodes
        for j in range(self.hiddenNodes):
            error = 0.0
            for k in range(self.outputNodes):
                error = error + output_deltas[k]*self.outputWeight[j][k]
            hidden_deltas[j] = dsigmoid(self.hiddenActivation[j]) * error

        # update output weights
        for j in range(self.hiddenNodes):
            for k in range(self.outputNodes):
                change = output_deltas[k]*self.hiddenActivation[j]
                self.outputWeight[j][k] = self.outputWeight[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.inputNodes):
            for j in range(self.hiddenNodes):
                change = hidden_deltas[j]*self.inputActivation[i]
                self.inputWeight[i][j] = self.inputWeight[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k] - self.outputActivation[k])**2
            
        return error


    def test(self, inputNodes):
        # Giving in the input values for predicting
        print(inputNodes, '->', self.update(inputNodes))
        return self.update(inputNodes)[0]       # return the output of the activation function of the output neuron after sending in the data into the network.

    def weights(self):
        print('Input weights:')
        for i in range(self.inputNodes):
            print(self.inputWeight[i])
        print()
        print('Output weights:')
        for j in range(self.hiddenNodes):
            print(self.outputWeight[j])

    def train(self, patterns, iterations = 1000, N = 0.5, M = 0.1):
        # N: learning rate, M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)











## ================================================================

# function for calculating nomalized price
def normalizePrice(price, minimum, maximum):
    return ((2*price - (maximum + minimum)) / (maximum - minimum))


# function for calculating denormalizzed price
def denormalizePrice(price, minimum, maximum):
    return (((price*(maximum-minimum))/2) + (maximum + minimum))/2

## ================================================================

def rollingWindow(seq, windowSize):
    it = iter(seq)
    win = [it.next() for cnt in xrange(windowSize)] # First window
    yield win
    for e in it: # Subsequent windows
        win[:-1] = win[1:]
        win[-1] = e
        yield win


# Calculate the moving averages of the data using a specified window size
def getMovingAverage(values, windowSize):
    movingAverages = []
    
    for w in rollingWindow(values, windowSize):
        movingAverages.append(sum(w)/len(w))

    print
    print "getMovingAverage() ---> ",movingAverages
    print
    return movingAverages


#In the list of values fetch the largest value
def getMinimums(values, windowSize):
    minimums = []

    for w in rollingWindow(values, windowSize):
        minimums.append(min(w))
    print
    print "getMinimums() ---> ",minimums
    print
    return minimums


# In the list of values fetch the smallest value
def getMaximums(values, windowSize):
    maximums = []

    for w in rollingWindow(values, windowSize):
        maximums.append(max(w))

    print
    print "getMaximums() ---> ",maximums
    print
    return maximums

## ================================================================

def getTimeSeriesValues(values, window):
    print
    print "getTimeSeriesValues() ---> ",values
    print
    movingAverages = getMovingAverage(values, window)
    minimums = getMinimums(values, window)
    maximums = getMaximums(values, window)

    returnData = []

    # build items of the form [[average, minimum, maximum], normalized price]
    for i in range(0, len(movingAverages)):
        inputNode = [movingAverages[i], minimums[i], maximums[i]]   # Creating the first part of the structured list

        price = normalizePrice(values[len(movingAverages) - (i + 1)], minimums[i], maximums[i])     # calculating the nomalized price


        print "getTimeSeriesValues() normalizePrice ---> ",price
        print

        outputNode = [price]
        tempItem = [inputNode, outputNode]
        returnData.append(tempItem)

    print "getTimeSeriesValues() returnData ---> ",returnData
    print

    return returnData

## ================================================================

def getHistoricalData(stockSymbol):
    historicalPrices = []   # a list that will contain the stock prices of previous days
    
    # login to API to get the historical data values
    urllib2.urlopen("http://api.kibot.com/?action=login&user=guest&password=guest")

    # get 14 days of data from API (business days only, could be < 10) try and read what the parameters are saying you'll understant what it means.
    url = "http://api.kibot.com/?action=history&symbol=" + stockSymbol + "&interval=daily&period=14&unadjusted=1&regularsession=1"


    apiData = urllib2.urlopen(url).read().split("\n")   #data returned by the api is string data type.It is converted into list by spliting on the basis of number of newlines present in the data returned. The returned data is in the format --> Date,Open,High,Low,Close,Volume eg. 04/13/2016,54.73,55.05,54.51,54.97,18942957

    for line in apiData:    #out of the 14 stock values choose each one of them one by one and then process every single day's value

        if(len(line) > 0):      #if the data is not empty of that day's stock value then move ahead

            tempLine = line.split(',')  #Since the each days values are sperated by commas (04/13/2016,54.73,55.05,54.51,54.97,18942957) go ahead and split them and make another temporary list. Note still all the data after splitting is still a string data type and not an integer or float
            price = float(tempLine[1])  # convert that day's stock price into float data type. We are not using any other data. We are only using that day's stock value and nothing else to predict the next days value.
            historicalPrices.append(price)  # append the stock prices in the list

    return historicalPrices     # return the list

## ================================================================

def getTrainingData(stockSymbol):
    historicalData = getHistoricalData(stockSymbol)     # fetch the last 14 days value using the kibot api. The data type of this variable will be a list. It will contain just a simple list of all the previous 14 days stock values.

    # reverse it so we're using the most recent data first, ensure we only have 9 data points
    historicalData.reverse()
    del historicalData[9:]  # Of the last 14 days data. Delete the oldest 5 days values. After doing this we'll get only 9 values left with us to work on.

    # get five 5-day moving averages, 5-day lows, and 5-day highs, associated with the closing price
    trainingData = getTimeSeriesValues(historicalData, 5)

    return trainingData


# Calculating the stuff of next 5 days for creating the predictionData
def getPredictionData(stockSymbol):
    historicalData = getHistoricalData(stockSymbol)

    # reverse it so we're using the most recent data first, then ensure we only have 5 data points
    historicalData.reverse()
    del historicalData[5:]

    # get five 5-day moving averages, 5-day lows, and 5-day highs
    predictionData = getTimeSeriesValues(historicalData, 5)


    # remove associated closing price
    predictionData = predictionData[0][0]

    print "Final prediction data -> ",predictionData
    print

    return predictionData

## ================================================================

def analyzeSymbol(stockSymbol):
    print "Analyzing Stock Symbol"
    startTime = time.time()     # store the present time in a variable called startTime

    print "Fetching the training dataset"
    trainingData = getTrainingData(stockSymbol)     # Get the training data
    
    print "Intializing Artificial Neural Network"
    network = NeuralNetwork(inputNodes = 3, hiddenNodes = 3, outputNodes = 1)       # Create an artificial neural network of 3 input nodes, 3 hidden nodes and 1 output node
    
    print "Training the neural network"
    network.train(trainingData)     # training the network on training data

    # get rolling data for most recent day -- this will go as an input into the ANN
    print "Getting the rolling data for the most recent day"
    predictionData = getPredictionData(stockSymbol)

    # get prediction
    print "Calculating the next day's value"
    returnPrice = network.test(predictionData)

    # de-normalize and return predicted stock price
    print "Denormalizing the Stock Price"
    predictedStockPrice = denormalizePrice(returnPrice, predictionData[1], predictionData[2])       #Denormalizing prices --> predictionData[1], predictionData[2] ---> min,max

    # create return object, including the amount of time used to predict
    returnData = {}
    returnData['price'] = predictedStockPrice   # stuffing the data into a dictionary
    returnData['time'] = time.time() - startTime    # calculate the time taken todo all the stuff

    return returnData

## ================================================================

if __name__ == "__main__":      # Checks whether the program is being imported or is being run form the command line
    print "\n\n\t\t Artificial Neural Network Stock Price Predictor\n\n"
    
    # a try-catch block for catching the keyboard interrupt signal (CTRL + C) to end the program.
    try:
        a = raw_input("Enter the stock symbol : ")      # Ask for input of the stock symbol to use.
        data = analyzeSymbol(a)         # Passing the input of the user to the function analyzeSymbol() which does further processing.

        # Just priting some shit
        print "\n\n########## Prediction ##############"
        print "Next day stock price for %s : ",data['price']
        print "Time taken to compute : ",data['time']
        print "####################################\n\n"
        raw_input("Press any key to exit.....")
        print

    except KeyboardInterrupt, e:        # If u detect a keyboard interrupt signal...end the program after priting two new lines into the console.
        print "\n\n"
        sys.exit()  # Close the program and return 0 as the exit code to the OS.
