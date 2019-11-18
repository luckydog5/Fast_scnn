import matplotlib.pyplot as plt 
import numpy as np 

class LearningRateDecay:
    def plot(self,epochs,title="Learning Rate Schedule"):
        lrs = [self(i) for i in epochs]
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs,lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")

class StepDecay(LearningRateDecay):
    def __init__(self,initAlpha=0.01,factor=0.25,dropEvery=10):
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self,epoch):
        exp = np.floor((1+epoch)/self.dropEvery)
        alpha = self.initAlpha*(self.factor**exp)
        return float(alpha)

class PolynomialDecay(LearningRateDecay):
    def __init__(self,maxEpochs=100,initAlpha=0.01,power=1.0):
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power

    def __call__(self,epoch):
        decay = (1-(epoch/float(self.maxEpochs)))**self.power
        alpha = self.initAlpha*decay
        return float(alpha)
