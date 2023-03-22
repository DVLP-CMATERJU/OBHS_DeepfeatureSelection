import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC

#FoodSource Class
class FoodSource:
    #constructor to initialise initial parameters of each food source
    def __init__(self, dim, trainPath, validationPath, trainLabelPath, validationLabelPath) -> None:
        self.trial = 0
        self.dim = dim
        self.solution = np.zeros(self.dim)    #real number 1D array filled with all zeros
        self.fit = 0
        self.prob = 0.
        #setting training,training-label,testing,testing-label paths
        self.trainPath = trainPath
        self.validationPath = validationPath
        self.trainLabelPath = trainLabelPath
        self.validationLabelPath = validationLabelPath
    
    #randomly initialise food sources
    def randomSolutionInitialise(self):
        for i in range(self.dim):
            r = np.random.rand()

            self.solution[i] = round(abs(r % 2)) % 2
        
        self.trial = 0
    
    #function to implement change in food source in each generation's bee phases
    def generate(self, j, phi, p):
        self.solution[j] = self.solution[j] + phi * (self.solution[j] - self.solution[p])
        self.solution[j] = round(abs(self.solution[j] % 2)) % 2

    #function to evaluate fitness of every food source
    def evaluate(self, initialise = False):
        f = self.calculateSVMAccuracy()
        #print("Output of SVMAccuracy:",f)
        if f < 0:
            fitNew = 1 + abs(f)
        else:
            fitNew = 1 / (1 + f)

        if not initialise:
            if fitNew > self.fit:
                self.fit = fitNew
                self.trial = 0

            else:
                self.trial += 1

        else:
            self.fit = fitNew
    
    #function to evaluate probability of choosing the food source
    def probability(self, maxFit):
        self.prob = 0.9 * (self.fit / maxFit) + 0.1
        return self.prob 
    
    #counting number of 1s in solution vector
    def count1(self):
        count=0
        for i in self.solution:
            if i > 0.5:
                count=count+1
        return count

    def generateSubset(self, path):
        data = pd.read_csv(path)
        data = data.iloc[1:, 1:]
        #in case all features are 0 valued
        cc=self.count1()
        #loop to check while there is not at least two 1s in solution vector
        #keep generating randomised solution
        while(cc<= 1):
            self.randomSolutionInitialise()
            cc=self.count1()
        #end of loop
        
        data.columns = [1 for i in range(self.dim)]
        # selecting those columns whose solution vector corresponding 
        # dimension value is 1
        # hence we get different svm accuracy for different food sources
        sub_data = data.iloc[: , data.columns == self.solution]
        #print("OL:",sub_data.shape)

        return sub_data
    
    #function to calculate SVM accuracy for a give dataset
    def calculateSVMAccuracy(self):
        #generating subset from training csv file
        trainSubData = self.generateSubset(self.trainPath)
        trainSubData = trainSubData.iloc[:, 1:]
        trainSubData = trainSubData.to_numpy()

        #generating subset from validation or testing csv file
        validationSubData = self.generateSubset(self.validationPath)
        validationSubData = validationSubData.iloc[:, 1:]
        validationSubData = validationSubData.to_numpy()

        #trainLabel = np.load(self.trainLabelPath, allow_pickle = True)
        #validationLabel = np.load(self.validationLabelPath, allow_pickle = True)
        trainLabel = pd.read_csv(self.trainLabelPath)
        trainLabel=trainLabel.iloc[1:,1:].to_numpy().ravel()
        validationLabel = pd.read_csv(self.validationLabelPath)
        validationLabel=validationLabel.iloc[1:,1:].to_numpy().ravel()  #ravel() to flattening the array

        rbfSVC = svm.SVC(kernel = "rbf")
        rbfSVC.fit(trainSubData, trainLabel)

        accuracy = rbfSVC.score(validationSubData, validationLabel)

        return accuracy



#class for Artificial Bee Colony Optimization
class ArtificialBeeColony:
    #constructor to initialise initial parameters of ABC Optimization
    def __init__(self, dim, swarmSize, generations, limit, trainPath, validationPath, trainLabelPath, validationLabelPath) -> None:
        self.dim = dim
        self.populationSize = swarmSize // 2
        self.generations = generations
        self.limit = limit
        

        self.population = []
        
        #setting training,training-label,testing,testing-label paths
        self.trainPath = trainPath
        self.validationPath = validationPath
        self.trainLabelPath = trainLabelPath
        self.validationLabelPath = validationLabelPath

        self.bestFit = 0
        self.bestFitIndex = -1
        self.bestSolution = np.zeros(self.dim)
        self.bestGeneration=1
    
    #function to implement employer bee phase for every bee
    def employerBeesPhase(self):
        for i in range(self.populationSize):
            j = np.random.randint(0, self.dim)
            p = np.random.randint(0, self.dim)
            while(j == p):
                p = np.random.randint(0, self.dim)
            self.population[i].generate(j, 0.1, p)
            self.population[i].evaluate()

    #function to implement onlooker bee phase for every bee
    def onlookerBeesPhase(self):
        m = 0
        n = 0
        maxFit = max([self.population[i].fit for i in range(self.populationSize)])

        while(m < self.populationSize):
            r = np.random.random()
            if r < self.population[n].probability(maxFit):
                j = np.random.randint(0, self.dim)
                p = np.random.randint(0, self.dim)
                while(j == p):
                    p = np.random.randint(0, self.dim)
                self.population[n].generate(j, 0.1, p)
                self.population[n].evaluate()
                m += 1
                n = (n + 1) % self.populationSize
            
            else:
                n = (n + 1) % self.populationSize
    
    #function to implement scout bee phase for every bee
    def scoutPhase(self, index):
        self.population[index].randomSolutionInitialise()
    
    #function to implement main algorithm optimally
    def optimize(self):
        #print("PLACE1")
        for i in range(self.populationSize):
            food = FoodSource(self.dim, self.trainPath, self.validationPath, self.trainLabelPath, self.validationLabelPath)
            food.randomSolutionInitialise()
            food.evaluate(initialise= True)
            self.population.append(food)    
        genfit=[]       #list to store best fitness value for every generation
        for i in range(self.generations):
            #print("PLACE2")
            self.employerBeesPhase()
            self.onlookerBeesPhase()

            for j in range(self.populationSize):
                if self.bestFit < self.population[j].fit:
                    self.bestFit = self.population[j].fit
                    self.bestFitIndex = j
                    self.bestSolution = self.population[j].solution
                    self.bestGeneration=i+1
            # print(i," Generation Fitness Value=>",self.bestFit)
            scoutSelection = []
            dtype = [('index', int), ('trial', int)]
            index = -1
            for j in range(self.populationSize):
                if self.population[j].trial > self.limit:
                    scoutSelection.append((j, self.population[j].trial))
            if len(scoutSelection) > 0:
                scoutSelection = np.array(scoutSelection, dtype=dtype)
                scoutSelection = np.sort(scoutSelection, order= 'trial')

                index = scoutSelection[-1][0]

                if index == self.bestFitIndex and len(scoutSelection) > 1:
                    index = scoutSelection[-2][0]
                    self.scoutPhase(index)

                elif index != self.bestFitIndex:
                    self.scoutPhase(index)

            print(f"Iteration {i} complete.\n",end="")
            print("Best Fitness => ",self.bestFit)
            genfit.append(self.bestFit)
        
        # print("All generation fitness values are:",genfit)
        #save all required data in respective files
        with open('result_fs.txt', 'w') as file:
            file.write(f"Best Fit: {self.bestFit} \n")
            file.write(f"Best f: {(1 / self.bestFit) - 1} \n")
            file.write(f"Best solution: {self.bestSolution} \n")
            file.write(f"Best fit's index: {self.bestFitIndex} \n")
            file.write(f"Best Generation:{self.bestGeneration}")
            for i in range(len(genfit)):
                file.write(str(i+1)+" Generation : "+str(genfit[i])+"\n")
        return genfit
        



if __name__ == "__main__":
    # Number of features
    dim = int(input("Enter dimension:"))             # Set according to problem (19 here)

    # Swarm Size, usually kept at equal to D so that population is equal to D / 2
    swarmSize = dim

    # Number of generations are usually 1000, but can set to anything
    generations = int(input("Enter number of generations:"))          # 1000 here

    # Limit is usually kept equal to N x D / 2
    limit = (swarmSize * dim) / 4

    #train, validation path, here, CSV files
    trainPath = "C:/Users/Subhradeep/OneDrive/Desktop/MCA_SEMS/4thSem/OPT_ALGO/data/absent_training.csv"
    validationPath = "C:/Users/Subhradeep/OneDrive/Desktop/MCA_SEMS/4thSem/OPT_ALGO/data/absent_validation.csv"

    #train label and validation label path, here, numpy files
    
    trainLabelPath = "C:/Users/Subhradeep/OneDrive/Desktop/MCA_SEMS/4thSem/OPT_ALGO/data/absent_training_label.csv"
    validationLabelPath = "C:/Users/Subhradeep/OneDrive/Desktop/MCA_SEMS/4thSem/OPT_ALGO/data/absent_validation_label.csv"

    #loop to execute ABC Optimization for every benchmark function present in benchmarks.py
    abc = ArtificialBeeColony(dim, swarmSize, generations, limit,
                                trainPath, validationPath, trainLabelPath, validationLabelPath)
    list_fit = abc.optimize()
    print(list_fit)
