import numpy
import pandas as pd
import sklearn
from sklearn import datasets,svm,metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import seaborn as sns



class Harmony:
    #constructor to initialise initial parameters of OBHS
    def __init__(self,HMS,data_inputs,data_outputs,HMCR,PAR,Maxitr):
        self.kf = KFold(5)
        self.data_inputs = data_inputs
        self.data_outputs = data_outputs
        self.HMS = HMS
        self.num_samples = data_inputs.shape[0]
        self.num_feature_elements = data_inputs.shape[1]
        self.HM_SHAPE = (self.HMS, self.num_feature_elements)
        self.HMCR = HMCR
        self.PAR = PAR
        self.Maxitr= Maxitr
        self.low = 0
        self.high = 2

        self.harmony_memory = []
        self.best_outputs = []
        self.best_opp_outputs = []
        self.best_solution = []
        self.best_solution_indices = -1
        self.best_solution_num_elements = 0
        self.best_solution_fitness = 0
        self.best_k = 0
        self.best_k_fitness = 0
    
    #method to randomly initialise candidate solutions
    def random_initialise(self):
        self.harmony_memory = numpy.random.randint(self.low, self.high, size = self.HM_SHAPE)

    #method to reduce features from csv for generate accuracy score
    def reduce_features(self,solution, features):
        selected_elements_indices = numpy.where(solution ==1)[0]
        reduced_features = features[:, selected_elements_indices]
        return reduced_features
    
    #method to calculate accuracy score of a data set
    def classification_accuracy(self,labels, predictions):
        correct = numpy.where(labels == predictions)[0]
        accuracy = correct.shape[0]/labels.shape[0]
        return accuracy
    
    #method to calculate population fitness
    def cal_pop_fitness(self,pop, features, labels, train_indices,val_indices, test_indices,classifier):
        test_accuracies = numpy.zeros(pop.shape[0])
        val_accuracies = numpy.zeros(pop.shape[0])
        idx = 0
        for curr_solution in pop:
            reduced_features = self.reduce_features(curr_solution, features)
            train_data = reduced_features[train_indices, :]
            val_data=reduced_features[val_indices,:]
            test_data = reduced_features[test_indices, :]

            train_labels = labels[train_indices]
            val_labels=labels[val_indices]
            test_labels = labels[test_indices]
            if classifier=='SVM':
                SV_classifier = sklearn.svm.SVC(kernel='rbf',gamma='scale',C=5000)
                SV_classifier.fit(X=train_data, y=train_labels)
                val_predictions = SV_classifier.predict(val_data)
                val_accuracies[idx] = self.classification_accuracy(val_labels, val_predictions)
                test_predictions = SV_classifier.predict(test_data)
                test_accuracies[idx] = self.classification_accuracy(test_labels, test_predictions)
                idx = idx + 1
            elif classifier == 'KNN':
                knn=KNeighborsClassifier(n_neighbors=8)
                knn.fit(train_data,train_labels)
                predictions=knn.predict(test_data)
                test_accuracies[idx]= self.classification_accuracy(test_labels,predictions)
                idx = idx + 1
            else :
                mlp = MLPClassifier()
                mlp.fit(train_data,train_labels)
                predictions=mlp.predict(test_data)
                test_accuracies[idx]= self.classification_accuracy(test_labels,predictions)
                idx = idx + 1
        return val_accuracies,test_accuracies
    
    #method to implement OBHS optimally
    def optimize(self):
        classifier="SVM"   
        NCHV = numpy.ones((1, self.num_feature_elements))
        fold=0
        for train_indices,test_val_indices in self.kf.split(self.data_inputs):
            fold=fold+1
            print("Fold : ",fold)
            val_indices,test_indices=train_test_split(test_val_indices,test_size=0.5,shuffle=True,random_state=8)
            best_test_outputs=[]

            #generate harmony and opposite harmony memory
            self.random_initialise()
            opposite_memory=1-self.harmony_memory
            total_memory=numpy.concatenate((self.harmony_memory,opposite_memory),axis=0)
            total_fitness,_ = self.cal_pop_fitness(total_memory,self.data_inputs,self.data_outputs,train_indices,val_indices,test_indices,classifier)
            fit_ind = numpy.argpartition(total_fitness, -self.HMS)[-self.HMS:]
            self.harmony_memory=total_memory[fit_ind,:]
            
            for currentIteration in range(self.Maxitr):
                NCHV = numpy.ones((1, self.num_feature_elements))
                print("Generation : ", currentIteration+1)
                
                fitness,test_fitness=self.cal_pop_fitness(self.harmony_memory,self.data_inputs,self.data_outputs,train_indices,val_indices,test_indices,classifier)
                best_test_outputs.append(numpy.max(test_fitness))
                print("Best Test result :",max(best_test_outputs))

                self.best_outputs.append(numpy.max(fitness))
                print("Best validation result : ", max(self.best_outputs))
                

                for i in range(self.num_feature_elements):
                    ran = numpy.random.rand()
                    if ran < self.HMCR:
                        index = numpy.random.randint(0, self.HMS)
                        NCHV[0, i] = self.harmony_memory[index, i]
                        pvbran = numpy.random.rand()
                        if pvbran < self.PAR:
                            pvbran1 = numpy.random.rand()
                            result = NCHV[0, i]
                            if pvbran1 < 0.5:
                                result =1-result

                    else:
                        NCHV[0, i] = numpy.random.randint(low=0,high=2,size=1)
                
                new_fitness,_ = self.cal_pop_fitness(NCHV, self.data_inputs, self.data_outputs, train_indices, val_indices,test_indices,classifier)
                if new_fitness > min(fitness):
                    min_fit_idx = numpy.where(fitness == min(fitness))
                    self.harmony_memory[min_fit_idx, :] = NCHV
                    fitness[min_fit_idx] = new_fitness

                opp_NCHV=1-NCHV
                new_opp_fitness,_= self.cal_pop_fitness(opp_NCHV,self.data_inputs, self.data_outputs, train_indices, val_indices,test_indices,classifier)
                if new_opp_fitness > min(fitness):
                    min_fit_idx = numpy.where(fitness == min(fitness))
                    self.harmony_memory[min_fit_idx, :] = opp_NCHV
                    fitness[min_fit_idx] = new_opp_fitness
                
            _,fitness = self.cal_pop_fitness(self.harmony_memory, data_inputs, data_outputs, train_indices,val_indices,test_indices,classifier)

            best_match_idx = numpy.where(fitness == numpy.max(fitness))[0]
            best_match_idx = best_match_idx[0]

            self.best_solution = self.harmony_memory[best_match_idx, :]
            self.best_solution_indices = numpy.where(self.best_solution == 1)[0]
            self.best_solution_num_elements = self.best_solution_indices.shape[0]
            self.best_solution_fitness = numpy.max(fitness)

            print("best_match_idx : ", best_match_idx)
            print("best_solution : ", self.best_solution)
            print("Selected indices : ", self.best_solution_indices)
            print("Number of selected elements : ", self.best_solution_num_elements)

            if self.best_solution_fitness > self.best_k_fitness:
              self.best_k = fold
              self.best_k_fitness = self.best_solution_fitness
            
            #generate dataframe and save into csv for selected features in every fold
            df=pd.read_csv('/content/sample_data/googlenet_mlbc.csv',usecols =self.best_solution_indices)
            df.to_csv("file"+str(fold)+".csv")
        print("Best Kfold : ",self.best_k," and Best Fitness : ",self.best_k_fitness)  
    
    #method to generate scatter plot
    def gen_plot(self): 
      x=pd.read_csv('/content/file'+str(self.best_k)+'.csv') 
      x=x.to_numpy()

      x=x[:,1:]

      y=pd.read_csv('/content/sample_data/mlbc_labels.csv')  
      y=y.to_numpy()

      y=y[:,1:]

      y=y[:,0]

      tsne = TSNE(n_components=2, verbose=0, random_state=123)
      z = tsne.fit_transform(x)   # problem showing all warnings



      df = pd.DataFrame()
      df["y"] = y
      df["comp-1"] = z[:,0]
      df["comp-2"] = z[:,1]


      sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),palette=sns.color_palette("hls", 4),
                      data=df).set(title="TSNE PLOT")



if __name__=="__main__":
    data_inputs1=pd.read_csv('/content/sample_data/googlenet_mlbc.csv')   
    #data_inputs2=pd.read_csv('/home/somenath/Desktop/sipkamed5/resnet18_sipakmed.csv')
    data_outputs=pd.read_csv('/content/sample_data/mlbc_labels.csv')
    data_inputs1=data_inputs1.to_numpy()
    #data_inputs2=data_inputs2.to_numpy()
    data_outputs=data_outputs.to_numpy()
    data_inputs=data_inputs1[:,1:]
    #data_inputs2=data_inputs2[:,1:]
    
    data_outputs=data_outputs[:,1]
    #data_inputs=numpy.concatenate((data_inputs1,data_inputs2),axis=1)
    # print(data_outputs.shape)
    # print(data_inputs.shape)
    
    HMCR = float(input("Enter HMCR : "))
    PCR = float(input("Enter PCR value : "))
    maxitr = int(input("Enter maximum number of iterations:"))
    obj = Harmony(100, data_inputs, data_outputs, HMCR, PCR, maxitr)
    # obj = Harmony(100, data_inputs, data_outputs, 0.9, 0.35, 100)
    obj.optimize()
    obj.gen_plot()