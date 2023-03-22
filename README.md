# OBHS_DeepfeatureSelection
- To run the OBHS code :
    1. change the training path and training label path under main function
    2. Also change training file path at the end of Optimize function to store best selected features in new 
        csv file
    3. You can change the file name as per your choice
    4. In gen_plot() function => function to generate scatter plot change label file written under variable y 
    5. Then run and input Harmony Memory Consideration Rate(HMCR), Pitch Adjustment Rate(PAR) and Number of 
        iterations
    6. random_initialise() function randomly initialses the population within 0 and 2. You can change the range 
        by updating self.low and self.high parameter.
    7. reduce_features() function to reduce csv features as per requirement
    8. classification_accuracy() function to calculate classifier's accuracy for fitness value of each memeber 
        of population
    9. cal_pop_fitness() function to calculate fitness of entire population
    10. Optimize() function implements the algorithm optimally.

  Reference:
    R. Sarkhel, A. K. Saha and N. Das, "An enhanced harmony search method for Bangla handwritten character recognition using region sampling," 
    2015 IEEE 2nd International Conference on Recent Trends in Information Systems (ReTIS), Kolkata, India, 2015, pp. 325-330, doi: 10.1109/ReTIS.2015.7232899.


- To run the ABCFeatureSelection code:
    1. input the feature dimensions as per csv file number of columns
    2. enter training path, trainingLabel path, validation path and validationLabel path as per your requirements
    3. FoodSource class is to create FoodSource for every population and calculate fitness values
    4. Under FoodSource class =>
        (a) randomSolutionInitialise() function to create random solution vectors for food sources
        (b) generateSubset() function to create subset from input csv files
        (c) calculateSVMAccuracy function to calculate fitness value using SVM classifier
        (d) count1() function to check if a population has at least two 1s in solution vector
    5. ArtificialBeeColony class to optimally implement the algorithmoptimally using optimize() function
    6. Under ArtificialBeeColony class employerBeesPhase() , onlookerBeesPhase() , scoutPhase() functions to 
        implements all the three phases of ABCFeatureSelection

  Developed By:
    Soumya Nasipuri
  Supervised By: 
    Prof. Nibaran Das

  Reference:
    Kiran, M. S. (2015). The continuous artificial bee colony algorithm for binary optimization. 
    Applied Soft Computing, 33, 15-23. doi:10.1016/j.asoc.2015.04.007
