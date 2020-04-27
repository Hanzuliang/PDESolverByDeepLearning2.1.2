import PDESolverByDeepLearning.hanzuliang as PDESolver
import tensorflow as tf

'''
example2: Second order differential equation
real solution: u(x)=x**5
u''(x)=20*x**3; u(-1)=-1; u(1)=1; x∈[-1,1]
'''

domain = [-1, 1]                                                                              #Domain
realSolution = lambda x: x**5                                                                 #Real solution
#If we do not know the true solution, please input the parameter as 'None'.
#realSolution = None
n = 100                                                                                       #Divide the domain into n sample points
#If there is an exception of 'Fail rename;Input/output error',please delete the last saved model parameter file 'CKPT' and train again.
StructureOfNeuralNetwork = [1, 10, 5, 2, 1]                                                   #Neural network structure
ImplicitSchemeOfEquation = lambda x, u: tf.gradients(tf.gradients(u, x)[0], x)[0] - 20*x**3   #It must be the implicit scheme of the equation,
                                                                                              #Where u is the predicted numerical solution we want to get
DirichletBCPoint = [-1, 1]
DirichletBCValue = [-1, 1]
numBatches = 30000                                                                             #Number of iterations
y_output = PDESolver.PDESolver(domain, n, realSolution, StructureOfNeuralNetwork,
                               ImplicitSchemeOfEquation, DirichletBCPoint, DirichletBCValue, numBatches)
print('The discrete solution predicted by Deep Learning is:')
print(y_output)
