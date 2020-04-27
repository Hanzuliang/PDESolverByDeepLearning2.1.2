import PDESolverByDeepLearning.hanzuliang as PDESolver
import tensorflow as tf

'''
example1: First order differential equation
real solution: u(x)= 5*x**3 + x**2 + 2*x + 1
u'(x) = 15*x**2 + 2*x + 2;  u(-1) = -5;  x∈[-1,1]
'''

domain = [-1, 1]                                                                    #Domain
realSolution = lambda x: 5*x**3 + x**2 + 2*x + 1                                    #Real solution
#If we do not know the true solution, please input the parameter as 'None'.
#realSolution = None
n = 100                                                                             #Divide the domain into n sample points
#If there is an exception of 'Fail rename;Input/output error',please delete the last saved model parameter file 'CKPT' and train again.
StructureOfNeuralNetwork = [1, 10, 1]                                               #Neural network structure
ImplicitSchemeOfEquation = lambda x, u: tf.gradients(u, x)[0] - 15*x**2 - 2*x - 2   #It must be the implicit scheme of the equation,
                                                                                    #Where u is the predicted numerical solution we want to get
DirichletBCPoint = [-1]                                                             #Dirichlet boundary conditions
DirichletBCValue = [-5]
numBatches = 30000                                                                   #Number of iterations
y_output = PDESolver.PDESolver(domain, n, realSolution, StructureOfNeuralNetwork,
                               ImplicitSchemeOfEquation, DirichletBCPoint, DirichletBCValue, numBatches)
print('The discrete solution predicted by Deep Learning is:')
print(y_output)
print(y_output[0])
