import PDESolverByDeepLearning.hanzuliang as PDESolver
import tensorflow as tf

"""
example3: Third order differential equation
real solution: u(x)=x**7 + 2*x**5 + 3*x**3 + x**2
u'''(x) = 210*x**4 + 120*x**2 + 18; u(-1)=-5; u(0)=0; u(1)=7; x∈[-1,1]
"""

domain = [-1, 1]                                                                              #Domain
realSolution = lambda x: x**7 + 2*x**5 + 3*x**3 + x**2                                        #Real solution
#If we do not know the true solution, please input the parameter as 'None'.
#realSolution = None
n = 100                                                                                       #Divide the domain into n sample points
#If there is an exception of 'Fail rename;Input/output error',please delete the last saved model parameter file 'CKPT' and train again.
StructureOfNeuralNetwork = [1, 50, 30, 10, 1]                                                 #Neural network structure
ImplicitSchemeOfEquation = lambda x, u: tf.gradients(tf.gradients(tf.gradients(u, x)[0], x)[0], x)[0]  - 210*x**4 - 120*x**2 - 18
                                                                                              #It must be the implicit scheme of the equation,
                                                                                              #Where u is the predicted numerical solution we want to get
DirichletBCPoint = [-1, 0, 1]
DirichletBCValue = [-5, 0, 7]
numBatches = 50000                                                                             #Number of iterations
y_output = PDESolver.PDESolver(domain, n, realSolution, StructureOfNeuralNetwork,
                               ImplicitSchemeOfEquation, DirichletBCPoint, DirichletBCValue, numBatches)
print('The discrete solution predicted by Deep Learning is:')
print(y_output)
