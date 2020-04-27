1. This operator is suitable for solving the problem of one-dimensional n-order differential equation with Dirichlet boundary conditions.


2. Input parameter description:
import tensorflow as tf
import PDESolverByDeepLearning.hanzuliang as PDESolver

def PDESolver(domain, n, realSolution, StructureOfNeuralNetwork, ImplicitSchemeOfEquation,
              DirichletBCPoint, DirichletBCValue, numBatches):
    '''
    :param domain: The domain of the definition of the equation.
    :param n: Discretize the domain into n grid points.
    :param realSolution: The true solution of the equation. If you do not know the true solution, please input the parameter as 'None'.
    :param StructureOfNeuralNetwork = [n1,...,ni,...,no]:
            Number of layers of neural network is len(StructureOfNeuralNetwork)
            n1: Number of neurons in input layer, whose value is equal to the number of variables in the equation
            ni: The number of neurons in the ith hidden layer, whose value is selected according to the
                complexity and oscillation of the equation.
            no: The number of neurons in the output layer must be 1.
    :param ImplicitSchemeOfEquation: Implicit scheme of differential equation.
    :param DirichletBCPoint = [x1,x2,x3,...,xn]
    :param DirichletBCValue = [u1,u2,u3,...,un]
    :param numBatches: Number of training iterations.
    :return y_output: The numerical solution predicted by Deep Learning is returned in the form of row vector.
    '''
 

3. Case: 
Case1.   First order differential equation
             real solution: u(x)= 5*x**3 + x**2 + 2*x + 1
             u'(x) = 15*x**2 + 2*x + 2;   u(-1) = -5;   x¡Ê[-1,1]

#Code of Case1
import tensorflow as tf
import PDESolverByDeepLearning.hanzuliang as PDESolver

domain = [-1, 1]                                                                                #Domain
realSolution = lambda x: 5*x**3 + x**2 + 2*x + 1                                   		#Real solution
#If we do not know the true solution, please input the parameter as 'None'.
#realSolution = None
n = 100                                                                             		#Divide the domain into n sample points
#If there is an exception of 'Fail rename;Input/output error',please delete the last saved model parameter file 'CKPT' and train again.
StructureOfNeuralNetwork = [1, 10, 1]                                                   	#Neural network structure
ImplicitSchemeOfEquation = lambda x, u: tf.gradients(u, x)[0] - 15*x**2 - 2*x - 2   #It must be the implicit scheme of the equation
DirichletBCPoint = [-1]                                                         	    	#Dirichlet boundary conditions
DirichletBCValue = [-5]	
numBatches = 30000                                                                	        #Number of iterations
y_output = PDESolver.PDESolver(domain, n, realSolution, StructureOfNeuralNetwork,
                               ImplicitSchemeOfEquation, DirichletBCPoint, DirichletBCValue, numBatches)
print('The discrete solution predicted by Deep Learning is:')
print(y_output)
print(y_output[0])


Case2.     Second order differential equation
	real solution: u(x)=x**5
	u''(x)=20*x**3; u(-1)=-1; u(1)=1; x¡Ê[-1,1]

#Code of Case2
import tensorflow as tf
import PDESolverByDeepLearning.hanzuliang as PDESolver

domain = [-1, 1]                                                                                #Domain
realSolution = lambda x: x**5                                                                   #Real solution
#If we do not know the true solution, please input the parameter as 'None'.
#realSolution = None
n = 100                                                                                                                            #Divide the domain into n sample points
#If there is an exception of 'Fail rename;Input/output error',please delete the last saved model parameter file 'CKPT' and train again.
StructureOfNeuralNetwork = [1, 10, 5, 2, 1]                                                     #Neural network structure
ImplicitSchemeOfEquation = lambda x, u: tf.gradients(tf.gradients(u, x)[0], x)[0] - 20*x**3     #The implicit scheme of the equation
DirichletBCPoint = [-1, 1]
DirichletBCValue = [-1, 1]
numBatches = 1000                                                                             	#Number of iterations
y_output = PDESolver.PDESolver(domain, n, realSolution, StructureOfNeuralNetwork,
                               ImplicitSchemeOfEquation, DirichletBCPoint, DirichletBCValue, numBatches)
print('The discrete solution predicted by Deep Learning is:')
print(y_output)


Case3.	Third order differential equation
	real solution: u(x)=x**7 + 2*x**5 + 3*x**3 + x**2
	u'''(x) = 210*x**4 + 120*x**2 + 18; u(-1)=-5; u(0)=0; u(1)=7; x¡Ê[-1,1]

#Code of Case3
import tensorflow as tf
import PDESolverByDeepLearning.hanzuliang as PDESolver

domain = [-1, 1]                                                                                #Domain
realSolution = lambda x: x**7 + 2*x**5 + 3*x**3 + x**2                         		        #Real solution
#If we do not know the true solution, please input the parameter as 'None'.
#realSolution = None
n = 100                                                                                         #Divide the domain into n sample points
#If there is an exception of 'Fail rename;Input/output error',please delete the last saved model parameter file 'CKPT' and train again.
StructureOfNeuralNetwork = [1, 50, 30, 10, 1]                                       		#Neural network structure
ImplicitSchemeOfEquation = lambda x, u: tf.gradients(tf.gradients(tf.gradients(u, x)[0], x)[0], x)[0]  - 210*x**4 - 120*x**2 - 18
DirichletBCPoint = [-1, 0, 1]
DirichletBCValue = [-5, 0, 7]
numBatches = 5000                                                                               #Number of iterations
y_output = PDESolver.PDESolver(domain, n, realSolution, StructureOfNeuralNetwork,
                               ImplicitSchemeOfEquation, DirichletBCPoint, DirichletBCValue, numBatches)
print('The discrete solution predicted by Deep Learning is:')
print(y_output)
