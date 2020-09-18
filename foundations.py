import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import ndarray
import streamlit as st

from typing import Callable
from typing import Dict
from typing import List

np.set_printoptions(precision=4)

'''
## Foundations

This is to build intuition for how a neural network, well works!

### Here are some basic functions:
'''

with st.echo():
    def square(x: ndarray) -> ndarray:
        return np.power(x, 2)

    def leaky_relu(x: ndarray) -> ndarray:
        return np.maximum(0.2 * x, x)

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 6))  # 2 Rows, 1 Col

    input_range = np.arange(-2, 2, 0.01)
    ax[0].plot(input_range, square(input_range))
    ax[0].plot(input_range, square(input_range))
    ax[0].set_title('Square function')
    ax[0].set_xlabel('input')
    ax[0].set_ylabel('input')

    ax[1].plot(input_range, leaky_relu(input_range))
    ax[1].plot(input_range, leaky_relu(input_range))
    ax[1].set_title('"ReLU" function')
    ax[1].set_xlabel('input')
    ax[1].set_ylabel('output')
    st.pyplot(fig)

r'''
## The derivative

Derivatives are pretty important to understand how deep learning works, as it is the concept you use to perform what
is commonly called 'back propagation' which we will return to later. Taking the derivative with respect to a variable, basically means you find out how much the output of a function changes, given a tiny increase in the input.

'''
st.latex(r'''\frac{df}{du}(a) = \lim_{\Delta \rightarrow 0}\frac{fa + \Delta - fa - \Delta}{2\times \Delta}''')
st.write("Something like this:")
st.latex(r'''\frac{df}{du}(a) = \lim_{\Delta \rightarrow 0}\frac{fa + 0,001 - fa - 0,001}{2\times 0,001}''')


with st.echo():
    def deriv(func: Callable[[ndarray], ndarray], input_: ndarray, diff: float = 0.001) -> ndarray:
        return (func(input_ + diff) - func(input_ - diff)) / (2 * diff)

'''
## The chain rule
When taking the derivative of nested functions like $f_{2}(f_{1}(x)) = y$ you have to use something called the chain rule.

I won't go to deeply into that, other than to say you probably need to go read up on it, if you don't know it!

Notation wise, it looks something like this:

'''

st.latex(r'''\frac{df_{2}}{du}(x) = \frac{df_{2}}{du}(f_{1}(x))\times \frac{df_{1}}{du}(x)''')

with st.echo():
    def sigmoid(x: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-x))

    Array_Function = Callable[[ndarray], ndarray]

    Chain = List[Array_Function]

    def chain_length_2(chain: Chain, x: ndarray) -> ndarray:
        assert len(chain) == 2

        f1 = chain[0]
        f2 = chain[1]

        return f2(f1(x))

    def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
        '''
        Uses the chain rule to compute the derivative of two nested functions:
        (f2(f1(x))' = f2'(f1(x)) * f1'(x)
        '''
        assert len(chain) == 2, \
        "This function requires 'Chain' objects of length 2"
        assert input_range.ndim == 1, \
        "Function requires a 1 dimensional ndarray as input_range"

        f1 = chain[0]
        f2 = chain[1]

        #df1/dx
        df1dx = deriv(f1, input_range)

        #df2/du (f1(x))
        df2du = deriv(f2, f1(input_range))

        return df2du * df1dx

    def plot_chain(ax, chain: Chain, input_range: ndarray) -> None:
        '''
        Plots a chain function - a function made up of 
        multiple consecutive ndarray -> ndarray mappings - 
        Across the input_range
        ax: matplotlib Subplot for plotting
        '''
        assert input_range.ndim == 1, \
        "Function requires a 1 dimensional ndarray as input_range"

        output_range = chain_length_2(chain, input_range)
        ax.plot(input_range, output_range)

    def plot_chain_deriv(ax, chain: Chain, input_range: ndarray) -> ndarray:
        '''
        Uses the chain rule to plot the derivative of a function consisting of two nested 
        functions.
        ax: matplotlib Subplot for plotting
        '''
        output_range = chain_deriv_2(chain, input_range)
        ax.plot(input_range, output_range)

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 8))  # 2 Rows, 1 Col

    chain_1 = [square, sigmoid]
    chain_2 = [sigmoid, square]

    PLOT_RANGE = np.arange(-3, 3, 0.01)
    plot_chain(ax[0], chain_1, PLOT_RANGE)
    plot_chain_deriv(ax[0], chain_1, PLOT_RANGE)

    ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"])
    ax[0].set_title("Function and derivative for\n$f(x) = sigmoid(square(x))$")

    plot_chain(ax[1], chain_2, PLOT_RANGE)
    plot_chain_deriv(ax[1], chain_2, PLOT_RANGE)
    ax[1].legend(["$f(x)$", "$\\frac{df}{dx}$"])
    ax[1].set_title("Function and derivative for\n$f(x) = square(sigmoid(x))$")
    st.pyplot(fig)

r'''
The chain rule can be used on expressions with arbitrary nesting, so let's do one more
with three functions.
'''

with st.echo():
    def chain_length_3(chain: Chain, x: ndarray) -> ndarray:
        assert len(chain) == 3, \
        "Length of input 'chain' should be 3"

        f1 = chain[0]
        f2 = chain[1]
        f3 = chain[2]

        return f3(f2(f1(x)))

    def chain_deriv_3(chain: Chain, input_range: ndarray) -> ndarray:
        assert len(chain) == 3, \
        "Length of input 'chain' should be 3"

        f1 = chain[0]
        f2 = chain[1]
        f3 = chain[2]
        
        #f1(x)
        f1_of_x = f1(input_range)

        #f2(f(x))
        f2_of_x = f2(f1_of_x)

        #df3du
        df3du = deriv(f3, f2_of_x)

        #df2du
        df2du = deriv(f2, f1_of_x)

        #df1dx
        df1dx = deriv(f1, input_range)

        return df3du * df2du * df1dx

    def plot_chain(ax,
               chain: Chain, 
               input_range: ndarray,
               length: int=2) -> None:
        '''
        Plots a chain function - a function made up of 
        multiple consecutive ndarray -> ndarray mappings - across one range
        
        ax: matplotlib Subplot for plotting
        '''
        
        assert input_range.ndim == 1, \
        "Function requires a 1 dimensional ndarray as input_range"
        if length == 2:
            output_range = chain_length_2(chain, input_range)
        elif length == 3:
            output_range = chain_length_3(chain, input_range)
        ax.plot(input_range, output_range)

    def plot_chain_deriv(ax,
                     chain: Chain,
                     input_range: ndarray,
                     length: int=2) -> ndarray:
        '''
        Uses the chain rule to plot the derivative of two nested functions.
        
        ax: matplotlib Subplot for plotting
        '''

        if length == 2:
            output_range = chain_deriv_2(chain, input_range)
        elif length == 3:
            output_range = chain_deriv_3(chain, input_range)
        ax.plot(input_range, output_range)

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 8))  # 2 Rows, 1 Col

    chain_1 = [leaky_relu, square, sigmoid]
    chain_2 = [leaky_relu, sigmoid, square]

    PLOT_RANGE = np.arange(-3, 3, 0.01)
    plot_chain(ax[0], chain_1, PLOT_RANGE, length=3)
    plot_chain_deriv(ax[0], chain_1, PLOT_RANGE, length=3)

    ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"])
    ax[0].set_title("Function and derivative for\n$f(x) = sigmoid(square(leakyRrelu(x)))$")

    plot_chain(ax[1], chain_2, PLOT_RANGE, length=3)
    plot_chain_deriv(ax[1], chain_2, PLOT_RANGE, length=3)
    ax[1].legend(["$f(x)$", "$\\frac{df}{dx}$"])
    ax[1].set_title("Function and derivative for\n$f(x) = square(sigmoid(leakyRelu(x)))$");

    st.pyplot(fig)

r'''
In _chain_deriv_3_ you can think of it like a very, very, very simple neural network, only
consisting of functions, where we do a 'forward' pass, when we compute $f_{1}$ and $f_{2}$,
and then we do a 'backward' pass, taking the derivatives using the chain rule. We take the
derivatives to check how each function contributed to the final output of the composed
functions.

## Functions with multiple inputs

In deep learning, you rarely have networks only taking one input, as you can think of the
number of inputs to the network, as the number of features you think are valuable to
the prediction you want to make. In simple terms, predicting house prices, on the size
of the house might work, but you would probably want other information too.

Therefor we we have to introduce the partial derivative, where you take derivative with
respect to each individual input, and keep the rest of the inputs constant.

If we have the following function:
'''

st.latex(r'''f(x,y) = \sigma(\alpha(x,y))''')

r'''
then the partial derivative with respect to $x$ is:
'''

st.latex(r'''\frac{\partial f}{\partial x} = \frac{\partial \sigma}{\partial u}(\alpha(x,y))\times \frac{\partial \alpha}{\partial x}((x,y)) = \frac{\partial \sigma}{\partial u}(x + y)\times \frac{\partial \alpha}{\partial x}((x,y))''')

r'''
and of course $\frac{\partial f}{\partial y}$ would be the same.

Now note that:
'''

st.latex(r'''\frac{\partial \alpha}{\partial x}((x,y) = 1''')

r'''
because for every unit increase i $x$, $\alpha$ increases by one unit, same holds for $y$.
'''

with st.echo():
    def multiple_inputs_add(x: ndarray,
                            y: ndarray,
                            sigma: Array_Function) -> float:
        
        assert x.shape == y.shape

        a = x + y
        return sigma(a)

    def multiple_inputs_add_backward(x: ndarray,
                                     y: ndarray,
                                     sigma: Array_Function) -> float:
        '''
        Computes the derivative of this simple function with respect to
        both inputs
        '''
        # compute the 'forward pass'
        a = x + y

        # compute the derivative, the 'backward pass'
        dsda = deriv(sigma, a)

        dadx, dady = 1, 1

        return dsda * dadx, dsda * dady

r'''
## Functions with multiple vector inputs

In deep learning we deal with functions whose inputs are _vectors_ and _matrices_, and knowing
all the common operations like matrix multiplication, dot product and transposes, are crucial
to understanding why deep learning works.

So instead of numbers, our input could be a vector of features, like so:
'''

st.latex(r'''X =
         \begin{bmatrix}
         x_{1}  x_{2} ... x_{n} 
         \end{bmatrix}''')

r'''
One of the most common operations in neural networks, is to form a 'weighted sum' of
these features, where the weights would emphasize certain features, and de-emphasize
others. In this example, you would multiply each feature with a weight, and thus
we have another vector of weights:
'''

st.latex(r'''W =
         \begin{bmatrix}
         w_{1}  w_{2} ... w_{n} 
         \end{bmatrix}''')

r'''
In this case, we would take the dot product of the two vectors, where, because
of the rules of dot products, we would have to tranpose the matrix, as $m \times n \cdot n \times m$.
Transposing a vector just 'stands'it up.
'''

st.latex(r'''W = \begin{bmatrix}
         w_{1}\\ 
         w_{2}\\ 
         ...\\ 
         w_{n}
         \end{bmatrix}''')

r'''
and so we could define the output like so:
'''

st.latex(r'''N = v(X,W) = X \times W^T = x_{1} \times w_{1} + x_{2} \times w_{2} + ... + x_{n} \times w_{n}''')

with st.echo():
    def matmul_forward(X: ndarray, W: ndarray) -> ndarray:
        assert X.shape[1] == W.shape[0], \
        "Shapes are not correct"

        N = np.dot(X, W)

        return N

    X = np.random.rand(1,3)
    W = np.random.rand(1,3)
    WT = W.T

    dot_product = matmul_forward(X, WT)

st.write("X = {0}, WT = {1}, and dot product = {2}".format(X, WT, dot_product))

r'''
## Derivatives of functions with multiple vector inputs

So given the function from earlier, $N = v(X,W)$, how do we then calculate $\frac{\partial N}{\partial X}$
and $\frac{\partial N}{\partial X}$?

How would we even imagine to find the derivative, with respect to a matrix? Well, matrices are just numbers
arranged in a structured way, so it really means "the derivative with respect to each element of the matrix",
so looking at this:

'''

st.latex(r'''N = v(X,W) = X \times W^T = x_{1} \times w_{1} + x_{2} \times w_{2} + ... + x_{n} \times w_{n}''')

r'''
we could say that
'''

st.latex(r'''\frac{\partial v}{\partial x_{1}} = w_{1}''')
st.latex(r'''\frac{\partial v}{\partial x_{2}} = w_{2}''')
st.latex(r'''...''')
st.latex(r'''\frac{\partial v}{\partial x_{n}} = w_{n}''')

r'''
The reasoning behind this, is similar to the linear example of $x + y$, because here when $x_{n}$ changes
by $\epsilon$ units, then $N$ will change by $w_{n} \times \epsilon$ units!

And thus!
'''

st.latex(r'''\frac{\partial v}{\partial X} = W^T''')

r'''
## Vector functions and their derivatives, let's take it further!

Usually in deep learning, doing forward and backward passes involve more than one operation, and multiple
functions, some of which are vector or matrix functios, and some of which apply a function elementwise
to the matrix they receive as input.

So let us now look at an example, keeping the in mind:
'''
st.latex(r'''v(X,W) =  x_{1} \times w_{1} + x_{2} \times w_{2} + x_{3} \times w_{3}''')

r'''
We will now look at a sequence of functions, $v$ from before, and now also $\sigma$, which will look
like the following:
'''

st.latex(r'''s = f(X,W) = \sigma(v(X,W)) = \sigma(x_{1} \times w_{1} + x_{2} \times w_{2} + x_{3} \times w_{3})''')

with st.echo():
    def matrix_forward_extra(X: ndarray,
                             W: ndarray,
                             sigma: Array_Function) -> ndarray:

        assert X.shape[1] == W.shape[0]
        
        # matrix multiplication
        N = np.dot(X,W)

        # feeding the output of the matrix multiplication through sigma
        S = sigma(N)

        return S

r'''
The backward pass, should be almost like the prior example, since $f(X,V)$ is a nested function,
specifically $\sigma(v(X,V))$.
A derivative with respect to X should then be:
'''
st.latex(r'''\frac{\partial f}{\partial X} = \frac{\partial \sigma}{\partial u}(v(X,W))\times \frac{\partial v}{\partial X}((X,Y))''')

r'''
The first part is simply:
'''

st.latex(r'''\frac{\partial \sigma}{\partial u}(v(X,W)) = \frac{\partial \sigma}{\partial u}(x_{1} \times w_{1} + x_{2} \times w_{2} + x_{3} \times w_{3})''') 

r'''
which is something we can always calculate, as it's just a continous function we are evaluating with the given arguments, so let's not focus on that.
From the earlier example, we also have the second term, so we can conclude the following:
'''

st.latex(r'''\frac{\partial f}{\partial X} = \frac{\partial \sigma}{\partial u}(v(X,W))\times \frac{\partial v}{\partial X}((X,Y)) = \frac{\partial \sigma}{\partial u}(x_{1} \times w_{1} + x_{2} \times w_{2} + x_{3} \times w_{3}) \times W^T''')

with st.echo():
    def matrix_function_backward_1(X: ndarray,
                                   W: ndarray,
                                   sigma: Array_Function) -> ndarray:

        assert X.shape[1] == W.shape[0]
        N = np.dot(X, W)
        S = sigma(N)
        dSdN = deriv(sigma, N)
        dNdX = np.transpose(W, (1,0))
        return np.dot(dSdN, dNdX)

    st.write(matrix_function_backward_1(X, WT, sigmoid))

r'''
## Two 2D matrix inputs

In deep learning, and in machine learning more generally, we mostly deal with operations that take two 2D arrays as input.
One which represents a batch of data, with the features we wanna use for training, and the other the weights. So let's try
and gain an intuition for that, let's suppose that:
'''

st.latex(r'''X = \begin{matrix}
             x_{11}  \ x_{12} \ x_{13} \\ 
             x_{21}  \ x_{22} \ x_{23} \\ 
             x_{31}  \ x_{32} \ x_{33} 
            \end{matrix}''')

st.write("and:")

st.latex(r'''W = \begin{matrix}
             w_{11}  \ w_{12} \ w_{13} \\ 
             w_{21}  \ w_{22} \ w_{23} \\ 
             w_{31}  \ w_{32} \ w_{33} 
            \end{matrix}''')

r'''
The coloumns of X could correspond to a dataset in which each observation has three features and each row
is each one such observation. 






