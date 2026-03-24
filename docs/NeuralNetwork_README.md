# Neural Networks: A Matrix-Notation Primer

This document explains the basic idea of a neural network and how it is trained, using vector and matrix notation.

A neural network is a parameterized function. It takes an input vector, transforms it through a sequence of layers, and produces an output such as a class score, a probability, or a numeric prediction.

## 1. Inputs and Parameters

Let the input be a vector

$$
x \in \mathbb{R}^{d_{\text{in}}}
$$

where $d_{\text{in}}$ is the number of input features.

A single fully connected layer uses a weight matrix and a bias vector:

$$
W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}, \qquad b \in \mathbb{R}^{d_{\text{out}}}
$$

The matrix $W$ determines how strongly each input feature contributes to each output feature. The bias $b$ shifts the output.

If we have a batch of inputs, we store them in a matrix

$$
X \in \mathbb{R}^{B \times d_{\text{in}}}
$$

where $B$ is the batch size.

## 2. Forward Pass

For one layer, the affine transformation is

$$
z = Wx + b
$$

or, for a whole batch,

$$
Z = XW^\top + \mathbf{1}b^\top
$$

where $\mathbf{1}$ is a column vector of ones used to broadcast the bias across the batch.

A neural network becomes nonlinear by applying an activation function:

$$
h = \phi(z)
$$

Common activations include ReLU, sigmoid, tanh, and GELU.

For a two-layer network, the computation is

$$
h^{(1)} = \phi\left(W^{(1)}x + b^{(1)}\right)
$$

$$
\hat{y} = W^{(2)}h^{(1)} + b^{(2)}
$$

Here, $\hat{y}$ is the model output. The superscripts label the layer index, not exponentiation.

### Example of a batch forward pass

For a batch $X$, a two-layer network can be written as

$$
H^{(1)} = \phi\left(X(W^{(1)})^\top + \mathbf{1}(b^{(1)})^\top\right)
$$

$$
\hat{Y} = H^{(1)}(W^{(2)})^\top + \mathbf{1}(b^{(2)})^\top
$$

This is the same computation applied to many examples in parallel.

## 3. What the Network Learns

The parameters of the model are the collection of all weights and biases:

$$
\theta = \{W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}, \dots\}
$$

These parameters are not fixed by hand. They are learned from data during training.

Intuitively:

- weights control how information moves between neurons
- biases shift the neuron responses
- activations make the model nonlinear, so it can represent complex functions

Without nonlinear activations, multiple matrix multiplications would collapse into one linear map, which would severely limit what the network can represent.

## 4. Output and Loss

The form of the output depends on the task.

### Regression

For regression, the output might be a single real number or a vector of real numbers. A common loss is mean squared error:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{B}\sum_{i=1}^{B} \|\hat{y}^{(i)} - y^{(i)}\|^2
$$

### Classification

For classification, the final layer often produces logits:

$$
z \in \mathbb{R}^{V}
$$

where $V$ is the number of classes.

These logits are turned into probabilities with **softmax**:

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{V} e^{z_j}}
$$

Then cross-entropy loss compares the probabilities to the true label:

$$
\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{V} y_k \log p_k
$$

If $y$ is one-hot, this reduces to the negative log probability assigned to the correct class.

## 5. Training

Training means adjusting the parameters $\theta$ so the loss becomes smaller on the training data.

The general loop is:

1. take a batch of inputs and targets
2. run the forward pass
3. compute the loss
4. compute gradients with backpropagation
5. update the parameters with an optimizer

### Backpropagation

Backpropagation uses the chain rule to compute derivatives of the loss with respect to every parameter:

$$
\nabla_\theta \mathcal{L}
$$

This gradient tells us how much each parameter contributed to the loss.

For example, if a parameter increases the loss, its gradient will point in the direction that should reduce the loss.

### Gradient Descent Update

The simplest update rule is gradient descent:

$$
\theta^{(k+1)} = \theta^{(k)} - \eta \nabla_\theta \mathcal{L}
$$

where:

- $\theta^{(k)}$ is the parameter set after step $k$
- $\eta$ is the learning rate
- $\nabla_\theta \mathcal{L}$ is the gradient of the loss

This says: move the parameters in the direction that reduces the loss.

### Mini-batch Training

In practice, the model is trained on mini-batches instead of one example at a time or the whole dataset at once.

If the batch contains $B$ examples, the loss is often averaged:

$$
\mathcal{L}_{\text{batch}} = \frac{1}{B}\sum_{i=1}^{B} \mathcal{L}^{(i)}
$$

Mini-batches are used because they are efficient on modern hardware and give stable gradient estimates.

### Adam and AdamW

Most modern neural networks do not use plain gradient descent. They use adaptive optimizers such as Adam or AdamW.

These methods keep running estimates of:

- the first moment of the gradient, which is like a smoothed average of the gradient
- the second moment of the gradient, which measures its magnitude

AdamW also applies weight decay in a decoupled way, which helps regularization.

## 6. Why Training Works

The key idea is that each update slightly improves the parameters, and many updates gradually make the network better at the task.

During training:

- the forward pass produces predictions
- the loss measures error
- backpropagation computes how to change the parameters
- the optimizer updates the weights and biases

Over time, the network learns internal representations that make the task easier.

## 7. Summary

A neural network is a composition of linear transformations and nonlinear activations:

$$
f(x;\theta) = f_L(\dots f_2(f_1(x)) \dots)
$$

Training adjusts the parameters $\theta$ so that the output of $f(x;\theta)$ matches the target data as well as possible.

In matrix form, the essential ideas are:

- inputs are vectors or batches of vectors
- layers are matrix multiplications plus bias terms
- activations add nonlinearity
- losses measure prediction error
- gradients and optimizers update the parameters
