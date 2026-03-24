# nanoGPT (Character-Level) - Mathematical Overview

This project trains a character-level autoregressive Transformer (GPT-style) in PyTorch and then samples text from the learned distribution.

For the general neural-network basics used throughout this document, including matrix notation, forward passes, losses, and parameter updates, see [NeuralNetwork_README.md](NeuralNetwork_README.md). This GPT document builds on that foundation and applies it to token embeddings, attention, and autoregressive generation.

## Language Modeling Objective

Given a tokenized character sequence

$$
(x_1, x_2, \dots, x_T), \quad x_t \in \{1,\dots,V\}
$$

where $V$ is the vocabulary size and $T$ the sequence length, the model learns the joint probability via the autoregressive factorization:

$$
P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

Here, $x_{<t}$ means "all tokens before position $t$", i.e.:

$$
x_{<t} = (x_1, x_2, \dots, x_{t-1})
$$

So at each step, the model predicts the next character using only past characters, never future ones.

This factorization turns one hard joint distribution over full strings into a sequence of easier conditional distributions. Instead of directly learning $P(x_1,\dots,x_T)$ as one object, the model learns local predictive rules like "given this prefix, what character is most likely next?" and multiplies those probabilities.

So the model is always solving next-token prediction from a growing context. In generation mode, this is exactly what happens token by token: predict one token from the current prefix, append it, and repeat.

Training maximizes log-likelihood (equivalently minimizes negative log-likelihood / cross-entropy):

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i} \log P_\theta\left(x_t^{(i)} \mid x_{<t}^{(i)}\right)
$$

In code, each window of length `block_size` is paired with a one-step-shifted target window, so the model predicts next characters at every position.

## Generation (Inference)
### 1. Token and Position Representations

Each input token index $x_t$ is mapped to an embedding vector:

$$
e_t = E[x_t] \in \mathbb{R}^{d}
$$

An embedding vector is a learned dense numeric representation of a discrete symbol (here, a character). Instead of treating a character as just an ID, the model maps it to a point in a $d$-dimensional space where geometric relationships can encode useful patterns. During training, these vectors are updated so characters used in similar contexts tend to develop related representations.

Simple example: suppose $d=4$ and your vocabulary includes `a`, `b`, and `c`. The embedding table may learn something like

$$
E[a] = [0.8, -0.1, 0.3, 1.2],\quad
E[b] = [0.7, -0.2, 0.4, 1.0],\quad
E[c] = [-0.6, 0.9, -0.1, -0.8]
$$

These numbers are not manually chosen features; they are optimized during training. In this toy example, `a` and `b` are closer to each other than to `c`, so the model can treat them as more contextually similar when predicting the next token.

where $d = n\_embd$.

A learned positional embedding is added:

$$
p_t = P[t] \in \mathbb{R}^{d}
$$

so that

$$
h_t^{(0)} = e_t + p_t\in \mathbb{R}^{d}
$$

$h_t^{(0)}$ is the representation of one token position $t$ at layer 0 (input to the first Transformer block), while

$$
H^{(0)} \in \mathbb{R}^{T \times d}
$$

is the full sequence representation obtained by stacking all token vectors row-wise:

$$
H^{(0)} =
\begin{bmatrix}
(h_1^{(0)})^\top \\
(h_2^{(0)})^\top \\
\vdots \\
(h_T^{(0)})^\top
\end{bmatrix}
$$

So the model input matrix for a sequence of length $T$ is $H^{(0)} \in \mathbb{R}^{T \times d}$.

### 2. Causal Self-Attention

Purpose of one attention head: learn one specific relation pattern between tokens (for example, local continuity, punctuation dependencies, or longer-range agreement). A head computes relevance with dot products because they are an efficient similarity measure in the learned feature space; [softmax](NeuralNetwork_README.md#Classification) then turns raw similarities into normalized, differentiable weights (a probability-like distribution over positions); and the weighted sum over $V$ produces a context vector that keeps useful information while suppressing irrelevant tokens.

For one attention head:

$$
Q = H W_Q, \quad K = H W_K, \quad V = H W_V
$$

with $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h}$ and head size $d_h$.

Interpretation of these matrices:

- $Q$ (queries): what each position is looking for in other positions.
- $K$ (keys): what each position offers as a matchable descriptor.
- $V$ (values): the information content each position contributes if selected.

Row-wise view at token position $i$:

$$
q_i = h_i W_Q, \quad k_j = h_j W_K, \quad v_j = h_j W_V
$$

Calculate Similarity:

The similarity score $s_{ij} = q_i \cdot k_j$ says how relevant token $j$ is when updating token $i$. The scaling by $\sqrt{d_h}$ keeps the scores from becoming too large when the head dimension grows, which makes softmax numerically more stable and helps gradients behave better. Then softmax over all $j$ gives attention weights, and the new representation for position $i$ is a weighted sum of value vectors $v_j$.

$$
S = \frac{QK^\top}{\sqrt{d_h}}
$$

A causal mask $M$ enforces autoregressive dependence (no look-ahead):

$$
S_{ij} = -\infty \;\text{for}\; j > i
$$

Attention weights then is smoothed:

$$
A = \operatorname{softmax}(S)
$$

Dropout is a regularization technique that randomly sets some attention weights to zero during training, which helps prevent the model from relying too heavily on a few paths and reduces overfitting.

$$
A_{ij} = 0 \;\text{for random}\; i,j
$$

Finally the output of the header is given by
$$
O = A V
$$

### 3. Multi-Head Attention

With $H$ heads:

$$
O^{(1)}, \dots, O^{(H)} \in \mathbb{R}^{T \times d_h}
$$

are concatenated:

$$
O_{\text{cat}} = \operatorname{Concat}(O^{(1)},\dots,O^{(H)}) \in \mathbb{R}^{T \times d}
$$

and projected:

$$
\operatorname{MHA}(H) = O_{\text{cat}} W_O,
$$

followed by dropout.

### 4. Feed-Forward Network (Position-Wise MLP)

The purpose of the FFN is to transform each token representation independently, adding nonlinearity and feature mixing after attention has gathered context from other positions.

Each position independently applies:

$$
\operatorname{FFN}(h) = W_2\,\operatorname{GELU}(W_1 h + b_1) + b_2,
$$

with expansion from $d$ to $4d$, then back to $d$, plus dropout. GELU (Gaussian Error Linear Unit) is a smooth activation function that keeps small negative values partially active instead of shutting them off completely, which often works well in Transformer models because it preserves more gradient information than a hard threshold like ReLU.

### 5. Transformer Block Dynamics (Pre-Norm)

The purpose of a Transformer block is to take the current token representations, mix information across positions with attention, refine each position with the feed-forward network, and do it in a way that trains stably thanks to residual connections and pre-layer normalization.

Each block uses pre-layer normalization and residual connections:

$$
\tilde{H} = H + \operatorname{MHA}(\operatorname{LN}(H)),
$$
$$
H' = \tilde{H} + \operatorname{FFN}(\operatorname{LN}(\tilde{H})).
$$

Stacking `n_layer` blocks yields the final hidden states.

### 6. Output Distribution and Loss

After final layer norm:

$$
Z_t = W_{\text{lm}} h_t + b_{\text{lm}} \in \mathbb{R}^{V}
$$

are logits over vocabulary. Logits are the raw, unnormalized scores the model assigns to each vocabulary token before softmax converts them into probabilities.

Token probabilities:

$$
P(x_{t+1}=k \mid x_{\le t}) = \frac{\exp(Z_{t,k})}{\sum_{j=1}^{V}\exp(Z_{t,j})}.
$$

Training uses cross-entropy between logits and ground-truth next token IDs.

### 7. Inference

At inference, only the latest context of length at most `block_size` is used: for $x$ being an encoded token,

$$
\text{context} = x_{t-B+1:t} with \quad B=\text{block\_size}.
$$

logits $z$ are the model's raw, unnormalized scores for each token in the vocabulary. They are not probabilities yet; softmax turns them into a distribution that can be sampled from. Then, for the next-token logits $z$:

1. Temperature scaling (with $\tau = \text{temperature}$):

$$
z' = z / \tau,
$$

2. Optional top-k truncation: keep only the largest $k$ logits, set others to $-\infty$.

3. Convert to probabilities with softmax and sample:

$$
probs = \operatorname{softmax}(z').
$$

4. Given the probabilities for each token in the vocabulary tensor, sample it to get the next token

$$
x_{t+1} \sim \operatorname{Categorical}(probs).
$$

This repeats autoregressively for the requested number of tokens.

## Training Procedure

This document describes how the model is trained from text windows to parameter updates. For the model architecture and generation procedure, see [GPT_inference.md](GPT_inference.md).

### 1. Supervised Next-Token Learning

Training uses teacher forcing: for every input window, the model predicts the next token at each position.

If a batch contains token sequences of length $T$, the model sees prefixes $x_{<t}$ and is trained to predict $x_t$ at each position. The loss is the average negative log-likelihood / cross-entropy over the batch:

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i} \log P_\theta\left(x_t^{(i)} \mid x_{<t}^{(i)}\right).
$$

Here $\theta$ denotes all trainable parameters of the model: the token embedding table, positional embedding table, attention projection matrices, attention output projection, feed-forward weights and biases, normalization scale and bias parameters, and the final output projection.

### 2. Mini-Batch Construction

The training script samples random contiguous windows from the text corpus. Each window of length `block_size` is paired with a one-step-shifted target window, so the model learns next-token prediction at every position in the sequence.

This creates many training examples from the same corpus and keeps computation efficient because the windows can be processed in parallel.

### 3. Parameter Updates with AdamW

As previously stated, the trainable parameters are collected into one set $\theta$ so they can be updated together by the optimizer. If $\theta^{(k)}$ denotes the parameter set after update step $k$, one optimization step can be written as

$$
	\theta^{(k+1)} = \theta^{(k)} - \eta \, \Delta_{\text{AdamW}}\bigl(\nabla_{\theta^{(k)}} \mathcal{L}\bigr),
$$

where $\eta$ is the learning rate and $\Delta_{\text{AdamW}}$ denotes the AdamW update rule with decoupled weight decay.

In practice, AdamW keeps running estimates of the first and second moments of the gradients, then applies the parameter update using those smoothed statistics rather than the raw gradient alone.

### 4. Training and Validation Evaluation

The script periodically estimates train and validation loss by averaging over `eval_iters` mini-batches.

Training loss measures how well the model fits the seen data. Validation loss estimates how well the model generalizes to held-out text. Comparing the two helps detect overfitting and makes it easier to choose checkpoints.

The main training loop alternates between forward passes, loss computation, backpropagation, optimizer updates, and periodic evaluation.

