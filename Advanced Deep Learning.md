

# Advanced Deep Learning

DIKU - ADL

## Lectures

### Week 1 - Neural Networks

- Neural Network basics
  - mathematics: chain rule, matrix multiplication
  - typology: feed-forward vs. recurrent NNs; supervised vs. unsupervised learning
  - elements: neurons, activation functions, layers, loss functions, backpropagation, gradient descent
  - training: optimization, regularization, early stopping, normalization

- Deep learning:
  - shallow NN vs. deep learning 
  - multiple levels of non-linear transformation
    - to learn more abstract representations

- Others
  - ML Ops
  - W&B
      

### Week 2 - CNNs, Unet, segmentation
- CNNs
    - components: convolutional layers (producing feature maps), pooling layers (interleaved), standard NN on the top
      - convolution layer
        - convolution
          - 1D kernel, 2D kernel (different from cross-correlation)
            - kernel flipping
        - ideas behind
          - sparse interaction
          - parameter sharing
          - equivariance
      - pooling layer
        - compute on feature map 
        - padding, stride
      - tricks: ensemble, dropout, residual learning, augmentation and invariance, batch normalization
- U-nets 
  - https://arxiv.org/pdf/1505.04597.pdf
  - ![Alt text](image.png)

### Week 3 - Generative models
- autoencoder
- VAE
- GAN
- diffusion (week 8)

### Week 4 - RNNs, transformers (sequential data processing)
- Word2Vec
- RNN: Vanila, Stacked, bi-directional, GRU, LSTM
  - ELMo
- Transformers
  - BERT
  - ViT


### Week 5 - Fairness
- metrics design
  - statistical parity
  - performance parity
  - accuracy parity


### Week 6 - In-context learning
- In-weight learning vs. In-context learning

### Week 7 - Interpretability, transparency, trustworthiness


### Week 8 - Denoising diffusions
- GAN framework: generator - discriminator
  - two losses: generator loss, discriminator loss
- Markov chain
- ELBO and KL divergence

## Assignments 

### Assignment 1 (Week 1+2, group) - NN, CNNs, U-nets, MLOps

- feedforward model
- CNN
- U-nets and MLOps (using W&B)

### Assignment 2 (Week 3, individual) - Autoencoders, CNNs

- autoencoder
- PCA
- coupling CNN

### Assignment 3 (Week 4+5, group) - RNNs

- RNN and LSTMs

### Assignment 4 (Week 6, individual) - SAM, Prompting

- SAM
- Prompt in segmentation

## Resources

### References
Check slids for more infomation.

### Study Notes

1. general:
  -  https://github.com/lijin-THU/notes-machine-learning
  -  https://github.com/dcetin/eth-cs-notes/tree/master 
  - https://github.com/navalnica/dl_sys_course_notes 
2. understanding optimization:
  - https://kunyuan827.github.io/teaching/
3. for more: 
  - https://github.com/PKUFlyingPig/CMU10-714
