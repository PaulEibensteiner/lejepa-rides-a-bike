# Question

I want to understand how much the Gaussian Isotropic regularization SIGReg on the latent representations in LeJepa can hinder performance.

To this end I want to implement and train three RL setups for riding a bike. 1. A simple 2 layer neural network that redicts only the variables relevant to the bike movement (bike lean and heading plus derivatives, no position etc.). 2. A simple 2 layer newtork that predicts the entire bike state (also predicts position etc.). 3. Implement a 4 layer NN that uses the SIGReg regularization on the outputs of layer two and predicts the full set of latent variables. All layers should be of latent variable dimensions. As a simplification, please keep the bike velocity fixed at all times (its neither a latent nor an action).

- I don't want to make reconstruction from pixels or similar work. All models work on the 'latent variables' they need i.e. position, speed, lean angle etc. of the bike
- I want to train the parameters by next state prediction with random action (handlebar angle) input.
- Then I want to use CEM to minimize the predicted trajectory and a perfect upright ride in a specified direction.
- To make it slightly challenging I want to use 'wind' i.e. gaussian perturbations as explained in the 2 neuron paper