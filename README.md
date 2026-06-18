# Lejepa riding a bike

based on LeWorldmodel ([implementation](https://github.com/lucas-maes/le-wm), [paper](https://arxiv.org/pdf/2603.19312v1)) and "It takes two Neurons to ride a bicycle" ([implementation](https://vhartmann.com/two-neurons-bike/), [paper](https://paradise.caltech.edu/~cook/papers/TwoNeurons.pdf)).

**This Repo is in a non-working state**

## Theory





Given are heading $\theta$, speed $s$, lean $\gamma$, handlebar angle $\alpha$, torque $\tau$, desired heading $\theta_d$

Control calculation i.e. formula that defines the required torque:

$$
\tau = c_2\cdot \big(\underbrace{ \overbrace{\sigma(c_1 \cdot(\theta_d - \theta))}^{\text{required heading diff}} - \gamma}_{\text{required lean velocity}}\big) - \underbrace{c_3 \dot \gamma}_{\text{current lean velocity}}
$$

„Torque is proportional to the required difference in lean velocity“ (1)

„Required lean is proportional to required heading difference (capped to not crash the bike)“

(1) is equivalent to

$$
\Delta \gamma_{t+1} - \Delta \gamma_t  = c_\tau\tau_t
$$

Which is equivalent to 

$$
\Delta \gamma_{t+1} = \Delta \gamma_t + c_\tau\tau_t
$$

with an additional term for gravity we have

$$
\Delta \gamma_{t+1} = \Delta \gamma_t + c_\tau\tau_t + c_{\gamma} \gamma_t
$$

And by second order Taylor expansion we can say

$$
\gamma_{t+1} = \gamma_t + \Delta \gamma_t + 1/2\cdot(c_\tau \tau_t + c_\gamma \gamma_t)
$$


From the physical motion of a bicycle we also know that the rate of change of the heading is proportional to the handlebar angle

$$
\Delta \theta = \underbrace{\frac v L}_{c_\theta} \cdot \tan(\alpha)
$$

since we assume that the handlebar angle is always near zero, we can approximate this linearly

$$
\Delta \theta = c_\theta \alpha
$$

and therefore

$$
\theta_{t+1} = \theta_t + c_\theta \alpha
$$

Also, the torque on the handlebar should be proportional to the acceleration of the angle

$$
\Delta \alpha_{t+1} - \Delta \alpha_t = c_\alpha \cdot \tau \\
\Delta \alpha_{t+1} = \Delta \alpha_t + c_{\alpha} \cdot \tau \\
\alpha_{t+1} = \alpha_t + \Delta \alpha_t + 1/2\cdot c_{\alpha} \cdot \tau
$$

So in total we have (the 1/2 can be factored in to the learnt constants $c$)

$$
\pmatrix{\theta'  \\ \gamma' \\ \Delta \gamma' \\ \alpha' \\ \Delta \alpha'} =
\pmatrix{
1 & 0 & 0 & c_\theta & 0 & 0 \\
0 & 1 & 1 & 0 & 0 & c_\tau \\
0 & 0 & 1 & 0 & 0 & c_\tau \\
0 & 0 & 0 & 1 & 1 & c_\alpha \\
0 & 0 & 0 & 0 & 1 & c_\alpha 
}
\pmatrix{\theta \\ \gamma \\ \Delta \gamma \\ \alpha \\ \Delta \alpha \\ \tau }
$$




- 

