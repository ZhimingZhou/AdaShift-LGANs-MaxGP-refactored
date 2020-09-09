# AdaShift, Lipschitz-GANs, MaxGP

This repo includes a joint implementation of AdaShift optimizer [1], LGANs [2], and MaxGP [3].

It achieves FID: 15.8800±0.4921 and Inception Score: 8.0367±0.0499 for unsupervised image generation of GANs in CIFAR-10.

The provided implementation of AdaShift (common/optimizer/AdaShift) is further developed version, which extends our discussion in [1].

That is, v_t can be any random variable that keeps the scale of the gradients and is independent of g_t.

We also provide a more efficient and elegant implementation of AdaShift, see AdaShift-new.py.

We use tensorflow 1.5 with python 3.5. 

You can refer to setting_cuda9_cudnn7_tensorflow1.5.sh to build up your environment. 

Try the code via running: python3 realdata_resnet.py. 

synthetic_real.py and synthetic_toy.py are the code we used for the synthetic experiments in [2] and [3].

[1] [AdaShift: Decorrelation and Convergence of Adaptive Learning Rate Methods](https://arxiv.org/abs/1810.00143)

[2] [Lipschitz Generative Adversarial Nets](https://arxiv.org/abs/1902.05687)

[3] [Towards Efficient and Unbiased Implementation of Lipschitz Continuity in GANs](https://arxiv.org/abs/1904.01184)
