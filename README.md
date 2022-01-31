# Simple Variational Diffusion Models

> Note! This is still work in progress. If you lke to contribute, please leave a pull request.

A simple (~600 line) PyTorch implementation of the "[Variational Diffusion Models](https://arxiv.org/abs/2107.00630)" paper by Kingma et al.
Much of it was inspired by the [lucidrains](https://github.com/lucidrains/denoising-diffusion-pytorch) and [revsic](https://github.com/revsic/jax-variational-diffwave) repositories.

[Most of the paper and code are documented on my blog post](https://davidruhe.github.io/notes/Variational-Diffusion-Models). If any part of either the code or the blog is not well-documented. Please let me know!

# Requirements
- Python 3.9
- CUDA 11.5

# Installation
1. `git clone `
2. `cd simple-variational-diffusion-models`
3. `source activate_py3.sh`
4. `python`

# To Do
1. Network adjustments reported in Appendix.
2. Fourier features.
3. Reproduce paper results.