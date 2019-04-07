# Fashion-MNIST
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=fashion_mnist --epoch=25 --adversarial_loss_mode=gan
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=fashion_mnist --epoch=25 --adversarial_loss_mode=gan --gradient_penalty_mode=dragan
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=fashion_mnist --epoch=25 --adversarial_loss_mode=lsgan
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=fashion_mnist --epoch=50 --adversarial_loss_mode=wgan --gradient_penalty_mode=wgan-gp --n_d=5

# CelebA
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=gan
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=gan --gradient_penalty_mode=dragan
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=lsgan
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=50 --adversarial_loss_mode=wgan --gradient_penalty_mode=wgan-gp --n_d=5

# Anime
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=anime --epoch=100 --adversarial_loss_mode=gan --gradient_penalty_mode=dragan
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=anime --epoch=200 --adversarial_loss_mode=wgan --gradient_penalty_mode=wgan-gp --n_d=5
