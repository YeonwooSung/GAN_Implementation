from transgan import TransGAN


tgan = TransGAN(
    z_dim=100,
    output_gim=32*32
)

z = torch.rand(100)  # random noise
pred = tgan(z)
