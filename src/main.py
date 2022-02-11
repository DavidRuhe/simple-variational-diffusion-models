import os
import math
import copy
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils import data
from tqdm import tqdm
import argparse


def gaussian_kl(mu1, var1, mu2=None, var2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if var2 is None:
        var2 = torch.ones_like(mu1)

    return 0.5 * (torch.log(var2 / var1) + (var1 + (mu1 - mu2).pow(2)) / var2 - 1)


def standard_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    """E.g., num=36, divisor=32 -> returns [32, 4]"""
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1), nn.GroupNorm(groups, dim_out), Mish()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)

        if self.mlp is not None:
            h += self.mlp(time_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(self.hidden_dim, dim, 1)

    def forward(self, x):
        b, _, h, w = x.shape

        # Get queries (Q), keys (K), values (V).
        qkv = self.to_qkv(x)

        # Reshape to get attention heads and their dimensionality for Q, K, V.
        qkv = qkv.reshape(b, 3, self.heads, self.dim_head, h * w)

        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        k = k.softmax(dim=-1)

        context = torch.matmul(k, v.permute(0, 1, 3, 2))  # V.T K
        out = torch.matmul(context.permute(0, 1, 3, 2), q)  # Q.T VT

        out = out.reshape(b, -1, h, w)
        return self.to_out(out)


class UNet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        device="cuda",
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        if out_dim is None:
            out_dim = channels

        self.final_conv = nn.Sequential(Block(dim, dim), nn.Conv2d(dim, out_dim, 1))

        self.to(device)
        self.device = device

    def forward(self, x, time):
        t = self.time_mlp(time) if self.time_mlp is not None else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


class PositiveLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.softplus = nn.Softplus()

    def forward(self, input: torch.Tensor):  # type: ignore
        return input @ self.softplus(self.weight) + self.softplus(self.bias)


class SNRNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_min = nn.Parameter(torch.tensor(-10.0))
        self.gamma_max = nn.Parameter(torch.tensor(20.0))

        self.softplus = nn.Softplus()

    def forward(self, t: torch.Tensor):  # type: ignore

        # Add start and endpoints 0 and 1.
        t = torch.cat([torch.tensor([0.0, 1.0], device=t.device), t])
        l1 = self.l1(t[:, None])
        l2 = torch.sigmoid(self.l2(l1))
        l3 = torch.squeeze(l1 + self.l3(l2), dim=-1)

        s0, s1, sched = l3[0], l3[1], l3[2:]

        norm_nlogsnr = (sched - s0) / (s1 - s0)

        nlogsnr = self.gamma_min + self.softplus(self.gamma_max) * norm_nlogsnr
        return -nlogsnr, norm_nlogsnr


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels=3,
        timesteps=1000,
        device="cuda",
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = timesteps
        self.snrnet = SNRNetwork()

        self.to(device)
        self.device = device

    def p_zs_zt(self, zt, t, s, clip_denoised: bool):

        logsnr_t, norm_nlogsnr_t = self.snrnet(t)
        logsnr_s, norm_nlogsnr_s = self.snrnet(s)

        alpha_sq_t = torch.sigmoid(logsnr_t)[:, None, None, None]
        alpha_sq_s = torch.sigmoid(logsnr_s)[:, None, None, None]

        alpha_t = alpha_sq_t.sqrt()
        alpha_s = alpha_sq_s.sqrt()

        sigmasq_t = 1 - alpha_sq_t
        sigmasq_s = 1 - alpha_sq_s

        alpha_sq_tbars = alpha_sq_t / alpha_sq_s
        sigmasq_tbars = sigmasq_t - alpha_sq_tbars * sigmasq_s

        alpha_tbars = alpha_t / alpha_s
        sigma_tbars = torch.sqrt(sigmasq_tbars)

        e_hat = self.denoise_fn(zt, norm_nlogsnr_t)
        sigma_t = sigmasq_t.sqrt()
        if clip_denoised:
            e_hat.clamp_((zt - alpha_t) / sigma_t, (zt + alpha_t) / sigma_t)

        mu_zs_zt = (
            1 / alpha_tbars * zt - sigmasq_tbars / (alpha_tbars * sigma_t) * e_hat
        )
        sigmasq_zs_zt = sigmasq_tbars * (sigmasq_s / sigmasq_t)

        return mu_zs_zt, sigmasq_zs_zt

    @torch.no_grad()
    def p_zs_zt_sample(self, zt, t, s, clip_denoised=True):

        batch_size = len(zt)

        mu_zs_zt, var_zs_zt = self.p_zs_zt(zt=zt, t=t, s=s, clip_denoised=clip_denoised)
        noise = torch.randn_like(zt)
        # No noise when s == 0:
        nonzero_mask = (1 - (s == 0).float()).reshape(
            batch_size, *((1,) * (len(zt.shape) - 1))
        )
        return mu_zs_zt + nonzero_mask * var_zs_zt.sqrt() * noise

    @torch.no_grad()
    def sample_loop(self, shape):

        batch_size = shape[0]
        z = torch.randn(shape, device=self.device)

        timesteps = torch.linspace(0, 1, self.num_timesteps)

        for i in tqdm(
            reversed(range(1, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):

            t = torch.full((batch_size,), timesteps[i], device=z.device)
            s = torch.full((batch_size,), timesteps[i - 1], device=z.device)

            z = self.p_zs_zt_sample(z, t=t, s=s)

        logsnr_0, _ = self.snrnet(torch.zeros((batch_size,), device=z.device))
        alpha_sq_0 = torch.sigmoid(logsnr_0)[:, None, None, None]
        sigmasq_0 = 1 - alpha_sq_0
        sigma_0 = sigmasq_0.sqrt()

        # Get p(x | z_0)
        d = 1 / 255
        X = torch.linspace(-1, 1, 256)
        p_x_z0 = []
        for x in X:
            if x == -1:
                p = standard_cdf((x + d - z) / sigma_0)
            elif x == 1:
                p = 1 - standard_cdf((x - d - z) / sigma_0)
            else:
                p = standard_cdf((x + d - z) / sigma_0) - standard_cdf((x - d - z) / sigma_0)
            p_x_z0.append(p)

        p_x_z0 = torch.stack(p_x_z0, dim=1)

        # Sample
        cumsum = torch.cumsum(p_x_z0, dim=1)
        r = torch.rand_like(cumsum)
        x = torch.max(cumsum > r, dim=1)[1]

        return x

    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size = self.image_size
        channels = self.channels
        return self.sample_loop((batch_size, channels, image_size, image_size))

    def q_zt_zs(self, zs, t, s=None):

        if s is None:
            s = torch.zeros_like(t)

        logsnr_t, norm_nlogsnr_t = self.snrnet(t)
        logsnr_s, norm_nlogsnr_s = self.snrnet(s)

        alpha_sq_t = torch.sigmoid(logsnr_t)[:, None, None, None]
        alpha_sq_s = torch.sigmoid(logsnr_s)[:, None, None, None]

        alpha_t = alpha_sq_t.sqrt()
        alpha_s = alpha_sq_s.sqrt()

        sigmasq_t = 1 - alpha_sq_t
        sigmasq_s = 1 - alpha_sq_s

        alpha_sq_tbars = alpha_sq_t / alpha_sq_s
        sigmasq_tbars = sigmasq_t - alpha_sq_tbars * sigmasq_s

        alpha_tbars = alpha_t / alpha_s
        sigma_tbars = torch.sqrt(sigmasq_tbars)

        return alpha_tbars * zs, sigma_tbars, norm_nlogsnr_t

    def prior_loss(self, x, batch_size):
        logsnr_1, _ = self.snrnet(torch.ones((batch_size,), device=x.device))
        alpha_sq_1 = torch.sigmoid(logsnr_1)[:, None, None, None]
        sigmasq_1 = 1 - alpha_sq_1
        alpha_1 = alpha_sq_1.sqrt()
        mu_1 = alpha_1 * x
        return gaussian_kl(mu_1, sigmasq_1).sum() / batch_size

    def data_likelihood(self, x, batch_size):
        logsnr_0, _ = self.snrnet(torch.zeros((1,), device=x.device))
        alpha_sq_0 = torch.sigmoid(logsnr_0)[:, None, None, None].repeat(*x.shape)
        sigmasq_0 = 1 - alpha_sq_0
        alpha_0 = alpha_sq_0.sqrt()
        mu_0 = alpha_0 * x
        sigma_0 = sigmasq_0.sqrt()
        d = 1 / 255
        p_x_z0 = standard_cdf((x + d - mu_0) / sigma_0) - standard_cdf((x - d - mu_0) / sigma_0)
        p_x_z0[x == 1] = 1 - standard_cdf((x[x == 1] - d - mu_0[x == 1]) / sigma_0[x == 1])
        p_x_z0[x == -1] = standard_cdf((x[x == -1] + d - mu_0[x == -1]) / sigma_0[x == -1])
        nll = -torch.log(p_x_z0)
        return nll.sum() / batch_size

    def get_loss(self, x):

        batch_size = len(x)

        e = torch.randn_like(x)
        t = torch.rand((batch_size,), device=self.device)

        mu_zt_zs, sigma_zt_zs, norm_nlogsnr_t = self.q_zt_zs(zs=x, t=t)

        zt = mu_zt_zs + sigma_zt_zs * e

        e_hat = self.denoise_fn(zt.detach(), norm_nlogsnr_t)

        t.requires_grad_(True)
        logsnr_t, _ = self.snrnet(t)
        logsnr_t_grad = autograd.grad(logsnr_t.sum(), t)[0]

        diffusion_loss = (
            -0.5
            * logsnr_t_grad
            * F.mse_loss(e, e_hat, reduction="none").sum(dim=(1, 2, 3))
        )
        diffusion_loss = diffusion_loss.sum() / batch_size
        prior_loss = self.prior_loss(x, batch_size)
        data_loss = self.data_likelihood(x, batch_size)

        loss = diffusion_loss + prior_loss + data_loss

        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = (
            *x.shape,
            x.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"

        return self.get_loss(x)


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        *,
        ema_decay=0.995,
        image_size=32,
        train_batch_size=32,
        train_lr=2e-5,
        train_total_steps=2**19,
        gradient_accumulate_every=2,
        step_start_ema=2048,
        update_ema_every=16,
        save_and_sample_every=1024,
        results_folder="./results/",
        save_n_images=36,
        save_models=False,
        device="cuda",
    ):
        super().__init__()
        self.model = diffusion_model
        self.device = device

        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_total_steps = train_total_steps
        self.save_and_sample_every = save_and_sample_every
        self.save_models = save_models
        self.save_n_images = save_n_images

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),  # Rescale to [-1, 1]
            ]
        )

        self.dataset = datasets.CIFAR10(
            "./data/", transform=self.transform, download=True
        )
        self.dataloader = cycle(
            data.DataLoader(
                self.dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True
            )
        )
        self.optimizer = optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
        else:
            self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"))
        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

    def save_and_sample(self):

        milestone = self.step // self.save_and_sample_every
        batches = num_to_groups(self.save_n_images, self.batch_size)
        all_images_list = list(
            map(lambda n: self.ema_model.sample(batch_size=n), batches)
        )
        all_images = torch.cat(all_images_list, dim=0)
        # all_images = (all_images + 1) * 0.5  # Normalize
        utils.save_image(
            all_images / 255,
            os.path.join(self.results_folder, f"sample-{milestone}.png"),
            nrow=int(math.sqrt(self.save_n_images)),
        )
        if self.save_models:
            self.save(milestone)

    def train(self):
        while self.step < self.train_total_steps:
            for i in range(self.gradient_accumulate_every):

                data, _ = next(self.dataloader)
                data = data.to(self.device)
                loss = self.model(data)
                (loss / self.gradient_accumulate_every).backward()

                print(
                    f"[{self.step} / {self.train_total_steps}] - Loss: {loss.item():.2f}"
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_and_sample_every == 0:
                self.save_and_sample()

            self.step += 1


def main(args):
    unet = UNet(dim=args.base_dim, device=args.device)

    diffusion = GaussianDiffusion(
        unet,
        image_size=args.image_size,
        timesteps=args.diffusion_steps,
        device=args.device,
    )

    trainer = Trainer(
        diffusion,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_total_steps=args.train_total_steps,
        gradient_accumulate_every=args.gradient_accumulation,
        ema_decay=args.ema_decay,
        results_folder=args.results_folder,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dim", default=64, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=1024)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_total_steps", type=int, default=2**19)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--results_folder", type=str, default="./results/")

    args = parser.parse_args()
    main(args)
