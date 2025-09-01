from models.generator_model import UNetGenerator
from models.discriminator import PatchGAN70, PatchGAN34
import torch
from loaders.data_loader import get_dataloaders
from utils.config import TrainingArgs
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from utils.config import DiscriminatorLosses, GeneratorLosses, TrainLosses, ValidationLosses
from utils.metrics import PSNR, SSIM
from utils.logging import TBLogger
from tqdm import tqdm
import math
import os

class CosineScalarScheduler:
    """
    Cosine anneal a scalar from start -> end over T_max steps.
    Call .get() to read current value, .step() after each G update.
    """
    def __init__(self, start: float, end: float, T_max: int):
        self.start = float(start)
        self.end = float(end)
        self.T_max = max(int(T_max), 1)
        self.t = 0

    def get(self) -> float:
        # lr(t) = end + 0.5*(start-end)*(1 + cos(pi * t / T_max))
        cos_term = 0.5 * (1.0 + math.cos(math.pi * self.t / self.T_max))
        return self.end + (self.start - self.end) * cos_term

    def step(self):
        if self.t < self.T_max:
            self.t += 1


class MyTrainer:
    def __init__(self, args: TrainingArgs):
        self.args = args

        # Number of epochs
        self.epochs = args.epochs

        # Path to the data
        self.data_path = args.data_path
        # Batch size
        self.batch_size = args.batch_size
        # Image size
        self.image_size = args.image_size
        # Device to use
        self.device = args.device

        # Generator and discriminator __init__ arguments
        self.generator__init__args = args.generator__init__args
        self.discriminator__init__args = args.discriminator__init__args

        # Generator and discriminator learning rates
        self.generator_lr = args.generator_lr
        self.discriminator_lr = args.discriminator_lr
        self.accumulation_steps = args.accumulation_steps

        # Generator and discriminator beta1 and beta2
        self.generator_beta1 = args.generator_beta1
        self.generator_beta2 = args.generator_beta2
        self.discriminator_beta1 = args.discriminator_beta1
        self.discriminator_beta2 = args.discriminator_beta2

        # Generator and discriminator lambda_gp and lambda_l1
        self.lambda_gp = args.lambda_gp
        self.lambda_l1 = args.lambda_l1
        self.lambda_mode_seeking = args.lambda_mode_seeking
        self.lambda_l1_low = args.lambda_l1_low
        self.lambda_mode_seeking_low = args.lambda_mode_seeking_low

        # Generator and discriminator models
        self.generator = UNetGenerator(**self.generator__init__args).to(self.device)
        self.discriminator = PatchGAN70(**self.discriminator__init__args).to(self.device)
        self.critic_n = args.critic_n
        self.original_critic_n = args.critic_n

        # Generator and discriminator optimizers
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.generator_lr, betas=(self.generator_beta1, self.generator_beta2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, betas=(self.discriminator_beta1, self.discriminator_beta2))

        # Train and validation dataloaders
        self.train_loader, self.val_loader = get_dataloaders(self.data_path, batch_size=self.batch_size, image_size=self.image_size)

        # Generator and discriminator schedulers which works for each step
        self.generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.generator_optimizer, T_max=self.epochs*len(self.train_loader)//self.accumulation_steps, eta_min=self.generator_lr * 0.1)
        self.discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.discriminator_optimizer, T_max=self.epochs*len(self.train_loader)*self.critic_n//self.accumulation_steps, eta_min=self.discriminator_lr * 0.1)
        self.scheduler_l1 = CosineScalarScheduler(self.lambda_l1, self.lambda_l1_low, self.epochs*len(self.train_loader)//self.accumulation_steps)
        self.scheduler_mode_seeking = CosineScalarScheduler(self.lambda_mode_seeking, self.lambda_mode_seeking_low, self.epochs*len(self.train_loader)//self.accumulation_steps)

        # Number of validation in one epoch
        self.val_freq = len(self.train_loader) // args.val_freq

        # Tensorboard logger
        self.tb_logger = TBLogger(log_dir=args.log_dir)

        self.g_accumulation_count = 0
        self.d_accumulation_count = 0

    def validate(self, batch_idx):
        if (batch_idx % self.val_freq != 0) or batch_idx<self.val_freq:
            return None
        self.generator.eval()
        self.discriminator.eval()
        g_losses = []
        l1_losses = []
        mode_seeking_losses = []
        distances = []
        psnrs = []
        ssims = []
        sample_generated_images = None
        sample_targets = None
        sample_original_images = None
        with torch.no_grad():
            for batch in self.val_loader:
                # Load images and targets
                images = batch['image'].to(self.device)
                targets = batch['target'].to(self.device)

                # Generate images
                z = self.generate_z(images.shape[0], 128)
                generated_images = self.generator(images, z)
                z_2 = self.generate_z(images.shape[0], 128)
                generated_images_2 = self.generator(images, z_2)

                # Save sample images
                if sample_generated_images is None:
                    sample_generated_images = torch.cat([generated_images, generated_images_2], dim=0)
                if sample_targets is None:
                    sample_targets = targets
                if sample_original_images is None:
                    sample_original_images = images

                # Calculate discriminator values
                d_real = self.discriminator(torch.cat([targets, images], dim=1))
                d_generated = self.discriminator(torch.cat([generated_images, images], dim=1))

                # Calculate l1 loss
                l1_loss = F.l1_loss(generated_images, targets)

                # Calculate mode-seeking loss
                num = F.l1_loss(generated_images, generated_images_2, reduction="mean")
                denom = F.l1_loss(z, z_2, reduction="mean")
                mode_seeking_loss = -(num / denom)

                # Calculate generator loss
                g_loss = -d_generated.mean() + self.lambda_l1 * l1_loss + self.lambda_mode_seeking * mode_seeking_loss

                # Calculate Wasserstein distance
                distance = d_real.mean() - d_generated.mean()

                # Calculate psnr
                psnr = PSNR(targets, generated_images)

                # Calculate ssim
                ssim = SSIM(targets, generated_images)

                # Save in lists
                g_losses.append(g_loss.item())
                l1_losses.append(l1_loss.item())
                mode_seeking_losses.append(mode_seeking_loss.item())
                distances.append(distance.item())
                psnrs.append(psnr)
                ssims.append(ssims)
        # Calculate average losses
        g_loss = sum(g_losses)/len(g_losses)
        l1_loss = sum(l1_losses)/len(l1_losses)
        mode_seeking_loss = sum(mode_seeking_losses)/len(mode_seeking_losses)
        distance = sum(distances)/len(distances)
        psnr = sum(psnrs)/len(psnrs)
        ssim = 0

        # Create generator losses
        generator_losses = GeneratorLosses(g_loss=g_loss, l1_loss=l1_loss, mode_seeking_loss=mode_seeking_loss, grad_norm=None)
        # Create validation losses
        validation_losses = ValidationLosses(generator_losses=generator_losses, psnr=psnr, ssim=ssim, sample_generated_images=sample_generated_images, sample_targets=sample_targets, sample_original_images=sample_original_images, distance=distance)
        return validation_losses
    
    def log_step(self, global_step: int, train_losses: TrainLosses, validation_losses: ValidationLosses):
        self.tb_logger.log(global_step, self.generator_optimizer, self.discriminator_optimizer, self.generator, train_losses, validation_losses, self.lambda_l1, self.lambda_mode_seeking, self.critic_n)

    def save_model(self, epoch):
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(self.generator.state_dict(), f'checkpoints/generator_epoch_{epoch}.pth')
        torch.save(self.discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch}.pth')

    def load_model(self, epoch):
        self.generator.load_state_dict(torch.load(f'checkpoints/generator_epoch_{epoch}.pth'))
        self.discriminator.load_state_dict(torch.load(f'checkpoints/discriminator_epoch_{epoch}.pth'))

    def train(self):
        for epoch in range(self.epochs):
            for batch_idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs}"):
                global_step = epoch * len(self.train_loader) + batch_idx

                if global_step % 500 == 0 or global_step < 25:
                    self.critic_n = 100
                else:
                    self.critic_n = self.original_critic_n
                train_losses = self.train_step(batch)
                validation_losses = self.validate(batch_idx)
                self.log_step(global_step, train_losses, validation_losses)
            self.save_model(epoch)

    def generate_z(self, batch_size, z_dim):
        return torch.randn(batch_size, z_dim, device=self.device)

    def gradient_penalty(self, real_data, generated_data, input_data):
        generated_data = generated_data.detach()
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(self.device)

        # Calculate probability of interpolated examples
        conditioned_interpolated = torch.cat([interpolated, input_data], dim=1)
        prob_interpolated = self.discriminator(conditioned_interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean(), gradients.norm(2, dim=1).mean()

    def discriminator_step(self, input_images, real_images):
        # Zero out the gradients
        if self.d_accumulation_count == 0:
            self.discriminator_optimizer.zero_grad()

        # Generate images
        with torch.no_grad():
            z = self.generate_z(input_images.shape[0], 128)
            generated_images = self.generator(input_images, z)

        # Concatenate input images with real and generated images
        conditioned_real_images = torch.cat([real_images, input_images], dim=1)

        # Predict real images
        real_preds = self.discriminator(conditioned_real_images)

        # Concatenate input images with generated images
        conditioned_generated_images = torch.cat([generated_images, input_images], dim=1)

        # Predict generated images
        generated_preds = self.discriminator(conditioned_generated_images)

        # Calculate distance
        distance = real_preds.mean() - generated_preds.mean()

        # Calculate gradient penalty
        gp_loss, grad_norm = self.gradient_penalty(real_images, generated_images, input_images)

        # Calculate Wasserstein loss
        em_loss = generated_preds.mean() - real_preds.mean()

        # Calculate total loss
        d_loss = em_loss + self.lambda_gp * gp_loss
        d_loss = d_loss / self.accumulation_steps
        self.d_accumulation_count += 1

        # Backpropagate
        d_loss.backward()
        if self.d_accumulation_count >= self.accumulation_steps:
            self.discriminator_optimizer.step()
            self.discriminator_scheduler.step()
            self.d_accumulation_count = 0
        # Log the losses
        return DiscriminatorLosses(d_loss=d_loss, em_loss=em_loss, gp_loss=gp_loss, grad_norm=grad_norm, distance=distance)

    def generator_step(self, input_images, real_images):
        if self.g_accumulation_count == 0:
            self.generator_optimizer.zero_grad()

        z = self.generate_z(input_images.shape[0], 128)
        generated_images = self.generator(input_images, z)

        z_2 = self.generate_z(input_images.shape[0], 128)
        generated_images_2 = self.generator(input_images, z_2)

        conditioned_generated_images = torch.cat([generated_images, input_images], dim=1)
        generated_preds = self.discriminator(conditioned_generated_images)
        l1_loss = F.l1_loss(generated_images, real_images)

        # Mode-Seeking Loss
        num = F.l1_loss(generated_images, generated_images_2, reduction="mean")
        denom = F.l1_loss(z, z_2, reduction="mean")
        mode_seeking_loss = -(num / denom)

        g_loss = -generated_preds.mean() + self.lambda_l1 * l1_loss + self.lambda_mode_seeking * mode_seeking_loss
        g_loss = g_loss / self.accumulation_steps
        g_loss.backward()
        self.g_accumulation_count += 1
        # Compute total grad norm
        total_norm = 0
        for p in self.generator.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)  # ||grad||_2
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if self.g_accumulation_count >= self.accumulation_steps:
            self.generator_optimizer.step()
            self.generator_scheduler.step()
            self.scheduler_l1.step(); self.lambda_l1 = self.scheduler_l1.get()
            self.scheduler_mode_seeking.step(); self.lambda_mode_seeking = self.scheduler_mode_seeking.get()
            self.g_accumulation_count = 0
        return GeneratorLosses(g_loss=g_loss, l1_loss=l1_loss, mode_seeking_loss=mode_seeking_loss, grad_norm=total_norm)

    def train_step(self, batch):
        images = batch['image'].to(self.device)
        targets = batch['target'].to(self.device)

        # Switch to training mode
        self.generator.train()
        self.discriminator.train()

        # Discriminate real and fake images
        em_losses = []
        gp_losses = []
        grad_norms = []
        distances = []
        d_losses = []
        for _ in range(self.critic_n):
            d_loss = self.discriminator_step(images, targets)
            d_losses.append(d_loss.d_loss)
            em_losses.append(d_loss.em_loss)
            gp_losses.append(d_loss.gp_loss)
            grad_norms.append(d_loss.grad_norm)
            distances.append(d_loss.distance)
        discriminator_losses = DiscriminatorLosses(d_loss=sum(d_losses)/len(d_losses), em_loss=sum(em_losses)/len(em_losses), gp_loss=sum(gp_losses)/len(gp_losses), grad_norm=sum(grad_norms)/len(grad_norms), distance=sum(distances)/len(distances))
        # Train generator
        generator_losses = self.generator_step(images, targets)
        # Create train losses
        train_losses = TrainLosses(discriminator_losses=discriminator_losses, generator_losses=generator_losses)
        return train_losses