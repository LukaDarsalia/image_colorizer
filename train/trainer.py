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

        # Generator and discriminator beta1 and beta2
        self.generator_beta1 = args.generator_beta1
        self.generator_beta2 = args.generator_beta2
        self.discriminator_beta1 = args.discriminator_beta1
        self.discriminator_beta2 = args.discriminator_beta2

        # Generator and discriminator lambda_gp and lambda_l1
        self.lambda_gp = args.lambda_gp
        self.lambda_l1 = args.lambda_l1

        # Generator and discriminator models
        self.generator = UNetGenerator(**self.generator__init__args).to(self.device)
        self.discriminator = PatchGAN70(**self.discriminator__init__args).to(self.device)
        self.critic_n = args.critic_n

        # Generator and discriminator optimizers
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.generator_lr, betas=(self.generator_beta1, self.generator_beta2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, betas=(self.discriminator_beta1, self.discriminator_beta2))

        # Train and validation dataloaders
        self.train_loader, self.val_loader = get_dataloaders(self.data_path, batch_size=self.batch_size, image_size=self.image_size)

        # Generator and discriminator schedulers which works for each step
        self.generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.generator_optimizer, T_max=self.epochs*len(self.train_loader), eta_min=self.generator_lr * 0.01)
        self.discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.discriminator_optimizer, T_max=self.epochs*len(self.train_loader), eta_min=self.discriminator_lr * 0.01)

        # Number of validation in one epoch
        self.val_freq = len(self.train_loader) // args.val_freq

        # Tensorboard logger
        self.tb_logger = TBLogger(log_dir=args.log_dir)

    def validate(self, batch_idx, end_of_epoch=False):
        if (batch_idx % self.val_freq != 0) or batch_idx<self.val_freq:
            return None
        self.generator.eval()
        self.discriminator.eval()
        g_losses = []
        l1_losses = []
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
                generated_images = self.generator(images)

                # Save sample images
                if sample_generated_images is None:
                    sample_generated_images = generated_images
                if sample_targets is None:
                    sample_targets = targets
                if sample_original_images is None:
                    sample_original_images = images

                # Calculate discriminator values
                d_real = self.discriminator(torch.cat([targets, images], dim=1))
                d_generated = self.discriminator(torch.cat([generated_images, images], dim=1))

                # Calculate l1 loss
                l1_loss = F.l1_loss(generated_images, targets)

                # Calculate generator loss
                g_loss = -d_generated.mean() + self.lambda_l1 * l1_loss

                # Calculate Wasserstein distance
                distance = d_real.mean() - d_generated.mean()

                # Calculate psnr
                psnr = PSNR(targets, generated_images)

                # Calculate ssim
                ssim = SSIM(targets, generated_images)

                # Save in lists
                g_losses.append(g_loss.item())
                l1_losses.append(l1_loss.item())
                distances.append(distance.item())
                psnrs.append(psnr)
                ssims.append(ssims)
        # Calculate average losses
        g_loss = sum(g_losses)/len(g_losses)
        l1_loss = sum(l1_losses)/len(l1_losses)
        distance = sum(distances)/len(distances)
        psnr = sum(psnrs)/len(psnrs)
        ssim = 0

        # Create generator losses
        generator_losses = GeneratorLosses(g_loss=g_loss, l1_loss=l1_loss, grad_norm=None)
        # Create validation losses
        validation_losses = ValidationLosses(generator_losses=generator_losses, psnr=psnr, ssim=ssim, sample_generated_images=sample_generated_images, sample_targets=sample_targets, sample_original_images=sample_original_images, distance=distance)
        return validation_losses
    
    def log_step(self, global_step: int, train_losses: TrainLosses, validation_losses: ValidationLosses):
        self.tb_logger.log(global_step, self.generator_optimizer, self.discriminator_optimizer, self.generator, train_losses, validation_losses)

    def train(self):
        for epoch in range(self.epochs):
            for batch_idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs}"):
                train_losses = self.train_step(batch)
                validation_losses = self.validate(batch_idx)
                global_step = epoch * len(self.train_loader) + batch_idx
                self.log_step(global_step, train_losses, validation_losses)
                self.generator_scheduler.step()
                self.discriminator_scheduler.step()

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
        self.discriminator_optimizer.zero_grad()

        # Generate images
        generated_images = self.generator(input_images)

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

        # Backpropagate
        d_loss.backward()
        self.discriminator_optimizer.step()
        # Log the losses
        return DiscriminatorLosses(d_loss=d_loss, em_loss=em_loss, gp_loss=gp_loss, grad_norm=grad_norm, distance=distance)

    def generator_step(self, input_images, real_images):
        self.generator_optimizer.zero_grad()
        generated_images = self.generator(input_images)
        conditioned_generated_images = torch.cat([generated_images, input_images], dim=1)
        generated_preds = self.discriminator(conditioned_generated_images)
        l1_loss = F.l1_loss(generated_images, real_images)
        g_loss = -generated_preds.mean() + self.lambda_l1 * l1_loss

        g_loss.backward()
        # Compute total grad norm
        total_norm = 0
        for p in self.generator.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)  # ||grad||_2
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.generator_optimizer.step()
        return GeneratorLosses(g_loss=g_loss, l1_loss=l1_loss, grad_norm=total_norm)

    def train_step(self, batch):
        images = batch['image'].to(self.device)
        targets = batch['target'].to(self.device)

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