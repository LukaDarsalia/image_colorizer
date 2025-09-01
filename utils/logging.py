from torch.utils.tensorboard import SummaryWriter
import torch
from utils.config import TrainLosses, ValidationLosses
from typing import Optional


class TBLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.tb = SummaryWriter(log_dir)
        self.global_step = 0

    def log_lambda(self, lambda_l1: float, lambda_mode_seeking: float, critic_n: int):
        self.tb.add_scalar("lambda/l1", lambda_l1, self.global_step)
        self.tb.add_scalar("lambda/mode_seeking", lambda_mode_seeking, self.global_step)
        self.tb.add_scalar("lambda/critic_n", critic_n, self.global_step)

    def log_lr(self, optimizer_g: torch.optim.Optimizer, optimizer_d: torch.optim.Optimizer):
        self.tb.add_scalar("lr/g", optimizer_g.param_groups[0]["lr"], self.global_step)
        self.tb.add_scalar("lr/d", optimizer_d.param_groups[0]["lr"], self.global_step)

    def log_losses(self, train_losses: TrainLosses, validation_losses: ValidationLosses):
        if validation_losses is not None:
            # log Wasserstein distance
            self.tb.add_scalar("val_loss/distance", validation_losses.distance, self.global_step)
            # log PSNR
            self.tb.add_scalar("val_loss/psnr", validation_losses.psnr, self.global_step)
            # log SSIM
            self.tb.add_scalar("val_loss/ssim", validation_losses.ssim, self.global_step)
            # log generator loss
            self.tb.add_scalar("val_loss/generator", validation_losses.generator_losses.g_loss, self.global_step)
            # log l1 loss
            self.tb.add_scalar("val_loss/l1", validation_losses.generator_losses.l1_loss, self.global_step)
            # log mode-seeking loss
            self.tb.add_scalar("val_loss/mode_seeking", -validation_losses.generator_losses.mode_seeking_loss, self.global_step)
        # log discriminator loss
        self.tb.add_scalar("train_loss/discriminator", train_losses.discriminator_losses.d_loss, self.global_step)
        # log em loss
        self.tb.add_scalar("train_loss/em", train_losses.discriminator_losses.em_loss, self.global_step)
        # log gp loss
        self.tb.add_scalar("train_loss/gp", train_losses.discriminator_losses.gp_loss, self.global_step)
        # log grad norm
        self.tb.add_scalar("train_loss/grad_norm", train_losses.discriminator_losses.grad_norm, self.global_step)
        # log train distance
        self.tb.add_scalar("train_loss/distance", train_losses.discriminator_losses.distance, self.global_step)
        # log generator grad norm
        self.tb.add_scalar("train_loss/generator_grad_norm", train_losses.generator_losses.grad_norm, self.global_step)
        # log generator l1 loss
        self.tb.add_scalar("train_loss/generator_l1", train_losses.generator_losses.l1_loss, self.global_step)
        # log generator mode-seeking loss
        self.tb.add_scalar("train_loss/generator_mode_seeking", -train_losses.generator_losses.mode_seeking_loss, self.global_step)
        # log generator g loss
        self.tb.add_scalar("train_loss/generator_g", train_losses.generator_losses.g_loss, self.global_step)

    def log_model(self, model: torch.nn.Module, sample_original_images: torch.Tensor):
        # log model architecture with only one input image
        self.tb.add_graph(model, input_to_model=sample_original_images[0:1])

    def log_images(self, images: torch.Tensor, sample_generated_images: torch.Tensor, sample_targets: torch.Tensor):
        self.tb.add_images("images/input", images, self.global_step)
        self.tb.add_images("sample/generated", sample_generated_images, self.global_step)
        self.tb.add_images("sample/targets", sample_targets, self.global_step)

    def log(self, global_step: int, optimizer_g: torch.optim.Optimizer, optimizer_d: torch.optim.Optimizer, model: torch.nn.Module, train_losses: TrainLosses, validation_losses: Optional[ValidationLosses], lambda_l1: float, lambda_mode_seeking: float, critic_n: int):
        self.global_step = global_step
        self.log_lr(optimizer_g, optimizer_d)
        self.log_lambda(lambda_l1, lambda_mode_seeking, critic_n)
        self.log_losses(train_losses, validation_losses)
        if validation_losses is not None:
            # self.log_model(model, validation_losses.sample_original_images)
            self.log_images(validation_losses.sample_original_images, validation_losses.sample_generated_images, validation_losses.sample_targets)
