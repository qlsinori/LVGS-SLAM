# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Scheduler Classes"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from nerfstudio.configs.base_config import InstantiateConfig


@dataclass
class SchedulerConfig(InstantiateConfig):
    """Basic scheduler config"""

    _target: Type = field(default_factory=lambda: Scheduler)
    """target class to instantiate"""


class Scheduler:
    """Base scheduler"""

    config: SchedulerConfig

    def __init__(self, config: SchedulerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        """Abstract method that returns a scheduler object.

        Args:
            optimizer: The optimizer to use.
            lr_init: The initial learning rate.
        Returns:
            The scheduler object.
        """


@dataclass
class MultiStepSchedulerConfig(SchedulerConfig):
    """Config for multi step scheduler where lr decays by gamma every milestone"""

    _target: Type = field(default_factory=lambda: MultiStepScheduler)
    """target class to instantiate"""
    max_steps: int = 1000000
    """The maximum number of steps."""
    gamma: float = 0.33
    """The learning rate decay factor."""
    milestones: Tuple[int, ...] = (500000, 750000, 900000)
    """The milestone steps at which to decay the learning rate."""


class MultiStepScheduler(Scheduler):
    """Multi step scheduler where lr decays by gamma every milestone"""

    config: MultiStepSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.config.milestones,
            gamma=self.config.gamma,
        )
        return scheduler


@dataclass
class ExponentialDecaySchedulerConfig(SchedulerConfig):
    """Config for exponential decay scheduler with warmup"""

    _target: Type = field(default_factory=lambda: ExponentialDecayScheduler)
    """target class to instantiate"""
    lr_pre_warmup: float = 1e-8
    """Learning rate before warmup."""
    lr_final: Optional[float] = None
    """Final learning rate. If not provided, it will be set to the optimizers learning rate."""
    warmup_steps: int = 0
    """Number of warmup steps."""
    max_steps: int = 100000
    """The maximum number of steps."""
    ramp: Literal["linear", "cosine"] = "cosine"
    """The ramp function to use during the warmup."""


class ExponentialDecayScheduler(Scheduler):
    """Exponential decay scheduler with linear warmup. Scheduler first ramps up to `lr_init` in `warmup_steps`
    steps, then exponentially decays to `lr_final` in `max_steps` steps.
    """

    config: ExponentialDecaySchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        if self.config.lr_final is None:
            lr_final = lr_init
        else:
            lr_final = self.config.lr_final

        def func(step):
            if step < self.config.warmup_steps:
                if self.config.ramp == "cosine":
                    lr = self.config.lr_pre_warmup + (lr_init - self.config.lr_pre_warmup) * np.sin(
                        0.5 * np.pi * np.clip(step / self.config.warmup_steps, 0, 1)
                    )
                else:
                    lr = (
                        self.config.lr_pre_warmup
                        + (lr_init - self.config.lr_pre_warmup) * step / self.config.warmup_steps
                    )
            else:
                t = np.clip(
                    (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps), 0, 1
                )
                lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init  # divided by lr_init because the multiplier is with the initial learning rate

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler


class slam_Scheduler:


    def __init__(self, warm_up_end=5000, learning_rate_alpha=0.05,
                 max_steps=50000, lr_init=0.0016 * 6, lr_final=1.6E-6,
                 lr_delay_mult=0.01, post=True):
        self.warm_up_end = warm_up_end
        self.learning_rate_alpha = learning_rate_alpha
        self.max_steps = max_steps
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_mult = lr_delay_mult
        self.post = post
    def update_mean_lr(self, iteration,optimizer,step,writer,camera):
        """Learning rate scheduling per step"""
        param_group = optimizer["means"].param_groups[0]
        assert len(optimizer["means"].param_groups) == 1


        lr = helper(
            iteration,
            lr_init=self.lr_init,
            lr_final=self.lr_final,
            lr_delay_mult=self.lr_delay_mult,
            max_steps=self.max_steps,
        )

        param_group["lr"] = lr
        writer.put_scalar(name=f"learning_rate/means", scalar=lr, step=step)
        return lr

        assert 1==0

    def update_post_lr(self, iteration,optimizer,step,writer,camera,post):
        """Learning rate scheduling per step"""
        param_group = optimizer["fields"].param_groups[0]
        assert len(optimizer["fields"].param_groups) == 1


        if camera.metadata["scene_index"] > 0 or not(post):
            param_group["lr"] = 0.0
            post = False
        writer.put_scalar(name=f"learning_rate/post", scalar=param_group["lr"], step=step)
        return post


def helper(
    step, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        # Disable this parameter
        return 0.0
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp


@dataclass
class CosineDecaySchedulerConfig(SchedulerConfig):
    """Config for cosine decay schedule"""

    _target: Type = field(default_factory=lambda: CosineDecayScheduler)
    """target class to instantiate"""
    warm_up_end: int = 5000
    """Iteration number where warmp ends"""
    learning_rate_alpha: float = 0.05
    """Learning rate alpha value"""
    max_steps: int = 300000
    """The maximum number of steps."""


class CosineDecayScheduler(Scheduler):
    """Cosine decay scheduler with linear warmup"""

    config: CosineDecaySchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        def func(step):
            if step < self.config.warm_up_end:
                learning_factor = step / self.config.warm_up_end
            else:
                alpha = self.config.learning_rate_alpha
                progress = (step - self.config.warm_up_end) / (self.config.max_steps - self.config.warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            return learning_factor

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler
