from __future__ import annotations
"""
通用训练器封装：
将原 scripts/train.py 中的核心训练循环抽象为 Trainer.fit，便于后续拓展（如加入验证钩子、混合精度等）。
"""

from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        criterion: nn.Module,
        device: torch.device,
        out_dir: Path,
        image_size: tuple[int, int],
        save_every: int = 10,
        config: Optional[Dict] = None,
        clip_grad: float = 0.0,
        model_type: str = 'grid_mlp',
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.out_dir = out_dir
        self.H, self.W = image_size
        self.save_every = save_every
        # 可选的保存到 checkpoint 的配置字典（用于推理时恢复模型超参/模式等）
        self.config = config if config is not None else {}
        # 梯度裁剪阈值（L2 norm）；<=0 则不裁剪
        self.clip_grad = float(clip_grad)
        self.model_type = str(model_type)

        self.best_loss: Optional[float] = None

    def train_one_epoch(self, loader, epoch: int = None, total_epochs: int = None) -> float:
        self.model.train()
        running_loss = 0.0
        count = 0
        if epoch is not None and total_epochs is not None:
            print(f"\n==== Epoch {epoch}/{total_epochs} ====")
        pbar = tqdm(loader, desc=f'Training Epoch {epoch}' if epoch is not None else 'Training')
        for batch in pbar:
            xy, t_feat, rgb = batch
            non_block = True if self.device.type == 'cuda' else False
            xy = xy.to(self.device, non_blocking=non_block)
            t_feat = t_feat.to(self.device, non_blocking=non_block)
            rgb = rgb.to(self.device, non_blocking=non_block)

            pred = self.model(xy, t_feat)
            loss = self.criterion(pred, rgb)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # 计算当前梯度范数并在需要时进行裁剪
            if self.clip_grad is not None and self.clip_grad > 0.0:
                # clip_grad_norm_ 返回裁剪前的总范数
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            else:
                # 计算未裁剪的梯度范数用于诊断
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

            self.optimizer.step()

            bs = xy.shape[0]
            running_loss += loss.item() * bs
            count += bs
            pbar.set_postfix(loss=running_loss / max(1, count))

        epoch_loss = running_loss / max(1, count)
        print(f"Epoch {epoch} finished. Avg loss: {epoch_loss:.6f}")
        return epoch_loss

    def _save_ckpt(self, tag: str, epoch: int):
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'image_size': (self.H, self.W),
            'model_type': self.model_type,
            'config': self.config,
        }
        torch.save(ckpt, self.out_dir / f'{tag}.ckpt')

    def fit(self, loader, start_epoch: int, epochs: int):
        for epoch in range(start_epoch, epochs + 1):
            epoch_loss = self.train_one_epoch(loader, epoch=epoch, total_epochs=epochs)

            # 调整学习率
            try:
                if self.scheduler is not None:
                    # ReduceLROnPlateau 需要监控值
                    if hasattr(self.scheduler, 'step'):
                        self.scheduler.step(epoch_loss)
            except Exception:
                pass

            current_lr = self.optimizer.param_groups[0].get('lr', None)
            if current_lr is not None:
                print(f"Epoch {epoch} lr={current_lr:.6g}")

            # 周期性保存
            if (epoch % self.save_every == 0) or (epoch == epochs):
                self._save_ckpt('last', epoch)

            # 保存最优
            if self.best_loss is None or epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self._save_ckpt('best', epoch)
