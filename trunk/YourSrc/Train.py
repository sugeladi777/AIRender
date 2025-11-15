import argparse
import os
from typing import Dict, List

import numpy as np
import OpenEXR
import torch
import torch.nn as nn
import torch.optim as optim

from Dataset import LightmapDataset
from Model import LightmapModel
import Utils


class Trainer:
    """封装训练流程，负责遍历数据集"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using device: {self.device}")
        self.dataset = LightmapDataset(args.dataset)
        os.makedirs("./Parameters", exist_ok=True)
        os.makedirs("./ResultImages", exist_ok=True)

    def run(self) -> None:
        total_psnr: List[float] = []
        total_ssim: List[float] = []
        total_lpips: List[float] = []
        for sample in self.dataset:
            lightmap_tag = f"{sample['level']}_{sample['id']}"
            print(f"training lightmap {lightmap_tag}")
            metrics = self._train_one(sample)
            total_psnr.extend(metrics["psnr"])
            total_ssim.extend(metrics["ssim"])
            total_lpips.extend(metrics["lpips"])
            print(
                f"metrics of lightmap {lightmap_tag}------------\n"
                f"PSNR: {np.mean(metrics['psnr'])}\n"
                f"SSIM: {np.mean(metrics['ssim'])}\n"
                f"LPIPS: {np.mean(metrics['lpips'])}\n"
                f"Model Size: {metrics['model_size']:.2f} MB\n"
                "-----------------------------------------"
            )
        self._print_total_metrics(total_psnr, total_ssim, total_lpips)

    def _train_one(self, sample: Dict) -> Dict:
        resolution = sample["resolution"]
        model = LightmapModel(
            hidden_dim=self.args.hidden_dim,
            feature_width=resolution["width"],
            feature_height=resolution["height"],
        ).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        coords = sample["coords"].to(self.device)
        values = sample["lightmap"].to(self.device)
        coord_dim = coords.shape[1]
        total_data = torch.cat([coords, values], dim=-1)
        total_data = total_data[torch.randperm(total_data.shape[0], device=self.device)]

        batch_start = 0
        for it in range(self.args.iterations):
            batch_end = min(batch_start + self.args.batch_size, total_data.shape[0])
            batch = total_data[batch_start:batch_end]
            pred = model(batch[:, :coord_dim])
            loss = criterion(pred, batch[:, coord_dim:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_start = batch_end if batch_end < total_data.shape[0] else 0
            if batch_start == 0:
                total_data = total_data[torch.randperm(total_data.shape[0], device=self.device)]
            if (it + 1) % 1000 == 0:
                print(f"iteration {it + 1} loss: {loss.item()}")

        self._save_parameters(model, sample)
        metrics = self._evaluate_and_export(model, sample)
        metrics["model_size"] = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
        return metrics

    def _save_parameters(self, model: LightmapModel, sample: Dict) -> None:
        # 保存模型参数以便离线加载
        params = []
        for param in model.parameters():
            params.append(param.detach().cpu().numpy().ravel())
        params_array = np.concatenate(params)
        out_path = f"./Parameters/model_{sample['level']}_{sample['id']}_params.bin"
        params_array.astype(np.float32).tofile(out_path)

    def _evaluate_and_export(self, model: LightmapModel, sample: Dict) -> Dict:
        model.eval()
        coords = sample["coords"].to(self.device)
        preds: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, coords.shape[0], self.args.batch_size):
                preds.append(model(coords[start : start + self.args.batch_size]))
        pred = torch.cat(preds, dim=0)

        time_count = sample["time_count"]
        height = sample["resolution"]["height"]
        width = sample["resolution"]["width"]
        pred = pred.reshape(time_count, height, width, 3).permute(0, 3, 1, 2)
        target = sample["lightmap"].reshape(time_count, height, width, 3).permute(0, 3, 1, 2).to(self.device)
        mask_tensor = torch.from_numpy(sample["mask"]).to(self.device)

        metrics = self._compute_metrics(pred, target, mask_tensor)
        self._save_reconstructions(pred, sample)
        return metrics

    def _compute_metrics(self, pred: torch.Tensor, target: torch.Tensor, mask_tensor: torch.Tensor) -> Dict:
        psnr_list: List[float] = []
        ssim_list: List[float] = []
        lpips_list: List[float] = []
        part_size = 256
        rows = (target.shape[2] + part_size - 1) // part_size
        cols = (target.shape[3] + part_size - 1) // part_size
        time_count = target.shape[0]
        for time_idx in range(time_count):
            pred[time_idx, :, mask_tensor[time_idx] <= 0] = 0
            for i in range(rows):
                for j in range(cols):
                    sr = i * part_size
                    er = min((i + 1) * part_size, target.shape[2])
                    sc = j * part_size
                    ec = min((j + 1) * part_size, target.shape[3])
                    lightmap_part = target[[time_idx], :, sr:er, sc:ec]
                    pred_part = pred[[time_idx], :, sr:er, sc:ec]
                    mask_part = mask_tensor[time_idx, sr:er, sc:ec]
                    if (mask_part >= 127).any().item() and lightmap_part.max() != 0:
                        psnr_list.append(Utils.cal_psnr(lightmap_part, pred_part, mask_part))
                        ssim_list.append(Utils.cal_ssim(lightmap_part, pred_part))
                        lpips_list.append(Utils.cal_lpips(lightmap_part, pred_part))
        return {"psnr": psnr_list, "ssim": ssim_list, "lpips": lpips_list}

    def _save_reconstructions(self, pred: torch.Tensor, sample: Dict) -> None:
        # 保存重建结果，便于肉眼检查
        pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
        height = sample["resolution"]["height"]
        width = sample["resolution"]["width"]
        for time_idx in range(sample["time_count"]):
            path = f"./ResultImages/reconstructed_{sample['id']}_{time_idx + 1:02d}.00.exr"
            header = OpenEXR.Header(width, height)
            exr = OpenEXR.OutputFile(path, header)
            exr.writePixels({
                c: pred_np[time_idx][..., i].tobytes()
                for i, c in enumerate(["R", "G", "B"])
            })
            exr.close()

    def _print_total_metrics(self, psnr: List[float], ssim: List[float], lpips: List[float]) -> None:
        print("metrics of total data set ---------------")
        print(f"PSNR of all lightmaps: {np.mean(psnr)}")
        print(f"SSIM of all lightmaps: {np.mean(ssim)}")
        print(f"LPIPS of all lightmaps: {np.mean(lpips)}")
        print("-----------------------------------------")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default="../Data/SimpleData")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
