'''
# @Time    : 2025/5/14 9:34
# @Author  : Fengcheng Ji
# @Function: 应保证范数小于1/sqrt(c)，Poincaré球双曲空间工具类
'''
import torch


class Hyperbolic:
    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 0.5, eps: float = 1e-5) -> torch.Tensor:
        """
        Möbius addition on the Poincaré ball:
        (1 + 2c <x, y> + c ||y||^2) x + (1 - c ||x||^2) y
        ------------------------------------------------
                 1 + 2c <x, y> + c^2 ||x||^2 ||y||^2
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + (c ** 2) * x2 * y2
        return num / (denom + eps)

    @staticmethod
    def mobius_mul(r, x: torch.Tensor, c: float = 0.5, eps=1e-5):
        norm_x = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
        sqrt_c = torch.sqrt(torch.tensor(c, dtype=x.dtype))
        scaled_norm = sqrt_c * norm_x
        atanh_scaled_norm = torch.atanh(scaled_norm)
        coef = torch.tanh(r * atanh_scaled_norm) / scaled_norm
        return coef * x

    @staticmethod
    def project(x: torch.Tensor, c: float = 0.5, eps: float = 1e-5) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
        return torch.tanh(torch.sqrt(torch.tensor(c)) * norm) * x / (torch.sqrt(torch.tensor(c)) * norm)

    @staticmethod
    def unproject(y: torch.Tensor, c: float = 0.5, eps: float = 1e-5) -> torch.Tensor:
        norm_y = torch.norm(y, dim=-1, keepdim=True).clamp_min(eps)
        sqrt_c = torch.sqrt(torch.tensor(c, dtype=y.dtype, device=y.device))
        r = (1.0 / sqrt_c) * torch.atanh(sqrt_c * norm_y)
        return r * y / norm_y

    @staticmethod
    def dist(x: torch.Tensor, y: torch.Tensor, c: float = 0.5, eps: float = 1e-5) -> torch.Tensor:
        """
        Hyperbolic distance in the Poincaré ball:
        d(x, y) = arccosh(1 + 2c ||x - y||^2 / ((1 - c ||x||^2)(1 - c ||y||^2)))
        """
        x2 = torch.sum(x * x, dim=-1)
        y2 = torch.sum(y * y, dim=-1)
        diff2 = torch.sum((x - y) ** 2, dim=-1)
        num = 2 * c * diff2
        denom = (1 - c * x2) * (1 - c * y2) + eps
        arg = 1 + num / denom
        arg = torch.clamp(arg, min=1 + eps)
        return torch.acosh(arg)

    @staticmethod
    def poincare_to_klein(x, c=0.5):
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        denominator = 1 + c * x_norm_sq
        x_klein = 2 * x / denominator
        return x_klein

    @staticmethod
    def klein_to_poincare(x_klein, c=0.5):
        x_norm_sq = torch.sum(x_klein ** 2, dim=-1, keepdim=True)
        denominator = 1 + torch.sqrt(1 - c * x_norm_sq)
        x_poincare = x_klein / denominator

        # 超出边界时映射回双曲空间（留1%余量）
        R = (1.0 / torch.sqrt(torch.tensor(c))) * 0.99
        norm = torch.norm(x_poincare, dim=-1)
        if norm > R:
            x_poincare = Hyperbolic.project(x_poincare, c)
        return x_poincare

    @staticmethod
    def einstein_midpoint(x_klein, c=0.5, dim=0):
        x_norm_sq = torch.sum(x_klein ** 2, dim=-1, keepdim=True)
        gamma = 1 / torch.sqrt(1 - c * x_norm_sq + 1e-10)
        weighted_sum = torch.sum(gamma * x_klein, dim=dim)
        gamma_sum = torch.sum(gamma, dim=dim)
        midpoint = weighted_sum / (gamma_sum + 1e-10)
        if torch.isnan(midpoint).any():
            print()
        return midpoint

    @staticmethod
    def compute_hyperbolic_prototype(embeddings, c=0.5, dim=0):
        bach_klein = Hyperbolic.poincare_to_klein(embeddings, c)
        midpoint_klein = Hyperbolic.einstein_midpoint(bach_klein, c, dim=dim)
        prototype = Hyperbolic.klein_to_poincare(midpoint_klein, c)
        return prototype
