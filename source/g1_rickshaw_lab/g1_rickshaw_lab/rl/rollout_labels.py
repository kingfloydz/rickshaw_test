"""Stateful gait and cart-lag labels for on-policy S1 rollouts."""

from __future__ import annotations

import math

import torch


class RolloutLabelTracker:
    """Derive guide-defined auxiliary labels from contact and velocity histories."""

    def __init__(
        self,
        num_envs: int,
        *,
        policy_dt: float = 0.02,
        lag_history: int = 61,
        maximum_lag_steps: int = 25,
        maximum_frequency_hz: float = 4.0,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_envs <= 0 or policy_dt <= 0.0:
            raise ValueError("num_envs and policy_dt must be positive")
        if lag_history < 3 or maximum_lag_steps < 1 or maximum_lag_steps >= lag_history:
            raise ValueError("cart-lag window must be longer than its maximum lag")
        self.num_envs = num_envs
        self.policy_dt = policy_dt
        self.maximum_lag_steps = maximum_lag_steps
        self.maximum_frequency_hz = maximum_frequency_hz
        self.device = torch.device(device)
        self.step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.previous_contact = torch.zeros(num_envs, 2, dtype=torch.bool, device=self.device)
        self.last_rise = torch.full((num_envs, 2), -1, dtype=torch.long, device=self.device)
        self.period_steps = torch.zeros(num_envs, 2, dtype=torch.long, device=self.device)
        self.last_side = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.v_ref_history = torch.zeros(num_envs, lag_history, device=self.device)
        self.cart_speed_history = torch.zeros_like(self.v_ref_history)
        self.history_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def reset(self, dones: torch.Tensor) -> None:
        ids = torch.nonzero(dones.reshape(-1).to(device=self.device, dtype=torch.bool), as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        self.step[ids] = 0
        self.previous_contact[ids] = False
        self.last_rise[ids] = -1
        self.period_steps[ids] = 0
        self.last_side[ids] = 0
        self.v_ref_history[ids] = 0.0
        self.cart_speed_history[ids] = 0.0
        self.history_count[ids] = 0

    def _cart_lag(self, active_speed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enough = self.history_count >= self.v_ref_history.shape[1]
        ref = self.v_ref_history - self.v_ref_history.mean(dim=1, keepdim=True)
        cart = self.cart_speed_history - self.cart_speed_history.mean(dim=1, keepdim=True)
        correlations = []
        for lag in range(self.maximum_lag_steps + 1):
            ref_slice = ref[:, : ref.shape[1] - lag] if lag else ref
            cart_slice = cart[:, lag:] if lag else cart
            numerator = torch.sum(ref_slice * cart_slice, dim=1)
            denominator = torch.sqrt(
                torch.sum(ref_slice.square(), dim=1) * torch.sum(cart_slice.square(), dim=1)
            ).clamp_min(1.0e-8)
            correlations.append(numerator / denominator)
        correlation = torch.stack(correlations, dim=1)
        lag = torch.argmax(correlation, dim=1)
        variable = (ref.square().mean(dim=1) > 1.0e-6) & (cart.square().mean(dim=1) > 1.0e-6)
        mask = enough & variable & active_speed
        normalized = lag.to(ref.dtype).unsqueeze(-1) / float(self.maximum_lag_steps)
        return normalized, mask.unsqueeze(-1)

    def update(
        self,
        contact: torch.Tensor,
        v_ref: torch.Tensor,
        cart_speed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if contact.shape != (self.num_envs, 2) or contact.dtype != torch.bool:
            raise ValueError(f"contact must be bool [{self.num_envs},2]")
        if v_ref.shape != (self.num_envs,) or cart_speed.shape != (self.num_envs,):
            raise ValueError(f"v_ref/cart_speed must be [{self.num_envs}]")
        contact = contact.to(self.device)
        v_ref = v_ref.to(self.device)
        cart_speed = cart_speed.to(self.device)
        rising = contact & ~self.previous_contact
        for side in range(2):
            side_rising = rising[:, side]
            previous = self.last_rise[:, side]
            interval = self.step - previous
            valid_interval = side_rising & (previous >= 0) & (interval > 0)
            self.period_steps[:, side] = torch.where(valid_interval, interval, self.period_steps[:, side])
            self.last_rise[:, side] = torch.where(side_rising, self.step, previous)
            self.last_side = torch.where(side_rising, torch.full_like(self.last_side, side), self.last_side)

        env_ids = torch.arange(self.num_envs, device=self.device)
        period = self.period_steps[env_ids, self.last_side]
        last = self.last_rise[env_ids, self.last_side]
        active_speed = v_ref.abs() >= 0.1
        gait_mask = (period > 0) & (last >= 0) & active_speed
        phase = 2.0 * math.pi * (self.step - last).to(v_ref.dtype) / period.clamp_min(1).to(v_ref.dtype)
        phase = torch.remainder(phase, 2.0 * math.pi)
        phase_target = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
        frequency_hz = 1.0 / (period.clamp_min(1).to(v_ref.dtype) * self.policy_dt)
        frequency_target = (frequency_hz / self.maximum_frequency_hz).clamp(0.0, 1.0).unsqueeze(-1)

        self.v_ref_history[:, :-1] = self.v_ref_history[:, 1:].clone()
        self.cart_speed_history[:, :-1] = self.cart_speed_history[:, 1:].clone()
        self.v_ref_history[:, -1] = v_ref
        self.cart_speed_history[:, -1] = cart_speed
        self.history_count.add_(1).clamp_(max=self.v_ref_history.shape[1])
        cart_lag_target, lag_mask = self._cart_lag(active_speed)

        self.previous_contact.copy_(contact)
        self.step.add_(1)
        return {
            "phase_target": phase_target,
            "frequency_target": frequency_target,
            "contact_target": contact.to(v_ref.dtype),
            "cart_lag_target": cart_lag_target,
            "gait_mask": gait_mask.unsqueeze(-1),
            "lag_mask": lag_mask,
        }


__all__ = ["RolloutLabelTracker"]
