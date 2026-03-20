"""Forward hook registration for capturing hidden states during generation."""

import torch
from typing import Optional


class ActivationCapturer:
    """Captures hidden state vectors at specified transformer layers during generation.

    Registers forward hooks on decoder layers. During autoregressive generation,
    captures the last-token hidden state at each step and moves it to CPU immediately
    to avoid GPU OOM on long sequences.

    Usage:
        capturer = ActivationCapturer([8])
        capturer.register(model)
        # ... run model.generate() ...
        activations = capturer.get_activations(8)  # (n_tokens, hidden_dim)
        capturer.remove_hooks()
    """

    def __init__(self, target_layers: list[int]):
        self.target_layers = target_layers
        self.captured: dict[int, list[torch.Tensor]] = {l: [] for l in target_layers}
        self._hooks = []
        self._prompt_length: Optional[int] = None

    def register(self, model) -> None:
        """Register forward hooks on target decoder layers.

        Supports Qwen2 architecture: model.model.layers[i] -> Qwen2DecoderLayer
        Also works with Llama: same .model.layers[i] pattern.
        """
        for layer_idx in self.target_layers:
            layer = model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def set_prompt_length(self, length: int) -> None:
        """Set the prompt token length so we can skip prompt activations.

        Call this after tokenizing the prompt but before generation.
        The first forward pass processes all prompt tokens at once;
        we skip it and only capture generated token activations.
        """
        self._prompt_length = length

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # Qwen2DecoderLayer output varies by transformers version:
            #   - older: tuple (hidden_states, ...)
            #   - transformers 5.x: raw Tensor of shape (batch, seq_len, hidden_dim)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            seq_len = hidden_states.shape[1]

            if self._prompt_length is not None and seq_len > 1:
                # Prefill pass: captures all prompt tokens at once.
                # Save the last position's hidden state — it generates the
                # first output token and would otherwise be lost.
                self.captured[layer_idx].append(
                    hidden_states[:, -1, :].detach().cpu().float().clone()
                )
                return

            # During generation, seq_len=1 for each new token.
            # Detach, move to CPU, clone to release GPU memory.
            self.captured[layer_idx].append(
                hidden_states[:, -1, :].detach().cpu().float().clone()
            )

        return hook_fn

    def get_activations(self, layer_idx: int) -> torch.Tensor:
        """Return (n_tokens, hidden_dim) tensor for a layer."""
        if not self.captured[layer_idx]:
            return torch.empty(0)
        return torch.cat(self.captured[layer_idx], dim=0).squeeze(1)

    def get_norms(self, layer_idx: int) -> torch.Tensor:
        """Return (n_tokens,) tensor of L2 norms for a layer."""
        activations = self.get_activations(layer_idx)
        if activations.numel() == 0:
            return torch.empty(0)
        return torch.norm(activations, dim=-1)

    def n_captured(self, layer_idx: int) -> int:
        return len(self.captured[layer_idx])

    def clear(self) -> None:
        for l in self.captured:
            self.captured[l].clear()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
