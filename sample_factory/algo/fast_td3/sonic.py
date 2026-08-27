from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch

SONIC_TOKEN_DIM = 64
SONIC_BODY_ACTION_DIM = 29


class SonicCudaDecoder:
    """Run the frozen SONIC decoder with GPU I/O binding."""

    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        ort.preload_dlls(directory="")
        provider_options = {
            "device_id": str(device.index),
        }
        session_options = ort.SessionOptions()
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=[("CUDAExecutionProvider", provider_options)],
        )
        self.session.disable_fallback()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # The actor also carries the 14 hand controls; SONIC consumes token controls only.
        decoder_input = torch.cat(
            (tokens[:, :SONIC_TOKEN_DIM], state), dim=1
        ).contiguous()
        decoded = torch.empty(
            (decoder_input.shape[0], SONIC_BODY_ACTION_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        binding = self.session.io_binding()
        binding.bind_input(
            self.input_name,
            "cuda",
            self.device.index,
            np.float32,
            tuple(decoder_input.shape),
            decoder_input.data_ptr(),
        )
        binding.bind_output(
            self.output_name,
            "cuda",
            self.device.index,
            np.float32,
            tuple(decoded.shape),
            decoded.data_ptr(),
        )
        binding.synchronize_inputs()
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return decoded
