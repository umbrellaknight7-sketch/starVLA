# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""CALVIN env-side adapter for the StarVLA websocket policy server.

CALVIN checkpoints in this example are trained with 7-D continuous action
labels that are already in the benchmark action space:

    dx, dy, dz, droll, dpitch, dyaw, gripper

Unlike LIBERO, this client does not request server-side action unnormalization
and does not remap the gripper channel. It only handles websocket I/O and
action-chunk caching.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


class CalvinModelClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10093,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta.get("action_chunk_size", 1))
        self.server_metadata = meta

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.task_description: Optional[str] = None
        self.raw_actions: Optional[np.ndarray] = None

        print(
            f"*** CALVIN policy client: action_chunk_size={self.action_chunk_size}, "
            f"server_meta={meta} ***"
        )

    def reset(self, task_description: Optional[str] = None) -> None:
        self.task_description = task_description
        self.raw_actions = None

    def step(self, example: dict, step: int = 0) -> dict:
        """Return one cached 7-D CALVIN action for the current env step.

        Args:
            example: StarVLA example with ``image`` and ``lang`` keys, and
                optional ``state`` only for state-conditioned checkpoints.
            step: rollout step counter; used to index cached action chunks.

        Returns:
            ``{"raw_action": {"world_vector": ..., "rotation_delta": ..., "gripper": ...}}``
        """
        task_description = example.get("lang")
        if task_description != self.task_description:
            self.reset(task_description)

        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.predict_action(
                {
                    "examples": [example],
                    "skip_action_unnorm": True,
                    "do_sample": False,
                    "use_ddim": self.use_ddim,
                    "num_ddim_steps": self.num_ddim_steps,
                }
            )
            if not response.get("ok", False):
                raise RuntimeError(f"CALVIN policy server error: {response.get('error', response)}")
            try:
                actions_batch = response["data"]["actions"]
            except KeyError as exc:
                raise KeyError(
                    f"Key 'actions' not found in response data: "
                    f"keys={list(response.get('data', {}).keys())}, full response={response}"
                ) from exc

            self.raw_actions = np.asarray(actions_batch, dtype=np.float32)[0]
            if self.raw_actions.ndim != 2 or self.raw_actions.shape[-1] != 7:
                raise ValueError(f"Expected CALVIN action chunk shape (T, 7), got {self.raw_actions.shape}")

        action = self.raw_actions[step % self.action_chunk_size]
        return {
            "raw_action": {
                "world_vector": np.array(action[:3], dtype=np.float32),
                "rotation_delta": np.array(action[3:6], dtype=np.float32),
                "gripper": np.array(action[6:7], dtype=np.float32),
            }
        }
