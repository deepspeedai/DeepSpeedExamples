# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rollout_prefill import calibrate, main, nonnegative_fit


class CalibrationFitTest(unittest.TestCase):

    def test_recovers_nonnegative_costs(self):
        matrix = [[1, 1, 1], [1, 2, 4], [1, 4, 16], [1, 8, 64]]
        np.testing.assert_allclose(nonnegative_fit(matrix, [5.5, 10, 22, 58]), [2, 3, 0.5])
        # Noisy measurements must not produce negative execution costs.
        np.testing.assert_allclose(nonnegative_fit(matrix, [4, 3, 2, 1]), [2.5, 0, 0], atol=1e-12)

    def test_calibrates_multihead_model_without_kv_head_setting(self):
        from deepspeed.accelerator import get_accelerator
        from transformers import OPTConfig, OPTForCausalLM

        config = OPTConfig(vocab_size=32,
                           hidden_size=32,
                           ffn_dim=64,
                           num_hidden_layers=2,
                           num_attention_heads=4,
                           max_position_embeddings=1024,
                           pad_token_id=0,
                           eos_token_id=None,
                           attn_implementation="sdpa")
        self.assertFalse(hasattr(config, "num_key_value_heads"))
        device = get_accelerator().device_name()
        model = OPTForCausalLM(config).to(device).eval()
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(4)
            with torch.inference_mode(), redirect_stdout(io.StringIO()):
                costs, measurements = calibrate(SimpleNamespace(module=model),
                                                SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                                3,
                                                0,
                                                device,
                                                repeats=1)
        finally:
            torch.set_num_threads(threads)
        self.assertEqual(len(measurements["prefill_shapes"]), 6)
        self.assertTrue(all(np.isfinite(value) and value >= 0 for value in costs.values()))

    def test_cli_rejects_stale_or_missing_source_hash(self):
        costs = dict.fromkeys(("prefill_fixed_cost_ms", "prefill_token_cost_ms", "prefill_attention_cost_ms",
                              "prefill_kv_cost_ms", "continuous_decode_cost_ms"), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            costs_path = Path(directory) / "costs.json"
            output_path = Path(directory) / "sweep.json"
            # Isolate file compatibility from measurements; the model/config remain real.
            with patch("rollout_prefill.get_accelerator", return_value=SimpleNamespace(device_name=lambda: "cpu")), \
                    patch("torch.cuda.get_device_name", return_value="test-device"), \
                    patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(multi_processor_count=1)), \
                    patch("rollout_prefill.calibrate", return_value=(costs, {})), \
                    patch("rollout_prefill.sweep", return_value={"smoke": {"auto_vs_best": 1.0}}) as sweep, \
                    redirect_stdout(io.StringIO()):
                with patch("sys.argv", ["rollout_prefill.py", "--calibrate", str(costs_path)]):
                    main()
                saved = json.loads(costs_path.read_text())
                args = ["rollout_prefill.py", "--cost-config", str(costs_path), "--output", str(output_path)]
                with patch("sys.argv", args):
                    main()
                self.assertEqual(sweep.call_count, 1)
                for source_hash in ("0" * 64, None):
                    with self.subTest(source_hash=source_hash):
                        calibration = json.loads(json.dumps(saved))
                        if source_hash is None:
                            calibration["metadata"].pop("source_sha256")
                        else:
                            calibration["metadata"]["source_sha256"] = source_hash
                        costs_path.write_text(json.dumps(calibration))
                        stderr = io.StringIO()
                        with patch("sys.argv", args), redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                            main()
                        self.assertEqual(error.exception.code, 2)
                        self.assertIn("source_sha256", stderr.getvalue())
                        self.assertEqual(sweep.call_count, 1)
                costs_path.write_text(json.dumps({"costs": saved["costs"]}))
                with patch("sys.argv", args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(sweep.call_count, 1)
                costs_path.write_text(json.dumps(saved))
                changed_cache = Path(directory) / "changed_cache.py"
                changed_cache.write_text("# A different static-cache implementation\n")
                with patch("rollout_prefill.cache_backend.__file__", str(changed_cache)), \
                        patch("sys.argv", args), redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(sweep.call_count, 1)
                for missing in (None, *costs):
                    with self.subTest(missing_cost=missing):
                        calibration = json.loads(json.dumps(saved))
                        if missing is None:
                            calibration.pop("costs")
                        else:
                            calibration["costs"].pop(missing)
                        costs_path.write_text(json.dumps(calibration))
                        with patch("sys.argv", args), redirect_stderr(io.StringIO()), \
                                self.assertRaises(SystemExit) as error:
                            main()
                        self.assertEqual(error.exception.code, 2)
                        self.assertEqual(sweep.call_count, 1)


if __name__ == "__main__":
    unittest.main()
