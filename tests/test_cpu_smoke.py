"""CPU integration checks; no paper-performance assertions."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from lib.cmd_utils import parse_main_args
from lib.data import SyntheticDataset, create_if_not_exist_dataset
from lib.models import dagVAE

ROOT = Path(__file__).resolve().parents[1]


class CPUSmokeTest(unittest.TestCase):
    def test_generator_model_and_training_entry(self):
        torch.manual_seed(1)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            data_args = '16_8_3_3_2_1_gauss_lrelu_t_f'
            path = create_if_not_exist_dataset(root=tmp, arg_str=data_args)
            other = create_if_not_exist_dataset(root=tmp + '/repeat', arg_str=data_args)
            with np.load(path) as a, np.load(other) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
                self.assertEqual(a['x'].shape, (128, 3))
                self.assertEqual(a['u'].shape, (128, 8))
                self.assertTrue(np.isfinite(a['x']).all())
                self.assertTrue(np.any(a['w'] != 0))
                self.assertTrue(np.any(a['w'][:, :, 0] != a['w'][:, :, 1]))
            dataset = SyntheticDataset(path)
            x, u, z, noise = next(iter(torch.utils.data.DataLoader(dataset, batch_size=16)))
            model = dagVAE(3, 3, 8, hidden_dim=16, n_layers=2, device='cpu')
            self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))
            forward = model(x, u)
            self.assertEqual(forward[2].shape, z.shape)
            self.assertEqual(forward[4].shape, noise.shape)
            before = {k: v.detach().clone() for k, v in model.named_parameters()}
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            losses = []
            for _ in range(3):
                optimizer.zero_grad()
                elbo, *outputs = model.elbo(x, u)
                self.assertTrue(torch.isfinite(elbo))
                self.assertTrue(all(torch.isfinite(t).all() for t in outputs))
                (-elbo).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(-elbo.item())
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.named_parameters()))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            # data-args must override an explicitly supplied nonexistent file.
            command = [sys.executable, str(ROOT / 'main.py'), '--file', '/nonexistent.npz',
                       '--data-args', data_args, '--max-iter', '3', '--batch-size', '16',
                       '--hidden-dim', '16', '--depth', '2', '--lr', '0.001']
            result = subprocess.run(command, cwd=tmp, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('training on cpu', result.stdout)
            self.assertIn('completed optimizer steps: 3', result.stdout)
            self.assertEqual(len(list(Path(tmp).glob('ckpt/*/*.pth'))), 3)
            self.assertTrue((Path(tmp) / 'log/log.json').exists())
            print('CPU optimizer losses:', losses)
            print(result.stdout)
            bad = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--preload-gpu'],
                                 cwd=tmp, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('--preload-gpu requires --cuda', bad.stderr)

    def test_local_and_legacy_launchers(self):
        self.assertFalse(parse_main_args([]).cuda)
        self.assertFalse(parse_main_args([]).preload)
        result = subprocess.run(['bash', str(ROOT / 'run'), '--help'], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(['bash', str(ROOT / 'run_slurm')], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing SLURM configuration', result.stderr)


if __name__ == '__main__':
    unittest.main()
