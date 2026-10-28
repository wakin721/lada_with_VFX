# SPDX-License-Identifier: AGPL-3.0
import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location('cuda_encoding_test', Path(__file__).resolve().parents[1] / 'lada/utils/cuda_video_writer.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class CudaConfigTests(unittest.TestCase):
    def test_preserves_explicit_quality_and_lookahead(self):
        config = module.encoder_config('hevc_nvenc', '-cuda-input 1 -preset p7 -tune hq -rc constqp -qp 20 -bf 3 -rc-lookahead 32')
        self.assertEqual(config, dict(codec='hevc', preset='P7', tuning_info='high_quality',
            rc='constqp', constqp=20, bf=3, lookahead=32))

    def test_vfr_disables_b_frames_without_changing_quality(self):
        config = module.encoder_config('hevc_nvenc', '-preset p7 -qp 18 -bf 3', True)
        self.assertEqual((config['bf'], config['constqp'], config['preset']), (0, 18, 'P7'))

    def test_unsupported_quality_settings_are_never_silently_dropped(self):
        for options in ('-aq-strength 6', '-b_ref_mode middle', '-preset', '-qp 52', '-rc vbr'):
            with self.subTest(options=options), self.assertRaises(ValueError):
                module.encoder_config('hevc_nvenc', options)

    def test_cuda_input_is_explicit_and_does_not_import_driver(self):
        self.assertTrue(module.uses_cuda_input('-preset p4 -cuda-input 1 -qp 23'))
        self.assertFalse(module.uses_cuda_input('-preset p4 -qp 23'))
        self.assertFalse(module.uses_cuda_input('-cuda-input 0'))


if __name__ == '__main__':
    unittest.main()
