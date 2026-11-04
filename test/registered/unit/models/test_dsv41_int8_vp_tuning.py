"""CPU validation for version-bound VP algorithm and layout configuration."""
import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from sglang.kernels.ops.gemm.dsv41_int8 import _vp_tuning
from sglang.test.ci.ci_register import register_cpu_ci
register_cpu_ci(est_time=10,suite='base-a-test-cpu')

class TestInt8VPTuning(unittest.TestCase):
 def config(self):
  return dict(format_version=1,arch='gfx936',compute_units=80,torch_version=str(torch.__version__),n=16160,k=5120,vp_size=8,weight_lda=16384,quant_warps=4,shapes={'40':dict(q_pitch=5376,algorithm='cf220000000000000000000000000000')})
 def read(self,config):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'vp.json';p.write_text(json.dumps(config))
   with patch.dict(os.environ,{'SGLANG_DSV41_INT8_VP_ALGO_CONFIG':str(p)}),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(gcnArchName='gfx936',multi_processor_count=80)):
    return _vp_tuning(torch.device('cuda:0'),16160,5120,8)
 def test_no_config_keeps_baseline(self):
  with patch.dict(os.environ,{'SGLANG_DSV41_INT8_VP_ALGO_CONFIG':''}),patch.object(torch.cuda,'get_device_properties',side_effect=AssertionError('No GPU query for default path')):
   self.assertIsNone(_vp_tuning(torch.device('cuda:0'),16160,5120,8))
 def test_matching_configuration(self):
  c=self.config();self.assertEqual(self.read(c),c)
 def test_wrong_runtime_rejected(self):
  c=self.config();c['torch_version']='other'
  with self.assertRaisesRegex(ValueError,'runtime'):self.read(c)
 def test_wrong_shape_rejected(self):
  c=self.config();c['n']=129280
  with self.assertRaisesRegex(ValueError,'matrix'):self.read(c)
 def test_invalid_weight_leading_dimension_rejected(self):
  c=self.config();c['weight_lda']=128
  with self.assertRaisesRegex(ValueError,'layout'):self.read(c)
 def test_invalid_pitch_rejected(self):
  c=self.config();c['shapes']['40']['q_pitch']=5121
  with self.assertRaisesRegex(ValueError,'pitch'):self.read(c)
 def test_invalid_algorithm_rejected(self):
  c=self.config();c['shapes']['40']['algorithm']='x'*32
  with self.assertRaisesRegex(ValueError,'encoding'):self.read(c)

if __name__=='__main__':unittest.main()
