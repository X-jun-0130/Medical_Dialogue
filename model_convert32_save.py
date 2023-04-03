import os
os.chdir('/Nlp_2023/Dialogue_Bloom/')

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict


convert_zero_checkpoint_to_fp32_state_dict('./results/checkpoint-5000/', './Bloom_Dia/pytorch_model.bin')
