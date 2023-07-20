# DeepSpeed Characterization Script

# Contents
   * [Introduction](#introduction)
   * [Usage](#usage)

# Introduction
The step 3 characterization script is intented to sweep across various training parameters. Currently, the following are parameters are swept:
<pre>
Zero Stage: 2, 3
Hybrid Engine: True, False
Offload: True, False
Lora: True, False
</pre>

The `run_step3_opt_sweep.sh` script passes configuration arguments to `run_1.3b_lora_swp.sh`, which can be extended to sweep beyond the parameters listed above (learning rate, weight decay, etc).

# Usage
The sweep script can be run as follows:
<pre>
DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning$ bash training_scripts/single_node/sweep/run_step3_opt_sweep.sh
</pre>
