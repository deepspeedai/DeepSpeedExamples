# DeepSpeed Characterization Script

# Contents
   * [Introduction](#introduction)
   * [Usage](#usage)

# Introduction
The step 1 characterization script sweeps across various training parameters. Currently, the following parameters are swept:
<pre>
Zero Stage: 2, 3
Offload: True, False
Lora: True, False
</pre>

The `run_step1_sweep.sh` script passes configuration arguments to `run_single.sh`, which can be extended to sweep beyond the parameters listed above (e.g. learning rate, weight decay, etc).

# Usage
The sweep script can be run as follows:
<pre>
DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning$ bash training_scripts/opt/single_node/sweep/run_step1_sweep.sh
</pre>
