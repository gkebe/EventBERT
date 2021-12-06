# EventBERT

This repository provides a modified version of the NVIDIA BERT Code for PyTorch. The main change is that this code does not require docker.

## Bash Scripts

### Script for pre-training
`bash scripts/run_pretraining_nodocker.sh -n <num_gpus> -g <gpu_choices> -p <master_port> -c <initial_checkpoint> -r <resume_training> -d <dataset_phase1> -e <dataset_phase2> -a <batch_size_phase1> -b <batch_size_phase_2> -x <gradient_accumulation_steps_phase1> -y <gradient_accumulation_steps_phase2> -w <training_steps_phase1> -z <training_steps_phase2>`

Where:
-   `<num_gpus>` is the number of GPUs to use for training. Must be equal to or smaller than the number of GPUs attached to your node.
-   `<gpu_choices>` is a list of the GPUs used for training. e.g: "0,1" to use GPU 0 and GPU 1.
-   `<master_port>` is the port.
-   `<init_checkpoint>` is the initial checkpoint to start pretraining from (Usually a BERT pretrained checkpoint)
-   `<resume_training>` if set to `true` and `<init_checkpoint>` is not set, training should resume from latest model in `/results/checkpoints`. if `<init_checkpoint>` is set, pretraining starts from there.  
-	`<dataset_phase1>` is the path to the hdf5 files used for pretraining phase 1.
-	`<dataset_phase2>` is the path to the hdf5 files used for pretraining phase 2.
-   `<batch_size_phase1>` is per-GPU batch size used for phase 1 training. Larger batch sizes run more efficiently, but require more memory.
-   `<batch_size_phase2>` is per-GPU batch size used for phase 2 training. Larger batch sizes run more efficiently, but require more memory.
-   `<gradient_accumulation_steps_phase1>` is an integer indicating the number of steps to accumulate gradients over during phase 1. Effective batch size = `training_batch_size` / `gradient_accumulation_steps`.
-   `<gradient_accumulation_steps_phase2>` is an integer indicating the number of steps to accumulate gradients over during phase 2.
-   `<training_steps_phase1>` is the total number of training steps during phase 1.
-   `<training_steps_phase2>` is the total number of training steps during phase 2.
	
### Script for F1-score evaluation of NSP and MLM
`bash scripts/run_pretraining_inference_nodocker.sh -n <num_gpus> -g <gpu_choices> -p <master_port> -d <dataset> -b <batch_size>`
Where:
-   `<num_gpus>` is the number of GPUs to use for training. Must be equal to or smaller than the number of GPUs attached to your node.
-   `<gpu_choices>` is a list of the GPUs used for training. e.g: "0,1" to use GPU 0 and GPU 1.
-   `<master_port>` is the port.
-	`<dataset>` is the path to the hdf5 files used for testing.
-   `<batch_size>` is per-GPU batch size used for testing. Larger batch sizes run more efficiently, but require more memory.



