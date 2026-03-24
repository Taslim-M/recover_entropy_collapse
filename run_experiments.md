We are expanding the context_0/chekcpoints outputs for various 02c_ outputs

First, you have 8 gpus, use them to run the different checkpoints for the olmo32b model. Use different ports for each model checkpoint.

Interested checkpoints: ['1e-4-step10790', '1e-4-step10000', '1e-4-step9000', '1e-4-step7000', '1e-4-step5000', '1e-4-step3000', '1e-4-step2000', '1e-4-step1000']

Sample command: CUDA_VISIBLE_DEVICES=0 vllm serve allenai/Olmo-3-32B-Think-SFT --revision 1e-4-step1000 --dtype bfloat16 --port 2010 --host 0.0.0.0

Then, for each of the 02c_ see if there are missing checkpoints for the 'autobiographical' format. Run these experiments.

In parallel, if a GPU is avaialble for, start running the same experiments for the T=0.3 and T=0.7.

In parallel, start experimenting for the 'default' parameter for the first-person-variant

For each GPU (i.e. checkpoint), make a list of pending experiments and execute sequentially when done.

Sample command: python run_stage2_personas.py  --persona-model allenai/Olmo-3-32B-Think-SFT --revision 1e-4-step3000 -vllm-url http://localhost:8000 --stage2-mode sft --temperature 0.5 --first-person-variant autobiographical