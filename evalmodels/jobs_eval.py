import os
from pathlib import Path
import sys 

import htcondor

# Made price high since cluster seems to be absurdly busy 
JOB_BID_SINGLE = 2000 # 100
JOB_BID_MULTI = 2000 # 150

def launch_lmeval_job(
        model_dir,
        tasks,
        save_file,
        additional_kwargs,
        JOB_MEMORY,
        JOB_CPUS,
        JOB_GPUS=1,
        use_bf16=False,
        GPU_MEM=None,
        JOB_BID=JOB_BID_SINGLE,
        rm_model=False,
        gen_kwargs=None,
        max_tokens=8192,
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/nchandak/logs/howtoqa/evals/global_mmlu/"
    
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
        
    # if not os.path.exists(base_save_dir):
    #     print("Base save dir:", base_save_dir)
    #     os.makedirs(base_save_dir)

    
    CLUSTER_LOGS_SAVE_DIR=Path(LOG_PATH)
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    # if "snapshots" not in model_dir and "nchandak" not in model_dir:
    #     model_dir += "/snapshots/model/"
    if max_tokens < 8192:
        max_tokens = 4096
        
    pretrained = f"pretrained={model_dir}"
    if 'falcon' not in model_dir.lower():
        pretrained += ',trust_remote_code=True'

    if use_bf16:
        pretrained += ',dtype=bfloat16'

    # if JOB_GPUS > 1:
    #     pretrained += ',parallelize=True'

    pretrained += f',max_model_len={max_tokens}'
    
    # Get model size from model_dir
    model_size = model_dir.split('/')[-1].split('-')[-1]
    if 'B' in model_size:
        params = int(model_size.replace('B', '')) 
    
    pretrained += f',tensor_parallel_size={JOB_GPUS},dtype=auto,gpu_memory_utilization=0.85'
  
    executable = 'lm_eval_rm.sh' if rm_model else 'lm_eval.sh'

    # Construct gen_kwargs argument if provided
    gen_kwargs_arg = ""
    if gen_kwargs:
        gen_kwargs_arg = f"--gen_kwargs {gen_kwargs}"
    
    # Print the arguments
    # print(f"pretrained: {pretrained}")
    # print(f"tasks: {tasks}")
    # print(f"save_file: {save_file}")
    # print(f"model_dir: {model_dir}")
    # print(f"additional_kwargs: {additional_kwargs}")
    
    # Construct job description
    job_settings = {
        "executable": executable,
        "arguments": (
            f"{pretrained} "
            f"{','.join(tasks)} "
            f"{save_file} "
            f"{model_dir} "
            f"{additional_kwargs} "
            f"{gen_kwargs_arg} "
        ),
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        "request_cpus": f"{JOB_CPUS}",  # how many CPU cores we want
        "request_gpus": f"{JOB_GPUS}",
        "request_memory": JOB_MEMORY,  # how much memory we want
        "request_disk": JOB_MEMORY,
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "nikhil.chandak@tuebingen.mpg.de",  # otherwise one does not notice an you can miss clashes
        "notification": "error",
    }

    if GPU_MEM is not None:
        job_settings["requirements"] = f"(TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}) && (CUDACapability >= 8.0)"
    else:
        job_settings["requirements"] = "CUDACapability >= 8.0"

    job_description = htcondor.Submit(job_settings)
    # print(job_description)
    
    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched eval experiment with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")

if __name__ == "__main__":
    from models_utils import models, get_n_gpus

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="mmlu_continuation")
    parser.add_argument('--base_save_dir', type=str, default="/fast/nchandak/qaevals/evals/cloze/")  
    # parser.add_argument('--base_model_dir', type=str, default=None)  
    parser.add_argument('--verbosity', type=str, help='Verbosity level for evaluation')
    parser.add_argument('--additional_args', type=str, default="", help='Additional arguments to pass to the evaluation')
    parser.add_argument('--thinking', action='store_true', help='Whether to use thinking mode generation parameters')
    parser.add_argument('--gen_kwargs', type=str, default=None, 
                      help='Generation parameters like temperature, top_p, etc. (format: "temperature=0.7,top_p=0.9")')
    parser.add_argument('--max_tokens', type=int, default=32768, help='Maximum number of tokens to use for generation')
    parser.add_argument('--no_args', action='store_true', help='Whether to use thinking mode generation parameters')

    args = parser.parse_args()

    default_tokens = 16384 * 2
    # Set generation parameters based on thinking mode
    if args.gen_kwargs is None:
        if args.no_args or "cloze" in args.task or "continuation" in args.task :
            args.gen_kwargs = ""
        else :
            if args.thinking:
                if args.max_tokens > default_tokens:
                    default_tokens = args.max_tokens 
                    
                # thinking mode parameters
                args.gen_kwargs = f"temperature=0.6,top_p=0.95,min_p=0,top_k=20,max_gen_toks={default_tokens},do_sample=true"
                # args.gen_kwargs += ",repetition_penalty=1.05"
            else:
                default_tokens = 2048
                # non-thinking mode parameters
                args.gen_kwargs = f"temperature=0.7,top_p=0.8,min_p=0,top_k=20,max_gen_toks=2048,do_sample=true"
        
            
    models = {k: v['model_path'] for k, v in models.items()}
    base_save_dir = args.base_save_dir

    task = args.task
    tasks = {args.task: {'args': ''}}

    if not os.path.exists(base_save_dir):
        print("Base save dir:", base_save_dir)
        os.makedirs(base_save_dir)

    shared_args = ""
    if args.verbosity:
        shared_args += f" --verbosity {args.verbosity}"
    if args.additional_args:
        shared_args += f" {args.additional_args}"
    
    for task_name, task_config in tasks.items():
        for model, model_dir in models.items():
            print(model, model_dir)
            
            # if '1.5b' in model_dir:
            #     continue

            # Create nested directory structure: base_save_dir/model/task/
            # model_task_dir = os.path.join(base_save_dir, model, task_name)
            model_task_dir = os.path.join(base_save_dir, task_name)
            if not os.path.exists(model_task_dir):
                os.makedirs(model_task_dir, exist_ok=True)
                
            # save_name = f"{model}-{task_name}"
            save_name = f"{model}"
            suffix = "thinking" if args.thinking else "non_thinking"
            save_file = os.path.join(model_task_dir, f"{save_name}_{suffix}")

            # if save file exists, skip
            # if os.path.exists(save_file):
            #     print("save file exists")
            #     continue

            # continue if the model directory does not exist 
            if not os.path.exists(model_dir) :
                print("model dir not exists")
                continue 

            # if the model dir is empty, skip
            if len(os.listdir(model_dir)) == 0:
                print("directory empty -- skipping")
                continue
            
            print(model + " exists!")
            # print("Saving at:", save_file)
            
            # continue 
            additional_kwargs = task_config['args'] if 'args' in task_config else ''
            additional_kwargs += shared_args

            if 'JOB_GPUS' in task_config:
                del task_config['JOB_GPUS']

            # if not os.path.exists(save_file):
            if True:
                GPU_MEM = model["GPU_MEM"] if "GPU_MEM" in model else None
                use_bf16 = GPU_MEM is not None and GPU_MEM > 39000
                print(f"Launching {save_file}")
                
                launch_lmeval_job(
                    model_dir=model_dir,
                    tasks=[task_config['task']] if 'task' in task_config else [task_name],
                    save_file=save_file,
                    additional_kwargs=additional_kwargs,
                    JOB_MEMORY="64GB",
                    JOB_CPUS="8",
                    JOB_GPUS=8, #get_n_gpus(model),
                    GPU_MEM=GPU_MEM,
                    JOB_BID=JOB_BID_MULTI if get_n_gpus(model) > 1 else JOB_BID_SINGLE,
                    gen_kwargs=args.gen_kwargs,
                    max_tokens=default_tokens,
                )
                # break
