import os
from pathlib import Path

import htcondor

JOB_BID_SINGLE = 25 # 100
JOB_BID_MULTI = 100

def launch_lm_label_job(
        JOB_MEMORY,
        JOB_CPUS,
        JOB_GPUS=1,
        JOB_BID=JOB_BID_SINGLE,
        GPU_MEM=None,
        python_file=None
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/nchandak/logs/classification/mcq/mmlu_pro/"

    CLUSTER_LOGS_SAVE_DIR=Path(LOG_PATH)
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    # model_dir += "/snapshots/model/"

    # executable = 'lm_countdown.sh'
    # executable = 'launch_train_job.sh'
    executable = f'launch_train_job.sh'

    # Construct job description
    job_settings = {
        "executable": executable,
        
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        "arguments": (
            f"{python_file}"
        ),
        
        # "request_cpus": f"{JOB_CPUS}",  # how many CPU cores we want
        # "request_gpus": f"{JOB_GPUS}",
        # "request_memory": JOB_MEMORY,  # how much memory we want
        # "request_disk": JOB_MEMORY,
        
        "request_gpus": f"{JOB_GPUS}",
        "request_cpus": f"{max(JOB_CPUS*JOB_GPUS, 32)}",  # how many CPU cores we want
        "request_memory": f"{JOB_MEMORY*JOB_GPUS}GB",  # how much memory we want
        "request_disk": f"{JOB_MEMORY*JOB_GPUS}GB",
        
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "nikhil.chandak@tuebingen.mpg.de",  # otherwise one does not notice an you can miss clashes
        "notification": "error",
    }

    if GPU_MEM is not None:
        job_settings["requirements"] = f"(TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}) && (CUDACapability >= 8.0)"
    else:
        job_settings["requirements"] = "CUDACapability >= 9.0"
        
    
    # job_settings["requirements"] = "CUDACapability >= 9.0"


    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched experiment with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")


if __name__ == "__main__":
    # from weak_models_utils import models
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--python_file', type=str, required=True)
    args = parser.parse_args()
    
    GPU_MEM = 45000
    # GPU_MEM = 35000
    
    launch_lm_label_job(
        JOB_MEMORY=64,
        JOB_CPUS=8,
        JOB_GPUS=4,
        JOB_BID=JOB_BID_MULTI,
        GPU_MEM=GPU_MEM,
        python_file=args.python_file
    )
        # break 
