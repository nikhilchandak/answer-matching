import os 

BASE_MODEL_PATH = "/fast/rolmedo/models/"
BASE_MODEL_PATH2 = "/fast/nchandak/models/"

models = {
        # 'llama-3.1-8b-it': {
        #     'hf': 'meta-llama/Meta-Llama-3.1-8B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 39000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        #     'lr': 2e-5,
        #     'isnew': True,
        # },
        
        # 'qwen2.5-0.5b-it': {
        #     'hf': 'Qwen/Qwen2.5-0.5B',
        #     'GPU_MEM': 45000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        
        # 'qwen2.5-1.5b': {
        #     'hf': 'Qwen/Qwen2.5-1.5B',
        #     'GPU_MEM': 45000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'qwen2.5-1.5b-it': {
        #     'hf': 'Qwen/Qwen2.5-1.5B',
        #     'GPU_MEM': 45000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        
        # 'qwen2.5-1.5b-math-it': {
        #     'hf': 'Qwen/Qwen2.5-1.5B-Math',
        #     'GPU_MEM': 45000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'qwen2.5-3b': {
        #     'hf': 'Qwen/Qwen2.5-3B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'qwen2.5-3b-it': {
        #     'hf': 'Qwen/Qwen2.5-3B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'qwen2.5-7b': {
        #     'hf': 'Qwen/Qwen2.5-7B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'qwen2.5-7b-math-it': {
        #     'hf': 'Qwen/Qwen2.5-7B-Math',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'qwen2.5-7b-it': {
        #     'hf': 'Qwen/Qwen2.5-7B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'qwen2.5-14b-it': {
        #     'hf': 'Qwen/Qwen2.5-14B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'qwen2.5-14b-it': {
        #     'hf': 'Qwen/Qwen2.5-14B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'qwen2.5-32b-it': {
        #     'hf': 'Qwen/Qwen2.5-32B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        #     'INF_GPUS': 4,
        #  },
        
        # 'qwen2.5-72b-it': {
        #     'hf': 'Qwen/Qwen2.5-72B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        #     'INF_GPUS': 4,
        #  },
        
        # 'Qwen3-0.6B': {
        #     'hf': 'Qwen/Qwen3-0.6B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        'Qwen3-1.7B': {
            'hf': 'Qwen/Qwen3-1.7B',
            'GPU_MEM': 91000,
            'INF_MEM': 45000,
            'per_device_eval_batch_size': 1,
            'per_device_train_batch_size': 1,
        },
        
        
        # 'Qwen3-8B': {
        #     'hf': 'Qwen/Qwen3-8B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'Qwen3-4B': {
        #     'hf': 'Qwen/Qwen3-4B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        
        # 'Qwen3-4B-Base': {
        #     'hf': 'Qwen/Qwen3-4B-Base',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'Qwen3-14B': {
        #     'hf': 'Qwen/Qwen3-14B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        # 'qwen2.5-7b-it': {
        #     'hf': 'Qwen/Qwen2.5-7B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        # },
        
        
        # 'Qwen3-32B': {
        #     'hf': 'Qwen/Qwen3-32B',
        #     'GPU_MEM': 91000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 1,
        #     'per_device_train_batch_size': 1,
        #     'INF_GPUS': 4,
        # },
        
        # 'qwen-2.5-14b': {
        #     'hf': 'Qwen/Qwen2.5-14B',
        #     'GPU_MEM': 45000,
        #     'INF_MEM': 45000,
        #     'per_device_eval_batch_size': 8,
        #     'per_device_train_batch_size': 8,
        #     'JOB_GPUS': 4,
        # },
        
}

# add model_path to every model
for model in models:
    if "Qwen3" in model:
        new_path = BASE_MODEL_PATH2 + model + "/"
    else:
        new_path = BASE_MODEL_PATH + model + "/"
    models[model]['model_path'] = new_path 
    extra = "snapshots/model/"
    if os.path.exists(new_path + extra):
        models[model]['model_path'] += extra

def get_model_name(model_dir):
    for model in models.keys():
        if model in model_dir:
            return model
    return None

def get_job_memory(model_dir):
    model_name = get_model_name(model_dir)
    if model_name in models:
        return models[model_name]['INF_MEM']
    return 90000  # default to max

def get_n_gpus(model):
    model_name = get_model_name(model)
    if model_name in models:
        if 'INF_GPUS' in models[model_name]:
            return models[model_name]['INF_GPUS']
    return 1
