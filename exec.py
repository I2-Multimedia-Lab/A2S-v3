import time
import subprocess
import os

def get_available_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits"
    output = subprocess.check_output(command, shell=True).decode("utf-8")
    memory = [int(x) for x in output.strip().split(',')]
    return memory # 单位为MB

def check_gpu_availability():
    total_memory, available_memory = get_available_gpu_memory()
    utilization_threshold = 0.8  # 设置显存利用率阈值为80%
    print(str(available_memory/total_memory*100) + '%')
    if (available_memory/total_memory) > utilization_threshold:
        return True
    else:
        return False

def train_model():
    # 在这里执行你的神经网络模型训练代码
    print("开始训练模型...")

# 主程序
while True:
    if check_gpu_availability():
        os.system('python3 train.py midnet --gpus=1 --stage=2 --trset=cdt --vals=ce,de,te,cr,dr,tr --olr > records/midnet_spr.txt')
        #os.system('python3 train.py a2s --gpus=1 --stage=1 --trset=cdt --vals=cr,ce > records/a2s_cdt_pcl-sd_p.txt')
        break

    print("GPU未空闲，等待5分钟...")
    time.sleep(300)  # 等待5分钟

print("训练完成")