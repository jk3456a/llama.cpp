#!/usr/bin/env python3
"""
GGUF模型基准测试脚本
自动测试指定目录下的所有*.gguf模型文件
支持CPU和GPU测试，排除有问题的TQ量化模型的GPU测试
"""

import os
import subprocess
import csv
import json
import sys
import re
import itertools
from datetime import datetime
from pathlib import Path
import multiprocessing

# 配置参数
MODEL_DIR = "/home/modelbest/workspace/lizhen/Models/need-test/"
LLAMA_BENCH_PATH = "./tools/llama-bench/llama-bench"  # llama-bench工具路径
PROMPT_TOKENS = [16]  # 测试的prompt token数量
N_GEN = [128]  # 生成token数量
REPETITIONS = [1]  # 重复次数
FLASH_ATTN = [1]  # 启用flash attention
BATCH_SIZE = [1, 2, 4, 8, 16, 32]


def find_gguf_files(directory):
    """查找指定目录下所有的.gguf文件"""
    if not os.path.exists(directory):
        print(f"警告: 目录不存在: {directory}")
        return []
    
    gguf_files = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.gguf'):
                    full_path = os.path.join(root, file)
                    gguf_files.append(full_path)
                    print(f"发现模型: {file}")
    except Exception as e:
        print(f"搜索文件时出错: {e}")
    
    return gguf_files

def should_test_gpu(model_path):
    """判断模型是否应该在GPU上测试"""
    model_name = os.path.basename(model_path).upper()
    # TQ量化模型在Metal后端有兼容性问题，只在CPU上测试
    if "TQ" in model_name:
        return False
    return True

def run_benchmark(model_path, prompt_tokens, n_gen, repetitions, flash_attn, batch_size, use_gpu=True):
    """运行单个模型的基准测试"""
    cmd = [
        LLAMA_BENCH_PATH,
        "-m", model_path,
        "-fa", str(flash_attn),
        "-p", str(prompt_tokens),  # -p 是 --n-prompt (prompt tokens)
        "-n", str(n_gen),
        "-r", str(repetitions),
        "-b", str(batch_size)
    ]
    
    # 如果是CPU测试，添加-ngl 0参数
    if not use_gpu:
        cmd.extend(["-ngl", "0"])
    
    backend_type = "GPU" if use_gpu else "CPU"
    
    # 设置环境变量
    env = os.environ.copy()
    env["QOS_CLASS_USER_INTERACTIVE"] = "1"
    
    try:
        print(f"运行测试: {os.path.basename(model_path)} | {backend_type} | Prompt tokens:{prompt_tokens} | 生成:{n_gen} | 重复:{repetitions} | FlashAttn:{flash_attn} | 批量:{batch_size}")
        print(f"命令: QOS_CLASS_USER_INTERACTIVE=1 {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print(f"✓ 测试成功")
            # 输出原始数据用于调试
            print("--- 原始输出 ---")
            print(result.stdout)
            print("--- 原始输出结束 ---")
            return result.stdout, backend_type, n_gen, repetitions, flash_attn, batch_size
        else:
            print(f"✗ 测试失败: {result.stderr}")
            return None, backend_type, n_gen, repetitions, flash_attn, batch_size
    except FileNotFoundError:
        print(f"✗ 找不到llama-bench工具: {LLAMA_BENCH_PATH}")
        return None, backend_type, n_gen, repetitions, flash_attn, batch_size
    except Exception as e:
        print(f"✗ 运行测试时出错: {e}")
        return None, backend_type, n_gen, repetitions, flash_attn, batch_size

def extract_tokens_per_second(output):
    """使用正则表达式提取tokens/s数据"""
    if not output:
        return []
    
    results = []
    try:
        # 正则表达式匹配tokens/s数据
        pattern = r'(\d+\.\d+)\s*±\s*(\d+\.\d+)'
        matches = re.findall(pattern, output)
        
        # 查找表格数据行（包含测试类型和性能数据）
        for line in lines:
            # 跳过表头和分隔符行
            if '|' not in line or '---' in line or 'model' in line or 'test' in line:
                continue
            
            # 分割表格列
            columns = [col.strip() for col in line.split('|')]
            
            # 确保有足够的列（至少8列：空，model，size，params，backend，ngl，n_batch，fa，test，t/s）
            if len(columns) >= 10:
                test_type = columns[8].strip()  # test列
                ts_column = columns[9].strip()  # t/s列
                
                # 提取性能数据 (例如: "159.21 ± 0.00")
                ts_match = re.match(r'([\d.]+)\s*±\s*([\d.]+)', ts_column)
                if ts_match and test_type:
                    avg_ts = float(ts_match.group(1))
                    stddev_ts = float(ts_match.group(2))
                    
                    result = {
                        'avg_ts': avg_ts,
                        'stddev_ts': stddev_ts,
                        'test_type': test_type
                    }
                    
                    # 根据测试类型确定phase
                    if test_type.startswith('pp'):
                        result['phase'] = 'prompt_processing'
                    elif test_type.startswith('tg'):
                        result['phase'] = 'token_generation'
                    else:
                        result['phase'] = 'unknown'
                    
                    results.append(result)
                    print(f"解析到: {test_type} -> {result['phase']}: {avg_ts:.2f} t/s")
        
        print(f"总共解析到 {len(results)} 个测试结果")
        
    except Exception as e:
        print(f"解析表格输出时出错: {e}")
        # 如果表格解析失败，尝试简单的数值提取
        try:
            # 备用方案：只提取数值，不区分phase
            pattern = r'(\d+\.\d+)\s*±\s*(\d+\.\d+)'
            matches = re.findall(pattern, output)
            print(f"备用方案：找到 {len(matches)} 个数值")
            
            for avg_ts, stddev_ts in matches:
                result = {
                    'avg_ts': float(avg_ts),
                    'stddev_ts': float(stddev_ts),
                    'phase': 'unknown'
                }
                results.append(result)
                
        except Exception as e2:
            print(f"备用解析也失败: {e2}")
    
    return results

def format_size(size_bytes):
    """格式化文件大小"""
    if isinstance(size_bytes, (int, float)) and size_bytes > 0:
        return f"{size_bytes / (1024**3):.2f} GiB"
    return "N/A"

def format_params(params):
    """格式化参数数量"""
    if isinstance(params, (int, float)) and params > 0:
        if params >= 1e9:
            return f"{params / 1e9:.2f} B"
        elif params >= 1e6:
            return f"{params / 1e6:.2f} M"
        else:
            return f"{params / 1e3:.2f} K"
    return "N/A"

# CSV文件名
CSV_FILENAME = "benchmark_results.csv"

def load_existing_results():
    """加载已存在的测试结果"""
    existing_tests = set()
    
    if not os.path.exists(CSV_FILENAME):
        return existing_tests
    
    try:
        with open(CSV_FILENAME, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 创建唯一标识符：模型名_后端_prompt tokens_生成tokens_重复次数_FlashAttn_批量大小
                test_key = f"{row['model_name']}_{row['backend']}_{row['context_length']}_{row['n_gen']}_{row['repetitions']}_{row['flash_attn']}_{row['batch_size']}"
                existing_tests.add(test_key)
        
        print(f"从CSV文件加载了 {len(existing_tests)} 个已完成的测试配置")
        
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
    
    return existing_tests

def is_test_already_done(model_name, backend, prompt_tokens, n_gen, repetitions, flash_attn, batch_size, existing_tests):
    """检查测试是否已经完成"""
    test_key = f"{model_name}_{backend}_{prompt_tokens}_{n_gen}_{repetitions}_{flash_attn}_{batch_size}"
    return test_key in existing_tests

def save_to_csv(row_data):
    """立即保存一行数据到CSV文件"""
    try:
        file_exists = os.path.exists(CSV_FILENAME)
        
        with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['model_name', 'backend', 'context_length', 'n_gen', 'repetitions', 'flash_attn', 'batch_size', 'phase', 'tokens_per_sec']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                print(f"创建CSV文件: {CSV_FILENAME}")
            
            writer.writerow(row_data)
            
    except Exception as e:
        print(f"保存CSV时出错: {e}")

def main():
    """主函数"""
    print("=== GGUF模型基准测试工具 ===")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    print("")
    
    # 检查llama-bench是否存在
    if not os.path.exists(LLAMA_BENCH_PATH):
        print(f"错误: 找不到llama-bench工具: {LLAMA_BENCH_PATH}")
        print("请先编译llama-bench工具")
        return False
    
    # 查找所有GGUF文件
    print(f"在目录 {MODEL_DIR} 中查找GGUF文件...")
    gguf_files = find_gguf_files(MODEL_DIR)
    
    if not gguf_files:
        print(f"在 {MODEL_DIR} 中未找到任何.gguf文件")
        return False
    
    print(f"\n找到 {len(gguf_files)} 个GGUF文件")
    
    # 生成所有参数组合
    param_combinations = list(itertools.product(
        PROMPT_TOKENS,
        N_GEN,
        REPETITIONS,
        FLASH_ATTN,
        BATCH_SIZE
    ))
    
    print(f"参数组合数量: {len(param_combinations)}")
    print("参数组合示例:")
    for i, combo in enumerate(param_combinations[:3]):  # 显示前3个组合
        prompt_tokens, n_gen, rep, flash, batch = combo
        print(f"  {i+1}. Prompt tokens:{prompt_tokens}, 生成:{n_gen}, 重复:{rep}, FlashAttn:{flash}, 批量:{batch}")
    if len(param_combinations) > 3:
        print(f"  ... 以及其他 {len(param_combinations) - 3} 个组合")
    
    # 计算总测试数
    total_tests = 0
    for model_path in gguf_files:
        if should_test_gpu(model_path):
            total_tests += len(param_combinations) * 2  # CPU和GPU
        else:
            total_tests += len(param_combinations)  # 只有CPU
    
    print(f"计划进行 {total_tests} 个基准测试")
    print("")
    
    # 加载已存在的测试结果
    existing_tests = load_existing_results()
    
    # 存储所有测试结果
    all_results = []
    current_test = 0
    skipped_tests = 0
    
    # 对每个模型运行测试
    for model_path in gguf_files:
        model_name = os.path.basename(model_path)
        print(f"\n--- 测试模型: {model_name} ---")
        
        # 判断是否测试GPU
        test_gpu = should_test_gpu(model_path)
        
        if not test_gpu:
            print(f"⚠️  {model_name} 包含TQ量化，跳过GPU测试")
        
        # 对每个参数组合运行测试
        for prompt_tokens, n_gen, repetitions, flash_attn, batch_size in param_combinations:
            # CPU测试
            current_test += 1
            backend_type = "CPU"
            
            # 检查是否已经测试过
            if is_test_already_done(model_name, backend_type, prompt_tokens, n_gen, repetitions, flash_attn, batch_size, existing_tests):
                print(f"\n进度: {current_test}/{total_tests} - ⏭️ 跳过已完成的测试: {model_name} | {backend_type} | 参数组合")
                skipped_tests += 1
            else:
                print(f"\n进度: {current_test}/{total_tests}")
                
                output, backend_type, n_gen_used, repetitions_used, flash_attn_used, batch_size_used = run_benchmark(
                    model_path, prompt_tokens, n_gen, repetitions, flash_attn, batch_size, use_gpu=False
                )
                
                if output:
                    results = extract_tokens_per_second(output)
                    print(f"📊 {model_name} | {backend_type} | 参数组合结果")
                    for result in results:
                        tokens_per_sec = result.get('avg_ts', 0)
                        phase = result.get('phase', 'unknown')
                        print(f"   {phase}: {tokens_per_sec:.2f} t/s")
                        
                        # 立即保存到CSV
                        csv_row = {
                            'model_name': model_name,
                            'backend': backend_type,
                            'context_length': prompt_tokens,
                            'n_gen': n_gen_used,
                            'repetitions': repetitions_used,
                            'flash_attn': flash_attn_used,
                            'batch_size': batch_size_used,
                            'phase': phase,
                            'tokens_per_sec': tokens_per_sec
                        }
                        save_to_csv(csv_row)
                        all_results.append(csv_row)
                else:
                    print("未获取到输出数据")
            
            # GPU测试（如果支持）
            if test_gpu:
                current_test += 1
                gpu_backend_type = "GPU"
                
                # 检查是否已经测试过
                if is_test_already_done(model_name, gpu_backend_type, prompt_tokens, n_gen, repetitions, flash_attn, batch_size, existing_tests):
                    print(f"\n进度: {current_test}/{total_tests} - ⏭️ 跳过已完成的测试: {model_name} | {gpu_backend_type} | 参数组合")
                    skipped_tests += 1
                else:
                    print(f"\n进度: {current_test}/{total_tests}")
                    
                    output, backend_type, n_gen_used, repetitions_used, flash_attn_used, batch_size_used = run_benchmark(
                        model_path, prompt_tokens, n_gen, repetitions, flash_attn, batch_size, use_gpu=True
                    )
                    
                    if output:
                        results = extract_tokens_per_second(output)
                        print(f"📊 {model_name} | {backend_type} | 参数组合结果")
                        for result in results:
                            tokens_per_sec = result.get('avg_ts', 0)
                            phase = result.get('phase', 'unknown')
                            print(f"   {phase}: {tokens_per_sec:.2f} t/s")
                            
                            # 立即保存到CSV
                            csv_row = {
                                'model_name': model_name,
                                'backend': backend_type,
                                'context_length': prompt_tokens,
                                'n_gen': n_gen_used,
                                'repetitions': repetitions_used,
                                'flash_attn': flash_attn_used,
                                'batch_size': batch_size_used,
                                'phase': phase,
                                'tokens_per_sec': tokens_per_sec
                            }
                            save_to_csv(csv_row)
                            all_results.append(csv_row)
                    else:
                        print("未获取到输出数据")

    print(f"\n✅ 完成！")
    print(f"📊 新增测试结果: {len(all_results)} 条")
    print(f"⏭️ 跳过已完成测试: {skipped_tests} 个")
    print(f"💾 结果已保存到: {CSV_FILENAME}")
    
    # 生成Markdown报告
    md_filename = "report.md"
    print(f"生成Markdown报告: {md_filename}")
    
    try:
        # 读取完整的CSV数据来生成报告
        csv_data = []
        if os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                csv_data = list(reader)
        
        with open(md_filename, 'w', encoding='utf-8') as mdfile:
            mdfile.write("# GGUF模型基准测试报告\n\n")
            mdfile.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            mdfile.write(f"**测试目录**: `{MODEL_DIR}`\n\n")
            mdfile.write(f"**参数配置**:\n")
            mdfile.write(f"- Prompt tokens: {PROMPT_TOKENS}\n")
            mdfile.write(f"- 生成token数: {N_GEN}\n")
            mdfile.write(f"- 重复次数: {REPETITIONS}\n")
            mdfile.write(f"- Flash Attention: {FLASH_ATTN}\n")
            mdfile.write(f"- 批量大小: {BATCH_SIZE}\n")
            mdfile.write(f"**参数组合总数**: {len(param_combinations)}\n\n")
            
            mdfile.write("## 测试结果\n\n")
            
            # 按模型分组
            models = {}
            for row in csv_data:
                model_name = row['model_name']
                if model_name not in models:
                    models[model_name] = []
                models[model_name].append(row)
            
            # 为每个模型生成一个表格
            for model_name in sorted(models.keys()):
                mdfile.write(f"### {model_name}\n\n")
                
                # 创建表格头
                mdfile.write("| 后端 | Prompt tokens | 生成tokens | 重复次数 | FlashAttn | 批量大小 | Prompt Processing (t/s) | Token Generation (t/s) |\n")
                mdfile.write("|------|---------------|------------|----------|-----------|----------|-------------------------|------------------------|\n")
                
                # 按参数组合组织数据
                model_data = {}
                for row in models[model_name]:
                    backend = row['backend']
                    prompt_tokens = int(row['context_length'])  # CSV中仍使用context_length字段名
                    n_gen = int(row['n_gen'])
                    repetitions = int(row['repetitions'])
                    flash_attn = int(row['flash_attn'])
                    batch_size = int(row['batch_size'])
                    phase = row['phase']
                    tokens_per_sec = float(row['tokens_per_sec'])
                    
                    key = (backend, prompt_tokens, n_gen, repetitions, flash_attn, batch_size)
                    if key not in model_data:
                        model_data[key] = {}
                    model_data[key][phase] = tokens_per_sec
                
                # 按参数组合排序输出
                for key in sorted(model_data.keys()):
                    backend, prompt_tokens, n_gen, repetitions, flash_attn, batch_size = key
                    data = model_data[key]
                    pp_speed = data.get('prompt_processing', 'N/A')
                    tg_speed = data.get('token_generation', 'N/A')
                    
                    pp_str = f"{pp_speed:.2f}" if isinstance(pp_speed, (int, float)) else str(pp_speed)
                    tg_str = f"{tg_speed:.2f}" if isinstance(tg_speed, (int, float)) else str(tg_speed)
                    
                    mdfile.write(f"| {backend} | {prompt_tokens} | {n_gen} | {repetitions} | {flash_attn} | {batch_size} | {pp_str} | {tg_str} |\n")
                
                mdfile.write("\n")
            
            # 生成统计信息
            mdfile.write("## 测试统计\n\n")
            mdfile.write(f"- **模型数量**: {len(models)}\n")
            mdfile.write(f"- **参数组合数**: {len(param_combinations)}\n")
            mdfile.write(f"- **总测试记录**: {len(csv_data)}\n")
            
            # 性能统计
            cpu_speeds = []
            gpu_speeds = []
            for row in csv_data:
                if row['phase'] == 'token_generation':  # 只统计token generation性能
                    speed = float(row['tokens_per_sec'])
                    if row['backend'] == 'CPU':
                        cpu_speeds.append(speed)
                    elif row['backend'] == 'GPU':
                        gpu_speeds.append(speed)
            
            if cpu_speeds:
                mdfile.write(f"- **CPU Token Generation平均速度**: {sum(cpu_speeds)/len(cpu_speeds):.2f} t/s\n")
                mdfile.write(f"- **CPU Token Generation最高速度**: {max(cpu_speeds):.2f} t/s\n")
            
            if gpu_speeds:
                mdfile.write(f"- **GPU Token Generation平均速度**: {sum(gpu_speeds)/len(gpu_speeds):.2f} t/s\n")
                mdfile.write(f"- **GPU Token Generation最高速度**: {max(gpu_speeds):.2f} t/s\n")
            
            # 添加模型兼容性信息
            mdfile.write("\n## 模型兼容性说明\n\n")
            tq_models = [model for model in models.keys() if "TQ" in model.upper()]
            if tq_models:
                mdfile.write("以下模型包含TQ量化，由于Metal后端兼容性问题，仅在CPU上测试：\n\n")
                for model in sorted(tq_models):
                    mdfile.write(f"- {model}\n")
        
        print("✓ Markdown报告生成成功")
    except Exception as e:
        print(f"✗ 生成Markdown报告失败: {e}")
    
    print(f"\n=== 测试完成 ===")
    print(f"结果文件:")
    if os.path.exists(CSV_FILENAME):
        print(f"- {CSV_FILENAME}")
    if os.path.exists(md_filename):
        print(f"- {md_filename}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 