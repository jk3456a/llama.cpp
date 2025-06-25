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
from datetime import datetime
from pathlib import Path

# 配置参数
MODEL_DIR = "/Users/yummy/workspace/models"
LLAMA_BENCH_PATH = "./tools/llama-bench/llama-bench"  # llama-bench工具路径
CONTEXT_LENGTHS = [1024, 2048, 4096]  # 测试的上下文长度
N_GEN = 16  # 生成token数量
REPETITIONS = 1  # 重复次数
FLASH_ATTN = 1  # 启用flash attention

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

def run_benchmark(model_path, context_length, use_gpu=True):
    """运行单个模型的基准测试"""
    # Q4_0模型使用更多重复次数以获得更稳定的结果
    model_name = os.path.basename(model_path).upper()
    repetitions = 5 if "Q4_0" in model_name else REPETITIONS
    
    cmd = [
        LLAMA_BENCH_PATH,
        "-m", model_path,
        "-fa", str(FLASH_ATTN),
        "-d", str(context_length),
        "-n", str(N_GEN),
        "-r", str(repetitions),
        "-t", "4"
    ]
    
    # 如果是CPU测试，添加-ngl 0参数
    if not use_gpu:
        cmd.extend(["-ngl", "0"])
    
    backend_type = "GPU" if use_gpu else "CPU"
    
    # 设置环境变量
    env = os.environ.copy()
    env["QOS_CLASS_USER_INTERACTIVE"] = "1"
    
    try:
        print(f"运行测试: {os.path.basename(model_path)} | {backend_type} | 上下文: {context_length}")
        print(f"命令: QOS_CLASS_USER_INTERACTIVE=1 {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print(f"✓ 测试成功")
            # 输出原始数据用于调试
            print("--- 原始输出 ---")
            print(result.stdout)
            print("--- 原始输出结束 ---")
            return result.stdout, backend_type, N_GEN, repetitions
        else:
            print(f"✗ 测试失败: {result.stderr}")
            return None, backend_type, N_GEN, repetitions
    except FileNotFoundError:
        print(f"✗ 找不到llama-bench工具: {LLAMA_BENCH_PATH}")
        return None, backend_type, N_GEN, repetitions
    except Exception as e:
        print(f"✗ 运行测试时出错: {e}")
        return None, backend_type, N_GEN, repetitions

def extract_tokens_per_second(output):
    """使用正则表达式提取tokens/s数据"""
    if not output:
        return []
    
    results = []
    try:
        # 正则表达式匹配tokens/s数据
        pattern = r'(\d+\.\d+)\s*±\s*(\d+\.\d+)'
        matches = re.findall(pattern, output)
        
        # 匹配测试类型 pp512 @ d1024 或 tg16 @ d1024
        test_pattern = r'(pp\d+|tg\d+)\s*@\s*d(\d+)'
        test_matches = re.findall(test_pattern, output)
        
        print(f"找到 {len(matches)} 个性能数据匹配")
        print(f"找到 {len(test_matches)} 个测试类型匹配")
        
        # 配对性能数据和测试类型
        for i, (avg_ts, stddev_ts) in enumerate(matches):
            result = {}
            result['avg_ts'] = float(avg_ts)
            result['stddev_ts'] = float(stddev_ts)
            
            if i < len(test_matches):
                test_type, context = test_matches[i]
                result['test_type'] = test_type
                result['context_length'] = int(context)
                
                if test_type.startswith('pp'):
                    result['phase'] = 'prompt_processing'
                elif test_type.startswith('tg'):
                    result['phase'] = 'token_generation'
            
            results.append(result)
            
    except Exception as e:
        print(f"正则表达式解析出错: {e}")
    
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
                # 创建唯一标识符：模型名_后端_上下文长度
                test_key = f"{row['model_name']}_{row['backend']}_{row['context_length']}"
                existing_tests.add(test_key)
        
        print(f"从CSV文件加载了 {len(existing_tests)} 个已完成的测试配置")
        
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
    
    return existing_tests

def is_test_already_done(model_name, backend, context_length, existing_tests):
    """检查测试是否已经完成"""
    test_key = f"{model_name}_{backend}_{context_length}"
    return test_key in existing_tests

def save_to_csv(row_data):
    """立即保存一行数据到CSV文件"""
    try:
        file_exists = os.path.exists(CSV_FILENAME)
        
        with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['model_name', 'backend', 'n_gen', 'repetitions', 'context_length', 'phase', 'tokens_per_sec']
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
    
    # 计算总测试数
    total_tests = 0
    for model_path in gguf_files:
        if should_test_gpu(model_path):
            total_tests += len(CONTEXT_LENGTHS) * 2  # CPU和GPU
        else:
            total_tests += len(CONTEXT_LENGTHS)  # 只有CPU
    
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
        
        for context_length in CONTEXT_LENGTHS:
            # CPU测试
            current_test += 1
            backend_type = "CPU"
            
            # 检查是否已经测试过
            if is_test_already_done(model_name, backend_type, context_length, existing_tests):
                print(f"\n进度: {current_test}/{total_tests} - ⏭️ 跳过已完成的测试: {model_name} | {backend_type} | 上下文:{context_length}")
                skipped_tests += 1
            else:
                print(f"\n进度: {current_test}/{total_tests}")
                
                output, backend_type, n_gen, repetitions = run_benchmark(model_path, context_length, use_gpu=False)
                if output:
                    results = extract_tokens_per_second(output)
                    print(f"📊 {model_name} | {backend_type} | 上下文:{context_length}")
                    for result in results:
                        tokens_per_sec = result.get('avg_ts', 0)
                        phase = result.get('phase', 'unknown')
                        print(f"   {phase}: {tokens_per_sec:.2f} t/s")
                        
                        # 立即保存到CSV
                        csv_row = {
                            'model_name': model_name,
                            'backend': backend_type,
                            'n_gen': n_gen,
                            'repetitions': repetitions,
                            'context_length': context_length,
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
                if is_test_already_done(model_name, gpu_backend_type, context_length, existing_tests):
                    print(f"\n进度: {current_test}/{total_tests} - ⏭️ 跳过已完成的测试: {model_name} | {gpu_backend_type} | 上下文:{context_length}")
                    skipped_tests += 1
                else:
                    print(f"\n进度: {current_test}/{total_tests}")
                    
                    output, backend_type, n_gen, repetitions = run_benchmark(model_path, context_length, use_gpu=True)
                    if output:
                        results = extract_tokens_per_second(output)
                        print(f"📊 {model_name} | {backend_type} | 上下文:{context_length}")
                        for result in results:
                            tokens_per_sec = result.get('avg_ts', 0)
                            phase = result.get('phase', 'unknown')
                            print(f"   {phase}: {tokens_per_sec:.2f} t/s")
                            
                            # 立即保存到CSV
                            csv_row = {
                                'model_name': model_name,
                                'backend': backend_type,
                                'n_gen': n_gen,
                                'repetitions': repetitions,
                                'context_length': context_length,
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
            mdfile.write(f"**上下文长度**: {', '.join(map(str, CONTEXT_LENGTHS))}\n\n")
            mdfile.write(f"**生成token数**: {N_GEN}\n\n")
            mdfile.write(f"**重复次数**: {REPETITIONS}\n\n")
            mdfile.write(f"**Flash Attention**: {'启用' if FLASH_ATTN else '禁用'}\n\n")
            
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
                mdfile.write("| 后端 | 上下文长度 | Prompt Processing (t/s) | Token Generation (t/s) |\n")
                mdfile.write("|------|------------|-------------------------|------------------------|\n")
                
                # 按后端和上下文长度组织数据
                model_data = {}
                for row in models[model_name]:
                    backend = row['backend']
                    context_length = int(row['context_length'])
                    phase = row['phase']
                    tokens_per_sec = float(row['tokens_per_sec'])
                    
                    key = (backend, context_length)
                    if key not in model_data:
                        model_data[key] = {}
                    model_data[key][phase] = tokens_per_sec
                
                # 按后端和上下文长度排序输出
                for (backend, context_length) in sorted(model_data.keys(), key=lambda x: (x[0], x[1])):
                    data = model_data[(backend, context_length)]
                    pp_speed = data.get('prompt_processing', 'N/A')
                    tg_speed = data.get('token_generation', 'N/A')
                    
                    pp_str = f"{pp_speed:.2f}" if isinstance(pp_speed, (int, float)) else str(pp_speed)
                    tg_str = f"{tg_speed:.2f}" if isinstance(tg_speed, (int, float)) else str(tg_speed)
                    
                    mdfile.write(f"| {backend} | {context_length} | {pp_str} | {tg_str} |\n")
                
                mdfile.write("\n")
            
            # 生成统计信息
            mdfile.write("## 测试统计\n\n")
            mdfile.write(f"- **模型数量**: {len(models)}\n")
            mdfile.write(f"- **上下文配置**: {', '.join(map(str, CONTEXT_LENGTHS))}\n")
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