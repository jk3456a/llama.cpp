#!/bin/bash

# GGUF模型基准测试运行脚本
# 用于编译llama-bench并运行基准测试

echo "=== LLAMA.CPP GGUF模型基准测试 ==="
echo ""

# 检查是否在llama.cpp根目录
if [ ! -f "CMakeLists.txt" ] || [ ! -d "tools/llama-bench" ]; then
    echo "错误: 请在llama.cpp根目录下运行此脚本"
    exit 1
fi

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到conda环境，请确保您在正确的环境中"
else
    echo "当前conda环境: $CONDA_DEFAULT_ENV"
fi

# 检查是否已编译llama-bench
if [ ! -f "tools/llama-bench/llama-bench" ]; then
    echo "llama-bench未找到，开始编译..."
    
    # 创建build目录
    mkdir -p build
    cd build
    
    # 配置和编译
    echo "配置CMake..."
    cmake .. -DCMAKE_BUILD_TYPE=Release
    
    if [ $? -ne 0 ]; then
        echo "CMake配置失败"
        exit 1
    fi
    
    echo "编译llama-bench..."
    make llama-bench -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    if [ $? -ne 0 ]; then
        echo "编译失败"
        exit 1
    fi
    
    # 复制到工具目录
    cp bin/llama-bench ../tools/llama-bench/
    cd ..
    
    echo "编译完成！"
else
    echo "llama-bench已存在，跳过编译"
fi

# 检查Python脚本是否存在
if [ ! -f "benchmark_gguf_models.py" ]; then
    echo "错误: 找不到benchmark_gguf_models.py脚本"
    exit 1
fi

# 设置执行权限
chmod +x benchmark_gguf_models.py
chmod +x csvtoreport.py

# 运行基准测试
echo ""
echo "开始运行基准测试..."
echo "这可能需要较长时间，请耐心等待..."
echo ""

python3 benchmark_gguf_models.py
python3 csvtoreport.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=== 测试完成 ==="
    echo "结果文件："
    [ -f "benchmark_results.csv" ] && echo "- benchmark_results.csv"
    [ -f "report.md" ] && echo "- report.md"
else
    echo "测试过程中出现错误"
    exit 1
fi 