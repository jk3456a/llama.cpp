# GGUF模型基准测试工具

这个工具集用于自动测试指定目录下的所有GGUF模型文件，并生成详细的性能报告。

## 文件说明

- `benchmark_models.py` - 主要的Python测试脚本
- `run_benchmark.sh` - Shell包装脚本，负责编译和运行
- `BENCHMARK_README.md` - 本说明文档

## 使用方法

### 1. 环境准备

确保您在llama.cpp的conda环境中：

```bash
conda activate llamacpp  # 或您的llama.cpp环境名称
```

### 2. 配置参数

编辑 `benchmark_models.py` 中的配置参数（如需要）：

```python
MODEL_DIR = "/Users/yummy/.lmstudio/models"  # 模型文件目录
DEPTHS = [1024, 2048, 1096]  # 测试深度
N_GEN = 16  # 生成token数量
REPETITIONS = 1  # 重复次数
FLASH_ATTN = 1  # 启用flash attention
```

### 3. 运行测试

在llama.cpp根目录下运行：

```bash
./run_benchmark.sh
```

脚本会自动：
1. 检查环境
2. 编译llama-bench工具（如果需要）
3. 查找所有.gguf文件
4. 运行基准测试
5. 生成CSV和Markdown报告

### 4. 查看结果

测试完成后会生成两个文件：

- `benchmark_results.csv` - 详细的CSV数据
- `report.md` - 格式化的Markdown报告

## 测试参数说明

基准测试使用以下参数：

- `-m` : 模型文件路径
- `-fa 1` : 启用Flash Attention
- `-d {depth}` : 设置上下文深度（1024, 2048, 1096）
- `-n 16` : 生成16个token
- `-r 1` : 每个测试重复1次
- `-o json` : 输出JSON格式便于解析

## 注意事项

1. 测试可能需要较长时间，具体取决于：
   - 模型数量
   - 模型大小
   - 硬件配置

2. 确保有足够的GPU内存来加载大型模型

3. 如果某个模型测试失败，会跳过继续测试其他模型

4. 所有测试结果都会汇总到最终报告中

## 故障排除

### 找不到llama-bench
- 确保在llama.cpp根目录运行
- 脚本会自动编译llama-bench

### 找不到模型文件
- 检查 `MODEL_DIR` 路径是否正确
- 确保目录中包含.gguf文件

### 测试超时
- 单个测试的超时时间是300秒
- 可以在脚本中调整 `timeout` 参数

### 内存不足
- 考虑减少GPU层数或使用CPU后端
- 可以修改测试参数来适应硬件限制

## 自定义配置

您可以根据需要修改 `benchmark_models.py` 中的参数：

```python
# 自定义测试深度
DEPTHS = [512, 1024, 2048, 4096]

# 增加重复次数以提高准确性
REPETITIONS = 3

# 调整生成token数量
N_GEN = 32

# 禁用Flash Attention
FLASH_ATTN = 0
```

重新运行脚本应用新配置。 