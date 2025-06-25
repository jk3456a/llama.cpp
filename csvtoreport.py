import csv
import os
from datetime import datetime

# 读取CSV数据
CSV_FILENAME = 'benchmark_results.csv'
csv_data = []
if os.path.exists(CSV_FILENAME):
    with open(CSV_FILENAME, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        csv_data = list(reader)

# 生成报告
with open('report.md', 'w', encoding='utf-8') as mdfile:
    mdfile.write('# GGUF模型基准测试报告\n\n')
    mdfile.write(f'**生成时间**: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\n\n')
    mdfile.write('**测试目录**: \`/Users/yummy/.lmstudio/models\`\n\n')
    mdfile.write('**上下文长度**: 1024, 2048, 4096\n\n')
    mdfile.write('**生成token数**: 16\n\n')
    mdfile.write('**重复次数**: 1\n\n')
    mdfile.write('**Flash Attention**: 启用\n\n')
    
    mdfile.write('## 测试结果\n\n')
    
    # 按模型分组
    models = {}
    for row in csv_data:
        model_name = row['model_name']
        if model_name not in models:
            models[model_name] = []
        models[model_name].append(row)
    
    # 为每个模型生成一个表格
    for model_name in sorted(models.keys()):
        mdfile.write(f'### {model_name}\n\n')
        
        # 创建表格头
        mdfile.write('| 后端 | 上下文长度 | Prompt Processing (t/s) | Token Generation (t/s) |\n')
        mdfile.write('|------|------------|-------------------------|------------------------|\n')
        
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
            
            pp_str = f'{pp_speed:.2f}' if isinstance(pp_speed, (int, float)) else str(pp_speed)
            tg_str = f'{tg_speed:.2f}' if isinstance(tg_speed, (int, float)) else str(tg_speed)
            
            mdfile.write(f'| {backend} | {context_length} | {pp_str} | {tg_str} |\n')
        
        mdfile.write('\n')

print('✓ Markdown报告生成成功')