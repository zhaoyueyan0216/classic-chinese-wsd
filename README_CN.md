# 古汉语词义消歧系统

## 项目概述

本项目实现了一个完整的古汉语词义消歧（WSD）管道，使用伪古文生成、Pyserini BM25检索和基于LLM的决策方法。该系统旨在准确识别上下文中古汉语词的正确义项。

## 系统架构

系统由以下组件组成：

1. **伪古文生成** - 生成伪古文示例以提高检索效果
2. **Pyserini BM25检索** - 从语料库中检索相关支持句
3. **嵌入重排** - 根据语义相似度对检索到的句子进行重排序
4. **基于LLM的Top-k过滤** - 使用LLM过滤top-k候选义项
5. **历史证据构建** - 从检索到的句子构建证据
6. **最终LLM决策** - 做出最终的词义消歧决策

## 项目结构

```
testbm25pipeline/
├── main.py                # 主测试脚本
├── pseudo_retrieval.py     # 伪古文生成和检索
├── embedding_utils.py      # 嵌入工具
├── llm.py                 # LLM交互工具
├── prototype.py           # 原型向量构建
├── schema.py              # 用法模式构建
├── decision.py            # 决策工具
├── requirements.txt       # 项目依赖
├── data/                  # 测试数据
│   ├── character/         # 字符级数据
│   └── compound/          # 复合词数据
├── guwenbert_output/      # GuwenBERT模型输出
└── bm25_index_zh/         # BM25索引目录
```

## 前提条件

- Python 3.8+
- PIP包管理器
- LLM API访问权限（如OpenAI、DeepSeek）
- Pyserini用于BM25检索
- 预构建的古汉语语料库BM25索引

## 安装

1. 克隆仓库：

```bash
git clone <repository-url>
cd testbm25pipeline
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 设置LLM API访问的环境变量（如果需要）：

```bash
# 例如，对于OpenAI
export OPENAI_API_KEY=your-api-key

# 对于DeepSeek
export DEEPSEEK_API_KEY=your-api-key
```

4. 确保BM25索引在`main.py`中指定的路径可用：

```python
index_dir = r"/mimer/NOBACKUP/groups/cik_data/yueyan/testfrozen_pipeline/bm25_index_zh"
```

## 使用方法

### 运行完整系统

要运行启用了所有组件的完整系统：

```bash
python main.py
```

### 运行消融实验

系统支持多个消融实验，以评估不同组件的贡献：

```bash
# 不使用Top-2过滤
python main.py ablation_no_top2

# 只使用Top-2过滤（无其他组件）
python main.py ablation_top2_only

# 无历史证据
python main.py ablation_no_history

# 不使用伪古文生成
python main.py ablation_no_pseudo

# 不使用用法模式
python main.py ablation_no_schema

# 不使用相似度计算
python main.py ablation_no_similarity
```

## 配置选项

系统通过`main.py`中的`BASE_CONFIG`字典进行配置：

| 配置选项 | 描述 | 默认值 |
|---------|------|--------|
| `use_top2_filter` | 使用LLM过滤top-2候选义项 | True |
| `use_pseudo_query` | 使用伪古文生成 | True |
| `use_bm25` | 使用Pyserini BM25检索 | True |
| `use_rerank` | 使用基于嵌入的重排序 | True |
| `use_schema` | 使用用法模式构建 | True |
| `use_similarity` | 使用相似度计算 | True |
| `use_supporting_sentences` | 使用支持句 | True |
| `final_decision_with_llm` | 使用LLM进行最终决策 | True |
| `use_final_cache` | 使用最终LLM决策的缓存 | False |

## 输出

系统生成Excel或CSV格式的详细测试结果，包括：

- 目标词
- 上下文句子
- 黄金义项ID和定义
- 预测义项ID
- 预测结果（正确/错误）
- 排除的义项
- 决策推理
- 候选义项数量
- LLM生成的Top-2义项

## 评估指标

系统使用准确率评估性能，计算方式为：

```
准确率 = 正确预测数量 / 测试用例总数
```

## 示例输出

```
古汉语词义消歧系统测试 - 使用伪古文生成
配置：full
==========================================================================================
配置详情：
  use_top2_filter: True
  use_pseudo_query: True
  use_bm25: True
  use_rerank: True
  use_schema: True
  use_similarity: True
  use_supporting_sentences: True
  final_decision_with_llm: True
  use_final_cache: False
==========================================================================================
加载了 1000 个测试上下文
使用 999 个义项进行测试
实际需要测试的目标词数量：99（已排序）
排序后的测试词：['严', '丹', '体', '余', '倕', '偶', '入', '冲', '凉', '劳', ...]

>>> 开始批量处理，共 10 批，每批 100 个测试用例

=== 批次 1 完成，累计结果：85/100 = 0.850 ===
=== 批次 2 完成，累计结果：172/200 = 0.860 ===
...

>>> 导出测试结果到Excel
✅ 成功导出测试结果到 pseudo_test_results_full_20260414_123456.xlsx
导出了 999 条测试记录
添加了统计信息行：测试用例数量=999，正确预测数量=867，准确率=0.868

==========================================================================================
测试完成：867/999 = 0.868
==========================================================================================
```

## 故障排除

1. **LLM API错误**
   - 确保API密钥设置正确
   - 检查网络连接
   - 验证API速率限制

2. **BM25索引问题**
   - 确保索引目录存在
   - 验证索引是否正确构建
   - 检查文件权限

3. **嵌入错误**
   - 确保嵌入模型可访问
   - 检查内存限制

4. **输出导出问题**
   - 安装所需依赖：`pip install openpyxl pandas`
   - 检查输出目录的写入权限

## 贡献

欢迎为改进系统做出贡献。请按照以下步骤：

1. Fork仓库
2. 创建新分支
3. 进行修改
4. 提交拉取请求

## 许可证

本项目采用MIT许可证。

## 致谢

- GuwenBERT用于古汉语语言建模
- Pyserini用于高效BM25检索
- LLM提供商用于语义理解
