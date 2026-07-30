"""
模块6：Jinja2 提示词模板
用于文档清洗、对话生成、查询增强的模板管理
所有 prompt 统一通过模板管理器渲染，便于维护和修改
"""

import os
from jinja2 import Template, FileSystemLoader, Environment
from config import OUTPUT_DIR


# ========== 模板定义 ==========

# 文档清洗模板
DOCUMENT_CLEANUP_TEMPLATE = Template("""你是一个文档清洗专家。请将以下原始文档内容清洗为结构化的Markdown格式。

要求：
1. 去除无意义的符号和空白
2. 保留所有有效信息
3. 使用清晰的标题层级
4. 保持原始内容的完整性

原始文档：
{{ raw_content }}

清洗后的Markdown：""")


# 产品信息提取模板
PRODUCT_EXTRACTION_TEMPLATE = Template("""从以下文档中提取{{ product_name }}的产品信息。

文档内容：
{{ content }}

请提取以下信息并输出JSON格式：
- 产品名称
- 产品类别
- 核心功能（至少3个）
- 技术参数
- 目标用户
- 产品亮点

输出格式：
```json
{
  "name": "",
  "category": "",
  "features": [],
  "specs": {},
  "target_users": "",
  "highlights": []
}
```""")


# FAQ 生成模板
FAQ_GENERATION_TEMPLATE = Template("""根据以下产品文档，生成5个常见问题及答案。

产品：{{ product_name }}
文档内容：
{{ content }}

请生成以下格式的FAQ：
1. Q: [问题]
   A: [答案]

2. Q: [问题]
   A: [答案]

...（共5组）""")


# 排障步骤模板
TROUBLESHOOT_TEMPLATE = Template("""根据以下产品文档，提取常见故障及排障步骤。

产品：{{ product_name }}
文档内容：
{{ content }}

请输出以下格式：
## 故障1：[故障描述]
- 原因：[可能原因]
- 解决方法：[排障步骤]

## 故障2：[故障描述]
...（至少3个常见故障）""")


# ========== 对话生成模板 ==========

# 智能对话模板（smart_assistant + chat_stream 共用）
CHAT_TEMPLATE = Template("""你是讯飞智能硬件产品的智能助手。你必须严格基于以下知识库文档回答用户问题。

【产品范围】
{{ product_context | default("全部产品", true) }}

【核心规则 - 违反将导致严重错误】
1. 禁止说"知识库中暂无"、"没有相关信息"、"无法回答"等放弃性语言
2. 即使文档中没有直接答案，也必须列出文档中所有与问题可能相关的内容
3. 如果文档提到了相关产品，必须列出该产品的所有已知参数和功能
4. 如果文档中有部分信息与问题相关，必须全部整理出来，逐条列出
5. 只要检索到任何文档，就必须输出文档中的内容，不能只说"没有"
6. 唯一允许说"暂无"的情况：知识库文档列表完全为空
7. 不要编造信息，但必须最大化利用已有文档中的每一个字

【回答格式要求 - 必须按以下结构输出】
1. **问题结论**：用一句话直接回答用户的问题
2. **操作步骤**（如果是操作类问题）：列出具体步骤，每步编号
3. **规格参数**（如果是参数类问题）：列出产品的规格参数
4. **注意事项**：如果有需要注意的事项，列出
5. **适用产品/版本**：说明该回答适用的产品型号和版本
6. **文档来源**：说明信息来源（如"产品基础信息"、"用户手册"等）
7. **相关视频**：如果知识库文档中提到了相关操作视频，列出视频标题（如"更多信息请参考产品操作视频"）
8. **相关问题**：根据当前问题，给出2~3个相关的延伸问题供用户参考

【如果文档中确实没有答案】
仍然要列出文档中实际包含的产品信息，并说明"以下为文档中与您问题相关的内容"

对话历史：
{{ history_context | default("（这是对话开始）", true) }}

知识库文档：
{{ kb_result }}

用户问题：{{ query }}

请按上述格式，基于文档内容回答（禁止说"没有"、"暂无"等）：""")


# ========== 查询增强模板 ==========

# HyDE 假设性文档生成模板
HYDE_TEMPLATE = Template("""请根据以下问题，写一段详细的回答（约200字）。
要求：回答要具体、包含产品名称、型号、技术参数、功能特点。模拟产品说明书的口吻。只输出回答内容。

问题：{{ query }}

回答：""")


# Query 重写模板
QUERY_REWRITE_TEMPLATE = Template("""将以下问题重写为更适合知识库检索的形式。
要求：保留核心语义，补充产品全称，使用书面语。只输出重写后的查询。

问题：{{ query }}

重写：""")


# 多扩展查询模板
QUERY_EXPAND_TEMPLATE = Template("""请将以下问题改写为{{ n }}个不同的检索查询，每行一个。
要求：
1. 围绕同一主题，从不同角度表述（如：功能、参数、价格、型号、对比等）
2. 使用不同的关键词组合
3. 包含产品全称和型号缩写
4. 只输出查询，每行一个，不要编号

问题：{{ query }}

{{ n }}个查询：""")


# LLM 重排模板
LLM_RERANK_TEMPLATE = Template("""判断以下文档与查询的相关性，只输出保留的文档编号。

查询：{{ query }}

候选文档：
{{ docs_text }}

规则：
- 只要文档包含与查询相关的信息就保留
- 只输出保留的文档编号，用逗号分隔
- 格式示例：2,5,1

保留的文档编号：""")


# 回答生成模板
ANSWER_TEMPLATE = Template("""你是讯飞产品知识库的智能助手。请根据以下所有文档内容回答用户问题。

【产品范围】
{{ product_context | default("全部产品", true) }}

【回答格式 - 必须按以下结构输出，每个部分都要有】
1. **问题结论**：用一句话直接回答用户的问题
2. **操作步骤**（如果是操作类问题）：列出具体步骤，每步编号
3. **规格参数**（如果是参数类问题）：列出产品的规格参数
4. **注意事项**：如果有需要注意的事项，列出
5. **适用产品/版本**：说明该回答适用的产品型号和版本
6. **文档来源**：说明信息来源（如"产品基础信息"、"用户手册"等）
7. **相关视频**：如果知识库文档中提到了相关操作视频，列出视频标题（如"更多信息请参考产品操作视频"）
8. **相关问题**：根据当前问题，给出2~3个相关的延伸问题供用户参考

【核心规则】
1. 优先基于知识库文档回答，禁止编造产品参数和功能
2. 如果知识库中有相关信息，必须详细提取并整理
3. 如果知识库中没有完全匹配的信息，但有相关产品信息，结合已有信息回答
4. 禁止说"知识库中暂无"、"没有相关信息"、"无法回答"等放弃性语言
5. 只要检索到任何文档，就必须输出文档中的内容，不能只说"没有"

知识库全部文档（共{{ doc_count }}个）：
{{ context }}

用户问题：{{ query }}

请按上述格式，基于文档内容回答（禁止说"没有"、"暂无"等）：""")


# ========== 模板管理器 ==========

class PromptTemplateManager:
    """提示词模板管理器"""

    def __init__(self):
        self.templates = {
            # 文档处理类
            "document_cleanup": DOCUMENT_CLEANUP_TEMPLATE,
            "product_extraction": PRODUCT_EXTRACTION_TEMPLATE,
            "faq_generation": FAQ_GENERATION_TEMPLATE,
            "troubleshooting": TROUBLESHOOT_TEMPLATE,
            # 对话生成类
            "chat": CHAT_TEMPLATE,
            # 查询增强类
            "hyde": HYDE_TEMPLATE,
            "query_rewrite": QUERY_REWRITE_TEMPLATE,
            "query_expand": QUERY_EXPAND_TEMPLATE,
            "llm_rerank": LLM_RERANK_TEMPLATE,
            "answer": ANSWER_TEMPLATE,
        }

    def render(self, template_name: str, **kwargs) -> str:
        """渲染指定模板"""
        if template_name not in self.templates:
            raise ValueError(f"模板 '{template_name}' 不存在。可用模板: {list(self.templates.keys())}")

        template = self.templates[template_name]
        print(f"  [Jinja2模板] 渲染模板: {template_name} | 参数: {list(kwargs.keys())}")
        return template.render(**kwargs)

    def list_templates(self) -> list:
        """列出所有可用模板"""
        return list(self.templates.keys())

    def add_template(self, name: str, template_str: str):
        """添加新模板"""
        self.templates[name] = Template(template_str)

    def save_template_example(self, template_name: str, output_dir: str = OUTPUT_DIR):
        """保存模板渲染示例"""
        os.makedirs(output_dir, exist_ok=True)

        examples = {
            "document_cleanup": {"raw_content": "示例原始文档内容..."},
            "product_extraction": {"product_name": "讯飞智能办公本X2", "content": "示例产品文档..."},
            "faq_generation": {"product_name": "讯飞AI录音卡", "content": "示例产品文档..."},
            "troubleshooting": {"product_name": "讯飞翻译机", "content": "示例产品文档..."},
            "chat": {"history_context": "用户: 讯飞有哪些产品", "kb_result": "文档1: 讯飞录音笔...", "query": "讯飞翻译机多少钱"},
            "hyde": {"query": "讯飞办公本有什么功能"},
            "query_rewrite": {"query": "这个东西咋用"},
            "query_expand": {"query": "讯飞翻译机", "n": 3},
            "llm_rerank": {"query": "翻译机", "docs_text": "[文档1] ...\n[文档2] ..."},
            "answer": {"doc_count": 5, "context": "文档1: ...\n文档2: ...", "query": "讯飞产品有哪些"},
        }

        if template_name in examples:
            rendered = self.render(template_name, **examples[template_name])
            file_path = os.path.join(output_dir, f"template_{template_name}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(rendered)
            print(f"  已保存示例: {file_path}")


# 全局实例
template_manager = PromptTemplateManager()


if __name__ == "__main__":
    print("Jinja2 提示词模板管理器")
    print(f"可用模板: {template_manager.list_templates()}")

    # 保存所有模板示例
    for name in template_manager.list_templates():
        template_manager.save_template_example(name)
