"""
模块6：Jinja2 提示词模板
用于文档清洗、数据提取、查询增强的模板管理
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


# 查询增强模板
QUERY_ENHANCE_TEMPLATE = Template("""将以下用户查询改写为更适合知识库检索的形式。

原始查询：{{ query }}
改写要求：{{ requirement }}

改写后的查询：""")


# ========== 模板管理器 ==========

class PromptTemplateManager:
    """提示词模板管理器"""

    def __init__(self):
        self.templates = {
            "document_cleanup": DOCUMENT_CLEANUP_TEMPLATE,
            "product_extraction": PRODUCT_EXTRACTION_TEMPLATE,
            "faq_generation": FAQ_GENERATION_TEMPLATE,
            "troubleshooting": TROUBLESHOOT_TEMPLATE,
            "query_enhance": QUERY_ENHANCE_TEMPLATE,
        }

    def render(self, template_name: str, **kwargs) -> str:
        """渲染指定模板"""
        if template_name not in self.templates:
            raise ValueError(f"模板 '{template_name}' 不存在。可用模板: {list(self.templates.keys())}")

        template = self.templates[template_name]
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
            "query_enhance": {"query": "这个东西咋用", "requirement": "使用书面语，补充产品全称"},
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
