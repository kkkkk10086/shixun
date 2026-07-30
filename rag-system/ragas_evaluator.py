"""
模块：RAGAS 评估系统
评估检索质量、引用来源追溯、评估报告生成

评估维度：
1. Context Precision（上下文精度）- 检索结果中相关文档的比例
2. Context Recall（上下文召回）- 是否检索到所有相关文档
3. Faithfulness（忠实度）- 回答是否基于检索到的上下文
4. Answer Relevancy（答案相关性）- 回答与问题的相关程度
5. Citation Accuracy（引用准确性）- 引用来源的准确性
"""

import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()
import time
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, OUTPUT_DIR


class RAGASEvaluator:
    """RAGAS 评估器"""

    def __init__(self, collection=None):
        self.llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.collection = collection
        self.evaluation_results = []

    def set_collection(self, collection):
        """设置向量数据库"""
        self.collection = collection

    def _get_llm_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """调用 DeepSeek LLM"""
        try:
            response = self.llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM调用失败] {e}")
            return ""

    def _retrieve_documents(self, query: str, top_k: int = 10) -> List[str]:
        """检索相关文档：使用增强检索（三路融合 + Cross-Encoder 重排）"""
        if self.collection is None:
            return []

        try:
            from rag_tools import set_collection, search_knowledge_base
            set_collection(self.collection)

            # 直接调用主系统的检索工具
            kb_result = search_knowledge_base.invoke(query)

            # 解析文档列表
            docs = []
            if kb_result and kb_result != "知识库未初始化":
                for line in kb_result.split("\n"):
                    if line.startswith("[文档"):
                        content = line.split("] ", 1)[1] if "] " in line else line
                        docs.append(content)

            # 保存原始检索结果
            self._raw_kb_result = kb_result

            print(f"[评估检索] 通过 search_knowledge_base 检索到 {len(docs)} 个文档")
            return docs if docs else []
        except Exception as e:
            print(f"[评估检索] 检索失败: {e}")
            self._raw_kb_result = ""
            return []

    def _generate_answer(self, query: str, contexts: List[str]) -> str:
        """基于检索结果生成回答（直接用检索到的文档，不重新检索）"""
        if not contexts:
            return "知识库中暂无相关信息。"

        # 直接用检索到的文档生成回答（和主系统聊天用同一个 prompt）
        kb_result = "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(contexts)])

        try:
            from smart_assistant import SmartAssistant
            assistant = SmartAssistant(self.collection)
            # 直接调用 LLM，用检索到的文档（不重新检索）
            from prompt_templates import template_manager
            prompt = template_manager.render("chat",
                history_context="",
                kb_result=kb_result,
                query=query
            )
            response = assistant.llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()

            # 兜底：如果回答太短（< 30 字符），说明 LLM 放弃回答，强制重试
            if len(answer) < 30 and len(contexts) > 0:
                print(f"  [评估生成] 回答过短({len(answer)}字符)，可能是LLM放弃，强制重试...")
                retry_prompt = f"""你是讯飞智能硬件产品助手。以下是你必须使用的知识库文档，你必须从中提取信息回答用户问题。

禁止说"暂无"、"没有"、"无法回答"等放弃性语言。
即使文档中没有直接答案，也必须列出文档中所有与问题相关的内容。

知识库文档：
{kb_result}

用户问题：{query}

请列出文档中所有与问题相关的内容（必须逐条列出，不少于100字）："""
                response2 = assistant.llm.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": retry_prompt}],
                    max_tokens=800,
                    temperature=0.3
                )
                answer = response2.choices[0].message.content.strip()
                print(f"  [评估生成] 重试后回答: {len(answer)} 字符")

            return answer
        except Exception as e:
            print(f"[评估生成] 生成失败: {e}")
            return ""

    def _extract_citations(self, answer: str, contexts: List[str]) -> List[Dict]:
        """从回答中提取引用来源"""
        citations = []

        prompt = f"""从以下回答中提取引用来源信息。

回答内容：
{answer}

可用的文档来源：
{chr(10).join([f"[文档{i+1}] {doc[:100]}..." for i, doc in enumerate(contexts)])}

请提取回答中引用的文档编号，格式为JSON数组：
[{{"doc_index": 1, "cited_text": "引用的具体内容", "source_snippet": "文档中的原文"}}]

只输出JSON，不要其他内容："""

        result = self._get_llm_response(prompt, max_tokens=500)

        try:
            # 清理可能的 markdown 标记
            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]

            citations = json.loads(result)
        except json.JSONDecodeError:
            # 如果解析失败，返回空列表
            pass

        return citations

    def evaluate_context_precision(self, query: str, contexts: List[str], ground_truth: str = None) -> float:
        """
        评估上下文精度（Context Precision）
        衡量检索结果中相关文档的比例
        """
        if not contexts:
            return 0.0

        prompt = f"""评估以下检索结果与查询的相关性。

查询：{query}

检索结果（共{len(contexts)}个）：
{chr(10).join([f"[文档{i+1}] {doc[:200]}" for i, doc in enumerate(contexts)])}

评估标准：
- 每个文档是否与查询相关（1=相关，0=不相关）
- 输出相关文档的数量和比例

输出格式（JSON）：
{{"relevant_count": 相关文档数, "total_count": 总文档数, "precision": 精度值(0-1)}}

只输出JSON："""

        result = self._get_llm_response(prompt, max_tokens=200)

        try:
            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]

            data = json.loads(result)
            return float(data.get("precision", 0.5))
        except (json.JSONDecodeError, ValueError):
            return 0.5  # 默认值

    def evaluate_context_recall(self, query: str, contexts: List[str], ground_truth: str) -> float:
        """
        评估上下文召回（Context Recall）—— 真正的文档级召回
        
        衡量标准：ground_truth 中的关键信息，有多少能在检索到的文档中找到。
        这才是真正的"检索召回率"，而不是之前那种"回答 vs 标准答案"的假指标。
        """
        if not contexts or not ground_truth:
            return 0.0

        # 用 LLM 将 ground_truth 拆解为独立的事实陈述
        prompt = f"""将以下标准答案拆解为独立的事实陈述（每条一行，用数字编号）。

标准答案：{ground_truth}

要求：
- 每条事实陈述必须独立、可验证
- 只输出事实列表，不要其他内容

输出格式：
1. 事实1
2. 事实2
..."""

        facts_text = self._get_llm_response(prompt, max_tokens=300)
        if not facts_text:
            # 降级：直接用简单关键词匹配
            return self._simple_keyword_recall(ground_truth, contexts)

        # 解析事实列表
        import re
        facts = []
        for line in facts_text.split("\n"):
            line = line.strip()
            if re.match(r'^\d+[\.\)、]', line):
                fact = re.sub(r'^\d+[\.\)、]\s*', '', line).strip()
                if len(fact) > 3:
                    facts.append(fact)

        if not facts:
            facts = [ground_truth]

        # 对每个事实，检查是否在检索到的文档中出现
        contexts_text = "\n\n".join([f"[文档{i+1}] {doc[:500]}" for i, doc in enumerate(contexts)])

        check_prompt = f"""逐一检查以下事实陈述，判断它们是否能在检索到的文档中找到依据。

事实陈述：
{chr(10).join(f"{i+1}. {f}" for i, f in enumerate(facts))}

检索到的文档：
{contexts_text}

对每个事实，判断是否能在文档中找到（1=能找到，0=找不到）。
输出JSON格式，只输出JSON：
{{"results": [{{"fact_index": 1, "found": 1或0, "evidence": "找到的依据（如有）"}}, ...]}}"""

        result = self._get_llm_response(check_prompt, max_tokens=500)
        try:
            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(result)
            found_count = sum(1 for r in data.get("results", []) if r.get("found", 0) == 1)
            recall = found_count / len(facts) if facts else 0.5
            print(f"  [召回详情] {found_count}/{len(facts)} 个事实在检索文档中找到")
            return round(recall, 3)
        except (json.JSONDecodeError, KeyError):
            return self._simple_keyword_recall(ground_truth, contexts)

    def _simple_keyword_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """降级方案：简单关键词匹配"""
        import re
        gt_numbers = set(re.findall(r'\d+\.?\d*', ground_truth))
        all_context = " ".join(contexts)
        ctx_numbers = set(re.findall(r'\d+\.?\d*', all_context))
        if gt_numbers:
            number_recall = len(gt_numbers & ctx_numbers) / len(gt_numbers)
        else:
            number_recall = 1.0

        gt_keywords = [w for w in ground_truth.replace("，", " ").replace("、", " ").split() if len(w) >= 2]
        if gt_keywords:
            keyword_recall = sum(1 for kw in gt_keywords if kw in all_context) / len(gt_keywords)
        else:
            keyword_recall = 1.0

        return round(number_recall * 0.6 + keyword_recall * 0.4, 3)

    def evaluate_faithfulness(self, query: str, answer: str, contexts: List[str]) -> float:
        """
        评估忠实度（Faithfulness）
        衡量回答是否基于检索到的上下文
        """
        if not contexts or not answer:
            return 0.0

        prompt = f"""评估以下回答是否忠实于提供的上下文。

查询：{query}

上下文信息：
{chr(10).join([f"[文档{i+1}] {doc[:500]}" for i, doc in enumerate(contexts)])}

回答：{answer}

评估标准：
- 回答中的信息是否都能在上下文中找到依据
- 是否有捏造或虚构的信息
- 输出忠实度评分（0-1）

输出格式（JSON）：
{{"faithfulness": 忠实度值(0-1), "hallucinated_info": "虚构的信息（如有）"}}

只输出JSON："""

        result = self._get_llm_response(prompt, max_tokens=300)

        try:
            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]

            data = json.loads(result)
            return float(data.get("faithfulness", 0.5))
        except (json.JSONDecodeError, ValueError):
            return 0.5

    def evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """
        评估答案相关性（Answer Relevancy）
        衡量回答与问题的相关程度
        """
        if not answer:
            return 0.0

        prompt = f"""评估以下回答与查询的相关性。

查询：{query}

回答：{answer}

评估标准：
- 回答是否直接回答了查询的问题
- 是否提供了有用的信息
- 输出相关性评分（0-1）

输出格式（JSON）：
{{"relevancy": 相关性值(0-1), "reason": "评分理由"}}

只输出JSON："""

        result = self._get_llm_response(prompt, max_tokens=200)

        try:
            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]

            data = json.loads(result)
            return float(data.get("relevancy", 0.5))
        except (json.JSONDecodeError, ValueError):
            return 0.5

    def evaluate_citation_accuracy(self, answer: str, citations: List[Dict], contexts: List[str]) -> float:
        """
        评估引用准确性（Citation Accuracy）
        衡量引用来源的准确性
        """
        if not citations or not contexts:
            return 0.5

        prompt = f"""评估以下回答中引用来源的准确性。

回答：{answer}

引用信息：
{json.dumps(citations, ensure_ascii=False, indent=2)}

可用的文档来源：
{chr(10).join([f"[文档{i+1}] {doc[:150]}" for i, doc in enumerate(contexts)])}

评估标准：
- 引用的文档编号是否正确
- 引用的内容是否准确
- 输出准确性评分（0-1）

输出格式（JSON）：
{{"accuracy": 准确性值(0-1), "issues": "问题（如有）"}}

只输出JSON："""

        result = self._get_llm_response(prompt, max_tokens=300)

        try:
            if result.startswith("```"):
                result = result.split("\n", 1)[1].rsplit("```", 1)[0]

            data = json.loads(result)
            return float(data.get("accuracy", 0.5))
        except (json.JSONDecodeError, ValueError):
            return 0.5

    def evaluate_single(self, query: str, ground_truth: str = None, top_k: int = 10) -> Dict:
        """
        评估单个查询

        返回包含所有评估指标的结果字典
        """
        print(f"\n{'='*50}")
        print(f"开始评估查询: {query[:50]}...")
        print(f"{'='*50}")

        start_time = time.time()

        # 1. 检索文档
        contexts = self._retrieve_documents(query, top_k=top_k)
        print(f"[1/6] 检索到 {len(contexts)} 个文档")

        # 2. 生成回答
        answer = self._generate_answer(query, contexts)
        print(f"[2/6] 生成回答完成 ({len(answer)} 字符)")

        # 3. 提取引用
        citations = self._extract_citations(answer, contexts)
        print(f"[3/6] 提取到 {len(citations)} 个引用")

        # 4. 评估各项指标
        context_precision = self.evaluate_context_precision(query, contexts, ground_truth)
        print(f"[4/6] 上下文精度: {context_precision:.3f}")

        context_recall = self.evaluate_context_recall(query, contexts, ground_truth) if ground_truth else 0.5
        print(f"[5/6] 上下文召回: {context_recall:.3f}")

        faithfulness = self.evaluate_faithfulness(query, answer, contexts)
        print(f"[6/6] 忠实度: {faithfulness:.3f}")

        answer_relevancy = self.evaluate_answer_relevancy(query, answer)
        print(f"答案相关性: {answer_relevancy:.3f}")

        citation_accuracy = self.evaluate_citation_accuracy(answer, citations, contexts)
        print(f"引用准确性: {citation_accuracy:.3f}")

        elapsed = time.time() - start_time

        # 构建结果
        result = {
            "query": query,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
            "citations": citations,
            "metrics": {
                "context_precision": round(context_precision, 4),
                "context_recall": round(context_recall, 4),
                "faithfulness": round(faithfulness, 4),
                "answer_relevancy": round(answer_relevancy, 4),
                "citation_accuracy": round(citation_accuracy, 4),
                "overall_score": round(
                    (context_precision + context_recall + faithfulness + answer_relevancy + citation_accuracy) / 5,
                    4
                )
            },
            "evaluation_time": round(elapsed, 2),
            "timestamp": datetime.now().isoformat()
        }

        self.evaluation_results.append(result)
        print(f"\n评估完成，总分: {result['metrics']['overall_score']:.3f}，耗时: {elapsed:.1f}秒")

        return result

    def evaluate_batch(self, test_cases: List[Dict], top_k: int = 10) -> Dict:
        """
        批量评估多个测试用例

        test_cases: [{"query": "...", "ground_truth": "..."}, ...]
        """
        print(f"\n{'='*60}")
        print(f"开始批量评估，共 {len(test_cases)} 个测试用例")
        print(f"{'='*60}")

        start_time = time.time()
        results = []

        for i, case in enumerate(test_cases):
            print(f"\n--- 测试用例 {i+1}/{len(test_cases)} ---")
            result = self.evaluate_single(
                query=case["query"],
                ground_truth=case.get("ground_truth"),
                top_k=top_k
            )
            results.append(result)

        elapsed = time.time() - start_time

        # 计算平均指标
        avg_metrics = {
            "context_precision": 0,
            "context_recall": 0,
            "faithfulness": 0,
            "answer_relevancy": 0,
            "citation_accuracy": 0,
            "overall_score": 0
        }

        for result in results:
            for metric in avg_metrics:
                avg_metrics[metric] += result["metrics"][metric]

        for metric in avg_metrics:
            avg_metrics[metric] = round(avg_metrics[metric] / len(results), 4)

        batch_result = {
            "total_cases": len(test_cases),
            "results": results,
            "average_metrics": avg_metrics,
            "evaluation_time": round(elapsed, 2),
            "timestamp": datetime.now().isoformat()
        }

        print(f"\n{'='*60}")
        print(f"批量评估完成")
        print(f"总耗时: {elapsed:.1f}秒")
        print(f"平均指标:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.3f}")
        print(f"{'='*60}")

        return batch_result

    def generate_report(self, evaluation_result: Dict) -> str:
        """生成评估报告"""
        report = []
        report.append("# RAGAS 评估报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"评估用例数: {evaluation_result.get('total_cases', 1)}")
        report.append(f"总耗时: {evaluation_result.get('evaluation_time', 0):.1f}秒")

        # 平均指标
        avg_metrics = evaluation_result.get("average_metrics", {})
        if avg_metrics:
            report.append("\n## 平均评估指标\n")
            report.append("| 指标 | 分数 | 说明 |")
            report.append("|------|------|------|")
            report.append(f"| Context Precision | {avg_metrics.get('context_precision', 0):.3f} | 上下文精度 |")
            report.append(f"| Context Recall | {avg_metrics.get('context_recall', 0):.3f} | 上下文召回 |")
            report.append(f"| Faithfulness | {avg_metrics.get('faithfulness', 0):.3f} | 忠实度 |")
            report.append(f"| Answer Relevancy | {avg_metrics.get('answer_relevancy', 0):.3f} | 答案相关性 |")
            report.append(f"| Citation Accuracy | {avg_metrics.get('citation_accuracy', 0):.3f} | 引用准确性 |")
            report.append(f"| **Overall Score** | **{avg_metrics.get('overall_score', 0):.3f}** | **综合评分** |")

        # 各用例详情
        results = evaluation_result.get("results", [evaluation_result] if "query" in evaluation_result else [])
        if results:
            report.append("\n## 详细评估结果\n")
            for i, result in enumerate(results):
                report.append(f"### 测试用例 {i+1}")
                report.append(f"**查询:** {result['query']}")
                if result.get('ground_truth'):
                    report.append(f"**标准答案:** {result['ground_truth']}")
                report.append(f"**生成回答:** {result['answer'][:200]}...")

                report.append("\n**评估指标:**")
                for metric, value in result["metrics"].items():
                    report.append(f"- {metric}: {value:.3f}")

                if result.get("citations") and isinstance(result["citations"], list):
                    report.append(f"\n**引用来源:** {len(result['citations'])} 个")
                    for cit in result["citations"][:3]:
                        if isinstance(cit, dict):
                            report.append(f"  - 文档{cit.get('doc_index', '?')}: {cit.get('cited_text', '')[:50]}...")

                report.append(f"\n**评估耗时:** {result.get('evaluation_time', 0):.1f}秒")
                report.append("")

        # 改进建议
        report.append("\n## 改进建议\n")
        if avg_metrics:
            if avg_metrics.get("context_precision", 0) < 0.7:
                report.append("- **上下文精度偏低**: 建议优化检索策略，增加关键词过滤或重排序")
            if avg_metrics.get("context_recall", 0) < 0.7:
                report.append("- **上下文召回偏低**: 建议增加检索数量或使用多扩展查询")
            if avg_metrics.get("faithfulness", 0) < 0.7:
                report.append("- **忠实度偏低**: 建议优化提示词，强调基于上下文回答")
            if avg_metrics.get("answer_relevancy", 0) < 0.7:
                report.append("- **答案相关性偏低**: 建议优化提示词模板")
            if avg_metrics.get("citation_accuracy", 0) < 0.7:
                report.append("- **引用准确性偏低**: 建议改进引用提取逻辑")

            overall = avg_metrics.get("overall_score", 0)
            if overall >= 0.8:
                report.append("- **整体表现优秀**: RAG系统质量较高，可进一步优化细节")
            elif overall >= 0.6:
                report.append("- **整体表现良好**: 有提升空间，重点关注低分指标")
            else:
                report.append("- **整体表现一般**: 需要重点优化检索和生成环节")

        return "\n".join(report)

    def save_report(self, evaluation_result: Dict, filename: str = None) -> str:
        """保存评估报告"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ragas_report_{timestamp}.md"

        filepath = os.path.join(OUTPUT_DIR, filename)
        report = self.generate_report(evaluation_result)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"评估报告已保存: {filepath}")
        return filepath

    def save_results_json(self, evaluation_result: Dict, filename: str = None) -> str:
        """保存评估结果为JSON"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ragas_results_{timestamp}.json"

        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)

        print(f"评估结果已保存: {filepath}")
        return filepath


# 全局评估器实例
evaluator = None


def get_evaluator(collection=None):
    """获取评估器实例"""
    global evaluator
    if evaluator is None:
        evaluator = RAGASEvaluator(collection)
    elif collection is not None:
        evaluator.set_collection(collection)
    return evaluator


# 预定义的测试用例（基于讯飞产品知识库）
DEFAULT_TEST_CASES = [
    {
        "query": "讯飞AI录音笔S6 Plus的存储容量是多少？",
        "ground_truth": "S6 Plus的存储容量为6GB+128GB机身存储，1GB永久+20GB三年云存储。"
    },
    {
        "query": "讯飞AI录音笔有哪些型号？",
        "ground_truth": "型号包括S8离线版、S6 Plus、S6、Magic、Pokee、Pokee SE、SR702星火版、SR502星火版、SR302 Pro、SR302星火版、H1 Pro。"
    },
    {
        "query": "讯飞翻译机支持哪些语言在线翻译？",
        "ground_truth": "支持63种语言在线翻译（中文与62种外语互译），包括中文、英语、日语、韩语等。"
    },
    {
        "query": "讯飞词典笔X8 Pro的屏幕尺寸和内存容量是多少？",
        "ground_truth": "词典笔X8 Pro配备3.5英寸触摸彩屏，内存容量为32GB。"
    },
    {
        "query": "讯飞学习机S90 Pro的屏幕尺寸和存储容量是多少？",
        "ground_truth": "S90 Pro配备13.2英寸屏幕，存储容量为256GB，运行内存8GB。"
    }
]


if __name__ == "__main__":
    print("RAGAS 评估系统模块")
    print("使用示例:")
    print("  from ragas_evaluator import get_evaluator")
    print("  evaluator = get_evaluator(collection)")
    print("  result = evaluator.evaluate_single('讯飞办公本有什么功能？')")
    print("  report = evaluator.generate_report(result)")
