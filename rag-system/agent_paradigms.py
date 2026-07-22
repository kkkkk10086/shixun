"""
模块7：Agent 范式实现
1. ReAct (Reasoning + Acting)
2. Plan-and-Solve
3. Reflection
"""

from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def get_llm():
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


# ============================================================
# 1. ReAct 范式（推理 + 行动）
# ============================================================

def react_agent(query: str, tools: dict) -> str:
    """
    ReAct 范式：推理 → 行动 → 观察 → 推理 → ...
    使用知识库文档，由 DeepSeek 分析回答
    """
    llm = get_llm()

    tool_desc = "\n".join([f"- {name}: {func.__doc__ or '无描述'}" for name, func in tools.items()])

    prompt = f"""你是一个智能助手，使用 ReAct 模式。

可用工具：
{tool_desc}

必须执行两步：
第1步：
Thought: 需要获取知识库文档
Action: search_knowledge
Action Input: 任意关键词
Observation: [返回所有文档]

第2步：
Thought: 需要将问题和文档发给 DeepSeek 分析
Action: analyze_with_llm
Action Input: 将以下内容发给 DeepSeek：问题：{query} + 上一步获取的文档内容
Observation: [DeepSeek 的分析回答]

最后：
Final Answer: 直接输出 DeepSeek 的回答

重要规则：
1. 必须先调用 search_knowledge，再调用 analyze_with_llm
2. 回答必须优先使用文档中存在的信息
3. 只有文档中完全没有相关信息时，才用通用知识补充
4. 不要说"文档中没有信息"，而是列出文档中找到的所有相关内容

用户问题：{query}

开始："""

    messages = [{"role": "user", "content": prompt}]
    max_rounds = 5

    for round_i in range(max_rounds):
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )

        output = response.choices[0].message.content.strip()
        print(f"\n[ReAct Round {round_i + 1}]\n{output}")

        # 检查是否有 Final Answer
        if "Final Answer:" in output:
            answer = output.split("Final Answer:")[-1].strip()
            return answer

        # 解析 Action
        if "Action:" in output and "Action Input:" in output:
            try:
                action_line = [l for l in output.split("\n") if l.strip().startswith("Action:")][0]
                action = action_line.split("Action:")[-1].strip()

                input_line = [l for l in output.split("\n") if l.strip().startswith("Action Input:")][0]
                action_input = input_line.split("Action Input:")[-1].strip()

                # 执行工具
                if action in tools:
                    observation = tools[action](action_input)
                else:
                    observation = f"工具 '{action}' 不存在"

                print(f"Observation: {observation[:200]}...")

                # 将结果加入对话
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": f"Observation: {observation}\n请继续思考。"})
            except Exception as e:
                messages.append({"role": "user", "content": f"执行出错: {e}，请重新思考。"})

    return "达到最大推理轮数，未得出最终答案。"


# ============================================================
# 2. Plan-and-Solve 范式（先规划，再执行）
# ============================================================

def plan_and_solve_agent(query: str, tools: dict = None) -> str:
    """
    Plan-and-Solve 范式：
    搜索知识库 → 发给 DeepSeek 分析 → 输出回答
    """
    llm = get_llm()

    # 1. 搜索知识库
    all_docs_text = ""
    if tools and "search_knowledge" in tools:
        all_docs_text = tools["search_knowledge"](query)
        print(f"\n[Plan-and-Solve] 获取到 {len(all_docs_text)} 字符文档")

    # 2. 发给 DeepSeek 分析
    if tools and "analyze_with_llm" in tools:
        analysis_prompt = f"""你是讯飞产品知识库的智能助手。请仔细阅读以下所有文档，找出与用户问题相关的所有产品信息。

重要规则：
1. 逐个检查每个文档，不要跳过任何一个
2. 查找所有包含用户查询关键词的文档（如"鼠标"、"键盘"、"翻译机"等）
3. 提取每个相关文档中的具体信息：产品名称、型号、价格、功能、参数
4. 如果找到了相关信息，必须列出，不要说"没有信息"
5. 只有当所有文档都确实没有相关信息时，才用通用知识补充

知识库文档：
{all_docs_text}

用户问题：{query}

直接回答："""
        answer = tools["analyze_with_llm"](analysis_prompt)
    else:
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": f"知识库：{all_docs_text}\n\n问题：{query}\n\n直接回答："}],
            max_tokens=800,
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()

    print(f"\n[Plan-and-Solve] 回答: {answer[:100]}...")
    return answer


# ============================================================
# 3. Reflection 范式（生成 → 反思 → 改进）
# ============================================================

def reflection_agent(query: str, tools: dict = None) -> str:
    """
    Reflection 范式：
    搜索知识库 → DeepSeek 分析 → 反思 → 改进
    """
    llm = get_llm()

    # 1. 搜索知识库
    all_docs_text = ""
    if tools and "search_knowledge" in tools:
        all_docs_text = tools["search_knowledge"](query)
        print(f"\n[Reflection] 获取到 {len(all_docs_text)} 字符文档")

    # 2. 发给 DeepSeek 分析
    if tools and "analyze_with_llm" in tools:
        analysis_prompt = f"""你是讯飞产品知识库的智能助手。请仔细阅读以下所有文档，找出与用户问题相关的所有产品信息。

重要规则：
1. 逐个检查每个文档，不要跳过任何一个
2. 查找所有包含用户查询关键词的文档（如"鼠标"、"键盘"、"翻译机"等）
3. 提取每个相关文档中的具体信息：产品名称、型号、价格、功能、参数
4. 如果找到了相关信息，必须列出，不要说"没有信息"
5. 只有当所有文档都确实没有相关信息时，才用通用知识补充

知识库文档：
{all_docs_text}

用户问题：{query}

直接回答："""
        initial_answer = tools["analyze_with_llm"](analysis_prompt)
    else:
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": f"知识库：{all_docs_text}\n\n问题：{query}\n\n直接回答："}],
            max_tokens=500,
            temperature=0.5
        )
        initial_answer = response.choices[0].message.content.strip()

    print(f"\n[Reflection] 初始回答: {initial_answer[:100]}...")

    # ===== 阶段3：反思 =====
    reflect_prompt = f"""评估以下回答的质量（1-10分），指出不足和改进建议。

问题：{query}
知识库文档：{all_docs_text[:1000] if all_docs_text else '无'}
回答：{initial_answer}

评估维度：准确性、完整性、清晰度
直接输出评估结果："""

    response = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": reflect_prompt}],
        max_tokens=400,
        temperature=0.3
    )
    reflection = response.choices[0].message.content.strip()
    print(f"\n[Reflection] 反思: {reflection[:100]}...")

    # ===== 阶段4：改进 =====
    if tools and "analyze_with_llm" in tools:
        improve_prompt = f"""根据反思意见和知识库信息，改进以下回答。

用户问题：{query}
知识库文档：{all_docs_text}
原始回答：{initial_answer}
反思意见：{reflection}

给出改进后的回答："""
        improved_answer = tools["analyze_with_llm"](improve_prompt)
    else:
        improve_prompt = f"""根据反思意见改进回答。

问题：{query}
原始回答：{initial_answer}
反思：{reflection}

改进后回答："""
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": improve_prompt}],
            max_tokens=600,
            temperature=0.5
        )
        improved_answer = response.choices[0].message.content.strip()

    print(f"\n[Reflection] 改进后: {improved_answer[:100]}...")

    return improved_answer


# ============================================================
# 统一 Agent 接口
# ============================================================

def run_agent(query: str, mode: str = "react", tools: dict = None) -> str:
    """
    统一 Agent 接口
    mode: react / plan_and_solve / reflection
    """
    if tools is None:
        tools = {}

    if mode == "react":
        return react_agent(query, tools)
    elif mode == "plan_and_solve":
        return plan_and_solve_agent(query, tools)
    elif mode == "reflection":
        return reflection_agent(query, tools)
    else:
        return f"未知模式: {mode}，支持: react, plan_and_solve, reflection"


if __name__ == "__main__":
    print("Agent 范式模块")
    print("支持模式: react, plan_and_solve, reflection")
