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

重要：第2步必须调用 analyze_with_llm，把问题和文档一起发给 DeepSeek 分析，不要自己判断有没有相关信息。

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

def plan_and_solve_agent(query: str, tools: dict) -> str:
    """
    Plan-and-Solve 范式：
    第一步：获取所有文档
    第二步：发给 DeepSeek 分析
    第三步：输出回答
    """
    llm = get_llm()

    # 1. 获取所有文档
    all_docs_text = ""
    if tools and "search_knowledge" in tools:
        all_docs_text = tools["search_knowledge"](query)
        print(f"\n[Plan-and-Solve] 获取到文档")

    # 2. 发给 DeepSeek 分析
    if tools and "analyze_with_llm" in tools:
        analysis_prompt = f"""你是讯飞产品知识库的智能助手。请根据以下所有文档内容回答用户问题。

重要：即使文档中没有完全匹配的信息，也要综合已有内容给出最接近的回答。

知识库全部文档：
{all_docs_text}

用户问题：{query}

直接回答（不要说"好的"、"作为助手"等开场白）："""
        answer = tools["analyze_with_llm"](analysis_prompt)
    else:
        gen_prompt = f"""请根据以下知识库信息回答用户问题。

知识库信息：
{all_docs_text if all_docs_text else '无'}

用户问题：{query}

直接回答："""
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": gen_prompt}],
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
    第一步：获取所有文档
    第二步：发给 DeepSeek 分析生成初始回答
    第三步：反思回答质量
    第四步：发给 DeepSeek 改进回答
    """
    llm = get_llm()

    # ===== 阶段1：获取所有文档 =====
    all_docs_text = ""
    if tools and "search_knowledge" in tools:
        all_docs_text = tools["search_knowledge"](query)
        print(f"\n[Reflection] 获取到文档")

    # ===== 阶段2：发给 DeepSeek 分析 =====
    if tools and "analyze_with_llm" in tools:
        analysis_prompt = f"""你是讯飞产品知识库的智能助手。请根据以下所有文档内容回答用户问题。

重要：即使文档中没有完全匹配的信息，也要综合已有内容给出最接近的回答。

知识库全部文档：
{all_docs_text}

用户问题：{query}

直接回答（不要说"好的"、"作为助手"等开场白）："""
        initial_answer = tools["analyze_with_llm"](analysis_prompt)
    else:
        # 降级：直接用 LLM
        gen_prompt = f"""请根据以下知识库信息回答用户问题。

知识库信息：
{all_docs_text if all_docs_text else '无'}

用户问题：{query}

直接回答："""
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": gen_prompt}],
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
