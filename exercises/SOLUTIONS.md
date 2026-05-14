# Solutions — A2A Multi-Agent Lab Exercises

> **Students:** Try your best before looking at these solutions!
> These solutions are meant to be checked **after** completing each exercise.

---

## Exercise 2: Thêm Tools và Knowledge Base

### Solution: `exercises/exercise_2_tools.py`

```python
"""Bài Tập 2: Thêm Tools và Knowledge Base — SOLUTION"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm

# Knowledge base
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include: (1) expectation damages; (2) consequential damages; (3) specific performance; "
            "(4) cover damages. Statute of limitations is typically 4 years (UCC § 2-725)."
        ),
    },
    # SOLUTION: Thêm entry về luật lao động Việt Nam
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "quyền lợi", "lương", "thưởng", "bảo hiểm"],
        "text": (
            "Theo Bộ luật Lao động Việt Nam 2019: "
            "(1) Hợp đồng lao động phải được giao kết bằng văn bản; "
            "(2) Thời hiệu khởi kiện việc sa thải hoặc bị đơn phương chấm dứt HĐLĐ trái pháp luật là 12 tháng; "
            "(3) Người sử dụng lao động có quyền đơn phương chấm dứt HĐLĐ nếu có lý do chính đáng; "
            "(4) Người lao động bị sa thải trái luật được khôi phục quyền hoặc hưởng lương trong thời gian chờ giải quyết."
        ),
    },
]


@tool
def search_legal_knowledge(query: str) -> str:
    """Tìm kiếm trong knowledge base pháp lý."""
    query_lower = query.lower()
    for entry in LEGAL_KNOWLEDGE:
        if any(kw in query_lower for kw in entry["keywords"]):
            return f"[{entry['id']}] {entry['text']}"
    return "Không tìm thấy thông tin liên quan."


# SOLUTION: Tạo tool check_statute_of_limitations
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ việc."""
    limitations = {
        "contract": "4 năm theo UCC § 2-725 (thương mại) hoặc theo Bộ luật Dân sự Việt Nam (3 năm cho nghĩa vụ dân sự thông thường)",
        "breach": "4 năm cho vi phạm hợp đồng thương mại (UCC § 2-725)",
        "employment": "12 tháng cho tranh chấp lao động theo Bộ luật Lao động Việt Nam 2019",
        "labor": "12 tháng cho tranh chấp về sa thải hoặc chấm dứt HĐLĐ trái pháp luật",
        "property": "10 năm đối với quyền đòi bồi thường thiệt hại về tài sản",
        "personal": "3 năm đối với yêu cầu bồi thường thiệt hại ngoài hợp đồng",
        "tax": "10 năm để truy thu thuế theo quy định của Việt Nam",
        "default": "Thời hiệu khởi kiện phụ thuộc vào loại vụ việc cụ thể. Vui lòng chỉ rõ loại (contract, breach, employment, labor, property, personal, tax)."
    }

    case_lower = case_type.lower()
    for key, value in limitations.items():
        if key in case_lower:
            return f"⏰ Thời hiệu khởi kiện cho '{case_type}': {value}"

    return limitations["default"]


async def main():
    load_dotenv()
    llm = get_llm()

    # SOLUTION: Thêm tool mới vào danh sách
    tools = [search_legal_knowledge, check_statute_of_limitations]
    llm_with_tools = llm.bind_tools(tools)

    question = "Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?"

    messages = [
        SystemMessage(content="Bạn là chuyên gia pháp lý. Sử dụng tools để tra cứu thông tin."),
        HumanMessage(content=question),
    ]

    print(f"Câu hỏi: {question}\n")

    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    # SOLUTION: Thêm xử lý cho check_statute_of_limitations
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"🔧 Gọi tool: {tool_call['name']}")
            tool_result = None

            if tool_call["name"] == "search_legal_knowledge":
                tool_result = search_legal_knowledge.invoke(tool_call["args"])
            elif tool_call["name"] == "check_statute_of_limitations":
                tool_result = check_statute_of_limitations.invoke(tool_call["args"])

            if tool_result:
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

        final_response = await llm_with_tools.ainvoke(messages)
        print(f"\n✅ Kết quả:\n{final_response.content}")
    else:
        print(f"\n✅ Kết quả:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())
```

### What was added:

| # | Change | Location |
|---|---|---|
| 1 | `labor_law` entry in `LEGAL_KNOWLEDGE` | Line 29-37 |
| 2 | `@tool check_statute_of_limitations` function | Line 44-71 |
| 3 | Added to `tools = [...]` list | Line 95 |
| 4 | Tool execution handler (`elif tool_call["name"] == "check_statute_of_limitations"`) | Line 113-114 |

### Concepts learned:
- `@tool` decorator creates a LangChain tool from a Python function
- Knowledge base with keyword matching for retrieval
- Binding multiple tools to an LLM with `.bind_tools()`
- Manual tool execution loop

---

## Exercise 4: Thêm Privacy Agent vào Multi-Agent System

### Solution: `exercises/exercise_4_multiagent.py`

```python
"""Bài Tập 4: Thêm Privacy Agent vào Multi-Agent System — SOLUTION"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm


def _last_wins(left: str | None, right: str | None) -> str:
    """Reducer: giá trị mới ghi đè giá trị cũ."""
    return right if right is not None else (left or "")


class State(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]  # SOLUTION: Đã có sẵn
    final_response: str


def law_agent(state: State) -> dict:
    """Agent phân tích pháp lý tổng quát."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia pháp lý. Phân tích câu hỏi sau:

{state['question']}

Tập trung vào: hợp đồng, trách nhiệm dân sự, quyền và nghĩa vụ pháp lý."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"law_analysis": response.content}


def check_routing(state: State) -> list[Send]:
    """Quyết định gọi agents nào dựa trên nội dung câu hỏi."""
    question_lower = state["question"].lower()
    tasks = []

    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))

    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))

    # SOLUTION: Thêm logic routing cho privacy_agent
    privacy_keywords = ["data", "privacy", "gdpr", "dữ liệu", "personal", "breach", "rò rỉ", "thông tin"]
    if any(kw in question_lower for kw in privacy_keywords):
        tasks.append(Send("privacy_agent", state))

    return tasks if tasks else [Send("aggregate_results", state)]


def tax_agent(state: State) -> dict:
    """Agent chuyên về thuế."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia thuế. Phân tích khía cạnh thuế trong câu hỏi:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: IRS, tax evasion, penalties, FBAR, FATCA."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"tax_analysis": response.content}


def compliance_agent(state: State) -> dict:
    """Agent chuyên về compliance."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia compliance. Phân tích khía cạnh tuân thủ:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: SEC, SOX, FCPA, AML, regulatory violations."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"compliance_analysis": response.content}


# SOLUTION: Implement privacy_agent
def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về bảo vệ dữ liệu cá nhân và GDPR. Phân tích khía cạnh privacy:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung vào:
- GDPR (General Data Protection Regulation)
- Quyền riêng tư dữ liệu cá nhân
- Nghĩa vụ bảo mật dữ liệu
- Hậu quả của việc rò rỉ dữ liệu (data breach)
- Quyền của chủ thể dữ liệu (data subject rights)
- Data breach notification requirements
- Các biện pháp bảo vệ và penalties"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}


def aggregate_results(state: State) -> dict:
    """Tổng hợp kết quả từ tất cả agents."""
    llm = get_llm()

    sections = []
    if state.get("law_analysis"):
        sections.append(f"📋 PHÂN TÍCH PHÁP LÝ:\n{state['law_analysis']}")
    if state.get("tax_analysis"):
        sections.append(f"💰 PHÂN TÍCH THUẾ:\n{state['tax_analysis']}")
    if state.get("compliance_analysis"):
        sections.append(f"✅ PHÂN TÍCH TUÂN THỦ:\n{state['compliance_analysis']}")
    # SOLUTION: Thêm privacy_analysis vào sections
    if state.get("privacy_analysis"):
        sections.append(f"🔒 PHÂN TÍCH BẢO MẬT DỮ LIỆU:\n{state['privacy_analysis']}")

    combined = "\n\n".join(sections)

    prompt = f"""Tổng hợp các phân tích sau thành một báo cáo pháp lý hoàn chỉnh:

{combined}

Câu hỏi gốc: {state['question']}

Hãy tạo một báo cáo ngắn gọn, có cấu trúc rõ ràng."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_response": response.content}


def build_graph() -> StateGraph:
    """Xây dựng multi-agent graph."""
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("law_agent", law_agent)
    graph.add_node("check_routing", check_routing)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    # SOLUTION: Thêm privacy_agent node
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("aggregate_results", aggregate_results)

    # Define edges
    graph.add_edge(START, "law_agent")
    graph.add_edge("law_agent", "check_routing")
    graph.add_conditional_edges("check_routing", lambda x: x)
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    # SOLUTION: Thêm edge từ privacy_agent đến aggregate_results
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    return graph.compile()


async def main():
    load_dotenv()

    question = "Nếu công ty bị rò rỉ dữ liệu khách hàng, hậu quả pháp lý và thuế là gì?"

    print("=" * 70)
    print("MULTI-AGENT SYSTEM với Privacy Agent")
    print("=" * 70)
    print(f"\nCâu hỏi: {question}\n")
    print("Đang xử lý qua các agents...\n")

    graph = build_graph()

    result = await graph.ainvoke({
        "question": question,
        "law_analysis": "",
        "tax_analysis": "",
        "compliance_analysis": "",
        "privacy_analysis": "",
        "final_response": "",
    })

    print("\n" + "=" * 70)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 70)
    print(result["final_response"])
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
```

### What was added:

| # | Change | Location |
|---|---|---|
| 1 | `privacy_analysis` field in `State` TypedDict | Line 30 — already present |
| 2 | Privacy routing logic in `check_routing()` | Lines 53-55 |
| 3 | Full `privacy_agent()` function implementation | Lines 95-113 |
| 4 | Privacy section in `aggregate_results()` | Lines 133-134 |
| 5 | `privacy_agent` node added to graph | Line 158 |
| 6 | Edge from `privacy_agent` to `aggregate_results` | Line 164 |

### Architecture after fix:

```
question
    │
    ▼
law_agent
    │
    ▼
check_routing ───► Send ──► tax_agent ──────────────────┐
    │                  │                                │
    └──────────► Send ──► compliance_agent ──────────────┼──► aggregate_results ──► END
                       │                                │
                       └────────► Send ──► privacy_agent ┘
```

### Concepts learned:
- LangGraph `StateGraph` with `TypedDict` for shared state
- `Annotated[str, _last_wins]` reducer pattern for parallel writes
- `Send` API for parallel agent dispatch
- `add_conditional_edges()` for dynamic routing
- Domain-specialized agents that run in parallel

---

## Bonus: When to Use Each Pattern

| Pattern | When to Use | Example |
|---|---|---|
| **Direct LLM** | Simple, stateless queries | FAQ bots |
| **LLM + Tools** | Need real data lookup | Knowledge base Q&A |
| **ReAct Agent** | Multi-step tasks, autonomous loop | Research agents |
| **Multi-Agent (in-process)** | Multiple domains, parallel processing | Legal analysis (this lab) |
| **Distributed A2A** | Production, independent scaling | Enterprise systems |

---

## Running the Solutions

```bash
# Exercise 2: Tools and Knowledge Base
cd exercises
py exercise_2_tools.py

# Exercise 4: Privacy Agent
cd exercises
py exercise_4_multiagent.py
```

Both exercises require a valid `OPENAI_API_KEY` in `.env`.
