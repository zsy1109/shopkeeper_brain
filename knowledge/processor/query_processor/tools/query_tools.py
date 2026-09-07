import json
from typing import Any, Dict, List

from knowledge.processor.query_processor.tools.base import AgentTool, ToolRegistry


class KnowledgeBaseTool(AgentTool):
    """从已检索+重排序后的知识库文档中获取内容。"""

    name = "search_knowledge_base"
    description = "从本地知识库中获取产品的技术文档、使用说明、规格参数等内容。知识库是产品PDF手册导入生成的。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "你想了解的具体技术问题或关键词",
            },
            "top_k": {
                "type": "integer",
                "description": "返回多少条相关结果（默认3，最多5）",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    def __init__(self, reranked_docs: List[Dict[str, Any]], item_names: List[str]):
        self._docs = reranked_docs or []
        self._item_names = item_names or []

    def run(self, query: str = "", top_k: int = 3) -> str:
        top_k = max(1, min(top_k, 5))
        if not self._docs:
            return "知识库中暂无相关文档。"

        results = []
        for i, doc in enumerate(self._docs[:top_k]):
            content = doc.get("content", "")
            title = doc.get("title", "")
            source = doc.get("source", "")
            score = doc.get("score")
            score_str = f" (相关度:{score:.3f})" if score is not None else ""
            results.append(
                f"[文档{i + 1}] 标题:{title} 来源:{source}{score_str}\n{content}"
            )

        header = f"知识库搜索结果（产品:{'、'.join(self._item_names) if self._item_names else '未知'}）:"
        return header + "\n\n" + "\n\n".join(results)


class WebSearchTool(AgentTool):
    """从互联网搜索实时信息。"""

    name = "search_web"
    description = "从互联网搜索实时信息，包括产品最新价格、库存状态、用户评价、行业新闻等时效性内容。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，建议包含产品名称和你想了解的具体信息",
            },
            "count": {
                "type": "integer",
                "description": "返回多少条搜索结果（默认3）",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    def __init__(self, web_docs: List[Dict[str, Any]]):
        self._docs = web_docs or []

    def run(self, query: str = "", count: int = 3) -> str:
        if not self._docs:
            return "本次查询未触发联网搜索，或联网搜索服务不可用。"

        count = max(1, min(count, 5))
        results = []
        for i, doc in enumerate(self._docs[:count]):
            title = doc.get("title", "")
            snippet = doc.get("snippet", "")
            url = doc.get("url", "")
            results.append(f"[结果{i + 1}] {title}\n{snippet}\n来源: {url}")

        return "联网搜索结果:\n\n" + "\n\n".join(results)


class RealTimePriceTool(AgentTool):
    """专门获取产品实时价格的工具。"""

    name = "get_realtime_price"
    description = "联网查询产品的最新市场价格、批量折扣、库存状态和购买渠道。适用于回答'价格多少'、'有没有降价'、'哪里买'等问题。"
    parameters = {
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "完整的产品名称，如'RS PRO RS-12 数字万用表'",
            },
        },
        "required": ["product_name"],
    }

    def __init__(self, web_docs: List[Dict[str, Any]]):
        self._docs = web_docs or []

    def run(self, product_name: str = "") -> str:
        if not self._docs:
            return "暂无法获取实时价格信息（联网搜索未触发或服务不可用）。"

        price_lines = []
        for doc in self._docs:
            snippet = doc.get("snippet", "")
            title = doc.get("title", "")
            url = doc.get("url", "")
            if any(kw in (snippet + title) for kw in ["价", "¥", "$", "￥", "元", "折扣", "库存", "现货", "购买"]):
                price_lines.append(f"- {title}: {snippet[:200]}  (来源: {url})")

        if not price_lines:
            all_lines = [
                f"- {d.get('title', '')}: {d.get('snippet', '')[:200]}  (来源: {d.get('url', '')})"
                for d in self._docs
            ]
            return (
                f"关于「{product_name}」的实时价格信息暂未获取到，但有以下联网搜索结果可参考:\n"
                + "\n".join(all_lines)
            )

        return f"「{product_name}」的实时价格/购买信息:\n" + "\n".join(price_lines)


class CompareProductsTool(AgentTool):
    """对比两款产品的参数。"""

    name = "compare_products"
    description = "对比两款相似产品的技术参数、功能差异、适用场景等。适合回答'A和B有什么区别'、'选A还是B'等问题。"
    parameters = {
        "type": "object",
        "properties": {
            "product_a": {
                "type": "string",
                "description": "第一款产品名称",
            },
            "product_b": {
                "type": "string",
                "description": "第二款产品名称",
            },
            "aspects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "重点对比的方面，如['规格参数', '价格', '适用场景']",
                "default": ["规格参数", "价格", "适用场景"],
            },
        },
        "required": ["product_a", "product_b"],
    }

    def __init__(self, reranked_docs: List[Dict[str, Any]], web_docs: List[Dict[str, Any]]):
        self._docs = reranked_docs or []
        self._web = web_docs or []

    def run(self, product_a: str = "", product_b: str = "", aspects: List[str] = None) -> str:
        aspects = aspects or ["规格参数", "价格", "适用场景"]

        relevant_docs = []
        for doc in self._docs:
            content = doc.get("content", "")
            if product_a in content or product_b in content:
                relevant_docs.append(content[:300])

        relevant_web = []
        for doc in self._web:
            text = doc.get("title", "") + " " + doc.get("snippet", "")
            if product_a in text or product_b in text:
                relevant_web.append(text[:200])

        return (
            f"请从以下资料中对比「{product_a}」和「{product_b}」在 {aspects} 方面的差异:\n\n"
            f"=== 知识库资料 ===\n"
            + ("\n".join(f"- {d}" for d in relevant_docs) if relevant_docs else "(无相关知识库文档)")
            + f"\n\n=== 联网资料 ===\n"
            + ("\n".join(f"- {w}" for w in relevant_web) if relevant_web else "(无相关联网资料)")
        )


def build_tools(
    reranked_docs: List[Dict[str, Any]],
    web_docs: List[Dict[str, Any]],
    item_names: List[str],
) -> List[AgentTool]:
    """根据 state 中的数据构建工具列表。"""
    ToolRegistry.clear()

    tools = [
        KnowledgeBaseTool(reranked_docs=reranked_docs, item_names=item_names),
        WebSearchTool(web_docs=web_docs),
        RealTimePriceTool(web_docs=web_docs),
        CompareProductsTool(reranked_docs=reranked_docs, web_docs=web_docs),
    ]

    for t in tools:
        ToolRegistry.register(t)

    return tools