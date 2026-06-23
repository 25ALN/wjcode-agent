import re
import logging
import urllib.request
import urllib.parse
from typing import List
from tools.base_tool import BaseTool, SAFE

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):

    name = "web_search"
    risk_level = SAFE
    description = (
        "联网搜索工具，在互联网上搜索指定关键词并返回结果摘要。"
        "适用于：查询实时信息（天气、新闻、股价）、查找技术资料、获取最新动态。"
        "参数 query 是搜索关键词，max_results 是最多返回条数（默认5）。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "最多返回条数（1-10，默认5）",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = 5, **kwargs) -> str:
        max_results = min(max(max_results, 1), 10)

        try:
            results = self._search_duckduckgo_html(query, max_results)

            if not results:
                return (
                    f"[搜索] 未找到与 '{query}' 相关的结果。\n"
                    "建议：尝试更简短的关键词，或换个说法再搜。"
                )

            output = [
                f"搜索: {query}",
                "Provider: duckduckgo",
                f"共 {len(results)} 条",
                "-" * 50,
            ]
            for i, r in enumerate(results, 1):
                output.append(f"{i}. {r.get('title', '无标题')}")
                snippet = r.get('snippet', '')
                if snippet:
                    output.append(f"   {snippet[:300]}")
                url = r.get("url", "")
                if url:
                    output.append(f"   {url}")
                output.append("")
            return "\n".join(output)

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return f"[错误] 搜索失败: {str(e)[:300]}"

    def _search_duckduckgo_html(self, query: str, max_results: int) -> List[dict]:
        """抓取 DuckDuckGo 的 HTML 搜索结果页

        DuckDuckGo 提供无 JS 的 HTML 版本：html.duckduckgo.com
        返回结构化的搜索结果（标题 + 摘要 + URL）。
        """
        params = urllib.parse.urlencode({"q": query})
        url = f"https://html.duckduckgo.com/html/?{params}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception:
            logger.warning("DuckDuckGo HTML 搜索失败，尝试 Lite 版本")
            return self._search_duckduckgo_lite(query, max_results)

        return self._parse_ddg_html(html, max_results)

    def _search_duckduckgo_lite(self, query: str, max_results: int) -> List[dict]:
        """DuckDuckGo Lite 版本（备用），更轻量的 HTML 页面"""
        params = urllib.parse.urlencode({"q": query})
        url = f"https://lite.duckduckgo.com/lite/?{params}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        return self._parse_ddg_lite(html, max_results)

    @staticmethod
    def _parse_ddg_html(html: str, max_results: int) -> List[dict]:
        """解析 DuckDuckGo HTML 搜索结果"""
        results = []

        # DuckDuckGo HTML 搜索结果结构：
        # <a rel="nofollow" class="result__a" href="...">标题</a>
        # <a class="result__snippet">摘要</a>

        # 匹配结果链接和标题
        link_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )
        # 匹配摘要
        snippet_pattern = re.compile(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (href, title) in enumerate(links):
            if i >= max_results:
                break
            title_clean = _clean_html(title)
            if not title_clean:
                continue

            snippet_clean = ""
            if i < len(snippets):
                snippet_clean = _clean_html(snippets[i])

            # 去掉 URL 中的 DuckDuckGo 重定向前缀
            url_clean = href
            if "uddg=" in url_clean:
                from urllib.parse import unquote
                match = re.search(r'uddg=([^&]+)', url_clean)
                if match:
                    url_clean = unquote(match.group(1))

            results.append({
                "title": title_clean,
                "snippet": snippet_clean,
                "url": url_clean,
                "provider": "duckduckgo_html",
            })

        return results

    @staticmethod
    def _parse_ddg_lite(html: str, max_results: int) -> List[dict]:
        """解析 DuckDuckGo Lite 搜索结果"""
        results = []

        link_pattern = re.compile(
            r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )
        snippet_pattern = re.compile(
            r'<span[^>]*class="[^"]*snippet[^"]*"[^>]*>(.*?)</span>',
            re.DOTALL | re.IGNORECASE,
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        # 过滤 DuckDuckGo 自身的链接
        skip_domains = ["duckduckgo.com", "spreadprivacy.com"]
        valid_links = [
            (href, title) for href, title in links
            if not any(d in href for d in skip_domains)
        ]

        for i, (href, title) in enumerate(valid_links):
            if i >= max_results:
                break
            title_clean = _clean_html(title)
            if not title_clean:
                continue

            snippet_clean = ""
            if i < len(snippets):
                snippet_clean = _clean_html(snippets[i])

            results.append({
                "title": title_clean,
                "snippet": snippet_clean,
                "url": href,
                "provider": "duckduckgo_lite",
            })

        return results


def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", str(text))
    replacements = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split()).strip()
