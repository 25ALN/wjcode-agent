import re
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TextSplitter:
    """
    用法：
        splitter = TextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_text(long_text)
        # 或者批量处理多个文档
        all_chunks = splitter.split_documents(["文档1内容...", "文档2内容..."])
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
    ):
        """
        Args:
            chunk_size: 每个 chunk 的目标最大字符数
            chunk_overlap: 相邻 chunk 之间的重叠字符数
            separators: 分割符层级列表，从粗到细。
            keep_separator: 是否在 chunk 末尾保留分割符
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator
        self.separators = separators or [
            "\n\n",     # 段落分隔（双换行）
            "\n",       # 段落分隔（单换行）
            "。",       # 中文句号
            "！",       # 中文感叹号
            "？",       # 中文问号
            "；",       # 中文分号
            ".",        # 英文句号
            "!",        # 英文感叹号
            "?",        # 英文问号
            ";",        # 英文分号
            " ",        # 空格（兜底）
        ]

    def split_text(self, text: str) -> List[str]:
        """
        将单篇文本切割为多个 chunk
        Args:
            text: 原始文本
        Returns:
            切割后的 chunk 列表
        """
        if not text or not text.strip():
            return []

        # 如果文本本身就不超过 chunk_size，直接返回
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # 递归分割,这个分割的会非常的碎
        splits = self._split_recursive(text, self.separators.copy())

        # 将分割后的片段合并成带重叠的 chunk，拼接上面的碎片让语义更加连贯
        chunks = self._merge_with_overlap(splits)

        return chunks

    def split_documents(self, documents: List[str]) -> List[str]:
        """
        批量切割多个文档，返回所有 chunk 的扁平列表

        Args:
            documents: 文档内容字符串列表
        Returns:
            所有文档切割后的 chunk 列表
        """
        all_chunks: List[str] = []
        for doc_idx, doc in enumerate(documents):
            if not doc or not doc.strip():
                logger.warning(f"文档 #{doc_idx} 为空，跳过")
                continue
            chunks = self.split_text(doc)
            all_chunks.extend(chunks)
            logger.debug(
                f"文档 #{doc_idx} (长度 {len(doc)}) -> {len(chunks)} 个 chunks"
            )
        logger.info(f"共 {len(documents)} 篇文档 -> {len(all_chunks)} 个 chunks")
        return all_chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        递归地按分隔符层级切割文本
        1. 从最粗粒度的分隔符开始尝试分割（如 \n\n）
        2. 如果某个片段仍然超过 chunk_size，则用下一级分隔符继续分割
        3. 如果到了最后一级分隔符仍然超长，则按固定长度强制截断
        """
        if not separators:
            # 没有分隔符可用，按固定长度强制切割
            return self._fixed_size_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]#保留之后的分隔符好进行后面的递归切割

        if separator == "":
            # 空分隔符意味着按字符切割
            return self._fixed_size_split(text)

        if separator in text:
            parts = text.split(separator)
            # 保留分隔符在片段末尾
            if self.keep_separator:
                parts = [p + separator for p in parts[:-1]] + [parts[-1]]
            # 过滤空字符串
            parts = [p for p in parts if p.strip()]
        else:
            # 当前分隔符不在文本中，直接用下一级
            return self._split_recursive(text, remaining_separators)

        # 检查每个片段是否需要进一步切割
        result: List[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                if part.strip():
                    result.append(part)
            else:
                sub_splits = self._split_recursive(part, remaining_separators)
                result.extend(sub_splits)

        return result

    def _fixed_size_split(self, text: str) -> List[str]:
        """
        固定大小强制切割
        按 chunk_size 直接截断，尽量在标点符号处断开以避免在词中间截断
        """
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # 尝试在 chunk 末尾附近找到更好的断点（标点符号处）
            chunk_slice = text[start:end]
            # 从后往前找标点符号
            better_end = self._find_better_break(chunk_slice)
            if better_end is not None:
                end = start + better_end + 1  # +1 保留标点

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end

        return chunks

    def _find_better_break(self, text_slice: str) -> Optional[int]:
        # 按优先级查找
        priority_punctuation = [
            "。", "！", "？", ".", "!", "?",   # 句子结束
            "；", ";", "，", ",",              # 子句分隔
            " ",                                 # 空格
        ]

        for punct in priority_punctuation:
            idx = text_slice.rfind(punct)
            if idx > self.chunk_size * 0.5:  # 至少在前半部分才接受
                return idx

        return None

    def _merge_with_overlap(self, splits: List[str]) -> List[str]:
        if not splits:
            return []

        chunks: List[str] = []
        current_chunk_parts: List[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            # 如果当前单个片段就超过 chunk_size，直接作为一个 chunk
            if split_len > self.chunk_size:
                # 先保存当前累积的 chunk
                if current_chunk_parts:
                    chunks.append("".join(current_chunk_parts).strip())#拼接列表中的所有元素
                    current_chunk_parts = []
                    current_length = 0
                # 长片段需要进一步强制切割
                sub_chunks = self._fixed_size_split(split)
                chunks.extend(sub_chunks)
                continue

            # 如果加上当前片段会超过 chunk_size，保存当前 chunk 并开始新的
            if current_length + split_len > self.chunk_size and current_chunk_parts:
                merged = "".join(current_chunk_parts).strip()
                if merged:
                    chunks.append(merged)
                # 新 chunk 从前一个 chunk 的末尾重叠部分开始
                overlap_text = merged[-self.chunk_overlap:] if len(merged) > self.chunk_overlap else merged
                current_chunk_parts = [overlap_text]
                current_length = len(overlap_text)

            current_chunk_parts.append(split)
            current_length += split_len

        # 最后一个 chunk
        if current_chunk_parts:
            merged = "".join(current_chunk_parts).strip()
            if merged:
                chunks.append(merged)

        return chunks


class Embedding:
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = False):
        """
        Args:
            model_name: BGE 模型名称（默认 bge-m3，多语言、8192 token 上下文）
            use_fp16: 是否使用半精度（节省显存，CPU 上设为 False）
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None  # 延迟加载

    @property#让方法可以想属性一样被访问也就是不加()
    def model(self):
        """延迟加载 BGE 模型"""
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"正在加载 BGE 模型: {self.model_name}")
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16
            )
            logger.info("BGE 模型加载完成")
        return self._model

    def encode_chunks(self, chunks: List[str]) -> List[List[float]]:
        if not chunks:
            return []

        logger.info(f"正在向量化 {len(chunks)} 个 chunks...")
        output = self.model.encode(chunks)
        dense_vecs = output['dense_vecs']
        # 转换为 Python float 列表
        embeddings = [vec.tolist() for vec in dense_vecs]
        logger.info(f"向量化完成，维度: {len(embeddings[0]) if embeddings else 0}")
        return embeddings

    def encode_query(self, query: str) -> List[float]:

        output = self.model.encode([query])
        return output['dense_vecs'][0].tolist()

    @staticmethod
    def cosine_similarity(
        query_vec: List[float],
        doc_vecs: List[List[float]]
    ) -> List[float]:
        """
        计算查询向量与多个文档向量的余弦相似度
        """
        import numpy as np

        query = np.array(query_vec, dtype=np.float32)
        docs = np.array(doc_vecs, dtype=np.float32)

        query_norm = query / (np.linalg.norm(query) + 1e-8)
        docs_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-8)

        scores = np.dot(docs_norm, query_norm)
        return scores.tolist()

    @staticmethod
    def top_k_indices(scores: List[float], k: int = 3) -> List[int]:
        """
        获取相似度分数中 top-k 的索引
        Args:
            scores: 相似度分数列表
            k: 返回前 k 个
        Returns:
            top-k 分数对应的原始索引列表
        """
        import numpy as np
        scores_arr = np.array(scores)
        # argsort 默认升序，取最后 k 个后反转得到降序
        indices = np.argsort(scores_arr)[-k:][::-1]
        return indices.tolist()


def create_default_splitter(
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> TextSplitter:
    """
    创建一个适用于中文 RAG 的默认切割器
    Args:
        chunk_size: chunk 大小（默认 600 字，适合 BGE-M3）
        chunk_overlap: 重叠大小（默认 100 字）
    Returns:
        配置好的 TextSplitter 实例
    """
    return TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            ". ",
            "! ",
            "? ",
            "; ",
            " ",
        ],
        keep_separator=True,
    )

#评估，后训练