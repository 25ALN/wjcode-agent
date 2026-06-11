"""
1. 文本切割（TextSplitter）
2. 向量化（Embedding / BGE-M3）
3. 余弦相似度检索（Cosine Similarity + Top-K）
"""

from typing import List, Optional, Tuple
import logging
from rag.embedding import TextSplitter, Embedding, create_default_splitter

logger = logging.getLogger(__name__)


class Retriever:
    """
    RAG 检索器

    管理文档库的完整生命周期：
    切割 -> 向量化 -> 存储 -> 检索
    """

    def __init__(
        self,
        splitter: Optional[TextSplitter] = None,
        embedder: Optional[Embedding] = None,
    ):
        """
        Args:
            splitter: 文本切割器（默认使用 create_default_splitter()）
            embedder: 向量化器（默认使用 Embedding()，即 BGE-M3）
        """
        self.splitter = splitter or create_default_splitter()
        self.embedder = embedder or Embedding()

        # 存储
        self.chunks: List[str] = []          # 切割后的文本块
        self.embeddings: List[List[float]] = []  # 文本块对应的向量
        self.original_docs: List[str] = []   # 原始文档（可选保留）

    def add_documents(self, docs: List[str]) -> int:
        """
        添加文档到向量库
        自动完成：切割 -> 向量化 -> 存储
        Args:
            docs: 文档列表（每个元素是一篇完整的文档）
        Returns:
            新增的 chunk 数量
        """
        if not docs:
            logger.warning("add_documents: 没有文档需要添加")
            return 0

        # 保留原始文档
        self.original_docs.extend(docs)

        # 切割文本
        logger.info(f"正在切割 {len(docs)} 篇文档...")
        new_chunks = self.splitter.split_documents(docs)
        logger.info(f"切割完成，得到 {len(new_chunks)} 个 chunks")

        if not new_chunks:
            logger.warning("切割后没有产生任何 chunk")
            return 0

        # 向量化
        logger.info(f"正在向量化 {len(new_chunks)} 个 chunks...")
        new_embeddings = self.embedder.encode_chunks(new_chunks)

        # 存储
        self.chunks.extend(new_chunks)
        self.embeddings.extend(new_embeddings)

        logger.info(
            f"文档添加完成。当前共 {len(self.chunks)} 个 chunks，"
            f"向量维度: {len(new_embeddings[0]) if new_embeddings else 'N/A'}"
        )

        return len(new_chunks)

    def query_retrieve(
        self,
        query: str,
        top_k: int = 3,
        return_scores: bool = False,
    ) -> List[str] | List[Tuple[str, float]]:
        """
        查询检索：对问题向量化后计算余弦相似度，返回最相关的文本 chunk
        Args:
            query: 查询文本
            top_k: 返回前 k 个最相关的 chunk
            return_scores: 是否同时返回相似度分数
        Returns:
            如果 return_scores=False: 返回文本 chunk 列表
            如果 return_scores=True: 返回 (chunk, score) 元组列表
        """
        if not self.chunks or not self.embeddings:
            logger.warning("向量库为空，无法检索")
            return []

        # 确保 top_k 不超过实际 chunk 数量
        top_k = min(top_k, len(self.chunks))

        logger.debug(f"检索查询: {query[:100]}...")
        query_vec = self.embedder.encode_query(query)

        scores = self.embedder.cosine_similarity(query_vec, self.embeddings)

        # 获取 top-k 索引
        top_indices = self.embedder.top_k_indices(scores, top_k)

        # 返回结果
        if return_scores:
            return [
                (self.chunks[idx], float(scores[idx]))
                for idx in top_indices
            ]
        else:
            return [self.chunks[idx] for idx in top_indices]

    def clear(self):
        """清空所有存储的文档和向量"""
        self.chunks.clear()
        self.embeddings.clear()
        self.original_docs.clear()
        logger.info("向量库已清空")

    @property
    def chunk_count(self) -> int:
        """当前存储的 chunk 数量"""
        return len(self.chunks)

    @property
    def doc_count(self) -> int:
        """当前存储的原始文档数量"""
        return len(self.original_docs)

    def __repr__(self) -> str:
        return (
            f"Retriever(docs={self.doc_count}, "
            f"chunks={self.chunk_count}, "
            f"model={self.embedder.model_name})"
        )

def create_retriever(
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    model_name: str = "BAAI/bge-m3",
) -> Retriever:
    """
    快速创建一个配置好的 Retriever 实例
    Args:
        chunk_size: chunk 大小
        chunk_overlap: chunk 重叠大小
        model_name: BGE 模型名称
    """
    splitter = create_default_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embedder = Embedding(model_name=model_name)
    return Retriever(splitter=splitter, embedder=embedder)