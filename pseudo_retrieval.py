# 基于伪古文生成 + Pyserini BM25召回 + embedding重排 的支持句检索功能

import json
from embedding_utils import get_embedding, cosine_similarity
from llm import cached_call_llm
from pyserini.search.lucene import LuceneSearcher

# 全局缓存：同一个 index_dir 只初始化一次 searcher
_SEARCHER_CACHE = {}


def load_pyserini_searcher(index_dir):
    """
    加载 Pyserini 搜索器（带缓存）
    """
    global _SEARCHER_CACHE

    if index_dir in _SEARCHER_CACHE:
        print(f"[Pyserini] 复用已有搜索器: {index_dir}")
        return _SEARCHER_CACHE[index_dir]

    print(f"[Pyserini] 初始化搜索器: {index_dir}")
    searcher = LuceneSearcher(index_dir)
    searcher.set_language("zh")
    _SEARCHER_CACHE[index_dir] = searcher
    return searcher


def safe_json_load(raw):
    try:
        return json.loads(raw)
    except Exception:
        return {}


def generate_pseudo_ancient_chinese(gloss, target_word, wsid=None, num_examples=3):
    """
    根据义项生成伪古文
    """
    print(f"\n[伪古文生成] 开始生成伪古文")
    print(f"[伪古文生成] 目标词: {target_word}")
    print(f"[伪古文生成] 义项ID: {wsid}")
    print(f"[伪古文生成] 义项解释: {gloss}")
    print(f"[伪古文生成] 生成数量: {num_examples}")

    gloss_lower = gloss.lower()
    if any(keyword in gloss_lower for keyword in [
        '行动', '行为', '动作', '做', '行', '走', '跑', '跳', '打', '杀', '吃', '喝', '穿', '戴',
        '拿', '放', '推', '拉', '抬', '扛', '挑', '背', '抱', '举', '扔', '抛', '接', '送', '取',
        '给', '买', '卖', '借', '还', '抢', '偷', '骗', '救', '帮', '助', '教', '学', '写', '读',
        '说', '听', '看', '闻', '尝', '摸', '抓', '握', '踢', '踩', '踏', '登', '爬', '游', '飞',
        '骑', '驾', '乘', '坐', '站', '躺', '卧', '趴', '蹲', '跪', '拜', '揖', '拱'
    ]):
        gloss_type = "action"
    elif any(keyword in gloss_lower for keyword in [
        '状态', '情况', '样子', '形态', '姿态', '表情', '心情', '情绪', '感觉', '感受', '体验',
        '经历', '过程', '阶段', '时期', '时间', '空间', '位置', '方向', '大小', '多少', '长短',
        '粗细', '高低', '远近', '快慢', '好坏', '善恶', '美丑', '真假', '虚实', '有无', '存亡',
        '生死', '兴衰', '成败', '荣辱', '得失', '利弊', '祸福', '吉凶', '贫富', '贵贱', '强弱',
        '胜负', '优劣'
    ]):
        gloss_type = "state"
    elif any(keyword in gloss_lower for keyword in [
        '抽象', '概念', '思想', '观念', '意识', '精神', '灵魂', '心灵', '情感'
    ]):
        gloss_type = "abstract"
    else:
        gloss_type = "entity"

    print(f"[伪古文生成] 义项类型: {gloss_type}")

    prompt = f"""
    你要为【古汉语词义消歧】生成"伪古文用法句"，用于后续 BM25 检索与 embedding rerank。
    目标不是写得华丽，而是生成能代表该义项的典型用法，并且便于与真实古文句子匹配。

    【目标词】{target_word}
    【义项定义】{gloss}
    【义项类型】{gloss_type}  （只能是 action / state / abstract / entity）

    生成要求（必须严格遵守）：
    1) 输出 4句"古汉语风格"的短句（每句 8–20 字），不使用现代汉语语序（禁止"的、了、在、把、被、因为、所以、但是"等）。
    2) 每句必须包含【目标词】且只出现 1 次。
    3) 不引入具体史实：禁止出现具体朝代、年号、真实人名、真实地名、具体书名篇名。
    4) 句式要多样化：至少包含两种不同句式。
    5) 语义要忠实于【义项定义】，不要生成其他义项的用法。

    请输出严格 JSON，不要任何多余文字：
    {{
      "pseudo_contexts": [
        {{"text": "...", "pattern": "句式标签", "notes": "一句话说明该句如何体现义项"}}
      ]
    }}
    """

    cache_tag = f"pseudo|word={target_word}|wsid={wsid}"
    response = cached_call_llm(prompt, cache_tag=cache_tag, model="deepseek-v3.2-exp")

    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    try:
        result = json.loads(response)
        pseudo_contexts = result.get("pseudo_contexts", [])
        print(f"[伪古文生成] 成功生成 {len(pseudo_contexts)} 条伪古文")

        pseudo_sentences = []
        for i, ctx in enumerate(pseudo_contexts):
            text = ctx.get("text", "")
            pattern = ctx.get("pattern", "")
            notes = ctx.get("notes", "")
            pseudo_sentences.append(text)
            print(f"[伪古文生成] 伪古文 {i+1}: {text} (句式: {pattern}) - {notes}")

        return pseudo_sentences[:num_examples]
    except Exception as e:
        print(f"[伪古文生成] 解析响应失败: {e}")
        return []


def build_bm25_query_from_pseudo(pseudo_sentences, target_word, gloss):
    """
    用伪古文 + 目标词 + 义项定义构造 BM25 query
    """
    pseudo_part = " ".join([s.strip() for s in pseudo_sentences if s and s.strip()])
    query = f"{target_word} {gloss} {pseudo_part}".strip()
    return query


def retrieve_supporting_sentences_with_pseudo_bm25(
    pseudo_sentences,
    target_word,
    gloss,
    searcher,
    bm25_top_k=200,
    final_top_k=10,
    threshold=0.5
):
    """
    用伪古文做 BM25 检索，再 embedding 重排
    """
    print(f"\n[伪古文BM25检索] 开始")
    print(f"[伪古文BM25检索] 目标词: {target_word}")
    print(f"[伪古文BM25检索] 义项解释: {gloss}")
    print(f"[伪古文BM25检索] 伪古文数量: {len(pseudo_sentences)}")
    print(f"[伪古文BM25检索] BM25召回数: {bm25_top_k}")
    print(f"[伪古文BM25检索] 最终返回数: {final_top_k}")

    if not pseudo_sentences:
        print("[伪古文BM25检索] 没有伪古文，返回空列表")
        return []

    query = build_bm25_query_from_pseudo(pseudo_sentences, target_word, gloss)
    print(f"[伪古文BM25检索] Query: {query[:200]}")

    hits = searcher.search(query, k=bm25_top_k)
    print(f"[伪古文BM25检索] BM25召回 {len(hits)} 条")

    if not hits:
        return []

    pseudo_embeddings = [get_embedding(s) for s in pseudo_sentences]

    reranked = []
    batch_size = 32
    batch_sentences = []
    batch_hits = []
    batch_items = []

    for rank, hit in enumerate(hits, start=1):
        doc = searcher.doc(hit.docid)
        if doc is None:
            continue

        raw = doc.raw()
        item = safe_json_load(raw)

        sentence = item.get("raw_text", "")
        if not sentence:
            continue

        batch_sentences.append(sentence)
        batch_hits.append(hit)
        batch_items.append((rank, item))

        if len(batch_sentences) >= batch_size or rank == len(hits):
            print(f"[伪古文BM25检索] 处理批次：{len(reranked) + 1} - {len(reranked) + len(batch_sentences)}")

            batch_embeddings = [get_embedding(sent) for sent in batch_sentences]

            for sent, sent_emb, hit, (rank, item) in zip(batch_sentences, batch_embeddings, batch_hits, batch_items):
                max_sim = 0.0
                for pseudo_emb in pseudo_embeddings:
                    sim = cosine_similarity(sent_emb, pseudo_emb)
                    if sim > max_sim:
                        max_sim = sim

                if max_sim >= threshold:
                    reranked.append({
                        "sentence": sent,
                        "similarity": max_sim,
                        "bm25_score": float(hit.score),
                        "docid": hit.docid,
                        "rank": rank,
                        "corpus": item.get("corpus", ""),
                        "genre": item.get("genre", "")
                    })

            batch_sentences = []
            batch_hits = []
            batch_items = []

    reranked.sort(key=lambda x: x["similarity"], reverse=True)
    final_results = reranked[:final_top_k]

    print(f"[伪古文BM25检索] 重排后返回 {len(final_results)} 条")
    for i, item in enumerate(final_results, 1):
        print(f"[伪古文BM25检索] Top-{i} (相似度: {item['similarity']:.4f}): {item['sentence'][:50]}...")

    return final_results


def get_supporting_sentences_with_pseudo_bm25(
    sense_gloss,
    target_word,
    index_dir,
    wsid=None,
    bm25_top_k=200,
    final_top_k=10,
    threshold=0.5
):
    """
    完整的支持句检索流程：生成伪古文 + BM25召回 + embedding重排
    """
    # 生成伪古文
    pseudo_sentences = generate_pseudo_ancient_chinese(sense_gloss, target_word, wsid=wsid)
    
    # 加载 Pyserini 搜索器（带缓存）
    searcher = load_pyserini_searcher(index_dir)
    
    # 使用伪古文进行 BM25 检索和重排
    supporting_sentences = retrieve_supporting_sentences_with_pseudo_bm25(
        pseudo_sentences,
        target_word,
        sense_gloss,
        searcher,
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
        threshold=threshold
    )
    
    return supporting_sentences


def retrieve_supporting_sentences_without_pseudo_bm25(
    target_word,
    gloss,
    searcher,
    bm25_top_k=200,
    final_top_k=10,
    threshold=0.5
):
    """
    不使用伪古文，直接用原始义项gloss做 BM25 检索，再 embedding 重排
    """
    print(f"\n[无伪古文BM25检索] 开始")
    print(f"[无伪古文BM25检索] 目标词: {target_word}")
    print(f"[无伪古文BM25检索] 义项解释: {gloss}")
    print(f"[无伪古文BM25检索] BM25召回数: {bm25_top_k}")
    print(f"[无伪古文BM25检索] 最终返回数: {final_top_k}")

    # 直接使用目标词和义项定义作为查询
    query = f"{target_word} {gloss}".strip()
    print(f"[无伪古文BM25检索] Query: {query[:200]}")

    hits = searcher.search(query, k=bm25_top_k)
    print(f"[无伪古文BM25检索] BM25召回 {len(hits)} 条")

    if not hits:
        return []

    # 使用义项定义的嵌入作为查询向量
    gloss_embedding = get_embedding(gloss)

    reranked = []
    batch_size = 32
    batch_sentences = []
    batch_hits = []
    batch_items = []

    for rank, hit in enumerate(hits, start=1):
        doc = searcher.doc(hit.docid)
        if doc is None:
            continue

        raw = doc.raw()
        item = safe_json_load(raw)

        sentence = item.get("raw_text", "")
        if not sentence:
            continue

        batch_sentences.append(sentence)
        batch_hits.append(hit)
        batch_items.append((rank, item))

        if len(batch_sentences) >= batch_size or rank == len(hits):
            print(f"[无伪古文BM25检索] 处理批次：{len(reranked) + 1} - {len(reranked) + len(batch_sentences)}")

            batch_embeddings = [get_embedding(sent) for sent in batch_sentences]

            for sent, sent_emb, hit, (rank, item) in zip(batch_sentences, batch_embeddings, batch_hits, batch_items):
                # 计算与义项定义的相似度
                sim = cosine_similarity(sent_emb, gloss_embedding)

                if sim >= threshold:
                    reranked.append({
                        "sentence": sent,
                        "similarity": sim,
                        "bm25_score": float(hit.score),
                        "docid": hit.docid,
                        "rank": rank,
                        "corpus": item.get("corpus", ""),
                        "genre": item.get("genre", "")
                    })

            batch_sentences = []
            batch_hits = []
            batch_items = []

    reranked.sort(key=lambda x: x["similarity"], reverse=True)
    final_results = reranked[:final_top_k]

    print(f"[无伪古文BM25检索] 重排后返回 {len(final_results)} 条")
    for i, item in enumerate(final_results, 1):
        print(f"[无伪古文BM25检索] Top-{i} (相似度: {item['similarity']:.4f}): {item['sentence'][:50]}...")

    return final_results


def get_supporting_sentences_without_pseudo_bm25(
    sense_gloss,
    target_word,
    index_dir,
    wsid=None,
    bm25_top_k=200,
    final_top_k=10,
    threshold=0.5
):
    """
    完整的支持句检索流程：不使用伪古文 + BM25召回 + embedding重排
    """
    print(f"\n[无伪古文检索] 开始")
    print(f"[无伪古文检索] 目标词: {target_word}")
    print(f"[无伪古文检索] 义项解释: {sense_gloss}")
    print(f"[无伪古文检索] 义项ID: {wsid}")
    
    # 加载 Pyserini 搜索器（带缓存）
    searcher = load_pyserini_searcher(index_dir)
    
    # 不使用伪古文，直接进行 BM25 检索和重排
    supporting_sentences = retrieve_supporting_sentences_without_pseudo_bm25(
        target_word,
        sense_gloss,
        searcher,
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
        threshold=threshold
    )
    
    return supporting_sentences