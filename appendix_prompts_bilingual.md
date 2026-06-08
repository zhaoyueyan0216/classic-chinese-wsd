# Appendix: Prompt Templates (Chinese with English Translation)

Most prompts are written in Chinese and submitted to the LLM in Chinese;
English translations are provided for reference. Two exceptions are
noted explicitly where they occur: the TopK candidate-filtering prompt
(`generate_topk_union`) is composed and submitted **in English**, and
the system message is in Chinese for every call (see §A.4).
Placeholders in curly braces (e.g. `{text}`) are filled at runtime.
All templates below are transcribed verbatim from the corresponding
`f"""..."""` strings in `experiments/main_experiment.py` and
`core/pseudo_retrieval.py` (the other experiment scripts import and
reuse these same functions/templates — see the notes in each section).

---

## A.1 Full Pipeline Prompts

### A.1.1 Pseudo-Classical Sentence Generation (generate_pseudo_ancient_chinese)

Called once per candidate sense before retrieval. Generates four short
"pseudo-classical" usage sentences from the (modern-Chinese) sense
gloss; these are concatenated with the target word and gloss to form
the BM25 query (`build_bm25_query_from_pseudo`) and are also used for
embedding-based reranking of retrieved sentences. A lightweight
keyword match first classifies the gloss into one of four types
(action / state / abstract / entity) and that label is inserted into
the prompt.

*(Note: in the source, this prompt's f-string is indented four spaces
inside the function body, so the literal text sent to the model carries
that indentation plus a leading/trailing blank line. Reproduced below
without that incidental whitespace for readability.)*

**Chinese (submitted to LLM):**

```
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
{"pseudo_contexts": [{"text": "...", "pattern": "句式标签", "notes": "一句话说明该句如何体现义项"}]}
```

**English translation:**

```
You are generating "pseudo-classical usage sentences" for [Ancient
Chinese Word Sense Disambiguation], to be used in subsequent BM25
retrieval and embedding reranking. The goal is not literary elegance,
but to produce typical usages representative of this sense that can be
matched against genuine classical-Chinese sentences.

[Target word] {target_word}
[Sense definition] {gloss}
[Sense type] {gloss_type}  (must be one of: action / state / abstract / entity)

Generation requirements (must be strictly followed):
1) Output 4 short "classical-Chinese-style" sentences (8-20 characters
   each), avoiding modern Chinese word order (no "的, 了, 在, 把, 被,
   因为, 所以, 但是", etc.).
2) Each sentence must contain the [target word] exactly once.
3) Do not introduce concrete historical facts: no specific dynasties,
   reign titles, real personal names, real place names, or titles of
   real books/chapters.
4) Vary the sentence patterns: include at least two distinct patterns.
5) Stay faithful to the [sense definition]; do not produce usages that
   belong to other senses.

Output strict JSON only, with no extra text:
{"pseudo_contexts": [{"text": "...", "pattern": "sentence-pattern label", "notes": "one sentence on how this instantiates the sense"}]}
```

*Called via `cached_call_llm(..., model="deepseek-v3.2-exp")`, which
overrides the project's default LLM model for this step only.*

---

### A.1.2 TopK Candidate Filtering (generate_topk_union)

Called three times per sample with different random seeds (42, 43, 44).
The union of selected wsids forms the candidate set for downstream
components. **Unlike the other prompts in this appendix, this prompt
is composed and submitted to the LLM directly in English** (the
f-string in the source code contains no Chinese).

**English (submitted to LLM):**

```
You are performing an [Ancient Chinese Word Sense Disambiguation Task].

[Target Sentence]
{text}

[Candidate Senses]
[1] wsid={wsid_1}: {gloss_1}
[2] wsid={wsid_2}: {gloss_2}
...

Please complete the following tasks:
1. Select the [most likely {k} senses] from the above options;
2. No final judgment required, only need to be "reasonable in the current context";
3. Provide a brief reason for each selected sense.

[Output Format (strict JSON)]
{"top_senses": [{"wsid": "...", "reason": "..."}]}
```

**中文参考译文：**

```
你正在执行【古汉语词义消歧任务】。

【目标句】
{text}

【候选义项】
[1] wsid={wsid_1}：{gloss_1}
[2] wsid={wsid_2}：{gloss_2}
...

请完成以下任务：
1. 从上述候选义项中，选出"最可能的{k}个义项"；
2. 无需做最终判断，只需在当前语境下"说得通"即可；
3. 为每个选中的义项简要说明理由。

【输出格式（严格JSON）】
{"top_senses": [{"wsid": "...", "reason": "..."}]}
```

---

### A.1.3 Sense Agent — Evidence Verification (sense_agent)

Called independently for each candidate sense (capped at the top 10
retrieved sentences). Filters retrieved sentences and assesses
contextual fit.

**Case 1: Retrieved sentences exist**

**Chinese (submitted to LLM):**

```
你是古汉语词义专家。

目标句：{text}
目标词：「{target_word}」
候选义项：{sense_gloss}

以下是为该义项检索到的历史例句：
1. {sentence_1}
2. {sentence_2}
...

请完成两个任务：

【任务1】筛选有效例句
从上面的例句中，找出「{target_word}」确实体现了「{sense_gloss}」
这个义项的句子。
注意：
- 语境高度相似的例句只选1句代表（避免重复）
- 无法判断义项归属的句子不选
- 宁可一句不选，也不选不确定的

【任务2】判断目标句
参考筛选出的有效例句的语境，判断目标句中「{target_word}」
是否符合「{sense_gloss}」这个义项。
如果有效例句为空，仅根据义项定义和目标句语境判断。

输出严格JSON，不要输出其他内容：
{
  "valid_examples": ["例句原文", ...],
  "evidence_status": "supported或noisy或unknown",
  "contextual_fit": "high或medium或low",
  "support": true或false或null,
  "confidence": "high或medium或low或unknown",
  "reason": "一句话说明：先判断目标句与义项定义是否契合，
             再说明历史例句是否提供辅助支持"
}
```

**English translation:**

```
You are an expert in classical Chinese lexical semantics.

Target sentence: {text}
Target word: "{target_word}"
Candidate sense: {sense_gloss}

The following historical sentences have been retrieved for this sense:
1. {sentence_1}
2. {sentence_2}
...

Please complete two tasks:

[Task 1] Filter valid examples
From the sentences above, identify those in which "{target_word}"
genuinely instantiates the sense "{sense_gloss}".
Notes:
- Select at most one representative sentence for highly similar contexts
- Do not select sentences whose sense assignment is uncertain
- When in doubt, select nothing rather than an uncertain example

[Task 2] Assess the target sentence
Using the context of the valid examples as reference, judge whether
"{target_word}" in the target sentence corresponds to
"{sense_gloss}". If no valid examples exist, judge based on the
sense definition and target sentence context alone.

Output strict JSON only:
{
  "valid_examples": ["verbatim sentence", ...],
  "evidence_status": "supported / noisy / unknown",
  "contextual_fit": "high / medium / low",
  "support": true / false / null,
  "confidence": "high / medium / low / unknown",
  "reason": "one sentence: first assess fit between target sentence
             and sense definition, then state whether historical
             examples provide supporting evidence"
}
```

**Case 2: No retrieved sentences (no LLM call; fixed return value)**

```
{
  "valid_examples": [],
  "evidence_status": "unknown",
  "contextual_fit": "unknown",
  "support": null,
  "confidence": "unknown",
  "reason": "未检索到含目标词的历史例句；
             这表示证据缺失，不表示该义项不符合目标句。"
}
```

*English: "No historical sentences containing the target word were
retrieved. This indicates an absence of evidence, not that the sense
is incorrect."*

---

### A.1.4 Aggregator — Main Decision (with evidence)

Called when at least one candidate sense has valid examples.
Implements three-step structured reasoning with an explicit bias check.
Note that the source prompt itself mixes Chinese step descriptions with
English step labels and several English-language "key principles" —
this is transcribed verbatim below, not a translation artefact.

**Chinese/English as submitted to LLM (verbatim from source):**

```
你是古汉语词义消歧专家。

目标句：{text}
目标词：「{target_word}」

各候选义项的历史证据和语境判断如下：

【候选义项1】
wsid: {wsid_1}
义项定义: {gloss_1}
证据状态: {evidence_status_1}
目标语境契合度: {contextual_fit_1}
置信度: {confidence_1}
判断理由: {reason_1}
经验证的代表性例句:
  · {valid_example_1}
  · {valid_example_2}

【候选义项2】
wsid: {wsid_2}
义项定义: {gloss_2}
...
经验证的代表性例句: 无（证据缺失，不作为排除理由）

请严格按以下顺序完成词义消歧：

第一步：Context-first 判断
仅根据目标句语境和义项定义，判断每个义项与目标句的契合度。
不要参考历史例句数量，也不要因为某义项没有历史例句而降低其合理性。

第二步：Evidence calibration
只检查历史例句是否与目标句语境高度同构。
历史证据只能作为辅助校准，不能覆盖第一步的语境判断。
有历史例句但语境不匹配的义项，不得加分。
无历史例句的义项，证据状态为 unknown，不等于错误或不支持。

第三步：Evidence-bias check
检查你是否因为某义项有历史例句而倾向选择它。
如果去掉历史例句后，另一个义项更符合目标句，则应优先选择语境契合度更高的义项。

重要原则：
1. Missing Evidence ≠ Rejection。
2. No evidence = Unknown, not Unsupported。
3. Evidence count is not probability。
4. Contextual compatibility has priority over historical evidence availability。
5. 如果某义项无历史证据但最符合目标句，应选择该义项。
6. 如果某义项有历史证据但与目标句语境不契合，不应选择该义项。

输出严格JSON：
{
  "step1": "仅基于目标句和义项定义的语境判断",
  "step2": "历史证据是否真正与目标句语境同构",
  "bias_check": "是否存在因有证据而偏向某义项的风险",
  "prediction": "wsid",
  "reason": "最终判断理由"
}
```

**English translation (of the Chinese portions):**

```
You are an expert in classical Chinese word sense disambiguation.

Target sentence: {text}
Target word: "{target_word}"

Historical evidence and contextual assessments for each candidate:

[Candidate Sense 1]
wsid: {wsid_1}
Sense definition: {gloss_1}
Evidence status: {evidence_status_1}
Contextual fit: {contextual_fit_1}
Confidence: {confidence_1}
Reasoning: {reason_1}
Verified representative examples:
  · {valid_example_1}
  · {valid_example_2}

[Candidate Sense 2]
wsid: {wsid_2}
Sense definition: {gloss_2}
...
Verified representative examples: None (absence of evidence is not
grounds for rejection)

Please perform disambiguation strictly in the following order:

Step 1 (Context-first assessment): Judge each sense's fit with the
target sentence based solely on context and sense definition. Do not
consider the number of historical examples, and do not penalise a
sense for lacking them.

Step 2 (Evidence calibration): Check only whether historical examples
are structurally parallel to the target sentence context. Historical
evidence serves as auxiliary calibration only and cannot override
Step 1. Senses with examples whose context does not match gain no
credit. Senses with no examples have "unknown" evidence status — this
is not equivalent to being unsupported.

Step 3 (Evidence-bias check): Examine whether you are inclined toward
a sense because it has historical examples. If removing the examples
would favour a different sense, prioritise the sense with higher
contextual fit.

Key principles:
1. Missing evidence ≠ rejection.
2. No evidence = unknown, not unsupported.
3. Evidence count is not probability.
4. Contextual compatibility takes priority over evidence availability.
5. Select a sense with no evidence if it best fits the context.
6. Do not select a sense with evidence if its context does not fit.

Output strict JSON:
{
  "step1": "contextual assessment based on target sentence and
            sense definitions only",
  "step2": "whether historical evidence is structurally parallel
            to the target sentence",
  "bias_check": "whether there is a risk of evidence-induced bias
                 toward a particular sense",
  "prediction": "wsid",
  "reason": "final disambiguation rationale"
}
```

---

### A.1.5 Aggregator — Missing-Evidence Re-verification

Triggered after the main decision (§A.1.4) whenever at least one
*non-selected* sense has no valid examples, to guard against
evidence-absence bias.

**Chinese (submitted to LLM):**

```
你是古汉语词义消歧专家。

你已初步选择义项 {initial_prediction}。

目标句：{text}
目标词：「{target_word}」

以下义项没有历史证据，可能被排除：
- wsid={wsid_x}：{gloss_x}（无历史证据）

请回答：
1. 你排除了上述哪些义项？
2. 排除它们的理由是什么？（是否仅因为它们缺乏历史证据？）

如果你的排除理由包含"缺乏历史证据"或类似表述，请重新评估。
重新思考：仅凭目标句语境和义项定义，这些被排除的义项是否
可能更符合？

重要原则：证据缺失不等于义项错误。必须仅凭语境契合度判断。

输出严格JSON：
{
  "excluded_check": "你是否仅因缺乏证据而排除了某些义项？",
  "reassessment": "重新评估后的判断",
  "final_prediction": "wsid（如果与初步选择不同则更新）",
  "final_reason": "最终判断理由"
}
```

**English translation:**

```
You are an expert in classical Chinese word sense disambiguation.

You have provisionally selected sense {initial_prediction}.

Target sentence: {text}
Target word: "{target_word}"

The following senses have no historical evidence and may have
been excluded:
- wsid={wsid_x}: {gloss_x} (no historical evidence)

Please answer:
1. Which of the above senses did you exclude?
2. What was your reason for exclusion? (Was it solely due to
   lack of historical evidence?)

If your reason for exclusion mentions "lack of historical evidence"
or similar, please re-evaluate. Reconsider: based solely on the
target sentence context and sense definitions, could any of the
excluded senses be a better fit?

Key principle: Absence of evidence does not mean the sense is
incorrect. Judgement must be based solely on contextual fit.

Output strict JSON:
{
  "excluded_check": "did you exclude any sense solely due to
                     lack of evidence?",
  "reassessment": "revised assessment after re-evaluation",
  "final_prediction": "wsid (update if different from provisional
                       selection)",
  "final_reason": "final disambiguation rationale"
}
```

*If this verification call's JSON cannot be parsed, the aggregator
silently falls back to the provisional `initial_prediction`/`reason`
from §A.1.4.*

---

### A.1.6 Aggregator — Degraded Mode (no evidence for any sense)

Triggered when **all** candidate senses have zero valid examples. In
this mode the prompt lists only the wsid and gloss of each candidate —
**no per-sense reasoning from the Sense Agent is forwarded** — and asks
the LLM to choose based purely on the target sentence and sense
definitions.

**Chinese (submitted to LLM):**

```
你是古汉语词义消歧专家。

目标句：{text}
目标词：「{target_word}」

历史证据检索未能找到有效例句。
请仅根据目标句的语境和以下义项定义进行判断。

候选义项：
wsid={wsid_1}: {gloss_1}
wsid={wsid_2}: {gloss_2}
...

请选出目标句中「{target_word}」最符合的义项。

输出严格JSON：
{"prediction": "wsid", "reason": "判断理由"}
```

**English translation:**

```
You are an expert in classical Chinese word sense disambiguation.

Target sentence: {text}
Target word: "{target_word}"

Historical evidence retrieval returned no usable examples.
Please judge based solely on the target sentence's context and the
following sense definitions.

Candidate senses:
wsid={wsid_1}: {gloss_1}
wsid={wsid_2}: {gloss_2}
...

Please select the sense that best fits the target word in the
target sentence.

Output strict JSON:
{"prediction": "wsid", "reason": "disambiguation rationale"}
```

---

## A.2 Baseline Prompts

### A.2.1 PureLLM (pure_llm configuration)

Implemented in `ablation_nopseudo_experiment.py` as a configuration
flag set, **not** as a separate stand-alone prompt: `pure_llm` reuses
the exact same pipeline function (`process_sample_with_union`) as the
full system, but with `use_top2_filter=False` and
`use_supporting_sentences=False`. Concretely this means:

- The complete sense inventory (`candidate_senses`) is used as the
  candidate set — TopK filtering (§A.1.2) is bypassed.
- No retrieval is performed; each Sense Agent slot is filled with a
  fixed placeholder result (`evidence_status="unknown"`,
  `valid_examples=[]`, `reason="未启用支持句检索"` — "sentence
  retrieval not enabled") rather than an LLM call.
- Because every candidate therefore has zero valid examples, the
  Aggregator always falls into **Degraded Mode (§A.1.6)** and selects
  directly from the full sense inventory using only the target
  sentence and sense definitions.

*No separate "PureLLM" prompt template exists in the codebase — the
baseline is realised entirely by disabling retrieval/filtering and
routing through the existing degraded-mode aggregator prompt.*

---

### A.2.2 NaiveRAG (Direct Sentence Retrieval)

Implemented in `naive_rag_experiment.py`. The target sentence itself
is used as the retrieval query, without sense-aware pseudo-classical
expansion (§A.1.1) or per-sense BM25 queries. Retrieved sentences
(filtered to those containing the target word, capped at 10) are
passed directly to the LLM alongside the *full* sense inventory
(no TopK filtering, no Sense Agent verification, no Aggregator).

**Chinese (submitted to LLM):**

```
你是古汉语词义消歧专家。

目标句：{text}
目标词：「{word}」

[检索到的历史例句]
1. {retrieved_sentence_1}
2. {retrieved_sentence_2}
...

候选义项：
wsid={wsid_1}: {gloss_1}
wsid={wsid_2}: {gloss_2}
...

请选出目标句中「{word}」最符合的义项。

输出严格JSON：
{"prediction": "wsid", "reason": "判断理由"}
```

*If no sentences are retrieved, the evidence section reads:*
`[未检索到含目标词的历史例句]`

**English translation:**

```
You are an expert in classical Chinese word sense disambiguation.

Target sentence: {text}
Target word: "{target_word}"

[Retrieved historical sentences]
1. {retrieved_sentence_1}
2. {retrieved_sentence_2}
...

Candidate senses:
wsid={wsid_1}: {gloss_1}
wsid={wsid_2}: {gloss_2}
...

Please select the sense that best fits the target word in the
target sentence.

Output strict JSON:
{"prediction": "wsid", "reason": "disambiguation rationale"}
```

*If no sentences are retrieved:*
`[No historical sentences containing the target word were retrieved]`

---

## A.3 Ablation Prompts

### A.3.1 Ablation A — w/o Verification (sense_agent_no_verify)

Implemented in `ablation_a_experiment.py`. The Sense Agent LLM call
(§A.1.3) is removed entirely — there is no prompt at this stage. All
retrieved sentences containing the target word are passed through
unfiltered as `valid_examples`, and `evidence_status` /
`contextual_fit` / `confidence` are assigned heuristically from the
example count (≥3 → "supported"/"medium"/"medium"; 1–2 →
"supported"/"low"/"low"; 0 → "unknown"/"unknown"/"unknown"). The
resulting agent results are then passed to the unmodified Aggregator
(§A.1.4–§A.1.6).

*No additional prompt template at the verification stage.*

---

### A.3.2 Ablation B — w/o Aggregator (vote_decision)

Implemented in `ablation_b_experiment.py`. The structured Aggregator
(§A.1.4–§A.1.6) is replaced by `vote_decision`, a single LLM call that
receives the concatenated Sense Agent outputs (wsid, gloss, reasoning,
and — where available — one representative example per sense) and
selects directly, without cross-sense structured reasoning, bias
checking, or a re-verification pass.

**Chinese (submitted to LLM):**

```
你是古汉语词义专家。

目标句：{text}
目标词：「{target_word}」

以下是各候选义项的独立分析结果：
义项1（wsid={wsid_1}）：{gloss_1}
判断：{reason_1}
例句：{valid_example_1}

义项2（wsid={wsid_2}）：{gloss_2}
判断：{reason_2}

请直接选出最符合目标句的义项，输出该义项的wsid。

输出严格JSON：
{"prediction": "wsid数字", "reason": "判断理由"}
```

**English translation:**

```
You are an expert in classical Chinese lexical semantics.

Target sentence: {text}
Target word: "{target_word}"

The following are the independent analyses for each candidate sense:
Sense 1 (wsid={wsid_1}): {gloss_1}
Assessment: {reason_1}
Example: {valid_example_1}

Sense 2 (wsid={wsid_2}): {gloss_2}
Assessment: {reason_2}

Please select the sense that best fits the target word in the
target sentence and output its wsid.

Output strict JSON:
{"prediction": "wsid", "reason": "disambiguation rationale"}
```

*Note: `main_experiment.py` also defines a non-LLM, confidence-ranking
`vote_decision` (picks the highest-confidence sense with `support ==
true`, falling back to the first candidate). It is not used by
`ablation_b_experiment.py`, which defines and imports its own
LLM-based version shown above.*

---

### A.3.3 No-History — w/o Historical-Evidence Retrieval

Implemented in `ablation_nohistory_experiment.py`
(`process_ablation_nohistory`). After TopK filtering (§A.1.2, k=2), no
retrieval and no Sense Agent call are performed at all — each
candidate is given a fixed placeholder evidence record
(`evidence_status="unknown"`, `valid_examples=[]`,
`reason="未启用历史证据检索"` — "historical evidence retrieval not
enabled"). Because every candidate has zero valid examples, the
unmodified Aggregator (§A.1.4–§A.1.6) always falls into **Degraded
Mode (§A.1.6)** and decides from the TopK-filtered sense definitions
alone.

*No additional prompt template — retrieval and verification are
skipped, and the existing degraded-mode aggregator prompt is reused.*

---

### A.3.4 NoPseudo — w/o Pseudo-Classical Query Expansion

Implemented in `ablation_nopseudo_experiment.py` (the `no_pseudo`
configuration of the multi-config sweep) and, as a standalone runner
restricted to this single configuration, in
`nopseudo_only_experiment.py`. The pipeline is otherwise identical to
the full system — TopK filtering, Sense Agent verification, and the
Aggregator all use the exact same prompt templates as §A.1.2–§A.1.6
(verified to be byte-identical f-strings in the source). The only
difference is at the retrieval stage: instead of generating
pseudo-classical example sentences (§A.1.1) and folding them into the
BM25 query, `get_supporting_sentences_without_pseudo_bm25` builds the
query directly from `+{target_word} {gloss}` (i.e. the modern-Chinese
sense gloss plus the target word, with no LLM-generated expansion and
no pseudo-sentence-based embedding rerank).

*No additional prompt template — only the retrieval query construction
differs; §A.1.1 (pseudo-sentence generation) is simply not invoked.*

---

## A.4 LLM Configuration

All experiments call `core.llm.call_llm` / `cached_call_llm`, which use
the same model and decoding settings (with one documented exception
for pseudo-sentence generation).

| Parameter | Value |
|-----------|-------|
| Language | Chinese for nearly all prompts; the TopK prompt (§A.1.2) is composed in English |
| Temperature | 0.0 (hard-coded in `call_llm`, regardless of the value passed in) |
| Random seed | 42 (TopK run 1), 43 (run 2), 44 (run 3); 42 for `sense_agent` calls; 42 for `call_llm` default; pseudo-sentence generation does not pass a seed |
| Response format | `{"type": "json_object"}` |
| System message | 你是一个中文语言模型，专门用于古汉语词义消歧任务。<br>("You are a Chinese-language model specialised in classical-Chinese word sense disambiguation.") |
| Pseudo-sentence model override | `generate_pseudo_ancient_chinese` calls `cached_call_llm(..., model="deepseek-v3.2-exp")`, overriding the project's configured default model for that step only |
| Caching | `cached_call_llm` computes an MD5 key over `cache_tag + prompt + sorted(kwargs)`; an identical key reuses the cached response instead of calling the LLM again |
