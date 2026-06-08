# Appendix: Prompt Templates (Chinese with English Translation)

All prompts are written in Chinese and submitted to the LLM in Chinese.
English translations are provided for reference.
Placeholders in curly braces (e.g. `{text}`) are filled at runtime.

---

## A.1 Full Pipeline Prompts

### A.1.1 TopK Candidate Filtering (generate_topk_union)

Called three times per sample with different random seeds (42, 43, 44).
The union of selected wsids forms the candidate set for downstream components.

**Chinese (submitted to LLM):**

```
你正在执行【古汉语词义消歧任务】。

【目标句】
{text}

【候选义项】
[1] wsid={wsid_1}：{gloss_1}
[2] wsid={wsid_2}：{gloss_2}
...

请完成以下任务：
1. 从上述候选义项中，选出【最可能的{k}个义项】；
2. 无需做最终判断，只需在当前语境下"说得通"即可；
3. 为每个选中的义项简要说明理由。

【输出格式（严格JSON）】
{"top_senses": [{"wsid": "...", "reason": "..."}]}
```

**English translation:**

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
2. No final judgment required — only needs to be plausible
   in the current context;
3. Provide a brief reason for each selected sense.

[Output Format (strict JSON)]
{"top_senses": [{"wsid": "...", "reason": "..."}]}
```

---

### A.1.2 Sense Agent — Evidence Verification (sense_agent)

Called independently for each candidate sense. Filters retrieved
sentences and assesses contextual fit.

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

### A.1.3 Aggregator — Main Decision (with evidence)

Called when at least one candidate sense has valid examples.
Implements three-step structured reasoning with an explicit bias check.

**Chinese (submitted to LLM):**

```
你是古汉语词义消歧专家。

目标句：{text}
目标词：「{target_word}」

各候选义项的历史证据和语境判断如下：

【候选义项1】
wsid：{wsid_1}
义项定义：{gloss_1}
证据状态：{evidence_status_1}
目标语境契合度：{contextual_fit_1}
置信度：{confidence_1}
判断理由：{reason_1}
经验证的代表性例句：
  · {valid_example_1}
  · {valid_example_2}

【候选义项2】
wsid：{wsid_2}
义项定义：{gloss_2}
...
经验证的代表性例句：无（证据缺失，不作为排除理由）

请严格按以下顺序完成词义消歧：

第一步：语境优先判断
仅根据目标句语境和义项定义，判断每个义项与目标句的契合度。
不要参考历史例句数量，也不要因为某义项没有历史例句而降低
其合理性。

第二步：证据校准
只检查历史例句是否与目标句语境高度同构。
历史证据只能作为辅助校准，不能覆盖第一步的语境判断。
有历史例句但语境不匹配的义项，不得加分。
无历史例句的义项，证据状态为"未知"，不等于错误或不支持。

第三步：证据偏差检验
检查你是否因为某义项有历史例句而倾向选择它。
如果去掉历史例句后，另一个义项更符合目标句，则应优先选择
语境契合度更高的义项。

重要原则：
1. 证据缺失不等于义项错误。
2. 无证据等于未知，不等于不支持。
3. 证据数量不等于正确概率。
4. 语境契合度优先于历史证据可用性。
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

**English translation:**

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
Verified representative examples: None
(absence of evidence is not grounds for rejection)

Please perform disambiguation strictly in the following order:

Step 1: Context-first assessment
Judge each sense's fit with the target sentence based solely on
context and sense definition. Do not consider the number of
historical examples, and do not penalise a sense for lacking them.

Step 2: Evidence calibration
Check only whether historical examples are structurally parallel
to the target sentence context. Historical evidence serves as
auxiliary calibration only and cannot override Step 1.
Senses with examples whose context does not match gain no credit.
Senses with no examples have unknown evidence status — this is
not equivalent to being unsupported.

Step 3: Evidence-bias check
Examine whether you are inclined toward a sense because it has
historical examples. If removing the examples would favour a
different sense, prioritise the sense with higher contextual fit.

Key principles:
1. Missing evidence ≠ incorrect sense.
2. No evidence = unknown, not unsupported.
3. Evidence count ≠ probability of correctness.
4. Contextual fit takes priority over evidence availability.
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

### A.1.4 Aggregator — Missing-Evidence Re-verification

Triggered after the main decision when at least one non-selected
sense has no valid examples, to guard against evidence-absence bias.

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

---

### A.1.5 Aggregator — Degraded Mode (no evidence for any sense)

Triggered when all candidate senses have no valid examples.
The improved version (used in all reported results) passes sense
agent reasoning alongside the sense definitions.

**Chinese (submitted to LLM):**

```
你是古汉语词义消歧专家。

目标句：{text}
目标词：「{target_word}」

历史证据检索未能找到有效例句。
以下是各候选义项的独立分析结果：

义项1（wsid={wsid_1}）：{gloss_1}
义项分析：{reason_1}

义项2（wsid={wsid_2}）：{gloss_2}
义项分析：{reason_2}

请仅根据目标句的语境、义项定义及分析进行判断，
选出最符合目标句的义项。

输出严格JSON：
{"prediction": "wsid", "reason": "判断理由"}
```

**English translation:**

```
You are an expert in classical Chinese word sense disambiguation.

Target sentence: {text}
Target word: "{target_word}"

Historical evidence retrieval returned no usable examples.
The following are the independent analyses for each candidate sense:

Sense 1 (wsid={wsid_1}): {gloss_1}
Analysis: {reason_1}

Sense 2 (wsid={wsid_2}): {gloss_2}
Analysis: {reason_2}

Please select the sense that best fits the target sentence, based
solely on the target sentence context, sense definitions, and the
analyses above.

Output strict JSON:
{"prediction": "wsid", "reason": "disambiguation rationale"}
```

---

## A.2 Baseline Prompts

### A.2.1 PureLLM (No TopK, No Retrieval)

The full sense inventory is provided as candidates. No retrieval,
no TopK filtering, no evidence. The LLM selects directly.

**Chinese (submitted to LLM):**

```
你是古汉语词义消歧专家。

目标句：{text}
目标词：「{target_word}」

以下是所有候选义项：
wsid={wsid_1}：{gloss_1}
wsid={wsid_2}：{gloss_2}
...

请直接选出目标句中「{target_word}」最符合的义项。

输出严格JSON：
{"prediction": "wsid", "reason": "判断理由"}
```

**English translation:**

```
You are an expert in classical Chinese word sense disambiguation.

Target sentence: {text}
Target word: "{target_word}"

The following are all candidate senses:
wsid={wsid_1}: {gloss_1}
wsid={wsid_2}: {gloss_2}
...

Please select the sense that best fits the target word in the
target sentence.

Output strict JSON:
{"prediction": "wsid", "reason": "disambiguation rationale"}
```

---

### A.2.2 NaiveRAG (Direct Sentence Retrieval)

The target sentence is used as the retrieval query without
sense-aware pseudo-classical expansion. Retrieved sentences
are passed directly to the LLM without verification or aggregation.
The full sense inventory is used as candidates (no TopK filtering).

**Chinese (submitted to LLM):**

```
你是古汉语词义消歧专家。

目标句：{text}
目标词：「{target_word}」

[检索到的历史例句]
1. {retrieved_sentence_1}
2. {retrieved_sentence_2}
...

候选义项：
wsid={wsid_1}：{gloss_1}
wsid={wsid_2}：{gloss_2}
...

请选出目标句中「{target_word}」最符合的义项。

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

### A.3.1 Ablation A — w/o Verification

The Sense Agent LLM call is removed. All sentences containing the
target word are passed directly to the Aggregator without semantic
filtering. The Aggregator prompts are identical to §A.1.3–A.1.5.

*No additional prompt template at the verification stage.*

---

### A.3.2 Ablation B — w/o Aggregator (vote_decision)

The Aggregator is replaced by a standard LLM that receives the
concatenated Sense Agent outputs and selects directly without
structured cross-sense reasoning.

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

---

### A.3.3 Ablation C — w/o Verification and w/o Aggregator

Both the Sense Agent verification and the Aggregator are removed.
Unfiltered retrieved sentences are concatenated per sense and
passed directly to a standard LLM.

**Chinese (submitted to LLM):**

```
你是古汉语词义消歧专家。

目标句：{text}
目标词：「{target_word}」

以下是各候选义项及其历史检索证据：

义项1（wsid={wsid_1}）：{gloss_1}
历史例句：{sent_1} | {sent_2} | ...

义项2（wsid={wsid_2}）：{gloss_2}
历史例句：无

请直接选出最符合目标句的义项，输出该义项的wsid数字。

输出严格JSON：
{"prediction": "wsid数字", "reason": "判断理由"}
```

**English translation:**

```
You are an expert in classical Chinese word sense disambiguation.

Target sentence: {text}
Target word: "{target_word}"

The following are the candidate senses and their retrieved
historical evidence:

Sense 1 (wsid={wsid_1}): {gloss_1}
Historical sentences: {sent_1} | {sent_2} | ...

Sense 2 (wsid={wsid_2}): {gloss_2}
Historical sentences: None

Please select the sense that best fits the target word in the
target sentence and output its wsid.

Output strict JSON:
{"prediction": "wsid", "reason": "disambiguation rationale"}
```

---

### A.3.4 NoPseudo — w/o Pseudo-Classical Query Expansion

The pipeline is identical to the full system. The only difference
is that the sense gloss (in modern Chinese) is submitted directly
to BM25 retrieval without pseudo-classical reformulation.
All downstream prompts (§A.1.2–A.1.5) are unchanged.

*No additional prompt template — only the retrieval query differs.*

---

## A.4 LLM Configuration

All experiments use the same model and decoding settings.

| Parameter | Value |
|-----------|-------|
| Language | Chinese (all prompts) |
| Temperature | 0.0 |
| Random seed | 42 (TopK run 1), 43 (run 2), 44 (run 3); 42 for all other calls |
| Response format | JSON object |
| System message | 你是一个中文语言模型，专门用于古汉语词义消歧任务。 |
| Caching | MD5-keyed file cache; identical prompt + tag → cached response reused |
