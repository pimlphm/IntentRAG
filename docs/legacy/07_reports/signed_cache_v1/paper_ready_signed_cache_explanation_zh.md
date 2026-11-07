# 可写入论文的 KV-cache 贡献：Admissibility-Signed KV Prefix

## 逻辑回答

前一步得到的是 `computational reuse != admissible evidence reuse`。下一步不能只是感叹“不等”，而要改变复用单位：不要让 KV cache 直接复用任意 prompt prefix，而要只复用带有 admissibility signature 的 canonical evidence-state prefix。

也就是说，缓存 key 不只是文本前缀，而应当变成：

`cache key = hash(admissibility signature) + canonical evidence-state prefix`

其中 admissibility signature 至少包括：

- aircraft/component scope;
- provenance anchor;
- authority cue;
- condition / applicability cue;
- action-prerequisite availability;
- revision/supersession proxy;
- evidence identity or evidence-set bucket.

这样，计算复用被限制在“可采纳性等价类”内部。只有当两个查询共享同一 admissibility signature 时，才允许复用同一段 evidence-state KV prefix。理论上，这使 computational reuse 向 admissible evidence reuse 对齐。

## 本地结果

本实验比较了 6 种 cache-key policy：

1. Plain prompt prefix;
2. State prefix;
3. Aircraft-signed;
4. Scope+authority signed;
5. Full admissibility signed;
6. Exact-evidence signed.

结果显示：

- 普通 state prefix 的 cache-hit opportunity 很高，hit-rate proxy = 1.000，但 admissibility purity 只有 0.565。也就是说它很容易复用，但共享组里仍混有不同 admissibility 状态。
- Full admissibility signed prefix 的 hit-rate proxy 仍然很高，为 0.991，同时 admissibility purity 提高到 1.000。这是最适合写成本文方法贡献的结果。
- Exact-evidence signed prefix 的 purity 也是 1.000，但 hit-rate proxy 降到 0.243。它最接近“计算复用等于证据复用”，但代价是复用机会大幅减少。

因此，本文可以提出一个中间策略：full admissibility signature。它不是最严格的 exact-evidence key，但在当前数据上达到了高 purity，同时几乎不损失 cache-hit opportunity。

## 可作为理论贡献的表述

本文的 KV-cache 贡献不是声称测量了真实 GPU KV cache 内部命中率，而是提出一种面向高风险证据生成的 cache-key 组织原则：admissibility-signed prefix caching。该原则要求可缓存前缀必须绑定到证据可采纳性签名，使得模型复用的不是任意上下文 token，而是当前查询可采纳的、结构化证据状态。

This changes the optimization target from maximizing prefix overlap to maximizing admissible prefix reuse.

## 可写入正文的英文段落

Computational reuse and evidential reuse can be aligned only if the reusable unit is changed. In ordinary prefix caching, two requests may share cached KV states whenever their serialized prompts share a prefix, even if the reused evidence is not equally applicable, authoritative, or action-supporting for the current query. We therefore define an admissibility-signed prefix: a canonical evidence-state prefix bound to a signature over aircraft/component scope, provenance, authority cues, condition cues, action-prerequisite availability, and evidence identity proxies. KV reuse is permitted only within the same signature. This converts cache reuse from a purely syntactic prefix-overlap mechanism into a constrained reuse of documentary evidence states.

## 可写入 Results 的英文段落

The signed-cache diagnostic shows a reuse-purity frontier. A state-prefix policy preserves maximal cache-hit opportunity (hit-rate proxy = 1.000), but its admissibility-vector purity is only 0.565. Binding the prefix to a full admissibility signature raises the purity to 1.000 while retaining nearly all cache opportunity (hit-rate proxy = 0.991). A stricter exact-evidence signature also reaches purity 1.000, but reduces hit-rate proxy to 0.243. These results indicate that admissibility signing can align computational reuse with evidence-state compatibility without collapsing reuse to exact evidence identity.

## 不能写成

- 不能写“本文测量了真实 KV cache hit rate”，因为当前是 tokenizer/grouping proxy。
- 不能写“KV cache 内部状态被证明等价于证据状态”，因为没有 layerwise tensor inspection。
- 不能写“action authorization 已经被验证”，因为 authority/action 仍是 proxy。
- 不能写“所有 RAG/RLM 系统都可直接使用该机制”，只能写成面向 evidence-grounded generation 的组织原则与项目数据诊断。

## 最稳妥的一句话贡献

The contribution is an admissibility-signed KV-prefix organization that makes cache reuse conditional on evidence-state compatibility, thereby turning prefix reuse from a purely computational optimization into a constrained form of reusable documentary support.

