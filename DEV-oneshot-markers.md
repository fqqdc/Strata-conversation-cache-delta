# DEV — one-shot markers（一次性请求标记）

`serve/oneshot_rules.py` 读运行配置里的规则，认出「这次调用留下的会话永远不会再被续用」的请求，
替客户端把话说出来（`strata_checkpoint: false` → 引擎的 `ckpt=0`），于是磁盘会话缓存**不为它落盘**。
**引擎一行没改**：`ckpt=0` 自 0.1.41 起就存在。

- 提交：`3c952ebf`，分支 `conversation-cache-delta`（基于 `8a720695` = 合并上游 main v0.1.44.1）。
- 新增 3 个文件：`serve/oneshot_rules.py`（527 行）、`serve/test_oneshot_rules.py`（543 行 / 66 项）、
  `serve/server.py` +75 行；**纯新增、零删除**。
- 实测机：RTX 5070 Ti 16G（sm_120）、Ryzen 9 9950X、64G RAM、Windows 11、CUDA 13.3；
  引擎 0.1.44.1，IQ3_XXS，512K 上下文，`--conversation-cache-disk-only --conversation-cache-disk-delta`。

---

## 1. 为什么要有它

磁盘专用模式下，一个会话被下一个请求切走时整份状态写进 spill 目录。有些请求是**结构上一次性的**，
它们留下的状态永远不会被续用，那笔写就是纯浪费——而引擎判不出来：保存时它只有 token，
不知道"这次调用是谁发的、还会不会回来"。

本机实测（同一台机、同一条主会话，修复前 → 现在）：

| 观测 | 修复前 | 现在 |
|---|---|---|
| DSH 起标题调用（126 prompt token） | `disk-saved 136 tokens (226 MiB, switch)`，逐会话 1–2 次，从不读回 | `marked by ds-title - 126 prompt tokens -> ckpt=0`，**零写盘** |
| DSH 上下文压缩（21176 prompt token） | 切换时 `disk-saved 24587 tokens (436 MiB, switch)`，从不读回 | `marked by ds-compaction - 21176 prompt tokens -> ckpt=0`，**零写盘** |
| 一整轮生产实例的 `disk-saved` | 每次各一笔 | **0** |
| 主会话自己的落盘与复用 | 正常 | 不变（19346/19364、19784/19968 = 99.9%） |

体积与 token 数无关：226 MiB 由两份稠密运行态主导，所以**很小的请求也会写出很大的文件**。

## 2. 机制：三行链路

```
规则命中  →  服务端给这次请求的采样字典写 strata_checkpoint: false
         →  sampling_keys() 翻成引擎键 ckpt=0                    (serve/server.py:1284)
         →  引擎按 ckpt=0 处理这次调用                            (src/program/generate.cpp:11013)
```

`ckpt=0` 的引擎语义（照抄源码注释的意思）：这次调用的 turn **不被后续请求续写、不被 park、
磁盘模式下不落盘**，也不在 turn 边界与 `--prompt-cache-every` 处留检查点
（`pp_tail_saved = reread_to > 0 || !req_ckpt`，`generate.cpp:11868`），但**仍可复用它匹配到的检查点**。
换句话说：这不是新增一种落盘行为，而是服务端把"客户端本来就能自己发的那个字段"替客户端发出来。

求值在 `generate()` 的第一行（`serve/server.py:1959`），拿到**整份请求体**和**已分词的 prompt id 序列**——
规则既能看字段，也能看 token（守卫和锚点用后者）。它只改采样字典的**副本**，原字典不动；
规则抛异常时打一行日志、请求照常发出（这个功能永远不能让请求失败）。

规则按配置顺序**全部**求值、不短路，命中集合写进日志行和内存计数器 `self.oneshot_marked`
（只在内存，`/metrics` 未暴露）。

## 3. 今天怎么用

唯一开关是运行配置里的 `oneshot_markers` 块（本机在 git-ignored 的 `strata-iq3_xxs.json`）：

```jsonc
"oneshot_markers": {
  "enabled": true,        // 关掉 = 回到发布行为
  "log": true,            // 每命中一次打一行
  "action": "ckpt0",      // 目前只有这一种动作
  "probe": false,         // 诊断用：打印请求体形状；不要带进生产
  "rules": [ /* 见下表 */ ]
}
```

| type | 匹配什么 | 参数 |
|---|---|---|
| `field` | 请求体字段（点分路径，数组下标写 `a.0.b`） | 可选 `equals`；JSON 类型严格（`false` ≠ `0` ≠ `"false"`） |
| `request_kind` | Codex 的 `x-codex-turn-metadata.request_kind`（或平铺的 `request_kind`） | `equals` |
| `system_text` | system 提示文本**前缀**（多条 system/`developer` 消息按顺序用 `\n` 拼起来比） | `prefix`；`match: "text"`(默认)/`"tokens"`；可选 `max_prompt_tokens` 守卫（**只能排除，不能选中**） |
| `user_text` | **最后一条** user 消息**前缀** | `prefix` |

当前配置里 3 条规则，各自的依据：

| id | 依据 | 说明 |
|---|---|---|
| `ds-title` | DSH 标题调用的 `developer` 文本前缀 | 每个新会话都发同一条固定文本 |
| `ds-compaction` | DSH 的压缩指令**就是本次调用的最后一条 user 消息** | 客户端构造如此，且该指令不再回放进后续会话 |
| `client-flag` | 客户端自己发 `strata_checkpoint: false` | 与手工调用同一个键；**今天没有任何客户端发它**，它不会自己命中（见下） |

未启用的类型仍然可用，只是本机没有启用：`request_kind` 面向 Codex 的 compaction 声明，
子代理那一类见 §7.1。`client-flag` 值得单独说清楚，因为它和另外两条不是一类东西：

- 它匹配的是**顶层请求体字段 `strata_checkpoint` 严格等于 `false`**（JSON 布尔；`0`、`"false"`、
  字段缺失都不算）。这个字段本来就是引擎侧的公开旋钮（`serve/server.py:1281`，issue #830）：
  客户端自己发它，`sampling_keys()` 就直接翻成 `ckpt=0`——**不需要本模块的任何一条规则**。
- 所以 `client-flag` 不新增行为，它做的是把"客户端自己声明的一次性调用"也纳入**同一份可见性**：
  命中会打出一行 `marked by client-flag …` 并计入 `self.oneshot_marked`，否则这类请求会静默地
  走另一条路（行为一样，但日志里看不出它是一次性调用）。
- 它也是面向未来的一格：若 `action` 以后不止 `ckpt=0`（例如引擎侧新增"留 RAM、不落盘"的键），
  客户端只说 `strata_checkpoint: false` 就不会自动获得新行为，而这条规则把身份显式写进了配置。
- 误伤风险为零：字段必须存在且恰为 `false`。
- 删掉它的后果：这类客户端**行为不变**（照样 `ckpt=0`），只是服务端不再有那行日志和计数。

**启动就会拒绝**（`MarkerConfigError`，进程退出，绝不退化成逐请求失效）：未知顶层键/规则键、
未知 `type`、未实现的 `action`、把启发式键（`max_tokens`/`tokens`/`message_count`/`idle_s`/`length`/
`round`…，`oneshot_rules.py:88`）写进规则、重复 `id`、`enabled: true` 却无规则、空前缀、键与类型不匹配。
反方向的容错是刻意的不对称：`match: "tokens"` 的规则拿不到 tokenizer 或前缀编码为空时，
**只丢这一条并打一行 note**，不影响启动和其它规则。

**没有热更新**：规则只在 `main()` 解析一次，改配置必须重启。

## 4. 怎么证明它生效（这一节最重要）

规则的命中和不命中在服务端**都是静默的**——服务 200、回复正常、日志干净，都可能意味着"一次都没标记"。
所以按阶梯走，最后两级不能省：

1. **单测**：`.\.venv\Scripts\python.exe -m unittest serve.test_oneshot_rules` → 66 项，~4 ms，
   不需要引擎、tokenizer、GPU。它覆盖规则语义、配置契约，以及两个**真实 DSH 请求体**（正例与反例）。
2. **离线启动**（mock 引擎，不加载模型、不占显存）：
   `.\.venv\Scripts\python.exe -X utf8 serve\server.py --engine mock --config strata-iq3_xxs.json --port 4399`
   → 期望 `[strata] oneshot markers on: action=ckpt0 log=True rules=N` 加逐条规则；
   把 `action` 改成未实现的值必须**启动即失败**（退出码 1，stderr 说明理由）。
3. **真机**：开一个新会话发一句话（触发标题调用），同一会话再聊 2–3 轮。服务端出现
   `marked by ds-title …`；引擎日志（`strata-iq3_xxs.log`）里**没有**对应的
   `disk-saved 1xx tokens (226 MiB, switch)`。
4. **对照**：主会话照常出现自己的 `disk-saved … (39x MiB, switch)`，`reused` 比例不降。

排障对照表：

| 日志行 | 含义 |
|---|---|
| `[strata] oneshot markers on: action=ckpt0 log=True rules=N` | 启动横幅，随后逐条列出规则读法 |
| `[strata] oneshot markers: marked by <id,…> - N prompt tokens, body B B -> ckpt=0` | **命中的唯一直接证据**（服务端 stdout） |
| `[strata] oneshot markers: rule evaluation failed (…)` | 规则抛异常，该请求按未标记发送（不该出现） |
| `[strata] oneshot_markers rule 'x' disabled: …` | token 锚点准备失败，规则被丢弃（不阻止启动） |
| 引擎 `prompt … , N checkpoints` | 被标记的调用不再新增检查点（对照未标记轮次） |
| 引擎 `conversation cache: disk-saved …` | 被标记会话**应当没有**这一笔 |

⚠ 服务端 stdout **不落文件**（`run-iq3_xxs-local.bat` 没重定向），`marked by` 只在控制台窗口里。
要留证据就把输出重定向到文件。

## 5. 要加规则、改行为，改哪里

**第一档：同类规则 → 只改配置，0 行代码。** 加进 `rules` 数组，重启，在横幅里确认它被列出，
再触发一次真实调用看 `marked by <你的 id>`。绝大多数需求到此为止。

**第二档：语义不够（要新的匹配方式）→ `oneshot_rules.py` 四处 + 测试。** 例如"在消息中段做 contains"
（子代理那一类要用）：`MATCHES`（L85）加模式 → `_parse_rule()` 对应分支（`user_text` 在 L246，
它现在明确拒绝 `match`）允许并校验 → 求值处（`user_text()` L346 或 `match()` L438 的分支）改成遍历
所有 user 消息 → `describe()`（L511）文案 → 在 `serve/test_oneshot_rules.py` 补正/反例。

**第三档：新规则类型 / 新动作。**
- 新类型：常量 + `RULE_TYPES`（L55）→ `Rule` 字段（L106）→ `_parse_rule()`（L182）→ `match()`（L438）
  → `describe()`（L511）→ 有新键就进 `ALLOWED_RULE`（L71）→ 测试。可用输入：整份请求体、prompt token id。
- 新动作（比如"留 RAM、永不落盘"）：`ACTIONS`（L59）+ `decision()`（L498）返回"要写的键"；
  `mark_one_shot()`（server.py:1202）不用改，它只负责把键写进采样字典——但**引擎必须认识那个键**，
  那是 `src/program/generate.cpp` 的采样键解析，属引擎改动。

## 6. 不能破坏的东西（改之前先读）

- **默认路径**：配置里没有 `oneshot_markers` 块（所有现存部署）→ `enabled=False` → `mark_one_shot()`
  原样返回采样字典，请求路径与发布版一致。这是本功能唯一的"不打扰"承诺。
- **不猜**：规则只认声明的身份（字段、请求种类、固定文本），不认大小/条数/空闲时间。
  这道线由 `HEURISTIC_KEYS`（L88）按名字强制。
- **失败方向不对称**：配置错误 → 启动即死；能力缺失 → 丢规则打 note。两者都不能反过来。
- **`__init__` 只在缺失时初始化**：懒加载引擎在第一个请求时会二次执行 `__init__`（`restart()`），
  一个朴素赋值会把 `main()` 挂上去的规则集抹掉——服务一切正常，**但一次都不标记**。
  移植时**没有**带上原实验的 wire 测试，所以改动 `__init__`/`main()` 装配后必须手工复核
  （横幅的 `rules=N` + 一次真实调用的 `marked by`）。
- **`system_text` 必须同时认 `developer`**：DSH 用 `developer` 角色承载系统提示（没有顶层 `system`
  字段），Strata 自己的适配器把 `developer` 当 `system`（`serve/frontend.py:526`，issue #56）。
  最初只认 `system` 的版本让**所有文本规则静默落空**。
- 上面两条的表现完全一样："服务一切正常、功能彻底没生效"。所以 §4 的第 3、4 级证据不能省。

## 7. 已知限制与下一步

1. **子代理一类没处理**（缺客户端信号）。判别特征已经找到并核实：DSH 给每个子会话注入一句
   `You are a delegated subagent: your permission scope was fixed when you were started …`
   （DSH 源码 `app.asar` 第 988199 行的 `SUBAGENT_DELEGATION_CONTEXT`，由 `applyChildComposition()`
   注册为 **runtime-context 贡献**——故意不放进系统提示，所以 `system_text` 永远看不见它；
   新建、fork/seed、`send_message` 续跑三条路径都带）。落地要的是 §5 第二档的 `contains` 模式。
   **但先别急着开**：把子代理标记成 `ckpt=0` 会让它在每一轮重读自己的上下文（只能命中共享根），
   代价随轮数近似平方增长；只有引擎侧"留 RAM、不落盘"的降级键才是两全。
2. **旧的标题条目不会被删**：盘上已存在的那个 226 MiB 条目由磁盘层自己的 LRU 回收，而且每次新会话的
   标题调用会**恢复**它（实测 `disk-restored 119 tokens (checkpoint, 226 MiB) in 305.4 ms`），
   恢复会刷新它的 `last_used`——它可能长期占着那 226 MiB。想更干净，需要在磁盘层加"不恢复一次性条目"
   的策略。
3. **无热更新**、**计数只在内存**（要长期观测得给 `/metrics` 加字段）。
4. 只在 Windows/CUDA 单 GPU 验证过；模块本身与平台无关，但没在 Linux/AMD/多 GPU 上测。
5. 上游若改动 `developer → system` 的映射，所有文本规则会再次静默落空（见 §6）。

## 8. 回滚

- **只关功能**：删掉配置块，或 `"enabled": false` → 立刻回到之前的行为；盘上已有条目不动，
  下一次需要空间时由磁盘层 LRU 回收。
- **回滚代码**：`git revert 3c952ebf`（纯新增，revert 干净）。引擎不需要任何操作。
- `probe: true` 只用于诊断，不要留在生产配置里。

## 9. 代码地图

| 文件 | 内容 |
|---|---|
| `serve/oneshot_rules.py` | 类型常量 `RULE_TYPES`（L55）、`ACTIONS`（L59）、`ALLOWED_TOP`/`ALLOWED_RULE`（L70-71）、`MATCHES`（L85）、`HEURISTIC_KEYS`（L88）、`Rule`/`Markers`（L106/L120）、`parse()`（L135）、`_parse_rule()`（L182）、`prepare_text_rules()`（L279）、`system_text()`（L311）、`user_text()`（L346）、`field_value()`（L396）、`request_kind()`（L415）、`match()`（L438）、`decision()`（L498）、`describe()`（L511） |
| `serve/server.py` | `mark_one_shot()`（L1202）、`generate()` 里的调用点（L1959）、`sampling_keys()` 的 `ckpt=0` 分支（L1284）、`_oneshot_encoder()`、`main()` 里的解析与装配（解析失败 → `SystemExit`，打印横幅，把规则集挂到每个引擎上） |
| `serve/test_oneshot_rules.py` | `RealSamples`（真实请求体正反例）、`DeclaredIdentities`、`DshCompaction`、`TheDecision`、`ConfigContract`、`TextAnchorPreparation`、`SystemTextExtraction`、`StartupLog` |
| `src/program/generate.cpp` | **未改动**：`ckpt=0` 语义（L11013）、tail 检查点（L11868） |

> 本文档取代已删除的实验记录（`EXPERIMENT-oneshot-cache-skip.md`、`HANDOFF-oneshot-cache-skip.md`；
> 原文备份在仓库外的 `strata-experiment-records-20261016.zip`）。
> 另注：本分支早先那条按**长度**拦截的 `--conversation-cache-max-tokens` 已随代码一并删除（判据从"长度"
> 换成"身份"），职责由 `ds-compaction` 承接；它在 `DEV-conversation-cache-delta.md` 里的记载也已同步清理。
