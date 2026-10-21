# DEV — conversation-cache-delta：分支特有功能的开发与维护文档

> 用途：维护和了解本分支相对主分支多出来的功能（磁盘专用对话缓存 + 差分链 + 磁盘层管理）。
> 整理自内部文档 `DELTA-1480.md`、`REPORT-1480.md`、`PROVENANCE.md`、`HANDOFF-1480.md`
> 等（这些文件之后会被删除，本文档是长期保留版）。
> 对应代码：fork `fqqdc/Strata-conversation-cache-delta`，分支 `conversation-cache-delta`，
> head `8a720695`（= #1480 + 差分 + 管理 + 合并上游 main v0.1.44.1）。
> 实测机：RTX 5070 Ti 16G（sm_120）、Ryzen 9 9950X、64G RAM、Windows 11、CUDA 13.3；
> 引擎二进制 `engine\strata.exe`（v0.1.44.1 + delta，构建自同一份代码）。

## 1. 相对主分支多了什么

| 层 | 内容 | 来源 / 状态 |
|---|---|---|
| #1271 | 溢出目录：RAM 缓存淘汰的会话写盘、侧车匹配、重启重索引 | 依赖 PR（OPEN） |
| #1269 | 流式恢复：读回会话文件时不把整份 K/V 放进 RAM | 依赖 PR（OPEN） |
| #1480 | 磁盘专用缓存：`--conversation-cache-disk-only` 等（见 §2） | 基线 PR（OPEN），已合并上面两条依赖 |
| 差分链（本分支） | `.dlog/.dchain` 仅追加落盘、压实、链恢复、崩溃安全 | 本分支，opt-in |
| 磁盘层管理（本分支） | 重启清理、按最后访问时间淘汰（LRU） | 本分支，只影响 disk-only 溢出层 |

- 不加新参数时与 #1480 行为一致（v1 会话文件字节不变、不产生 `.dlog/.dchain`）。
- #1480 本身也是 opt-in：必须显式给 `--conversation-cache-spill-dir`（+ `--conversation-cache-disk-only`）。

## 2. 参数与约束

| 参数 | 作用 |
|---|---|
| `--conversation-cache-spill-dir DIR` | 开启磁盘层（会话的磁盘目录） |
| `--conversation-cache-disk-only` | 磁盘专用模式：无 RAM 预算、**不按大小拒绝**任何会话；要求同有 spill-dir、`--prompt-cache`（默认 6=开）、`--conversation-cache-disk-mib > 0`；**仅单 GPU**，不与 `--batch`/`--peer-device` 同用（否则启动日志说 "it is off"） |
| `--conversation-cache-disk-mib N` | 磁盘预算（默认 8192 MiB，`0`=关）；按「基文件 + 已提交日志」计 |
| `--conversation-cache-min-free-mib N` | 只用于**瞬态缓冲**（1 个最深 checkpoint 运行态 + 2×16 MiB 块）；被拒时文件保留不删 |
| `--conversation-cache-disk-delta` | 本分支差分落盘（需与 disk-only 同用；单独给会 warning `needs --conversation-cache-disk-only; it is off`） |

- 观测：引擎 INFO 行 `conversation_cache_disk_delta=1`；`/metrics.conversation_cache.disk_delta`。
- 当前测试配置 `strata-iq3_xxs.json` 的关键 args：`--conversation-cache-disk-only --conversation-cache-disk-delta
  --conversation-cache-spill-dir C:\strata-conv-spill-1480 --conversation-cache-disk-mib 16384 --conversation-cache-min-free-mib 2560`
  （另有 `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --max-context 262144 --kv q4_0 --vision --vram-reserve-mib 1548`）。

## 3. 运行时机制

### 3.1 基础：磁盘专用缓存（#1480 + #1269 + #1271）

- **切换走**：出会话的 K/V 从设备池**流式直写盘**（无主机镜像）→ `strata-conv-N.sess`。
- **切换回**：流式恢复两遍——第 1 遍只读 + 校验全文件（不改 GPU）；第 2 遍按 16 MiB 分块 H2D 应用，
  首块之后 fail-stop（中途出错不会用半份数据继续）。
- **匹配语义（测试时务必注意）**：磁盘命中要求「返回的 prompt 与保存时的 id 序列构成前缀」。
  重发**完全相同**的历史（切回旧会话）或**追加**一轮都命中；在相同位置改一个尾 token 会导致不命中、全量重读。
- 预算满时淘汰最老（保留前缀条目）；监控卡片（web / `api_monitor`）已适配磁盘计数。

### 3.2 差分保存（本分支）

追加的前提（任一不满足 → 全量写整份）：
1. 该会话来自盘上已有链（`disk_kv_prefix > 0` 且链路径存在、是普通会话而非前缀条目）；
2. 返回 prompt 的前缀与链一致（`entry_matches_prefix`）；
3. 能算出各 K/V part 的差分起点（`conversation_snapshot_sources_delta`）。

- **起点**：`unchanged = min(disk_kv_prefix, live tokens)`，再对齐到第一个不早于它的页/行边界（逐 part）。
  `unchanged` 的来源：盘上恢复成功 = `resume`（应用的 checkpoint 长度）；在同一 live 会话上继续 =
  `min(之前值, read_from)`；其他（冷读、rewind、checkpoint mount）= 0（回退全量）。
- **记录内容**：live 运行态 + 最深 checkpoint + token/image ids + 自起点起的 K/V。运行态为稠密数组，
  与上下文长度无关（本模型两份约各 118 MB）。
- **上限与决策**（`src/program/generate.cpp` 保存侧，`compact_at = max(64 MiB, 基文件 × 2/5)`）：
  - 记录估计溢出（状态过大）→ 日志 + 全量；
  - `已提交日志 + 估计 > compact_at` 且日志非空 → **压实**（全量写新基，旧链删除）；
  - 仅首条记录就超限（短会话）→ 全量（日志 `delta record would be ... over ... chain limit`）；
  - 否则追加 `.dlog` 记录（日志 `(X MiB, delta)`）。
- **恢复**：读 `.dchain` + sidecar 确定链 → 校验基 + 全部记录（逐记录 payload 哈希）→ 按序应用，
  后写覆盖先写 → 恢复运行态。链不可证明 → 整条链 + sidecar 删除、回来重新读 prompt（fail-closed）。

### 3.3 磁盘层管理（本分支）

- **启动清理**（`open()`，引擎重启时）删除四类垃圾，计数在启动日志 `N stale files wiped`：
  1. 写一半的临时文件（点前缀会话写临时文件 `.strata-conv-N.sess.<hex>.tmp`、`.meta.tmp`、`.dchain.tmp`）；
  2. 没有 `.meta` 的 `.sess/.dlog/.dchain` 孤儿；
  3. **异身份**缓存（model/config 指纹与本引擎不符——换模型、换 `--kv`、升级引擎后；本就不能恢复）；
  4. 坏链（清单校验失败、基大小不符、日志短于提交点）。
  其余条目全部保留并重新索引。
  ⚠ 同一 spill 目录被两个不同配置的引擎**并发共用会互删对方缓存**——目录按配置专用。
- **LRU 淘汰**：每次保存、每次**成功恢复**都打时间戳（`system_clock` 毫秒，持久化在 sidecar v2，
  跨重启保持；每次恢复多一次 ~MB 级 sidecar 原子写）。预算超限按 `last_used` 升序淘汰：
  普通会话先于前缀条目、正在恢复（pinned）的跳过、同值按原顺序；v1 旧 sidecar 的未知年龄视为最老。
  运行期在保存日志行尾追加 `evicted=N`；启动时压回预算的计数在 `E disk evictions`。

## 4. 文件与格式（维护重点）

一条链（同一 stem）：

| 文件 | 内容 |
|---|---|
| `strata-conv-N.sess` | 基文件：普通 v1 会话文件（header/payload/trailer，payload 哈希 + 结束标记），最后一次全量写 |
| `strata-conv-N.dlog` | 追加日志：若干「差分记录」（自定界、带哈希）；调用方只认已提交长度 |
| `strata-conv-N.dchain` | 清单：基文件大小 + 日志已提交长度；**原子替换写入**（提交点） |
| `strata-conv-N.meta` | sidecar：匹配元数据 + `last_used` 时间戳（v2；读兼容 v1 旧文件，视为最老） |

**差分记录布局（小端）**：magic `'STRSDLT\1'`、version、model、config、first_token，随后是与会话文件
相同的 meta 前缀（geometry、层范围、cvec、live checkpoint、checkpoints），再是每层：K/V 头 + 每部分
`u64 offset, u64 size, bytes`（`offset` = 该 part 内本记录数据的起点，页对齐），最后 `u64 载荷哈希 + 'STRSEND\1'`。
同一会话中 `offset` 之前的内容由基文件/更早的记录提供。

**写入顺序（崩溃一致性）**：追加记录 → 原子写 `.dchain`（提交点）→ 原子写 sidecar。
- 在任一步之间崩溃：`.dchain` 未更新则记录视为不存在（下次追加前先截断到提交点，覆盖孤儿字节）；
- 已提交但 sidecar 落后：恢复读到最后一条记录以它为准（匹配用 sidecar 只会给更短的 resume，仍自洽）；
- 链无法证明（基大小不符、日志短于提交长度、清单校验失败、最后记录与 sidecar 不符）→ 整条链连同
  sidecar 删除，回来时重新读 prompt。

**记账/删除**：预算、`erase`、`drop_superseded`、淘汰都会把 `.sess/.dlog/.dchain/.meta` 一起处理。

## 5. 代码地图

| 文件 | 内容 |
|---|---|
| `include/strata/core/conversation_file.hpp` | `SessionKvSource.first_bytes`；`SessionDeltaRecord`；`session_delta_write/read/apply/record_bytes` |
| `src/core/conversation_file.cpp` | 记录编解码、偏移/长度校验、载荷哈希；`RawFile::open_append/seek`（两平台） |
| `include/strata/core/conversation_spill.hpp`、`src/core/conversation_spill.cpp` | 链的索引/追加/清单/清理；`spill_delta`、`entry_info`、`entry_matches_prefix`、`disk_evictions`；启动清理；LRU `touch`/`entry_last_used` |
| `include/strata/core/conversation_snapshot.hpp`、`src/core/conversation_state.cpp` | `conversation_snapshot_sources_delta`（各 K/V part 起点 = 页/行对齐的 unchanged 边界） |
| `src/program/generate.cpp` | `--conversation-cache-disk-delta`；保存侧链跟踪与差分/全量选择（约 L8429 起）；恢复侧基+记录两遍；压实/跳过日志 |
| `serve/server.py` | `/metrics` 卡片里的 `disk_delta` 等字段（解析引擎磁盘日志） |
| `src/core/conversation_file_test.cpp`、`conversation_spill_test.cpp` | CPU 单测（记录往返/哈希拒绝、链/清理/LRU/v1 兼容等） |

## 6. 日志与观测（排障对照）

| 日志行 | 含义 |
|---|---|
| `spill dir ready (N conversations, X MiB, F stale files wiped, E disk evictions)` | 开目录：重索引结果 + 清理数 + 启动淘汰数 |
| `disk-saved N tokens (X MiB, switch) ... evicted=N` | 全量保存 |
| `disk-saved N tokens (X MiB, delta) ... evicted=N` | 差分追加 |
| `disk-restored N tokens (checkpoint, X MiB) in T ms (read pass R ms)` | 恢复成功（T 总时长、R 只读校验时长） |
| `disk save skipped (not enough RAM for the save's transient buffers)` | 瞬态内存不足被拒（**文件保留**，可稍后重试；必要时下调 `--conversation-cache-min-free-mib`） |
| `delta record would be ... over the ... chain limit; writing the whole file` | 短会话按设计回退全量 |
| `delta chain holds A of B MiB; compacting into a new file` | 压实：写新基、删旧链 |
| `delta save skipped (...)` / `delta save failed (...); writing the whole file` | 差分失败回退全量 |
| `discard unusable disk conversation (...)` | 坏文件/链被丢弃（fail-safe；之后冷读一次） |

`/metrics.conversation_cache` 关键字段：`disk_only`、`disk_delta`、`disk_files`、`disk_mib`、
`disk_saves`、`disk_restores`、`disk_saved_tokens`、`disk_written_mib`、`disk_refused`、`disk_discarded`、
`evictions`、`last_event`/`last_tokens`。

## 7. 构建、测试、复现

- **引擎**：`START-HERE.bat --setup --build --family qwen --model IQ3_XXS --context 262144 --kv q4_0 --vision gpu --gguf-dir "<GGUF 目录>" --no-start --yes`
  产物 `engine\strata.exe`（+`strata-vision.exe`）。注意别把 setup 默认写入的 `--kv-resident` 留在配置里。
- **CPU 单测**（vcvars64 + Ninja）：`cmake -S . -B build-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_GGML_DIR=<ws>\third_party\llama.cpp`，
  构建并运行 `conversation_spill_test`、`conversation_file_test`、`conversation_cache_test`（当前 129/223/4214 全绿）。
- **场景脚本**（`scenario-1480.py`、`test-delta-restart.py`、`test-lru-live.py` 等）本地已删，
  可从本地备份分支取回：`git show disk-only-cache:scenario-1480.py > scenario-1480.py`（CSV/其余脚本同理）。
  测试要按「重发相同历史 / 追加」设计（见 §3.1 匹配语义）。
- **测试服务**：`.\.venv\Scripts\python.exe serve\server.py --engine strata --config strata-iq3_xxs.json --port 4321 --open`；
  引擎日志在 `strata-iq3_xxs.log`。单 GPU：启动前先确认没有别的引擎在跑。
- **从源码重编引擎（含本期自检补丁）**：仓库里已有配置好的 Ninja 构建目录 `build/`
  （MSVC 18 + CUDA 13.3，`CMAKE_CUDA_ARCHITECTURES=120`）。只改 `src/core/*.cpp` 时：

  ```
  cmd /c "call \"D:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" >nul && cmake --build build --target strata"
  ```

  产物 `build\strata.exe`。**注意**：① `cl.exe` 不在 PATH，必须先 call vcvars64；② 别在引擎正在
  运行时重编（链接器要写同一个 exe）；③ 测补丁时用一份指向 `build\strata.exe` 的独立配置
  （如 `strata-iq3_xxs-512k-patched.json`），不要覆盖 `engine\strata.exe`。
  只改一个 .cpp 时重编 + 链接约 20 s，无新增警告（该文件仅有 3 条既有的 `strerror` C4996）。


## 8. 已知限制与后续方向

1. **记录体积由运行态主导**：245K 的一条差分约 227 MiB 中，K/V 尾巴只有 ~0.1 MiB，其余是两份稠密
   运行态（live + 最深 checkpoint，各约 118 MB，尺寸不随上下文变化）。要再降一个数量级，需要给状态
   数组做**脏页跟踪**（内核侧改动），本轮未做。
2. **短会话**单条记录即超过链上限，按设计回退全量（差分对它们基本无收益）。
3. 压实阈值固定 `max(64 MiB, 基×2/5)`，未按收益自适应；图像会话与 pinned 前缀的差分链**未实测**。
4. 每次成功恢复会重写一次 sidecar（打时间戳，~MB 级原子写）。
5. 目前仅在 Windows/CUDA（单 GPU）验证；Linux/AMD/多 GPU、长跑 TBW 未测。
6. 观察（仅一次、未复现）：一次全量写出的 496 MiB 会话文件在下一次读取时报 payload checksum
   mismatch → 按设计丢弃、冷读一次（11 s）、重写后正常；其后 8 次写读循环未复现。
7. **同一现象的第二次观察（2026-10-15，512K 上下文 4 GiB 规模，间歇、非必然）**：450K × 2
   交替测试中，动作序列 `A冷读 → B冷读 → 切回A → 切回B` 出现
   `discard unusable disk conversation (session file: payload checksum mismatch)`，
   **B 的恢复失败 → 冷读 187 s**（A 恢复正常，9.9 s）。同一序列在同一台机器上立即重跑
   （清空 spill 目录、同样 450K）**两边全部正常**（A 7.9 s、B 5.2 s）；40K 规模下两种尺寸
   （571.01 / 579.79 MiB，基础文件尺寸不同）也完整跑通、零报错。故**与文件大小无相关性、
   非确定性**，两次观察都发生在「恢复后又会话被写回盘」的写读循环中。
   - 可疑机制（未证实）：`ConversationSpillCache::read_chain()` 只校验基础文件的**字节大小**
     （`conversation_spill.cpp:467`），不校验其**内容**；若基础 `.sess` 被同尺寸重写，链条清单
     仍然通过，直到读回基础文件时其自身 payload 哈希才暴露不一致。本次失败现场中
     `strata-conv-1.sess` 的 mtime 停留在最初保存时刻（未被重写），与该猜测**不符**，故机制待定。
   - 影响：恢复失败按 fail-closed 设计丢弃条目并冷读，无数据损坏；代价是一次约 185 s 的重读
     （450K）。若需要更保守的日常配置，可去掉 `--conversation-cache-disk-delta` 回退到 #1480 行为。
8. **机制已定位到一句话（2026-10-15 晚，自检补丁 + 复现）**：已给引擎加上载荷校验和自检——
   失败瞬间用独立句柄重读同一文件并记录两侧哈希（`src/core/conversation_file.cpp`，
   `session_verify_payload_diagnostic`，挂在 `read_impl` / `read_stream_impl` /
   `apply_stream_impl` 三处失败分支；日志前缀 `strata: session file check:`）。
   用打补丁的二进制（`build\strata.exe`，未覆盖 `engine\strata.exe`）在 450K 规模跑 12 次
   A/B 交替（`loop-switch-512k.mjs`），**1 次触发**，自检输出：

   ```
   … strata-conv-1.sess: size 4401650080 (the failing read saw 4401650080, unchanged since the read),
     payload 4401650000, trailer ok, end marker ok;
     the trailer holds 0x6b108c5654498713, the failing read computed 0xe08c0ef7af3a1c84,
     a fresh read computes 0x6b108c5654498713 -> the file was REWRITTEN while the read was in flight
   ```

   - **排除**：文件被写坏、撕裂写、截断、尺寸不符、链条清单/元数据校验误判、写侧 K/V 源错误
     （大小/载荷长度/尾部哈希/结束标记全部正常，独立重读能复算出与尾部一致的哈希）。
   - **确认**：在读校验与应用两遍之间，该基础文件被换成**内容不同、字节长度相同**的另一份。
     长度相同这一点与「保存同一会话的同一状态、经临时文件原子发布」相符（`write_impl` 用
     `Publisher` + `publish`，从不原地改写）。
   - **下一步要查的点**：`generate.cpp:8432` 的保存分支（差分追加 vs 全量写）与
     `conversation_spill` 的基础文件名选择——即「切换走时保存的那个会话，其基础文件路径是否
     可能落到正在被读的那个会话上」，或全量写与读两遍窗口重叠。
   - 复核工具：`verify-session-hash.mjs`（Node 独立复刻引擎的 4 通道哈希 P1–P5/round1，
     直接校验任意 `.sess` 的大小、头部校验、载荷哈希、结束标记；对健康文件报
     `payloadHashOk: true`，对人为破坏字节的副本报 `false`）。
9. **写侧被排除，故障指向读路径（2026-10-15 晚，写侧 trace + 五轮压测）**。
   给写入路径加了 trace（`STRATA_SESSION_TRACE=1`）：`session_trace`（`conversation_file.cpp`，
   覆盖全量写的临时文件发布、差分记录追加）与 `spill_trace`（`conversation_spill.cpp`，
   带基文件大小与追加区间）。压测汇总（8 轮 A/B + 4 轮 A/B/C，共 40 次请求，仅 1 次故障）：

   | trace 事件 | 次数 | 落点 |
   |---|---:|---|
   | `full-save-publish` | 9 | 全部是 `next_stem()` 产生的**新** `.sess` |
   | `delta-append` | 33 | **全部是 `.dlog`**，没有一次落在 `.sess` |
   | `chain-commit` | 33 | 对应的 `.dchain` |

   - **由此排除**：任何写路径在读期间改写基础文件（若写过，trace 必然留下该文件名的记录，
     而诊断里那句"被重写"只是自检当时对「本次读算出的哈希 ≠ 文件现在的哈希」的解释，
     不是观测到的写事件）。`next_stem()`（`conversation_spill.cpp:289`）还会跳过任何已存在的
     `.sess/.meta/.dlog/.dchain`，全量写在设计上不可能覆盖旧文件。
   - **读路径自证的结论**：文件大小、载荷长度、尾部哈希、结束标记全部正常，同尺寸、
     最后写入时间未移动，**独立重读能复算出与尾部一致的哈希**——即"读出坏字节、重读就好"。
     这与「文件损坏/写侧 bug」不相容，只能解释为**读路径偶发读到不一致字节**。
   - **复现率**：`A/B` 两会话全冷读场景约 1/12；常规交替压测约 1/40（五轮共 40 次请求，
     1 次触发）。冷读与首次回读的组合最易触发。
   - **下一步方向**：① 给读路径也加 trace（文件名、大小、`nFileBytesRead`），确认失败那次
     实际读到的字节数与预期是否一致；② 在读校验失败时做一次**重试**（重读并重算哈希），
     若重试必成功，则读取的不确定性（驱动/固件/介质层）基本可以定案，同时把 187 s 的
     冷读惩罚降为一次约 2–3 s 的重试。
10. **读侧定位器已就位，但尚未取到一次有效分歧（2026-10-15 深夜）**。新增
    `session_locate_divergence()`（公开 API，`conversation_file.hpp`）：把失败读的哈希状态与文件
    字节按 1 MiB 分段对拍，在首个不一致的分段内逐字节二分，给出**首个分歧的载荷偏移**、
    **分歧长度**，以及两侧各自消费的字节数；失败点还会打印解析器状态（已消费 / 剩余 / 两侧哈希）。
    判定把三种情况分开：字节数不同（序列必然不同）、偏移+长度确定（真正的字节分歧）、完全一致。
    - **教训（工具链）**：期间两次"构建成功"其实**没有替换二进制**——有引擎进程仍在运行，链接器报
      `LNK1104: 无法打开文件 strata.exe`，而我的输出过滤器把这个错误滤掉了，于是我基于旧仪表的
      数据下了结论。**规矩**：重编前确认 `Get-Process strata` 为空；构建后**必须**校验二进制里的
      仪表字符串（`[regex]::Matches($txt,'<标记>')`）与 `LastWriteTime`，不要只看 ninja 是否报错。
    - **CPU 单测暂时跑不通**：`conversation_file_test`（原版源码、与本期改动无关）在本机新编译后以
      `0xC0000409` 崩溃，而 2026-10-14 构建的同源二进制 223 checks 全过。`build-tests` 的缓存指向
      VS 2022 BuildTools，其 STL 头已要求更新的编译器（STL1001）；另建 `build-tests3`（VS 18）同样
      崩溃，疑为运行时/CRT 环境问题，待查。**在单测能跑之前，定位器只有端到端压测一条验证途径。**
    - 已验证的部分：`verify-session-hash.mjs` 独立复算健康文件（`payloadHashOk: true`）与人为破坏
      副本（`false`），证明**文件级**自检方向正确；定位器本身仍待验证。
11. **仪表已重做并逐级验证通过（2026-10-15 深夜）**。旧的"分段对拍"设计被判为**错误并删除**：它要求
    调用方交出「读取**开始**时刻」的哈希状态，而实现传的是「**失败**时刻」的状态（`read_impl` 里
    `in.hash` 早已吃完整个载荷），于是它必然在第一个分段就误报分歧——**此前那条"字节数差 1 MiB"的
    结论因此作废**。教训：这类"需要调用方在正确时刻抓状态"的 API 天生容易用错。
    - **新设计**（`session_file_payload_digest()`，`conversation_file.hpp/.cpp`）：只回答一个问题
      「盘上的载荷现在哈希成什么、共多少字节」；判定权在调用方——把失败读自己算出的摘要与它比较。
      两个值各自独立计算，**没有状态对齐问题**。失败路径只读文件一遍。
    - **验证阶梯（先轻后重，全部通过）**：
      1. **秒级自测** `tools/verify-locate.cpp`（10 checks，exit 0）：盘上载荷的摘要必须等于文件自己
         trailer 里的摘要；翻转 1 字节必须改变摘要；超长载荷必须被拒。构建：
         `cl /nologo /EHsc /std:c++20 /O2 /I include tools/verify-locate.cpp src/core/conversation_file.cpp /Fe:verify-locate.exe`
      2. **受控端到端触发**：停引擎 → 翻转某个 `.sess` 载荷里的 1 字节（尺寸与 trailer 都不动，用
         `verify-session-hash.mjs` 确认 `payloadHashOk: false`）→ 重启 → 用 `trigger-one.mjs` 发
         A/B/A（切换才会写盘）。日志给出可信判定：读消费了全部载荷、trailer 与盘上摘要一致、
         **失败读的摘要也等于盘上摘要** → "读到的不是这些字节"；随后 fail-closed 丢弃并冷读。
      3. **低强度 A/B**（40K token，4 次请求）：2 次冷读 + 2 次盘上恢复（1.2 s / 0.9 s），无崩溃、
         无诊断触发、`evicted=0`。
    - **仍未知**：真实偶发故障的根因未定（只验证了工具对**植入**损坏的判定正确）。诊断能证明
      "读到的字节 ≠ 盘上的字节"，但**无法重建失败读当时读到了什么**；要拿到那一侧，需要在读取路径
      加一个可开关的转储。
    - **复现配方**（写盘与选条目的机制，踩过的坑）：磁盘保存只在**切换会话**时发生——同一条会话连发
      两次只会命中 **RAM** 缓存，盘上不会出现文件；必须 A→B→A 交替才会产生 `disk-saved`，并且
      发起方要**复用完全相同的 prompt**（`trigger-one.mjs` 不含 nonce，因此可重复），否则前缀匹配
      会落到别的条目上、构造好的损坏文件读不到。
    - **引擎启动会清掉上一次的缓存条目**（多次实测：预先构造好的损坏文件在重启后消失，日志报
      `spill dir ready (0 conversations, …)`）。所以"造缓存 → 植入损坏 → 重启读取"必须一步完成：
      用 `tools/controlled-segment-test.ps1`（`-Segment N`），它把三步串起来并**只读本次运行新增的
      日志**（引擎日志跨轮累积，否则会读到旧结论）。
12. **探针的语义修正与受控测试的定性结论（2026-10-15 深夜）**。
    - **最初的探针设计是错的**：它把 `In::raw()` 收到的字节当成"载荷的前 N MiB"。实际上载荷里的
      K/V 大块是**从设备池直接流进文件**的（`out.source(...)`，见 `put_payload`），解析器的哈希看到
      的是一串**跳过这些区段**的字节，与文件里的"前 64 MiB"根本不是同一串——逐段比较必然全错
      （这就是"总是报 segment 0"的原因）。
    - **改成同语义比较**：`session_probe_verify_against_file()` 用**同一个解析器**重走文件并在同样的
      边界取摘要，两边才可比；"读文件的前 N MiB"这种做法在原理上就不可能对。
    - **受控测试（在 segment 3 植入 1 字节损坏）的结论**：`the same walk of the file reproduces all 8
      recorded segment boundaries (598317096 bytes)`，读取消费的字节数相同、前 32 字节一致、盘上摘要自洽。
    - **合并得到的定性结论**：真实偶发故障是**读取路径偶发读到不一致的字节**，而不是文件损坏、写侧
      bug 或尺寸问题——所有"文件侧"的证据都是好的，唯独那一次读取算出的摘要不对。仍缺的一环是
      "失败读当时究竟读到了什么"。






## 9. 关键实测（摘要）

| 场景 | #1480：每次切换写 | 本分支：每次切换写 | 返回 |
|---|---|---|---|
| 245K × 2 交替 | 2332 MiB（0.82–0.86 s） | **227 MiB（0.13–0.16 s），约 10.3×** | `reused = N-7`、0 拒绝 |
| 121K + 50K 交替 | 1266 / 654 MiB | **226 / 225 MiB，约 5.6×** | `reused = N-7`、0 拒绝 |

- 恢复：单基/短链约 1.3–2.3 s；链带 3–4 条记录时 2.8–3.6 s（读校验覆盖基+记录；压实后回落）。
- 链跨引擎重启可恢复（重启后重发同一 prompt：`reused = 244940`）；压实实测（`908/934 MiB` → 新基）。
- 管理实测：重启保留全部 14 个真实缓存、清掉植入的假临时文件；恢复 P1 后缩预算重启 → 淘汰 P2（最久未用）。
- 单测 223/129/4214 全绿；合并 v0.1.44.1 后四档场景（5K/121K/225K/245K）复跑全绿、0 拒绝。

## 10. 回滚

- 从配置里删掉 `--conversation-cache-disk-delta` 并重启 → 回到 #1480 行为（盘上已有 `.dlog/.dchain`
  在未删除时被忽略）；删除 spill 目录内文件可彻底清缓存。
- 会话文件 v1 格式未动；sidecar 写 v2、读兼容 v1。**降级到只认 v1 的旧二进制前先清缓存**
  （v2 sidecar 会被旧读取器当作坏文件清掉）。
- 管理增强（启动清理、LRU）是本 fork 对 disk-only 溢出层的行为，默认生效、无参数开关。
