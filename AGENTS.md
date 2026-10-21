# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5, Unsloth, Gyro and NVFP4
variants, and Qwen3.6-35B-A3B) on a normal PC: one NVIDIA or AMD graphics card plus system RAM on Windows or Linux,
or an Apple Silicon Mac. Version 0.2.2, engine 0.1.47. It has three parts:

- **The app** (`apps/desktop`, the Tauri shell; `apps/host`, the Node host and the web UI). It is for people who do
  not want a terminal. The installer has the engine inside.
- **The Terminal** (`strata.py`, `strata_cli/`, `strata_control/`): the `strata` command. It sets up a model, serves
  it and watches it. Agents use this part.
- **The engine and the server** (`src/`, `include/`, `metal/`, `sycl/` for the C++/CUDA/HIP/SYCL/Metal engine;
  `serve/` for the Python server with an OpenAI- and Anthropic-compatible API and a web page). `setup.py`, started by
  `START-HERE.bat` or `setup.sh`, is the older one-click installer from source.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For an agent that has a shell, the short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh                  # Linux, Mac
   irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
   ```

   On Windows, open a new terminal afterwards so that `strata` is on the PATH.
2. Check what would happen. Nothing is changed:

   ```sh
   strata launch --plan --model q2_0
   ```

3. Set up and serve without questions. It downloads the engine for the card and the model (tens of GB), starts the
   server, prints the addresses and returns. The server keeps running.

   ```sh
   strata launch --model q2_0 --yes --no-tui
   ```

   Useful flags (all of them: `strata launch --help`): `--family`, `--model-path` (files the user already has),
   `--data-dir`, `--context 32k|64k|128k`, `--vision off|on|gpu|cpu`, `--parallel N`, `--gpu 0,1`,
   `--access local|lan|cloudflare|tailscale`, `--port`, `--api-key-env NAME`, `--new-key`, `--save-preset NAME`,
   `--output json`. `strata launch --status` shows what is serving, `strata launch --stop` stops it, and
   `strata launch --show-key` prints the API key.
4. Connect the user's apps: `strata connect` lists them, `strata connect <app> --check` sends one prompt and one tool
   call the way that app does.

Keep the server on `127.0.0.1` unless the user asks for more. Beyond it an API key is recommended: `--access lan` makes one for you, and `--no-api-key` serves without one (a warning, never a refusal; on a public tunnel the warning is stronger). The old route (clone
the repository and run `START-HERE.bat --yes` or `./setup.sh --yes`) still works and is the fallback when the
Terminal cannot run. Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The app,
  the packages and the update feed: [packaging/README.md](packaging/README.md) and [packaging/UPDATES.md](packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `tests/terminal` (`python -m pytest tests/terminal`). The app's UI
  tests run with `npm test` in `apps/host`. Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

## 本机安装记录（2026-10-18 重写，以 `strata-iq3_xxs.json` 为准，仅此 PC）

> 旧记录里的 0.1.41、256K、`--conversation-cache-mib 8192`、2560 MiB 门槛等已过时，统一按当前生产配置重写。
> 上游主线那套安装（`Strata-main`）的记录仍留在它自己的 AGENTS.md 里，两边不要混用 config/engine。

### 硬件与目录

- RTX 5070 Ti 16 GB（sm_120，驱动 616.56）、Ryzen 9 9950X（AVX-512）、62 GB RAM、Windows 11。
- 本 fork 生产目录：`F:\Users\Emanon\Desktop\opencode workspace\Strata-1480`（分支 `conversation-cache-delta`，
  合并上游 v0.1.45 的提交 `2a32c037`）；数据目录 `...\opencode workspace\Strata-data`（`packs\iq3_xxs`、`mtp\rt`）。
- GGUF 在 `D:\Program Files\LM Studio Files\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF\`：配置直接引用，**不要移动/删除**。

### 当前生产配置（`strata-iq3_xxs.json`）

- 模型：Qwen3.8-Flash-Next **IQ3_XXS**（GSQ-RCO，两个分片已核对 SHA-256）；`--native`/`--ple-gguf` 指向上面的 GGUF 目录。
- 上下文 **512K**：`--max-context 524288` + `--rope-scaling yarn --rope-scale 2`（超过训练长度 262144 的部分靠 yarn）。
- K/V：`--kv q4_0`，`--kv-resident 32768`——每个 QSA 层 32K cell 常驻 VRAM，其余流式放内存
  （启动日志：`32768 of 524288 cells per QSA layer in VRAM, the K/V in 3.38 GiB of pinned RAM`）。
- 图片：`--vision`（GPU 编码器，`vision.max_tokens 1024`，复用本地 `mmproj-Qwen3.8-Flash-Next-BF16.gguf`，无需下载）。
- 性能：`--expert-cache auto`（本机约 3512 槽）、`--prefill auto`、`--spec 4 --spec-min-p 0.5`、
  `--mtp Strata-data\mtp\rt`、`--expert-profile data\expert-profile.bin`。
- 显存：`--vram-reserve-mib 1548`（手动调过；setup 默认写 700——旧记录实测 256K 时加载后只剩 ~176 MiB 并报 LOW，512K 只会更紧）。
  512K 加载完实测空闲 ~1035 MiB（日志 `1035 MiB of VRAM free with everything loaded`）。
- 对话缓存（本 fork 的核心差异；**disk-only，没有 `--conversation-cache-mib`**）：
  `--conversation-cache-disk-only --conversation-cache-disk-delta`
  `--conversation-cache-spill-dir D:\strata-conv-disk --conversation-cache-disk-mib 32768 --conversation-cache-min-free-mib 512`
  - 切走的会话流式写盘、回来的会话流式读回；RAM 里不留 parked 快照。
  - `--conversation-cache-disk-delta`：只追加变化的 K/V（base + `.dlog` + `.dchain`，超过阈值自动压实）。
  - 32 GiB 预算按 sidecar 里的 `last_used` 做 LRU 淘汰；512 MiB 是**读回时**的物理内存下限（不够就跳过这次命中、保留文件）。
  - 每个会话的 root 作为 pinned 前缀条目常驻；`pin=N` 的兄弟请求按 pinned 前缀计数。
  - 实测（`DEV-conversation-cache-delta.md` §9，本机）：245K 交替每次切换 2332 → **227 MiB（约 10.3×）**；
    121K+50K 交替 1266/654 → **226/225 MiB（约 5.6×）**；恢复单基/短链 1.3–2.3 s，3–4 条记录的链 2.8–3.6 s。
- 观测：`"api_monitor": true`（`/api-monitor` 保留最近 100 条请求）；引擎与 serve 日志都写 `strata-iq3_xxs.log`。
- 一次性请求：`"oneshot_markers"` 3 条规则（DSH 标题、`strata_checkpoint: false`、DSH compaction）→ 发 `ckpt=0`，
  本就不会被复用的请求不写盘。

### 引擎与构建

- `engine\strata.exe` = **0.1.45**（合并上游 v0.1.45 后的 `build\strata.exe`，SHA256 `3B5C9A37…`）；
  `engine\BUILD.json` 的 `version` 也是 0.1.45。`engine\strata-vision.exe` 仍是 0.1.44.1 构建——合并没有动
  `tools/vision`，`vision_src` 未变，可以继续用。
- 部署前备份：`engine\backup-0.1.44.1\`（strata.exe / strata-vision.exe / BUILD.json）。
- 从源码重编（Ninja + MSVC 18 + CUDA 13.3，`CMAKE_CUDA_ARCHITECTURES=120`）：
  `cmd /c "call \"D:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat\" >nul && cmake --build build --target strata"`
  产物 `build\strata.exe`。**不要**跑全量 `cmake --build build`：上游 `src/kernels/hc_persist_parity.cpp` 在 Windows
  用了 POSIX `setenv/unsetenv`，会报 C3861。
- 单测：`build-tests`（CUDA OFF、`STRATA_BUILD_CONVERSATION_TESTS=ON`）：`conversation_cache_test` 4772 项、
  `conversation_file_test` 223、`conversation_spill_test` 129、`conversation_memory_test` 25、
  `conversation_split_failure_test` 全过（0.1.45 合并后本机实测）。
- 部署：确认没有 `strata.exe` 在跑 → `Copy-Item build\strata.exe engine\strata.exe -Force` → 更新
  `engine\BUILD.json` 的 `version`/`src` → 重启 server。引擎在跑时不能重编（链接器要写同一个 exe）。

### 启动/停止

- `run-iq3_xxs.bat` → `0.0.0.0:4321` + bat 内写死的 API key，`--open`；只本机用 `run-iq3_xxs-local.bat` →
  `127.0.0.1:4321` 无 key。config 里的 `"port": 8080` 会被 bat 的 `--port 4321` 覆盖。
- 关窗口即停；后台进程：`Stop-Process (Get-NetTCPConnection -LocalPort 4321 -State Listen).OwningProcess`。
- **改 config 必须重启整个 server**（引擎参数在 server 启动时固定；unload/load 和引擎崩溃自动重启都复用旧参数）。

### 版本与缓存兼容（重要）

- 会话文件身份包含 `STRATA_VERSION`：换引擎版本 = 旧 spill 目录按异身份作废。0.1.45 首次启动实测
  `spill dir ready (0 conversations, 0 MiB, 13 stale files wiped, 0 disk evictions)`，约 16 GB 旧缓存被清，冷启动一次。
- 回滚旧引擎不会找回旧缓存（已被清）；回滚 = 换回 `engine\strata.exe` + `BUILD.json` + 重启。
- sidecar 仍是 v2；彻底清缓存 = 停服后删 `D:\strata-conv-disk\*`。

### 更新流程

- `UPDATE.bat` / `setup.py --update` 先 `git pull --ff-only`，而 AGENTS.md 常驻本地修改 → 会被拒：
  先 `git stash push`，更新完 `git stash pop`。
- `--setup` 重跑会重写 config（丢掉 `--vram-reserve-mib 1548`、`--kv-resident 32768` 和全部 disk-only 参数），必须按上面的配置补回。
  改设置必须带 `--gguf-dir "D:\Program Files\LM Studio Files\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
  --family qwen --model IQ3_XXS`，否则会去 `Strata-data\models` 触发 ~70 GB 重新下载。
- 重建默认配置的完整命令（512K）：`START-HERE.bat --setup --family qwen --model IQ3_XXS --context 524288 --kv q4_0
  --vision gpu --gguf-dir "D:\Program Files\LM Studio Files\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF" --no-start --yes`
  （>256K 时 setup 会自己加 yarn；跑完把「当前生产配置」里 setup 不写的项补回，再重启。）
- 网络：直连 GitHub/Hugging Face 很慢（几十 KB/s），代理 `127.0.0.1:7897` 正常；MTP 大文件走代理偶发
  SSL `UNEXPECTED_EOF`，重跑同一条 setup 命令即断点续传，不要删文件。
- 同机 LM Studio 的 `unsloth\Qwen3.8-Flash-Next-GGUF`（UD-IQ3_XXS）**不可用**：IQ3_XXS 只有 ISTA-DASLab 的 GSQ-RCO 文件。

### Qwen3.6-35B-A3B（2026-10-18 新增，独立引擎 `engine-q36\`，端口 4322）

- 模型：`D:\Program Files\LM Studio Files\models\unsloth\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`，
  22,853,663,008 B，SHA-256 `55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1` —— 与 `setup.py` 的
  `QWEN36_FILES` 逐字节一致（架构 `qwen35moe`，40 层 + 1 个 MTP 块，训练长度 262,144）。**文件本身可用，就是官方 pin 的那份。**
- **`engine\strata.exe` 跑不了它**：那份是旧 cache 编的（`STRATA_Q6K_EXPERTS=OFF`、`STRATA_MMQ_KQUANTS=OFF`），加载到
  `blk.34` 直接拒绝：`blk.34.experts are Q4_K/Q6_K ... build with -DSTRATA_Q6K_EXPERTS=ON`。用 `build-q36\` 重编：
  `-DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_NATIVE_EXPERTS=ON -DCMAKE_CUDA_ARCHITECTURES=120
  -DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3`（**漏掉 CUDAToolkit_ROOT 会链到 v11.8 的
  cudart，报 LNK2005**），产物 `strata.exe` + **`strata_kq.dll`（必须同目录）** 放 `engine-q36\`；`engine\` 未动。
- 配置 `strata-qwen36-ud-q4_k_xl.json`（`--max-context 65536 --prefill auto --mtp 2 --vision`，`arch: qwen35moe`），
  `run-qwen36-local.bat` → 127.0.0.1:4322。tokenizer 从 GGUF 生成到 `Strata-data\packs\qwen36-ud-q4_k_xl\tokenizer`：
  `STRATA_GGUF_PY=third_party\llama.cpp\gguf-py .venv\Scripts\python.exe tools\strata_tokenizer.py --gguf <gguf> --out <pack>`。
- 图片用 `mmproj-F16.gguf`（899,283,584 B，SHA `71f3cbc1…` 与 setup 的 pin 一致，2026-10-18 补下）；F32 版也能用
  （已验证读图），只是多占 ~1 GiB 显存。setup 的 `--gguf-dir` 还要求 `.done` 标记，所以这套是手写配置。
- **上下文阶梯（本机实测，vision 开、F16、chunk 2048）**：32K → 剩 6.22 GiB、**59/256 槽**；64K → 剩 2.87 GiB、**14 槽**；
  96K → 剩 0、**0 槽**（专家全走 CPU，仍能起）；**128K / 256K 起不来**（`not enough VRAM left for the hybrid experts'
  buffers and slots (1256 MiB needed, 0 MiB free …)`）。32K→64K 实测吃掉 3.35 GiB 显存（KV 理论只有 1.41 GiB，其余是
  随上下文增长的缓冲）。解码：greedy+`--mtp 2` 在 32K 和 64K 都是 **101.6 tok/s**（152/160 drafts）；采样（无草稿）
  32K 53–58、64K **45.9**、96K 47.3 tok/s。
- **没有 KV 量化、没有 KV 流式**：q36 这条路径的 K/V 是 fp32 设备内存（`float* kc/vc`，45 KB/token），
  `--kv q4_0` / `--kv-resident` / `--rope-scaling` 只属于 Flash-Next 引擎路径，写进 q36 配置会被 serve 模式忽略
  （打印 `ignoring …`）。所以 256K 在这张 16 GB 卡上不可能：光 KV 就 11 GiB。要 256K 得等这条路径支持 KV 量化/流式，
  或换 24–32 GB 卡。
- 本机实测：4.40 GiB dense 上 GPU（1.4 s）、18.75 GiB 专家 pinned 在内存；**pp2048 2,450–2,874 tok/s**；
  图片 260 image tokens，编码约 0.1–0.5 s（F16 GPU 编码器）。
- 内存：引擎 private ~36 GB / working set ~44 GB（+ vision 进程），跑起来时可用物理内存 ≈ 0 → **不能与 Flash-Next
  同时跑**（Flash-Next 自己也要 39.97 GiB expert arena + 3.38 GiB pinned KV）。
- 其它限制：没有 yarn（上限就是训练长度 262,144）；本 fork 的 disk-only 会话缓存只属于 Flash-Next 引擎路径，
  q36 的 serve 参数解析不认 `--conversation-cache-*`，会话只常驻引擎 session；`"parallel"`/`--batch` 可用（这里默认 1）。
- 想统一成一个引擎：按 setup 官方开关重编 `engine\strata.exe`（同样两个开关 + 复制 `strata_kq.dll`），两边都能跑；
  代价是 K-quant MMQ 实例多占 ~36 MiB VRAM（CMakeLists 注释里的实测）。
