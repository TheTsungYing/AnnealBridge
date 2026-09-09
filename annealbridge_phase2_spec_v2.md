# AnnealBridge — Phase 2 開發規格 (v2)

> v2 變更摘要（相對於 v1）
> - 對齊 Phase 1 v2：統一 `errors: list[SolveError]`，沿用 `infeasibility_proven`、`is_exhaustive`、`penalty_multiplier`
> - `ExecutionPolicy` 移入 orchestration 並注入 `OptimizationService`；`ServerSettings` 只在 composition root（CLI / MCP）使用，core 不讀 env
> - 明訂檢查順序：validate → backend/policy → compile → resource limits → solve
> - 新增 remote 專屬 error codes（`EMBEDDING_FAILED`、`REMOTE_AUTH_FAILED`、`REMOTE_TIMEOUT` 等）
> - Leap Hybrid 單一 sample、`min_time_limit` 的處理方式
> - Capabilities 檢查 D-Wave availability 不得產生網路請求
> - Streamable HTTP 預設綁 `127.0.0.1`；新增 `max_concurrent_solves`
> - Problem Validator 新增 warnings 與 `estimated_compiled_variables`
> - 新增 token 外洩掃描測試
> - MCP SDK pin `mcp>=2,<3`，整合測試改用 v2 `Client(mcp)` in-memory
> - Import package 統一改名 `annealbridge`

---

## 1. Phase 2 目標

Phase 1 已建立 IR、Schema、BQM Compiler、Validator、Penalty Strategy、SA / Exact backend、OptimizationService、CLI、Top-K ranking。

Phase 2 目標：

> **把 Optimization Core 封裝成 MCP Server，讓 AI Agent 透過標準 MCP Tool 呼叫，並加入 D-Wave 遠端 annealing backend。**

```text
Claude / Codex / AI Agent
          │ MCP
          ▼
┌────────────────────────────────┐
│        AnnealBridge MCP        │
│  get_optimization_capabilities │
│  validate_optimization_problem │
│  solve_optimization            │
└───────────────┬────────────────┘
                ▼
       OptimizationService  ◄── ExecutionPolicy
                │
          Solver Registry
    ┌───────────┼────────────────┐
    ▼           ▼                ▼
  Exact         SA        Remote (D-Wave)
                          ┌─────┴─────┐
                          ▼           ▼
                     Leap Hybrid     QPU
```

MCP 層不得重新實作 compilation、validation、penalty、ranking、retry、objective evaluation。MCP 只是 **interface adapter**。

---

## 2. Phase 2 Scope

1. MCP Python SDK v2 integration（`mcp>=2,<3`）
2. MCP Server：三個 tools，structured input / output
3. stdio transport（預設）+ Streamable HTTP（開發 / 未來部署）
4. MCP in-memory integration tests（`Client(mcp)`）
5. `ExecutionPolicy` + `ServerSettings`
6. `SolverRegistry`
7. `DWaveQPUBackend`、`LeapHybridBQMBackend`
8. Remote credential safety、remote retry policy、resource limits
9. `SolverExecutionMetadata` + whitelist sanitization
10. Problem Validator warnings
11. Optional（opt-in）D-Wave integration tests
12. CLI 移至 `interfaces/cli`，新增 `annealbridge capabilities`
13. README：MCP setup、D-Wave setup、security
14. Phase 1 regression 全綠

---

## 3. Phase 2 明確不做

LLM API、自然語言 → OptimizationProblem、prompt parsing、RAG、database、Web UI、job queue、background jobs、result persistence、authentication system、multi-tenant、CQM、integer / real variables、OR-Tools、Fujitsu DA、自動 solver 推薦、自動 cost estimation、MCP prompts、MCP elicitation、production SaaS deployment。

自然語言理解仍由 **MCP Host 的 LLM** 負責。

---

## 4. Architecture 與 Dependency Direction

```text
models ← validation ← compiler ← solvers ← orchestration ← interfaces (cli / mcp)
                                                  ▲
                                             config (composition root only)
```

規則：

- `orchestration` 定義 `ExecutionPolicy`（純 Pydantic model），不讀 env。
- `config/settings.py` 用 pydantic-settings 讀 env 並建構 `ExecutionPolicy`；**只有** `interfaces/*` 可 import `config`。
- Core（models / validation / compiler / solvers / penalty / orchestration）不得 import `mcp`、`dwave.cloud`、`config`、`interfaces`。
- `solvers/dwave_qpu.py` 與 `solvers/leap_hybrid_bqm.py` 可 import `dwave.system`，但必須 lazy import（在 `solve()` 內或 module 層 try/except），使 `pip install annealbridge` 不裝 D-Wave 也能 import registry。

用一個 import-linter 或簡單 pytest（掃 `annealbridge/{models,validation,compiler,penalty,orchestration}` 的 import）強制執行。

---

## 5. Package 命名

Step 1 將 import package 由 `optimizer` 改名為 `annealbridge`，發行名同為 `annealbridge`。理由：MCP 公開後 entry point、host 設定檔、README 全都會寫死名稱，之後再改成本高。改名為機械性 sed + 跑 Phase 1 tests。

---

## 6. Dependencies

```toml
[project]
dependencies = [
    "pydantic>=2",
    "pydantic-settings>=2",
    "dimod",
    "dwave-samplers",
]

[project.optional-dependencies]
mcp   = ["mcp>=2,<3"]
dwave = ["dwave-system"]
all   = ["mcp>=2,<3", "dwave-system"]
dev   = ["pytest", "pytest-anyio", "mcp[cli]>=2,<3"]   # cli extra 提供 MCP Inspector

[project.scripts]
annealbridge     = "annealbridge.interfaces.cli.main:main"
annealbridge-mcp = "annealbridge.interfaces.mcp.server:main"
```

注意 mcp v2 會帶入 `opentelemetry-api`、`httpx2` 等新依賴，這是 SDK 的 hard dependency，接受即可；但 **不得**把 problem payload 放進 span attributes（見 §31 logging）。

---

## 7. 目錄調整

```text
src/annealbridge/
├── models/            (Phase 1)
├── compiler/          (Phase 1)
├── validation/        (Phase 1 + warnings)
├── penalty/           (Phase 1)
├── solvers/
│   ├── base.py        (+ SolverCapabilities)
│   ├── exact.py
│   ├── simulated_annealing.py
│   ├── dwave_qpu.py           (new)
│   ├── leap_hybrid_bqm.py     (new)
│   ├── metadata.py            (new: sanitizer)
│   └── registry.py            (new)
├── orchestration/
│   ├── optimizer.py
│   └── policy.py              (new: ExecutionPolicy)
├── config/
│   └── settings.py            (new: ServerSettings → ExecutionPolicy)
└── interfaces/
    ├── cli/
    │   └── main.py            (moved from cli.py)
    └── mcp/
        ├── server.py
        ├── tools.py
        └── models.py          (capabilities / validation result models)

tests/
├── unit/                      (Phase 1 + policy / registry / sanitizer / settings)
├── scenarios/                 (Phase 1)
├── architecture/
│   └── test_import_boundaries.py
├── mcp/
│   ├── test_tools_list.py
│   ├── test_capabilities.py
│   ├── test_validate.py
│   ├── test_solve_exact.py
│   ├── test_solve_sa.py
│   └── test_stdio_entrypoint.py
├── remote_mock/
│   ├── test_dwave_qpu_mock.py
│   ├── test_leap_hybrid_mock.py
│   └── test_credential_leak.py
└── remote_live/               (opt-in, @pytest.mark.remote)
    ├── test_dwave_qpu_live.py
    └── test_leap_hybrid_live.py
```

---

## 8. ExecutionPolicy（orchestration/policy.py）

```python
class ExecutionPolicy(BaseModel):
    allow_remote: bool = False
    allow_remote_retries: bool = False
    exact_max_variables: int = 24          # compiled（含 slack）
    max_qpu_reads: int = 1000
    max_qpu_annealing_time_us: float = 2000.0
    max_remote_time_seconds: int = 300     # hybrid time_limit 上限
    max_concurrent_solves: int = 4
    enabled_backends: set[str] | None = None   # None = registry 全部
```

`OptimizationService.__init__(..., policy: ExecutionPolicy = ExecutionPolicy())`。Phase 1 v2 在 `ExactSolverBackend` 建構子的上限 **移除**，改由 policy 統一管理。

> 2026-09-09 review 修正：新增 `max_local_reads` / `max_sweeps` / `max_local_retries` / `max_remote_retries` / `max_top_k` 五個上限欄位，見 3a spec §11.4。

---

## 9. ServerSettings（config/settings.py）

```python
class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANNEALBRIDGE_")
    allow_remote: bool = False
    allow_remote_retries: bool = False
    exact_max_variables: int = 24
    max_qpu_reads: int = 1000
    max_qpu_annealing_time_us: float = 2000.0
    max_remote_time_seconds: int = 300
    max_concurrent_solves: int = 4
    http_host: str = "127.0.0.1"
    http_port: int = 8000

    def to_policy(self) -> ExecutionPolicy: ...
```

Env：`ANNEALBRIDGE_ALLOW_REMOTE`、`ANNEALBRIDGE_ALLOW_REMOTE_RETRIES`、`ANNEALBRIDGE_EXACT_MAX_VARIABLES`、`ANNEALBRIDGE_MAX_QPU_READS`、`ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US`、`ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS`、`ANNEALBRIDGE_MAX_CONCURRENT_SOLVES`、`ANNEALBRIDGE_HTTP_HOST`、`ANNEALBRIDGE_HTTP_PORT`。

D-Wave credential **不在** ServerSettings 內，交給 Ocean 原生 config（§18）。

---

## 10. Solver Backend Interface 擴充

```python
class SolverCapabilities(BaseModel):
    name: str
    remote: bool
    heuristic: bool
    exhaustive: bool
    supports_seed: bool
    supports_num_reads: bool
    supports_time_limit: bool
    supported_model_types: list[str]        # ["bqm"]
    returns_multiple_samples: bool          # leap_hybrid_bqm = False
    description: str

class SolverBackend(Protocol):
    @property
    def capabilities(self) -> SolverCapabilities: ...
    def is_available(self) -> tuple[bool, str | None]:   # (available, reason_if_not)
        """Must not perform network I/O."""
    def solve(self, compiled: CompiledProblem, prefs: SolverPreferences) -> RawSolverResult: ...
```

Phase 1 的 `name` / `is_exhaustive` 由 `capabilities.name` / `capabilities.exhaustive` 取代（保留 property 作 alias 也可）。

`is_available()` 規則：

- exact / SA：永遠 `(True, None)`
- D-Wave backends：檢查 `dwave.system` 可 import，且 `dwave.cloud.config.load_config()` 能解出非空 token（不呼叫 `Client.get_solvers()`、不建立連線）。回傳 reason 只能是分類字串如 `"dwave-system not installed"`、`"D-Wave credentials not configured"`，**不含任何 config 值**。

---

## 11. SolverRegistry

```python
class SolverRegistry:
    def __init__(self, backends: dict[str, SolverBackend]): ...
    def get(self, name: str) -> SolverBackend: ...        # KeyError → UNKNOWN_BACKEND
    def names(self) -> list[str]: ...
    @classmethod
    def default(cls) -> "SolverRegistry": ...            # 註冊四個；remote 的以 lazy import 建立
```

不要建立 factory-of-factory。`RawSolverResult` 加 `metadata: SolverExecutionMetadata | None`。

---

## 12. SolverPreferences 擴充（向後相容）

```python
class DWaveQPUOptions(BaseModel):
    annealing_time_us: float | None = None
    chain_strength: float | None = None      # None → Ocean 預設 (uniform_torque_compensation)
    auto_scale: bool = True

class LeapHybridBQMOptions(BaseModel):
    time_limit_seconds: float | None = None  # None → sampler 最小值

class SolverPreferences(BaseModel):
    backend: Literal["simulated_annealing", "exact", "dwave_qpu", "leap_hybrid_bqm"] = "simulated_annealing"
    num_reads: int = 100
    num_sweeps: int = 1000
    seed: int | None = None
    top_k: int = 5
    max_retries: int = 3
    penalty_multiplier: float = 2.0
    dwave_qpu: DWaveQPUOptions | None = None
    leap_hybrid_bqm: LeapHybridBQMOptions | None = None
```

禁止在 OptimizationProblem 中出現 token / endpoint / profile / region 等任何 credential 或連線設定；validator 不需檢查（schema 本來就沒這些欄位），但 README 明寫。

Phase 1 的 knapsack.json / assignment.json 必須不改一字仍可 parse。

---

## 13. SolveResult 擴充

```python
class SolveError(BaseModel):
    code: str
    path: str | None = None
    message: str
    retryable: bool = False
    recommended_action: str | None = None

class SolveResult(BaseModel):
    status: Literal[
        "success", "infeasible", "invalid_problem", "solver_error",
        "backend_unavailable", "resource_limit_exceeded", "configuration_error",
    ]
    backend: str | None
    objective_direction: Literal["minimize", "maximize"] | None
    solutions: list[Solution]
    attempts: list[SolveAttempt]
    infeasibility_proven: bool = False
    errors: list[SolveError] = []            # 取代 Phase 1 v2 的 ProblemError；不另設 error 欄位
    warnings: list[SolveError] = []          # 例如 REMOTE_RETRIES_DISABLED
    metadata: SolverExecutionMetadata | None = None
    message: str | None = None
```

`ProblemError` 改為 `SolveError` 的 alias 以保相容（欄位為超集）。

### 13.1 Status → 條件

| status | 何時 |
|---|---|
| `invalid_problem` | Problem Validator 失敗 |
| `backend_unavailable` | backend 未知、未安裝、無 credential、policy 未允許 remote、不在 `enabled_backends` |
| `resource_limit_exceeded` | compiled 變數 > exact 上限；`num_reads` > QPU 上限；`annealing_time_us` 超限；`time_limit` 超限；並行 solve 超過上限 |
| `configuration_error` | 讀到但無法解析的 D-Wave config、無效 profile 等 |
| `solver_error` | embedding 失敗、remote timeout、remote auth 被拒、sampler exception |
| `infeasible` / `success` | 同 Phase 1 |

### 13.2 Error codes（至少）

`UNKNOWN_BACKEND`、`BACKEND_NOT_INSTALLED`、`REMOTE_DISABLED`、`REMOTE_CREDENTIALS_MISSING`、`BACKEND_DISABLED_BY_POLICY`、`EXACT_VARIABLE_LIMIT`、`QPU_READS_LIMIT`、`QPU_ANNEALING_TIME_LIMIT`、`REMOTE_TIME_LIMIT`、`CONCURRENCY_LIMIT`、`EMBEDDING_FAILED`、`REMOTE_AUTH_FAILED`、`REMOTE_TIMEOUT`、`REMOTE_SOLVER_ERROR`、`REMOTE_RETRIES_DISABLED`（warning）。

每個 code 都必須有固定的 `recommended_action` 文字，集中在一個 dict，方便 Agent 學習。例如 `EMBEDDING_FAILED` → "Problem too dense or too large for QPU embedding; reduce variables/constraints, or use leap_hybrid_bqm or simulated_annealing."

---

## 14. OptimizationService 流程（Phase 2 版）

```text
1. validate problem                → invalid_problem
2. registry.get(backend)           → backend_unavailable (UNKNOWN_BACKEND)
3. policy.enabled_backends check   → backend_unavailable
4. if caps.remote and not policy.allow_remote → backend_unavailable (REMOTE_DISABLED)
5. backend.is_available()          → backend_unavailable / configuration_error
6. acquire concurrency slot（non-blocking；滿則 resource_limit_exceeded CONCURRENCY_LIMIT）
7. λ = initial_penalty
8. compile
9. resource limits（需 compiled 資訊）:
     exhaustive → num_variables <= exact_max_variables
     dwave_qpu  → num_reads <= max_qpu_reads; annealing_time_us <= max
     leap_hybrid_bqm → time_limit <= max_remote_time_seconds
   → resource_limit_exceeded（不得 clamp）
10. solve → 任何 backend exception → solver_error（含分類 code）
11. process candidates（Phase 1 §25）
12. feasible → success
13. exhaustive → infeasible (proven=True)
14. remote and not allow_remote_retries → infeasible (proven=False) + warning REMOTE_RETRIES_DISABLED
15. attempt > max_retries → infeasible
16. λ = next_penalty；回到 8
```

**2026-09-09 review F-14 修正**：步驟 9 的 exhaustive 變數上限先以
`estimate_model_variables(problem, model_type)` 在 compile 前擋（估算與編譯結果
相等是 estimates 的契約），compile 後的檢查保留為最終保證；code / message 不變。

不得 silent fallback：使用者指定 `dwave_qpu` 而不可用，就回 `backend_unavailable`，不改用 SA。

Concurrency slot 用 `threading.BoundedSemaphore`，屬 service 實例狀態；CLI 單次呼叫不受影響。

---

## 15. DWaveQPUBackend（solvers/dwave_qpu.py）

```python
from dwave.system import DWaveSampler, EmbeddingComposite   # lazy
```

- `EmbeddingComposite(DWaveSampler())`；embedding 由 Ocean 處理，呼叫端不負責 qubit mapping。
- 傳入 `num_reads`、`annealing_time`（µs）、`chain_strength`（None 則不傳，交給 Ocean 預設）、`auto_scale`。
- Chain break 使用 Ocean 預設 `majority_vote`；回傳的 logical samples 才是 `RawSolverResult.samples`。
- `seed` 不支援：`supports_seed=False`；若 problem 帶 seed，validator 產 warning `SEED_IGNORED`，不報錯。
- Exception 映射：`ValueError`/`EmbeddingError`（找不到 embedding）→ `EMBEDDING_FAILED`；`SolverAuthenticationError` → `REMOTE_AUTH_FAILED`；`RequestTimeout` → `REMOTE_TIMEOUT`；其他 → `REMOTE_SOLVER_ERROR`。Exception message 必須經 redaction（§19）再放進 SolveError。

### 15.1 Hard penalty vs chain strength

兩者完全獨立：hard penalty 是 business constraint → QUBO 係數，由 PenaltyStrategy 決定；chain strength 是 logical → physical embedding 參數，由 `DWaveQPUOptions` 或 Ocean 決定。程式碼中不得由一方推導另一方。

### 15.2 QPU 適用性 warning

Squared penalty 會讓 constraint 內所有變數兩兩耦合（clique），slack 再增加變數。Problem Validator 在 `backend == dwave_qpu` 時，若 estimated compiled variables > 150 或最大 constraint 涉及變數 > 30，產 warning `DENSE_FOR_QPU`（門檻為常數，可調）。這是 heuristic，不阻擋。

---

## 16. LeapHybridBQMBackend（solvers/leap_hybrid_bqm.py）

```python
from dwave.system import LeapHybridSampler   # lazy
```

- `time_limit`：使用者未給 → `sampler.min_time_limit(bqm)`；有給但低於最小值 → 回 `solver_error`/`REMOTE_SOLVER_ERROR`？**不**：改為在 backend 內以 `max(user, min_time_limit)` 取值並在 metadata 記錄 `effective_time_limit`；高於 policy 上限 → `resource_limit_exceeded`（service 層檢查）。
- Hybrid sampler **通常只回 1 個 sample**：`returns_multiple_samples=False`。Service 照常跑 pipeline，Top-K 自然 ≤ 1。README 與 capabilities 說明此事，避免 Agent 以為 top_k 沒作用是 bug。
- `num_reads` / `num_sweeps` / `seed` 對 hybrid 無意義：忽略並產 warning `PARAMETER_IGNORED`。

---

## 17. SolverExecutionMetadata 與 Sanitization

```python
class SolverExecutionMetadata(BaseModel):
    backend: str
    remote: bool
    solver_id: str | None = None
    logical_variables: int | None = None
    logical_interactions: int | None = None
    num_reads_requested: int | None = None
    effective_time_limit_seconds: float | None = None
    timing_us: dict[str, float] = {}               # whitelist keys only
    average_chain_break_fraction: float | None = None
    embedding_max_chain_length: int | None = None
```

`solvers/metadata.py` 提供 `sanitize_sampleset_info(info: dict, backend: str) -> SolverExecutionMetadata`：

- timing 只取 whitelist：`qpu_access_time`、`qpu_sampling_time`、`qpu_anneal_time_per_sample`、`qpu_programming_time`、`total_post_processing_time`、`run_time`、`charge_time`。
- 一律轉成 float / int / str，任何非 JSON 型別丟棄。
- **不得**直接 `metadata = sampleset.info`。
- Unit test：餵一個含 nested object、bytes、未知 key 的 fake info，斷言輸出只剩 whitelist。

---

## 18. D-Wave Credential

使用 Ocean 原生機制：D-Wave config file、`DWAVE_API_TOKEN`、`DWAVE_API_ENDPOINT`、`DWAVE_API_REGION`、`DWAVE_API_SOLVER`、`DWAVE_PROFILE`。

程式碼禁止 `DWaveSampler(token=...)`、`LeapHybridSampler(token=...)`；也不得在 ServerSettings 定義 token 欄位。AnnealBridge 不管理 token。

---

## 19. Credential Redaction

`solvers/metadata.py`（或 `redaction.py`）提供 `redact(text: str) -> str`：

- 讀取當前 Ocean config 中解析出的 token 值（若有），在任何要進入 SolveError / log / metadata 的字串中以 `***` 取代。
- 額外用 regex 遮蔽 `DEV-[A-Za-z0-9]{20,}`、`token=[^\s&]+`、`Authorization: [^\n]+` 形式。
- 所有 remote backend 的 exception → SolveError 路徑必經 `redact()`。

任何情況下 token 不得出現在：MCP result、SolveResult、log、exception message、metadata、trace、test fixture、span attributes。

---

## 20. Problem Validator 擴充：warnings 與 estimate

`ProblemValidationResult`（`interfaces/mcp/models.py` 引用，實際定義在 `validation/`）：

```python
class ProblemValidationResult(BaseModel):
    valid: bool
    errors: list[SolveError] = []
    warnings: list[SolveError] = []
    estimated_compiled_variables: int | None = None   # 原始變數 + slack bits（純算術，不建 BQM）
    objective_scale: float | None = None
```

Warnings（至少）：

| code | 條件 |
|---|---|
| `SOFT_WEIGHT_SMALL` | soft weight < objective_scale × 0.01 |
| `LARGE_SLACK_RANGE` | 單一 inequality 的 slack bits > 10 |
| `EXACT_NEAR_LIMIT` | backend=exact 且 estimated variables > exact_max_variables × 0.8（需 policy；MCP 層傳入） |
| `EXACT_OVER_LIMIT` | backend=exact 且 estimated variables > 上限（validate 階段先提醒，solve 時仍會 `resource_limit_exceeded`） |
| `DENSE_FOR_QPU` | §15.2 |
| `SEED_IGNORED` / `PARAMETER_IGNORED` | backend 不支援該參數 |
| `DUPLICATE_TERM_MERGED` | Phase 1 v2 §8 的合併情形 |
| `REDUNDANT_CONSTRAINT` | inequality 永遠成立 |

`validate_optimization_problem` 的回傳必須讓 Agent 能在花 quota 前修正問題或改 backend。

---

## 21. MCP Server（interfaces/mcp/server.py）

```python
from mcp.server import MCPServer
import anyio

mcp = MCPServer("AnnealBridge")

def build_service(settings: ServerSettings) -> OptimizationService: ...   # composition root
```

Server 不含任何 optimization logic：MCP request → Pydantic → `service` → Pydantic → structured output。

Structured output 由 SDK v2 依 return type annotation 自動產生；tool 直接 `return result`（Pydantic model），**不得** `return json.dumps(...)`。若 SDK 需要顯式旗標才產 output schema，再加，並以 §26 測試驗證。

Tool 皆為 `async def`，CPU / remote 工作以 `await anyio.to_thread.run_sync(service.solve, problem)` 執行，不改寫 core 為 async。

---

## 22. Tool — get_optimization_capabilities

Input：無。Output：

```python
class BackendCapability(BaseModel):
    name: str                 # 2026-09-09 review F-22：registry key（= solver.backend 用的名字）
    available: bool           # 套件 + credential（無網路呼叫）
    enabled: bool             # policy 允許
    unavailable_reason: str | None
    remote: bool
    heuristic: bool
    exhaustive: bool
    supports_seed: bool
    returns_multiple_samples: bool
    limits: dict[str, float | int]     # e.g. {"max_variables": 24} / {"max_reads": 1000}
    description: str

class OptimizationCapabilities(BaseModel):
    schema_version: str                # 取自 OptimizationProblem.version 的 Literal，不硬寫
    supported_variable_types: list[str]
    supported_constraint_operators: list[str]
    supported_objective_terms: list[str]
    inequality_requires_integer_coefficients: bool   # True
    backends: list[BackendCapability]
    problem_json_schema: dict          # OptimizationProblem.model_json_schema()
```

由 `registry` + `policy` 建構，MCP 層不 hard-code backend 能力。**不得**求解，不得對 QPU 發任何 request。

---

## 23. Tool — validate_optimization_problem

```python
@mcp.tool()
async def validate_optimization_problem(problem: OptimizationProblem) -> ProblemValidationResult:
    """..."""
```

只做 Pydantic + semantic validation + warnings + estimate。不 compile 成 BQM、不 solve、不打 remote。

---

## 24. Tool — solve_optimization

```python
@mcp.tool()
async def solve_optimization(problem: OptimizationProblem) -> SolveResult:
    """..."""
```

Docstring（Agent 依此判斷何時呼叫，必須完整）：

```text
Solve a structured binary combinatorial optimization problem.

Call this only after translating the user's request into explicit binary
variables, an objective (linear/quadratic, minimize or maximize), and hard
or soft linear constraints. Do not pass natural-language requirements.

Inequality constraints (<=, >=) require integer coefficients and right-hand
sides. Soft constraint weights are in objective units.

Leave solver.penalty_multiplier at its default unless a previous result was
infeasible on a remote backend; hard constraint penalties are managed by the
server. Use get_optimization_capabilities to see which backends are enabled;
call validate_optimization_problem first when planning to use a remote backend.

Returns ranked feasible solutions with per-constraint evaluations, or a
structured error with a recommended_action.
```

Pydantic validation 失敗（型別錯）時，SDK 會自動回 tool error；semantic 失敗由 service 回 `invalid_problem`。兩者都必須能在 §26 測試中觀察到。

---

## 25. Transport 與 Entry Point

```bash
annealbridge-mcp                                  # stdio（預設）
annealbridge-mcp --transport streamable-http      # 綁 127.0.0.1:8000
annealbridge-mcp --transport streamable-http --host 0.0.0.0 --port 8080
```

- Streamable HTTP 預設 host `127.0.0.1`；要對外必須明示 `--host`。
- stdio 模式下 SDK v2 會把 stdout 導向 stderr，但程式碼仍不得 `print()`。
- README 警告：不得將未認證的 Streamable HTTP endpoint 直接暴露到公網。
- Phase 2 不做 auth。

---

## 26. MCP Tests（tests/mcp/，in-memory）

使用 SDK v2 `Client(mcp)`，不起 subprocess、不開 port：

```python
from mcp import Client
async with Client(mcp) as client:
    tools = await client.list_tools()
    result = await client.call_tool("solve_optimization", {"problem": knapsack_json})
    assert result.structured_content["status"] == "success"
```

必測：

- `tools/list` 恰有三個 tool；`solve_optimization` input schema 含 `problem` → `variables/objective/constraints/solver`；output schema 對應 `SolveResult`。
- `structured_content` 存在且為 dict（非 JSON string）。
- Exact 經 MCP 解 knapsack → success、objective == 17、無 `__` 開頭變數。
- SA 固定 seed 經 MCP → success、feasible。
- `validate_optimization_problem` 對 unknown variable → `valid=False` 且 errors 含 `UNKNOWN_VARIABLE`。
- 指定 `dwave_qpu` 且 policy 未開 → `backend_unavailable` / `REMOTE_DISABLED`，且 `solutions == []`（無 fallback）。
- 型別錯誤 payload（如 `coefficient: "abc"`）→ SDK tool error（斷言 `result.is_error`）。
- `get_optimization_capabilities` 不觸發任何 backend `solve()`（用 spy）。
- `test_stdio_entrypoint.py`：以 `Client` 的 stdio subprocess 模式啟動 `annealbridge-mcp`，執行 `tools/list`（唯一允許起 subprocess 的測試）。

直接呼叫 Python function `solve_optimization(...)` 不算 MCP integration test。

---

## 27. Remote Mock Tests（tests/remote_mock/，regular CI）

**不得**呼叫真實 D-Wave。以 fake sampler（實作 `sample(bqm, **kw)` 回 `dimod.SampleSet`，`info` 含 fake timing）注入 backend：

- BQM 傳入、`num_reads` / `annealing_time` / `chain_strength` / `time_limit` 轉送正確
- SampleSet → RawSolverResult 轉換正確；logical 變數含 slack 再由 service 移除
- metadata sanitized（§17）
- exception 分類正確（fake sampler raise 各類 exception）
- `allow_remote_retries=False` 時只有 1 attempt 且有 `REMOTE_RETRIES_DISABLED` warning
- `num_reads` 超限 → `resource_limit_exceeded`，訊息含 requested / maximum，未 clamp

`test_credential_leak.py`：設 `DWAVE_API_TOKEN=DEV-FAKE-TOKEN-1234567890abcdefghij`，fake sampler raise 含該 token 的 exception，執行 solve；斷言 `result.model_dump_json()`、caplog 全部 records、所有 `SolveError.message` 皆不含該字串。

---

## 28. Remote Live Tests（tests/remote_live/，opt-in）

```python
pytestmark = pytest.mark.remote
```

`conftest.py`：無 `DWAVE_API_TOKEN` 或未帶 `-m remote` 時 skip（此 skip 是允許的，需註解）。用最小 knapsack，`num_reads` 極小。

`pytest` 預設（`-m "not remote"`）不消耗任何 Leap quota。

---

## 29. CI

- 普通 PR：unit、scenarios、architecture、mcp、remote_mock。
- `remote_live` 只在 `workflow_dispatch` 手動觸發，`DWAVE_API_TOKEN` 從 secrets 注入，fork PR 不可取得。
- 禁止 `printenv`、`env` dump 或任何 debug secrets 步驟。

---

## 30. CLI 擴充（interfaces/cli/main.py）

```bash
annealbridge solve problem.json --backend dwave_qpu
annealbridge solve problem.json --backend leap_hybrid_bqm
annealbridge capabilities
annealbridge export-schema
```

CLI 由 `ServerSettings` 建 policy，與 MCP 共用同一個 `build_service()`。`capabilities` 輸出：

```text
Backend               Available  Enabled  Remote  Limits
exact                 yes        yes      no      max_variables=24
simulated_annealing   yes        yes      no
dwave_qpu             no         no       yes     max_reads=1000   (dwave-system not installed)
leap_hybrid_bqm       no         no       yes     max_time=300s    (D-Wave credentials not configured)
```

不顯示任何 config 值。

---

## 31. Logging

Remote solve 記錄：backend、problem name、logical variables / interactions、attempt、num_reads、time_limit、solver_id、feasible count、best ranking_score。所有 message 經 `redact()`。

不記錄：token、credentials、整個 environment、完整 problem JSON（可能含業務資料）。OpenTelemetry span attributes 同樣不得含 problem payload 或 credential。

---

## 32. Agent Self-Correction Boundary

- `invalid_problem` → Agent 修 variables / objective / constraints
- `backend_unavailable` → Agent 換 backend（依 capabilities）
- `resource_limit_exceeded` → Agent 縮小問題或換 backend
- `infeasible` 且 `infeasibility_proven=False` 且 remote → Agent 可改 `penalty_multiplier` 或先用 SA 驗證
- `infeasible` 且 `infeasibility_proven=True` → Agent 應告知使用者 constraints 互相矛盾

Agent 不應也不能設定 hard penalty λ 本身；λ 仍由 PenaltyStrategy 產生。

---

## 33. 不要做的事情（Phase 2 特別注意）

- 讓 MCP 層重做任何 optimization logic
- 新增 `solve_qubo` / `compile_problem_to_qubo` 等公開 QUBO API
- 新增 `compile_qubo`、`calculate_penalty`、`retry_solver`、`rank_samples`、`sample_qpu` 等細碎 tool
- silent fallback 到 SA
- silent clamp `num_reads` / `time_limit`
- `metadata = sampleset.info` 原封回傳
- `DWaveSampler(token=...)`
- 把 hard penalty 與 chain strength 混用或互相推導
- capabilities 打網路
- core 讀 env 或 import `config` / `mcp`
- 為 `max_retries=3` 自動送四次 QPU request
- 回傳 JSON string 充當 structured output
- 加 CQM / integer variables / DA

---

## 34. README

至少：Architecture（core / interfaces 分離圖）、Installation（三種 extras）、MCP host setup（Claude Desktop / Codex 的 stdio 設定範例、`annealbridge-mcp` 指令）、MCP Inspector 開發流程（`mcp dev`）、Agent flow 示意（自然語言 → Agent → OptimizationProblem → MCP → AnnealBridge → validated solution → Agent）、簡化 tool input 範例、D-Wave setup（`dwave setup` / `dwave config create` / env；不含真 token；`ANNEALBRIDGE_ALLOW_REMOTE=true`）、Security（remote 預設關、retry 預設關、HTTP 綁 localhost、token 不進任何輸出）、Limitations（Phase 1 v2 五條 + "Leap Hybrid returns a single sample" + "QPU embedding may fail for dense problems"）、Testing（`pytest`、`pytest -m remote`）。

---

## 35. Development Order

每步跑完相關測試再進下一步。

1. Phase 1 tests 全綠 → package 改名 `annealbridge`、CLI 移至 `interfaces/cli` → tests 仍全綠；加 `tests/architecture/test_import_boundaries.py`
2. `ExecutionPolicy`、`ServerSettings.to_policy()`、注入 `OptimizationService`；exact 上限改由 policy；tests
3. `SolverCapabilities`、`is_available()`、`SolverPreferences` / `SolveResult` / `SolveError` 擴充；Phase 1 JSON 相容測試
4. `SolverRegistry`（先 exact + SA）；service 改用 registry；Phase 1 tests 全綠
5. `metadata.py`（sanitizer + redact）+ tests
6. `LeapHybridBQMBackend` + mock tests
7. `DWaveQPUBackend` + mock tests + exception mapping
8. Service 流程 §14 完整實作：policy 檢查、resource limits、remote retry、concurrency；`test_credential_leak.py`
9. Problem Validator warnings + `estimated_compiled_variables`
10. `mcp` extra、`MCPServer`、`build_service()`
11. `get_optimization_capabilities` + in-memory test
12. `validate_optimization_problem` + test
13. `solve_optimization`：exact end-to-end、SA、remote disabled、型別錯誤
14. stdio / Streamable HTTP entry point、`--host/--port`、stdio subprocess test
15. `remote_live` tests（opt-in）、CI workflow
16. README、CLI `capabilities`

---

## 36. Acceptance Criteria

- [ ] Phase 1 所有 tests 通過；Phase 1 example JSON 未改仍可 parse
- [ ] import package 為 `annealbridge`；import boundary 測試通過
- [ ] `mcp>=2,<3`；core 不 import `mcp` / `dwave.cloud` / `config`
- [ ] `pip install annealbridge` 不裝 mcp / dwave-system，registry 仍可 import
- [ ] MCP Server 可啟動；stdio 與 Streamable HTTP 可用；HTTP 預設 127.0.0.1
- [ ] 三個 tool 可用；input / output schema 由 Pydantic 產生；`structured_content` 為 dict
- [ ] Exact 與 SA 皆可經 MCP in-memory `Client` 求解
- [ ] `ExecutionPolicy` 由 service 注入；CLI 與 MCP 共用 `build_service()`
- [ ] remote 預設 disabled；remote retry 預設 disabled 並產 warning
- [ ] 指定不可用 backend 不 fallback；限制超出不 clamp
- [ ] Exact 上限、QPU reads / annealing time、hybrid time、concurrency 皆有 guard
- [ ] `DWaveQPUBackend` 與 `LeapHybridBQMBackend` 實作完成，mock tests 通過
- [ ] Exception → 分類 error code + recommended_action
- [ ] metadata 經 whitelist；token 經 redaction；`test_credential_leak.py` 通過
- [ ] hard penalty 與 chain strength 分離
- [ ] remote samples 仍經 independent validator；ranking 不用 energy；slack 不外洩
- [ ] Problem Validator 產 warnings 與 estimate
- [ ] capabilities 不打網路（spy 測試）
- [ ] live tests opt-in；CI 不自動送 QPU；secrets 不給 fork
- [ ] README 含 MCP setup、D-Wave setup、Security、Limitations
- [ ] `pytest` 全過，無 skip / xfail（除 remote_live 的註解 skip）

---

## 37. 最重要的 Architecture Rule

```text
          ┌──── CLI
Core ─────┤
          └──── MCP
```

任何 Python 程式都應能 `service.solve(problem)` 而不需要 MCP Server。若加 MCP 需要動到 compiler / solver / validator / retry，代表 Phase 1 / 2 architecture 不合格。Phase 3（CQM、Digital Annealing、integer variables、solver routing）只應新增 compiler 與 backend，不動 IR 與 interfaces。
