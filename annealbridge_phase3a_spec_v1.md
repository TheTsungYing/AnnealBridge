# AnnealBridge — Phase 3a 開發規格 (v1)

> v1 修訂紀錄（2026-09-02）：初稿完成後經兩個獨立審查（架構原則 / 程式可行性）修正 25 處，主要：`cqm.num_variables` 是方法改用 `len(cqm.variables)`；`ANNEALBRIDGE_LIMITS` 非 JSON 的例外型別是 `pydantic_settings.SettingsError`；`gate_errors` 改收 backend 以維持 lazy availability；`limits` 通道由假第五 backend 以自訂 key 端到端驗證；grep 測試改為 AST 整值比對並排除 docstring；開發順序把 policy limits 提前到 validator 之前、每步同步更新 error catalog 總數；registry 順序明定並修正 `recommend` 範例；新增 §33.2 對 outline 的偏離表。

> 本規格依 `annealbridge_phase3_spec_outline.md`（含 §8 使用者決策，2026-09-02）撰寫，體例沿用 `annealbridge_phase2_spec_v2.md`。
> 撰寫時遵守 `annealbridge_PROJECT_OVERVIEW.md` §五的核心原則，特別是原則 4（核心不認識特定求解器）、原則 5（不偷偷幫使用者做決定）、原則 7（不蓋空殼）。
>
> **Phase 3 已拆為 3a / 3b（決策 1）。本文件只涵蓋 3a：**
> - Step 0：償還兩項架構債（backend 參數/上限宣告、validator 依 capabilities 判斷）
> - CQM compiler + Leap Hybrid CQM backend
> - Solver routing（只建議、不代選）
>
> **3b（整數變數 IR 1.1 + Fujitsu Digital Annealer）另寫 `annealbridge_phase3b_spec_v1.md`，本文件只在 §32 留下 3b 需要的接點，不實作。**
>
> 四項使用者決策（outline §8）在本文件中的落點：
> | 決策 | 落點 |
> |---|---|
> | 1. 拆 3a / 3b | 全文；§3 明確不做整數變數與 Fujitsu |
> | 2. `ExecutionPolicy` 舊欄位與 env 名保留為相容層，內部轉通用結構 | §11 |
> | 3. Fujitsu DA 照做（歸 3b） | §32.2 只留接點 |
> | 4. routing 納入 3a、排最後 | §23–§25、§30 步驟 9 |

---

## 0. 現況（實作者必讀）

- Phase 1 + 2 完成，`pytest` 892 passed；套件 `annealbridge`；git main 最新 commit `b0eb7ba`（前一個程式碼 commit 為 `8d56366`）。
- 已安裝：dimod 0.12.22、dwave-samplers 1.8.0、numpy 2.5.2、pydantic 2.13.4、mcp 2.1.1；`dwave-system` 在開發機**未安裝**（remote 測試走 mock）。
- 現有擴充點只有三個 Protocol：`ModelCompiler`、`SolverBackend`、`PenaltyStrategy`。3a **不新增第四個 Protocol**；只擴充這三個的欄位/方法，以及在 `SolverCapabilities` 上加宣告式欄位。
- 目前 backend 名稱硬寫的位置（Step 0 要清掉的全部清單，實作完成後 `grep` 測試 §13.2 必須為零）：
  - `interfaces/capabilities.py` `_policy_limits`：`"exact"` / `"dwave_qpu"` / `"leap_hybrid_bqm"` if-chain
  - `validation/problem_validator.py`：`SEED_IGNORING_BACKENDS`、`_POSITIVE_BACKEND_OPTION_FIELDS`、`_warn_backend_fit` 的 `== "exact"` / `== "dwave_qpu"`、`_warn_ignored_parameters` 的 `== "leap_hybrid_bqm"`
  - `orchestration/optimizer.py` `_preference_limit_errors`：用 capabilities flag 分派，卻直接讀 `preferences.dwave_qpu` / `preferences.leap_hybrid_bqm` 子模型
  - `orchestration/optimizer.py` `_AVAILABILITY_MAP`：以 D-Wave 專屬 reason 字串常數對應 status/code
- 已重現的漂移（Step 0 的驗收基準）：
  1. 第五個 backend 若 `remote=True, supports_num_reads=True`，capabilities 回 `limits={}`、solve 卻回 `QPU_READS_LIMIT`。
  2. `exact` 宣告 `supports_seed=False`，但不在 `SEED_IGNORING_BACKENDS`，帶 seed 時不產 `SEED_IGNORED`。

---

## 1. Phase 3a 目標

> **讓「加一個新求解器」真的只需要加一個模組，並用第一個非 BQM 的路徑（CQM）證明這件事；同時給 Agent 一個只建議、不代選的 backend 推薦工具。**

```text
Claude / Codex / AI Agent
          │ MCP
          ▼
┌────────────────────────────────────┐
│          AnnealBridge MCP          │
│  get_optimization_capabilities     │
│  validate_optimization_problem     │
│  recommend_backend        (new)    │
│  solve_optimization                │
└───────────────┬────────────────────┘
                ▼
       OptimizationService  ◄── ExecutionPolicy（通用 limits）
          │           │
   compilers{bqm,cqm} │  依 backend.capabilities.supported_model_types 選 compiler
                      ▼
               Solver Registry
    ┌──────┬──────┬───────────┬────────────────┬────────────────┐
    ▼      ▼      ▼           ▼                ▼                ▼
  exact   SA   dwave_qpu  leap_hybrid_bqm  leap_hybrid_cqm (new)
  [bqm]  [bqm]   [bqm]        [bqm]            [cqm]
```

---

## 2. Phase 3a Scope

1. Step 0a：`SolverCapabilities` 搬到 `models/`、`AvailabilityStatus` 結構化、validator 改以 capabilities 注入判斷、`OptimizationService.validate()`、CLI `validate`
2. Step 0b：backend 以 `parameter_limits` 宣告受限參數；`ExecutionPolicy` 通用 `limit()` / `limits_for()`（舊欄位、env 名保留）；service 與 capabilities view 讀同一份宣告；架構測試（假第五 backend、名稱 grep）
3. `CompiledProblem.model_type` + `ModelCompiler` 擴充（`model_type`、`uses_hard_penalty`）
4. `CQMCompiler`（hard constraint 原生、無 penalty、無 slack）
5. service 依 capabilities 選 compiler；CQM 路徑 attempt 恆 1、penalty 記 `None`
6. `LeapHybridCQMBackend` + `LeapHybridCQMOptions` + mock / live tests
7. 遠端 backend 共用的 Ocean helper 抽到 `solvers/ocean.py`
8. Solver routing：`orchestration/routing.py`、`OptimizationService.recommend()`、MCP tool `recommend_backend`、CLI `recommend`
9. Error catalog 新 code、README、Phase 1/2 regression 全綠

---

## 3. Phase 3a 明確不做

整數變數、實數變數、無界整數、IR 版本變更（`version` 仍為 `Literal["1.0"]`）、Fujitsu Digital Annealer、自動 cost estimation、自動選 backend 後直接求解、OR-Tools / Gurobi 等精確 MILP、job queue、persistence、auth、多租戶、Web UI、LLM 呼叫、MCP prompts / elicitation、把 Leap 官方變數/constraint 上限寫進 validator 檢查（§17.6）、放寬 CQM 路徑的 `NON_INTEGER_INEQUALITY`（§21.3）。

---

## 4. Architecture 與 Dependency Direction

```text
models ← validation ← compiler ← solvers ← orchestration ← interfaces (cli / mcp)
                                                  ▲
                                             config (composition root only)
```

規則（全部沿用 Phase 2 §4，並新增）：

- `SolverCapabilities`、`AvailabilityStatus`、`ParameterLimit`、`ModelType` 移至 `models/capabilities.py`（純 Pydantic）。理由與 `SolverExecutionMetadata` 搬到 `models/metadata.py` 相同：`validation` 與 `orchestration/policy.py` 都要用到，而 `validation` 不得 import `solvers`。`solvers/base.py` 從 `models` re-export，既有 `from annealbridge.solvers import SolverCapabilities` 仍可用。
- `orchestration` **不得 import 任何具體 backend 模組**（`solvers.exact`、`solvers.dwave_qpu`、`solvers.leap_hybrid_*`、`solvers.simulated_annealing`），只能經 `solvers.registry` / `solvers.base` / `solvers.metadata`；`solvers/__init__.py` 對 `REASON_*` 的 re-export **保留**（既有測試 import 它們），只是 `orchestration` 不再 import。
- `orchestration` **可以** import 具體 compiler（`compiler.bqm`、`compiler.cqm`），但**只限 `orchestration/optimizer.py` 建立預設 `compilers` 清單那一處**；`routing.py`、`limits.py` 與 `optimizer.py` 其餘程式一律走 `ModelCompiler` Protocol（§13.3 以測試強制）。這是對 outline §4「不得 import 任何具體 compiler」的刻意縮限，理由：compiler 是核心自己的元件而非外掛，service 總得在某處知道預設有哪些 compiler，否則就要再蓋一個 compiler registry（原則 7）。記於 §33.2。
- service、validator、interfaces 內**不得出現 backend 名稱字串比對**；允許依 `capabilities` 欄位分派。§13.2 以 grep 測試強制。
- 所有遠端 backend 共用 `solvers/metadata.py`（availability / classify / redact / sanitize）與新的 `solvers/ocean.py`（§17.5）；exception → code 表可各自定義。
- 每個新 error code / warning code 都要進 `error_catalog` 且有 `recommended_action`；routing 的 reason code 有獨立的固定說明表（§23.4）。
- 測試紅線不變：MCP 測試走 `Client`、無 skip / xfail（live 除外）、Phase 1/2 example JSON 不改一字。

---

## 5. 目錄調整

```text
src/annealbridge/
├── models/
│   ├── capabilities.py        (new: ModelType, ParameterLimit, AvailabilityStatus, SolverCapabilities)
│   ├── compiled.py            (model_type、hard_penalty: float | None、ConstraintTrace.native)
│   ├── problem.py             (+ LeapHybridCQMOptions、backend Literal + "leap_hybrid_cqm")
│   ├── solution.py            (SolveAttempt.penalty: float | None)
│   ├── metadata.py            (+ sampler_reported_feasible)
│   └── error_catalog.py       (+ 新 codes)
├── validation/
│   ├── problem_validator.py   (capabilities 注入；移除所有 backend 名稱)
│   └── recommendation.py      (new: BackendRecommendation / BackendRecommendationResult 模型)
├── compiler/
│   ├── base.py                (ModelCompiler + model_type / uses_hard_penalty)
│   ├── objective.py           (new: build_objective_bqm，兩個 compiler 共用)
│   ├── bqm.py                 (model_type="bqm"；objective 改用 objective.py)
│   └── cqm.py                 (new: CQMCompiler)
├── solvers/
│   ├── base.py                (re-export models.capabilities；is_available() -> AvailabilityStatus)
│   ├── metadata.py            (dwave_availability() -> AvailabilityStatus；REASON_* 與 re-export 保留)
│   ├── ocean.py               (new: call_ocean、resolved、LazySampler、共用 exception 表)
│   ├── leap_hybrid_cqm.py     (new)
│   ├── registry.py            (+ leap_hybrid_cqm)
│   └── ...                    (既有 backend 改用 ocean.py、宣告 parameter_limits)
├── orchestration/
│   ├── policy.py              (limit() / limits_for() / limits dict)
│   ├── limits.py              (new: gate_errors、preference_limit_errors、read_preference)
│   ├── routing.py             (new: recommend())
│   └── optimizer.py           (compilers、model_type 分派、validate()、recommend())
├── config/settings.py         (+ limits: dict[str, float]；舊欄位不變)
└── interfaces/
    ├── capabilities.py        (limits 改讀 policy.limits_for(caps)；刪 _policy_limits)
    ├── cli/main.py            (+ validate、recommend)
    └── mcp/tools.py           (+ recommend_backend；validate 改呼叫 service.validate)

tests/
├── unit/
│   ├── test_capabilities_model.py     (new)
│   ├── test_availability.py           (new)
│   ├── test_policy.py                 (+ limit / limits_for / limits dict)
│   ├── test_settings.py               (+ ANNEALBRIDGE_LIMITS、pydantic_settings.SettingsError 轉換)
│   ├── test_capabilities_view.py      (is_available() 改 AvailabilityStatus)
│   ├── test_solvers.py / test_registry.py (同上；registry 五個名稱)
│   ├── test_error_catalog.py          (每步同步更新 code 總數)
│   ├── test_limits.py                 (new: orchestration/limits.py 純函式)
│   ├── test_problem_validator_full.py (改為 capabilities 注入)
│   ├── test_service_validate.py       (new)
│   ├── test_cqm_compiler.py           (new)
│   ├── test_service_cqm_flow.py       (new: 以 FakeLocalCQMBackend 走完整流程)
│   ├── test_routing.py                (new)
│   └── test_cli.py                    (+ validate / recommend)
├── architecture/
│   ├── test_import_boundaries.py      (+ orchestration 不 import 具體 backend)
│   ├── test_no_backend_names.py       (new: grep)
│   └── test_fifth_backend.py          (new: 假 backend 零改動)
├── fakes/                             (new: 測試用 backend / sampler，非 production)
│   ├── __init__.py
│   ├── local_cqm_backend.py           (FakeLocalCQMBackend，包 dimod.ExactCQMSolver)
│   └── declared_backend.py            (FakeDeclaredBackend，第五 backend)
├── scenarios/
│   └── test_knapsack_cqm.py           (new)
├── mcp/
│   ├── test_tools_list.py             (四個 tool)
│   ├── test_capabilities.py           (backend 名單改五個)
│   ├── test_validate.py               (+ capabilities 相關 warnings)
│   └── test_recommend.py              (new)
├── remote_mock/
│   ├── conftest.py                    (+ FakeCQMSampler；make_remote_available 改回 AvailabilityStatus)
│   ├── test_dwave_qpu_mock.py / test_leap_hybrid_mock.py (is_available() 斷言改型別)
│   ├── test_service_remote_flow.py    (availability 未知 reason 案例改 category="unavailable")
│   ├── test_leap_hybrid_cqm_mock.py   (new)
│   └── test_credential_leak.py        (+ CQM backend)
└── remote_live/
    └── test_leap_hybrid_cqm_live.py   (new, opt-in)
```

---

## 6. Dependencies

不新增套件。`LeapHybridCQMSampler` 隨 `dwave-system` 提供，仍屬 `dwave` extra；`dimod.ConstrainedQuadraticModel` 與 `dimod.ExactCQMSolver` 在核心依賴 `dimod>=0.12` 內（實測 0.12.22 皆存在）。

`pyproject.toml` 不變，除了 markers 說明可補「remote: live D-Wave tests（含 CQM）」。

---

## 7. `models/capabilities.py`

```python
from typing import Literal
from pydantic import BaseModel, Field

ModelType = Literal["bqm", "cqm"]

class ParameterLimit(BaseModel):
    """一個受 policy 上限管制的使用者參數（backend 自行宣告）。"""
    preference: str      # SolverPreferences 的點路徑，如 "num_reads"、"dwave_qpu.annealing_time_us"
    limit: str           # ExecutionPolicy 通用 limit key，如 "reads"（§11.1）
    error_code: str      # 超限時的 catalog code，如 "QPU_READS_LIMIT"

AvailabilityCategory = Literal[
    "available", "not_installed", "credentials_missing", "config_invalid", "unavailable"
]

class AvailabilityStatus(BaseModel):
    category: AvailabilityCategory
    detail: str | None = None        # 分類文字（如 "dwave-system not installed"），不含任何 config 值
    error_code: str | None = None    # backend 可指定；None 則 service 依 category 取預設（§8.2）

    @property
    def available(self) -> bool:
        return self.category == "available"

class SolverCapabilities(BaseModel):
    # Phase 2 §10 欄位，不變
    name: str
    remote: bool
    heuristic: bool
    exhaustive: bool
    supports_seed: bool
    supports_num_reads: bool
    supports_time_limit: bool
    supported_model_types: list[ModelType]      # 偏好順序；service 取第一個有 compiler 的（§16.1）
    returns_multiple_samples: bool
    description: str
    # 3a 新增，皆有預設值，既有 backend 宣告不需全部改
    supports_num_sweeps: bool = False           # 只有 SA 為 True
    requires_embedding: bool = False            # dwave_qpu 為 True → DENSE_FOR_QPU
    parameter_limits: list[ParameterLimit] = Field(default_factory=list)

    @property
    def preferred_model_type(self) -> ModelType:
        return self.supported_model_types[0]
```

`supported_model_types` 至少一個元素（validator 檢查）。`ModelType` 是 Literal 而非 str：新 model type 是核心的事（要有 compiler），不是外掛可以隨意宣告的。

### 7.1 五個 backend 的宣告

| backend | model types | supports_num_sweeps | requires_embedding | parameter_limits |
|---|---|---|---|---|
| `exact` | `["bqm"]` | False | False | `[]`（變數上限由 `exhaustive` flag 驅動，§12.2） |
| `simulated_annealing` | `["bqm"]` | **True** | False | `[]` |
| `dwave_qpu` | `["bqm"]` | False | **True** | `("num_reads", "reads", "QPU_READS_LIMIT")`、`("dwave_qpu.annealing_time_us", "annealing_time_us", "QPU_ANNEALING_TIME_LIMIT")` |
| `leap_hybrid_bqm` | `["bqm"]` | False | False | `("leap_hybrid_bqm.time_limit_seconds", "time_seconds", "REMOTE_TIME_LIMIT")` |
| `leap_hybrid_cqm` | `["cqm"]` | False | False | `("leap_hybrid_cqm.time_limit_seconds", "time_seconds", "REMOTE_TIME_LIMIT")` |

表中的 tuple 是縮寫，程式碼一律寫 `ParameterLimit(preference=..., limit=..., error_code=...)`（pydantic 對 `list[ParameterLimit]` 不接受 tuple）。

`simulated_annealing` 雖 `supports_num_reads=True`，但**不宣告** reads 上限：本機 reads 不受 policy 管，這正是 Phase 2 用 `caps.remote and caps.supports_num_reads` 特判想表達的事，改成宣告後不需要特判。

---

## 8. `is_available()` 結構化

### 8.1 Protocol 變更

```python
class SolverBackend(Protocol):
    @property
    def capabilities(self) -> SolverCapabilities: ...
    def is_available(self) -> AvailabilityStatus: ...        # 原 tuple[bool, str | None]
    @property
    def name(self) -> str: ...
    @property
    def is_exhaustive(self) -> bool: ...
    def resolve_time_limit(self, compiled_problem: CompiledProblem, preferences: SolverPreferences) -> float | None: ...
    def solve(self, compiled_problem: CompiledProblem, preferences: SolverPreferences) -> RawSolverResult: ...
```

參數名維持 Phase 2 的 `compiled_problem`，不改。

- 本機 backend：`AvailabilityStatus(category="available")`。
- `solvers/metadata.py` 的 `dwave_availability()` 改回傳 `AvailabilityStatus`：`not_installed` / `credentials_missing` / `config_invalid`（`error_code="DWAVE_CONFIG_INVALID"`）/ `available`。`detail` 沿用 `REASON_*` 常數文字。三個 D-Wave backend 共用。
- 仍不得有網路 I/O、不得快取。

### 8.2 service 對應表（依 category，不依字串）

| category | status | 預設 error code（backend 未指定 `error_code` 時） |
|---|---|---|
| `not_installed` | `backend_unavailable` | `BACKEND_NOT_INSTALLED` |
| `credentials_missing` | `backend_unavailable` | `REMOTE_CREDENTIALS_MISSING` |
| `config_invalid` | `configuration_error` | `BACKEND_CONFIG_INVALID`（新，§20） |
| `unavailable` | `backend_unavailable` | `BACKEND_UNAVAILABLE` |

`_AVAILABILITY_MAP` 改成以 category 為 key；`orchestration` 不再 import `REASON_*`（常數本身與 `solvers/__init__` 的 re-export 保留）。錯誤訊息格式不變：`Backend '<name>' is unavailable: <detail>`；`detail is None` 時沿用 `no reason reported`。

既有測試的對應：`tests/remote_mock/conftest.py` 的 `make_remote_available` 改回 `AvailabilityStatus(category="available")`；模擬 D-Wave config 壞掉的 fake 必須連 `error_code="DWAVE_CONFIG_INVALID"` 一起給（與 `dwave_availability()` 相同），`test_service_remote_flow.py` 對該 code 的期望才不變；Phase 2 的「未知 reason」與 `(False, None)` 案例改為 `category="unavailable"`，期望 `BACKEND_UNAVAILABLE`。

### 8.3 capabilities view

`BackendCapability.unavailable_reason = status.detail if not status.available else None`。輸出格式與 Phase 2 相同，既有 MCP / CLI 測試不需改期望值。

---

## 9. Problem Validator 改以 capabilities 判斷

### 9.1 簽名

```python
def validate_problem_full(
    problem: OptimizationProblem,
    *,
    capabilities: SolverCapabilities | None = None,
    max_compiled_variables: int | None = None,
    model_type: ModelType | None = None,
) -> ProblemValidationResult: ...
```

- `capabilities is None`：只做 backend 無關的檢查（errors、`SOFT_WEIGHT_SMALL`、`LARGE_SLACK_RANGE`、`REDUNDANT_CONSTRAINT`、`DUPLICATE_TERM_MERGED`），estimate 以 BQM 路徑計算，所有 backend-fit / ignored-parameter warnings 跳過。
- `exact_max_variables` 參數**移除**（改名 `max_compiled_variables`，只在 `capabilities.exhaustive` 時套用）。此函式的直接呼叫者只有 MCP tool 與測試，MCP tool 改走 `service.validate()`（§10）。
- `model_type`：呼叫端（service）告知實際會走的 compiler 路徑；`None` 時退回 `capabilities.preferred_model_type`（無 capabilities 則 `"bqm"`）。validator 不認識 compilers，所以由 service 傳入，避免多 model type 的 backend 搭配自訂 `compilers` 時 validator 猜錯路徑。

### 9.2 estimate 依 model type

```python
model_type = model_type or (capabilities.preferred_model_type if capabilities else "bqm")
estimated = estimate_compiled_variables(problem) if model_type == "bqm" else len(problem.variables)
```

CQM 路徑沒有 slack，`estimated_compiled_variables == len(variables)`。`ProblemValidationResult` 新增 `model_type: ModelType | None = None`（記錄估算依據）。

### 9.3 warnings 對應表（取代 Phase 2 §20 以 backend 名稱條件的列）

| code | 觸發條件（全部以 capabilities 欄位表達） |
|---|---|
| `EXACT_OVER_LIMIT` | `caps.exhaustive` 且 `max_compiled_variables` 非 None 且 `estimated > max` |
| `EXACT_NEAR_LIMIT` | `caps.exhaustive` 且 `max_compiled_variables` 非 None 且 `max × 0.8 < estimated ≤ max` |
| `DENSE_FOR_QPU` | `caps.requires_embedding` 且（`estimated > 150` 或最大 constraint 變數數 > 30） |
| `SEED_IGNORED` | `solver.seed is not None` 且 `not caps.supports_seed` |
| `PARAMETER_IGNORED`（`solver.num_reads`） | `num_reads` 非預設 且 `not caps.supports_num_reads` |
| `PARAMETER_IGNORED`（`solver.num_sweeps`） | `num_sweeps` 非預設 且 `not caps.supports_num_sweeps` |
| `PARAMETER_IGNORED`（`solver.penalty_multiplier`） | 非預設 且 `model_type == "cqm"`（CQM 路徑無 hard penalty，§16.3） |
| `PARAMETER_IGNORED`（`solver.max_retries`） | 非預設 且（`model_type == "cqm"` 或 `caps.exhaustive`）——兩者依 §16.3 attempt 都恆 1 |
| `PARAMETER_IGNORED`（`solver.<block>.*`） | 使用者填了某個 backend option block（如 `dwave_qpu`），但該 block 欄位名 ≠ `caps.name`（填錯 backend 的選項）。**命名契約**：option block 的欄位名必須等於該 backend 的 `capabilities.name`（現有四個皆如此，`leap_hybrid_cqm` 亦同）；比對對象是 `caps.name` 而非 registry key，自訂 registry 用別的 key 註冊時此 warning 以 `caps.name` 為準。沒有 option block 的 backend（exact、SA、§13.1 的 fake）不受影響 |
| 其餘（`SOFT_WEIGHT_SMALL`、`LARGE_SLACK_RANGE`、`REDUNDANT_CONSTRAINT`、`DUPLICATE_TERM_MERGED`） | 不變，backend 無關 |

`EXACT_*` / `DENSE_FOR_QPU` 的 code 名稱**保留**（Agent 已學過的詞彙），但訊息文字改為以能力描述：「the exhaustive backend limit」、「a backend that requires minor-embedding」。

**行為變更（刻意）**：`exact` 帶 `seed` 現在會產 `SEED_IGNORED`，帶非預設 `num_reads` / `num_sweeps` 會產 `PARAMETER_IGNORED`。這是修正漂移 2，不是回歸；`examples/*.json` 都用預設值，輸出不變。既有測試若斷言 `warnings == []` 且問題帶這些參數，更新期望值並在測試註解引用本節。

### 9.4 `_POSITIVE_BACKEND_OPTION_FIELDS` 通用化

改為由 `SolverPreferences.model_fields` 反射：凡 annotation 為 `<BaseModel 子類> | None` 的欄位視為 option block（同時處理 `types.UnionType` 與 `typing.Union` 兩種寫法；實測 pydantic 2.13 下 `dwave_qpu` 的 origin 是 `types.UnionType`，`seed: int | None` 正確地不被視為 block），其所有 `float | None` / `int | None` 欄位若非 None 必須有限且 > 0，否則 `INVALID_SOLVER_PREFERENCE`（path `solver.<block>.<field>`）。`bool` 欄位（`auto_scale`）不檢查。新增 `leap_hybrid_cqm` block 後 validator 零改動。「> 0」是 3a 全部 option 欄位的共同規則；3b 若出現合法為 0 或負數的 option 欄位，改為在欄位上以 `Field(gt=0)` 宣告、validator 只讀 constraint，屆時再改（§32）。

### 9.5 移除

`SEED_IGNORING_BACKENDS`、`HYBRID_IGNORED_PARAMETER_FIELDS`、`_POSITIVE_BACKEND_OPTION_FIELDS` 常數全部刪除。`problem_validator.py` 內不得殘留任何 backend 名稱字串。

---

## 10. `OptimizationService.validate()` 與 CLI `validate`

```python
class OptimizationService:
    def validate(self, problem: OptimizationProblem) -> ProblemValidationResult:
        try:
            caps = self._registry.get(problem.solver.backend).capabilities
        except KeyError:
            caps = None
        model_type = self._select_model_type(caps) if caps is not None else None   # §16.1，無 compiler 時 None
        return validate_problem_full(
            problem,
            capabilities=caps,
            max_compiled_variables=int(self._policy.limit("variables")),
            model_type=model_type,
        )
```

- 不 compile、不 solve、不打網路、不佔 concurrency slot。
- MCP `validate_optimization_problem` 改為一行 `state.service.validate(problem)`；policy → validator 的接線只存在於 service（Phase 2 §37：任何 Python 程式不經 MCP 也能得到同樣行為）。
- 未知 backend（自訂 registry 才可能）：`caps=None`，結果仍可能 `valid=True`；solve 時才回 `UNKNOWN_BACKEND`。`ProblemValidationResult.warnings` 加一條 `UNKNOWN_BACKEND`（warning 而非 error，因 validator 不判 backend 存在性）。

CLI：

```bash
annealbridge validate problem.json                 # 人類可讀
annealbridge validate problem.json --json          # ProblemValidationResult JSON
annealbridge validate problem.json --backend exact # 同 solve 的 --backend 覆寫
```

人類可讀輸出：

```text
Problem:   knapsack
Backend:   exact  (model type: bqm)
Valid:     yes
Estimated compiled variables: 7
Objective scale: 27

Warnings (1):
  [EXACT_NEAR_LIMIT] solver.backend: Estimated compiled variables (20) are above 80% of the exhaustive backend limit (24)
    recommended action: ...
```

exit code：valid → 0；invalid → 1；檔案 / JSON / schema 錯 → 2（與 `solve` 一致）。

---

## 11. `ExecutionPolicy` 通用 limits（決策 2）

### 11.1 Limit keys

固定詞彙，與 backend 無關：

| key | 意義 | 舊欄位（相容層） | env（不變） |
|---|---|---|---|
| `variables` | exhaustive backend 的編譯後變數上限 | `exact_max_variables` | `ANNEALBRIDGE_EXACT_MAX_VARIABLES` |
| `reads` | 單次遠端 reads 上限 | `max_qpu_reads` | `ANNEALBRIDGE_MAX_QPU_READS` |
| `annealing_time_us` | 單次 anneal 時間上限 | `max_qpu_annealing_time_us` | `ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US` |
| `time_seconds` | 遠端 time_limit 上限（所有 hybrid） | `max_remote_time_seconds` | `ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS` |

### 11.2 模型

```python
class ExecutionPolicy(BaseModel):
    # Phase 2 §8 欄位全部保留、預設值與 bounds 不變
    allow_remote: bool = False
    allow_remote_retries: bool = False
    exact_max_variables: int = Field(default=24, ge=1)
    max_qpu_reads: int = Field(default=1000, ge=1)
    max_qpu_annealing_time_us: float = Field(default=2000.0, gt=0)
    max_remote_time_seconds: int = Field(default=300, ge=1)
    max_concurrent_solves: int = Field(default=4, ge=1)
    enabled_backends: set[str] | None = None
    # 3a：通用結構。key 不得與 11.1 的四個相容 key 重複（validator 拒絕，避免兩個來源）
    limits: dict[str, float] = Field(default_factory=dict)   # 每個值必須有限且 > 0

    def limit(self, key: str) -> float | int | None:
        """通用查詢：先查 limits，再查相容欄位；都沒有回 None。"""

    def limits_for(self, capabilities: SolverCapabilities) -> dict[str, float | int]:
        """capabilities view 與 service 共用的唯一來源（§12.3）。"""
```

- 3a 內建 backend 只用四個相容 key，`limits` 預設空；它在 3a 的消費者是 §13.1 的假第五 backend（宣告自訂 key `iterations`，端到端證明自訂 key 可用），不是空殼。3b 的 Fujitsu 走同一條路。
- `ServerSettings` 加 `limits: dict[str, float] = {}`，env `ANNEALBRIDGE_LIMITS`（pydantic-settings 以 JSON 解析，實測 `'{"iterations": 100000}'` 可用）；`to_policy()` 原樣傳入。舊 env 名一個都不改。
- **例外型別（實測 pydantic-settings 2.15）**：env 值不是合法 JSON 時拋的是 `pydantic_settings.SettingsError`（`ValueError` 子類，**沒有** `.errors()`），不是 `pydantic.ValidationError`。`load_settings()` 必須同時捕捉兩者並轉成專案的 `annealbridge.config.SettingsError`：`ValidationError` 沿用逐欄位格式，`pydantic_settings.SettingsError` 用 `f"Invalid server settings: {exc}"` 一行，兩者都走 CLI / MCP 既有的 exit 2 路徑。
- **型別規則**：`limit(key)` 回相容欄位自己的型別（`variables` 為 `int`，其餘 `int` / `float` 依欄位），`limits` dict 的值一律 `float`；`limits_for()` 原樣輸出（自訂 key 會是 `100000.0`，可接受）；需要 `int` 的呼叫端（§10 的 `max_compiled_variables`）自行 `int()`。
- **不新增任何 backend 命名的 policy 欄位**；3b 加 backend 時只能走 `limits`。

### 11.3 宣告與 policy 的一致性

`OptimizationService.__init__` 走訪 registry 全部 backend 的 `parameter_limits`，任何 `limit` key 在 policy 查不到值（`limit(key) is None`）→ `raise ValueError("backend 'x' declares limit 'iterations' but the policy has no value for it")`。這是 composition root 的錯誤（`build_state` 轉成 `SettingsError` 一併印到 stderr、exit 2），不是 solve 時的結構化錯誤：一個宣告了上限卻沒有上限值的遠端 backend 不可以靜默地無上限執行。

### 11.4 2026-09-09 review 修正（F-02 / F-07）：新增五個內建 limit key

review 指出仍有一批呼叫端可控的參數沒有任何上限：本機取樣的 `num_reads` / `num_sweeps` 可以無限拉高而長時間佔住 concurrency slot；`max_retries` 即使 opt-in 遠端重試後也無上限，可以把單一請求放大成任意多次廠商提交；`top_k` 則無限制地放大回傳體積。因此在 §11.1 的四個 key 之後再加五個，形制與相容層完全相同（`max_<key>` 欄位 + `ANNEALBRIDGE_MAX_<KEY>` env），一樣被 `limits` validator 拒收，所以每個 key 仍只有一個來源。

| key | 欄位 | env | 預設 | 適用 |
|---|---|---|---|---|
| `local_reads` | `max_local_reads` | `ANNEALBRIDGE_MAX_LOCAL_READS` | 100000 | `simulated_annealing` 宣告 `ParameterLimit`（`num_reads` → `LOCAL_READS_LIMIT`） |
| `sweeps` | `max_sweeps` | `ANNEALBRIDGE_MAX_SWEEPS` | 100000 | `simulated_annealing` 宣告 `ParameterLimit`（`num_sweeps` → `SWEEPS_LIMIT`） |
| `local_retries` | `max_local_retries` | `ANNEALBRIDGE_MAX_LOCAL_RETRIES` | 10 | 所有 `remote=False` backend 的 `max_retries`（`RETRY_LIMIT`） |
| `remote_retries` | `max_remote_retries` | `ANNEALBRIDGE_MAX_REMOTE_RETRIES` | 3 | 所有 `remote=True` backend 的 `max_retries`（`RETRY_LIMIT`） |
| `top_k` | `max_top_k` | `ANNEALBRIDGE_MAX_TOP_K` | 1000 | 所有 backend 的 `top_k`（`TOP_K_LIMIT`） |

- `local_reads` / `sweeps` 走 §11.3 的宣告路徑：backend 自己宣告，policy 只提供值。
- `local_retries` / `remote_retries` / `top_k` **不屬於任何 backend**，由 service 層依 capabilities 的 flag 驅動：retry 上限以 `remote` 旗標二選一（`ExecutionPolicy.retries_limit_key()`），`top_k` 對每個 backend 都套用。`max_retries` 的檢查與該次是否真的會重試無關——`allow_remote_retries=False` 時超限一樣拒絕，opt-in 之後上限也不會被解除。
- `limits_for()` 因此對**每個** backend 在原有 key 之後追加 `max_local_retries` / `max_remote_retries`（依 `remote`）與 `max_top_k`。capabilities view 與 CLI 表格是同一份輸出，所以新 key 也會出現在那裡；key 順序仍是「宣告項在前、service 層在後」。
- 超限一律 `status="resource_limit_exceeded"` + 對應 code，值不砍，backend 不被呼叫。相關新增 error code：`LOCAL_READS_LIMIT`、`SWEEPS_LIMIT`、`RETRY_LIMIT`、`TOP_K_LIMIT`，皆非 retryable。
- 兩個 retry 上限的 bound 是 `ge=0`（`max_retries: 0` 是合法請求），其餘三個為 `ge=1`。
- §11.2 的「**不新增任何 backend 命名的 policy 欄位**」原則未變：這五個欄位全部以參數／範疇命名（local / remote、reads / sweeps / retries / top_k），沒有一個綁定 backend 名稱。

同批 review 另加（F-07）`PENALTY_OVERFLOW`：hard penalty 倍增或編譯結果溢出浮點範圍時，service 回 `resource_limit_exceeded` 並帶此 code，不再落到 `SOLVER_ERROR`；（F-08）`penalty_multiplier` 同時改為 `allow_inf_nan=False`，validator 訊息改為 `must be a finite number > 0`（`INVALID_SOLVER_PREFERENCE`）。

---

## 12. Service 通用 limit 檢查（`orchestration/limits.py`）

純函式，service 與 routing 共用；不 import 任何具體 backend。

```python
def read_preference(preferences: SolverPreferences, path: str) -> float | int | None:
    """點路徑取值；中途遇到 None（option block 未填）回 None；路徑不存在 → ValueError（宣告錯誤，測試抓）。"""

def preference_limit_errors(
    capabilities: SolverCapabilities, preferences: SolverPreferences, policy: ExecutionPolicy
) -> list[SolveError]:
    """§16.2 步驟 8：走訪 capabilities.parameter_limits，超限 → catalog_error(decl.error_code, ...)。不 clamp。
    policy.limit(decl.limit) 為 None → ValueError（宣告與 policy 不一致；service 建構時已擋，此處是純函式自己的防線）。"""

def gate_errors(
    backend_name: str, backend: SolverBackend, policy: ExecutionPolicy
) -> tuple[str, str, list[SolveError]] | None:
    """§16.2 步驟 3–5：enabled_backends → allow_remote → backend.is_available()，依序短路；
    前兩關擋下時不呼叫 is_available()（D-Wave 的 availability 會讀 config 檔，順序語意與 Phase 2 相同）。
    回 (status, reported_backend_name, errors) 或 None。reported_backend_name：BACKEND_DISABLED_BY_POLICY 用
    registry key（backend_name），其餘用 capabilities.name（Phase 2 既有行為，測試釘住）。"""
```

### 12.1 `_preference_limit_errors` 改寫

```python
for decl in caps.parameter_limits:
    value = read_preference(preferences, decl.preference)
    maximum = policy.limit(decl.limit)        # __init__ 已保證非 None
    if value is not None and value > maximum:
        errors.append(_limit_error(decl.error_code, decl.preference, value, maximum))
```

訊息格式沿用 `"{label} {value} exceeds the server maximum of {maximum}"`；label 改用點路徑（`dwave_qpu.annealing_time_us`）。既有測試只斷言訊息內含 requested / maximum 數值，不斷言 label 文字，故不需改。

### 12.2 flag 驅動的兩個編譯後檢查（不變，但改讀通用 key）

- `caps.exhaustive` 且 `compiled.num_variables > policy.limit("variables")` → `EXACT_VARIABLE_LIMIT`
- `caps.remote and caps.supports_time_limit` 且 `backend.resolve_time_limit(...) > policy.limit("time_seconds")` → `REMOTE_TIME_LIMIT`

這兩個是「需要 compiled 或 backend 才算得出來」的限制，無法用純 preference 宣告，所以維持 flag 驅動；規格明列這是唯二例外。

### 12.3 `ExecutionPolicy.limits_for(caps)`

```python
result = {}
if caps.exhaustive:
    result["max_variables"] = self.limit("variables")
for decl in caps.parameter_limits:
    result[f"max_{decl.limit}"] = self.limit(decl.limit)
if caps.remote and caps.supports_time_limit:
    result["max_time_seconds"] = self.limit("time_seconds")
return result
```

四個既有 backend 的輸出與 Phase 2 `_policy_limits` **逐 key 相同**（`max_variables` / `max_reads` + `max_annealing_time_us` / `max_time_seconds`），既有 capabilities 測試不需改期望值。`interfaces/capabilities.py` 的 `_policy_limits` 刪除，改呼叫 `policy.limits_for(caps)`。CLI `capabilities` 表格格式不變。

---

## 13. 架構測試（Step 0 的驗收）

### 13.1 `tests/architecture/test_fifth_backend.py`

`tests/fakes/declared_backend.py` 定義 `FakeDeclaredBackend`：`name="fake_declared"`、`remote=True`、`supports_num_reads=True`、`supports_time_limit=False`、`supported_model_types=["bqm"]`、無 option block，
`parameter_limits=[ParameterLimit(preference="num_reads", limit="iterations", error_code="FAKE_ITERATIONS_LIMIT")]`（**自訂 key**，證明 §11 的 `limits` 通道端到端可用；`FAKE_ITERATIONS_LIMIT` 只在測試內註冊到 catalog，或以 `catalog_error` 對未知 code 回 `recommended_action=None` 的既有行為處理），`is_available()` 永遠 available，`solve()` 回一個固定 feasible sample。以 `SolverRegistry.default()` 的五個 backend + 它組自訂 registry，policy 為 `ExecutionPolicy(allow_remote=True, limits={"iterations": 1000})`，斷言：

- `build_capabilities(...)` 對它回 `limits == {"max_iterations": 1000.0}`（漂移 1 修正：capabilities 與 service 同源）。
- `num_reads=5000` solve → `resource_limit_exceeded` / `FAKE_ITERATIONS_LIMIT`；`num_reads=10` → `success`。
- `service.validate()` 對它：帶 `seed` → `SEED_IGNORED`（不再需要名單）。
- `service.recommend()` 的清單含它且 `usable=True`。
- 上述全部**不修改** `optimizer.py`、`limits.py`、`routing.py`、`capabilities.py`、`problem_validator.py`、`policy.py`（測試本身就是證明：這些檔案裡沒有它的名字）。
- 同一個 fake 搭配 `ExecutionPolicy(allow_remote=True)`（沒有 `iterations`）→ `OptimizationService(...)` 建構時 `ValueError`，訊息含 backend 名與 key。

### 13.2 `tests/architecture/test_no_backend_names.py`

以 AST 掃描下列檔案，規則：任何 `ast.Constant` 字串節點的**整值**等於 `"exact"`、`"simulated_annealing"`、`"dwave_qpu"`、`"leap_hybrid_bqm"`、`"leap_hybrid_cqm"` 之一即違規（整值相等，不是子字串；docstring 與註解不在範圍——docstring 是 `Expr` 陳述式裡的 Constant，掃描時排除）：

- `orchestration/**`、`validation/**`、`interfaces/capabilities.py`、`interfaces/mcp/tools.py`、`interfaces/cli/main.py`

允許出現的位置：`models/problem.py`（`SolverPreferences.backend` Literal 與 option block 欄位名）、`models/error_catalog.py`、`solvers/<backend>.py`（自己的 `name`）、`solvers/registry.py`、README、測試。

整值規則不會抓到訊息文字裡的名字，所以另外要求（人工 + code review，不靠測試）：`problem_validator.py` 的 `_WARNING_RECOMMENDED_ACTIONS`（`EXACT_NEAR_LIMIT` / `EXACT_OVER_LIMIT` / `DENSE_FOR_QPU` 目前提到 `simulated_annealing`、`leap_hybrid_bqm`）與 `optimizer.py` 的 `EXACT_VARIABLE_LIMIT` 訊息（「exact solver limit」）改寫成能力描述（「the exhaustive backend」「a local heuristic backend」「a hybrid backend」），與 §9.3 的訊息規則一致。`models/error_catalog.py` 的 `recommended_action` 文字可以提 backend 名（它是給 Agent 的固定詞彙，且在 models 層）。

### 13.3 `test_import_boundaries.py` 新增

- `orchestration/**` 不得 import `annealbridge.solvers.exact` / `.simulated_annealing` / `.dwave_qpu` / `.leap_hybrid_bqm` / `.leap_hybrid_cqm`（可 import `annealbridge.solvers`、`.base`、`.registry`、`.metadata`）。
- `annealbridge.compiler.bqm` / `.cqm`（以及 `from annealbridge.compiler import BQMCompiler, CQMCompiler`）只允許出現在 `orchestration/optimizer.py`；`orchestration/routing.py`、`orchestration/limits.py`、`orchestration/policy.py` 不得 import 它們。

---

## 14. `CompiledProblem` 多型別與 `ModelCompiler` 擴充

```python
class ConstraintTrace(BaseModel):
    constraint_id: str
    constraint_type: Literal["hard", "soft"]
    operator: Literal["==", "<=", ">="]
    source_description: str | None
    generated_variables: list[str]
    penalty: float | None            # 原 float；CQM hard constraint 為 None，soft 為 weight
    slack_range: int | None
    redundant: bool = False
    native: bool = False             # 新：True = 以模型原生 constraint 表達，無 penalty 項
    compiler: str

class CompiledProblem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_type: ModelType = "bqm"    # 新，預設 bqm 保相容
    model: Any
    original_problem: OptimizationProblem
    internal_variables: set[str]
    constraint_trace: list[ConstraintTrace]
    hard_penalty: float | None       # 原 float；CQM 為 None
    objective_scale: float
    num_variables: int
```

```python
class ModelCompiler(Protocol):
    @property
    def model_type(self) -> ModelType: ...
    @property
    def uses_hard_penalty(self) -> bool: ...
    def compile(self, problem: OptimizationProblem, hard_penalty: float | None) -> CompiledProblem: ...
```

- `BQMCompiler`：`model_type="bqm"`、`uses_hard_penalty=True`；`hard_penalty is None` → `CompilationError`（呼叫端錯誤，不是問題錯誤）。輸出與 Phase 1/2 **位元一致**（既有 `test_bqm_compiler.py` 不改）。
- `CQMCompiler`：`model_type="cqm"`、`uses_hard_penalty=False`；`hard_penalty` 必須為 None，否則 `CompilationError`。

---

## 15. `CQMCompiler`（`compiler/cqm.py`）

### 15.1 輸出

`dimod.ConstrainedQuadraticModel`：

1. 每個 `Variable` → `cqm.add_variable("BINARY", name)`（3a 只有 binary；3b 在此分支加 `INTEGER`，§32.1）。
2. Objective：與 `BQMCompiler._compile_objective` 相同的符號規則（maximize 全數取負、含 constant）建成 `dimod.BinaryQuadraticModel`，`cqm.set_objective(bqm)`。這段抽成 `compiler/objective.py` 的 `build_objective_bqm(objective) -> BinaryQuadraticModel` 供兩個 compiler 共用（兩個實作共用 → 允許新模組）。
3. 每條 constraint：
   - `coefficients = accumulate_terms(constraint.terms)`，去除 0 係數。
   - 若無非零係數：constraint 是常數。`0 <op> rhs` 成立 → trace `redundant=True`、不加入；不成立 → `CompilationError`（validator 的 `TRIVIALLY_INFEASIBLE` 應已擋下，這是一致性防線，比照 `COMPILATION_FAILED` 語意）。
   - 否則 `lhs = dimod.BinaryQuadraticModel(coefficients, {}, 0.0, "BINARY")`，
     `cqm.add_constraint_from_model(lhs, sense=constraint.operator, rhs=constraint.rhs, label=constraint.id, weight=None | constraint.weight, penalty="quadratic")`。
   - hard：`weight=None`（dimod 語意：必須滿足）；trace `penalty=None, native=True`。
   - soft：`weight=constraint.weight, penalty="quadratic"`；trace `penalty=weight, native=True`。`penalty="quadratic"` 的能量貢獻為 `weight × violation²`，與 `solution_validator` 的 `weighted_penalty` 同一公式（§21.2 測試證明）。dimod 文件註明 `"quadratic"` 只適用 binary 變數，3b 加整數時必須重新決定（§32.1）。
   - label 用 `constraint.id`；validator 已保證唯一。
4. `CompiledProblem(model_type="cqm", model=cqm, internal_variables=set(), hard_penalty=None, objective_scale=compute_objective_scale(objective), num_variables=len(cqm.variables))`。注意 dimod 0.12.22 的 `ConstrainedQuadraticModel.num_variables` 是**方法**不是屬性（實測），直接塞會是 bound method；用 `len(cqm.variables)`。

### 15.2 不做的事

- 不做 slack、不做 squared penalty、不呼叫 `PenaltyStrategy`。
- 不呼叫 `dimod.cqm_to_bqm`（那是把 CQM 降級成 BQM，違反本路徑存在的意義；測試可以用它做交叉驗證，production 不可）。
- 不改 `NON_INTEGER_INEQUALITY`（§21.3）。
- 不提供 compiler option。

### 15.3 Determinism

同一 problem 兩次 compile，`cqm.variables` 順序、`constraint_labels` 順序、每條 constraint 的 `lhs` 係數相同；`objective_scale` 與 BQM 路徑相同（同一函式）。

---

## 16. Service：compiler 選擇、penalty 與 retry 對 CQM

### 16.1 建構與選擇

```python
class OptimizationService:
    def __init__(
        self,
        compilers: Iterable[ModelCompiler] | None = None,   # 取代 Phase 2 的 compiler=
        penalty_strategy: PenaltyStrategy | None = None,
        registry: SolverRegistry | None = None,
        policy: ExecutionPolicy = ExecutionPolicy(),
    ) -> None:
        compilers = list(compilers) if compilers is not None else [BQMCompiler(), CQMCompiler()]
        # 以 model_type 建索引；重複 → ValueError
```

- `compiler=` 關鍵字**移除**（唯一使用者 `test_service_validation_consistency.py` 改為 `compilers=[FakeFailingCompiler()]`，fake 必須宣告 `model_type="bqm"` **與** `uses_hard_penalty=True`，否則 §16.2 步驟 9 / §16.3 讀屬性會 AttributeError）。
- `_select_model_type(caps) -> ModelType | None`：依 `caps.supported_model_types` 順序取第一個有 compiler 的 type；`_select_compiler(caps)` 據此取 compiler。都沒有 → `configuration_error` / `NO_COMPILER_FOR_MODEL_TYPE`（新 code）。**這是 service 唯一依 model type 分派的地方，且依 capabilities，不依名稱。** `service.validate()`（§10）與 routing（§23.3）用同一個 `_select_model_type`，三處對「會走哪條路徑」的答案一致。
- 選定的 `compiled.model_type` 記入 log 與（§22）metadata。

### 16.2 流程（Phase 2 §14 的 3a 版）

```text
1. validate problem                                → invalid_problem
2. registry.get(backend)                           → backend_unavailable (UNKNOWN_BACKEND)
3–5. gate_errors(...)（enabled / allow_remote / availability by category）
6. concurrency slot
7. compiler = _select_compiler(caps)               → configuration_error (NO_COMPILER_FOR_MODEL_TYPE)
8. preference_limit_errors(caps, prefs, policy)    → resource_limit_exceeded（宣告驅動）
9. λ = strategy.initial_penalty(problem) if compiler.uses_hard_penalty else None
10. compile(problem, λ)
11. exhaustive → num_variables <= limit("variables")；remote+time_limit → resolve_time_limit <= limit("time_seconds")
12. solve → solver_error（含分類 code）
13. process_candidates（不變：去重、independent validator、ranking）
14. feasible → success
15. exhaustive → infeasible (proven=True)；**維持** Phase 2 條件：只有 raw.num_samples > 0 才算 proven
16. not compiler.uses_hard_penalty → infeasible (proven=False)，attempt 恆 1，無 REMOTE_RETRIES_DISABLED warning
17. remote and not allow_remote_retries → infeasible + REMOTE_RETRIES_DISABLED（僅 uses_hard_penalty 路徑；**維持** max_retries > 0 才發 warning）
18. attempt > max_retries → infeasible
19. λ = next_penalty；回到 10
```

步驟 3–5 的 gate 順序、availability 的 lazy 評估、回報的 backend 名稱規則，全部沿用 Phase 2（見 §12 `gate_errors` docstring）。

### 16.3 `_max_attempts`

```python
if backend.is_exhaustive: return 1
if not compiler.uses_hard_penalty: return 1      # 沒有可調的 lever，重試無意義
if caps.remote and not policy.allow_remote_retries: return 1
return 1 + preferences.max_retries
```

CQM 路徑 infeasible 的 message：`"No feasible solution found: the constraint-model backend returned no sample satisfying the hard constraints under independent validation; infeasibility is not proven"`。

### 16.4 `SolveAttempt.penalty: float | None`

CQM 路徑記 `None`。JSON 輸出多一個 `null` 的可能，Phase 1/2 的既有結果形狀不變。CLI `_render_human` 的 infeasible 區塊對 `None` 印 `penalty=-`。

### 16.5 不變的事

`process_candidates`、`deduplicate_samples`、`validate_batch`、ranking、tie-break 一行不改。CQM sampleset 的 `is_feasible` 欄位**不進入**這條管線（§17.3）。

---

## 17. `LeapHybridCQMBackend`（`solvers/leap_hybrid_cqm.py`）

### 17.1 Capabilities

```python
SolverCapabilities(
    name="leap_hybrid_cqm", remote=True, heuristic=True, exhaustive=False,
    supports_seed=False, supports_num_reads=False, supports_time_limit=True,
    supported_model_types=["cqm"], returns_multiple_samples=True,
    parameter_limits=[ParameterLimit(preference="leap_hybrid_cqm.time_limit_seconds", limit="time_seconds", error_code="REMOTE_TIME_LIMIT")],
    description=(
        "D-Wave Leap cloud hybrid constrained-quadratic-model solver. Hard constraints are "
        "submitted natively (no penalty, no slack) and soft constraints as weighted "
        "constraints; the solver returns several samples. The solver's own feasibility flag "
        "is ignored: every sample is re-validated against the original problem. Leap limits "
        "(5,000,000 variables, 100,000 constraints, minimum time_limit 5 s) are enforced by "
        "the solver, not by this server."
    ),
)
```

`returns_multiple_samples=True`：CQM hybrid solver 回傳多個 sample（與 BQM hybrid 的單一 sample 不同），top_k 有意義。

### 17.2 Sampler

- `_default_sampler_factory`：`from dwave.system import LeapHybridCQMSampler; return LeapHybridCQMSampler()`，lazy import。
- `resolve_time_limit`：`max(user, sampler.min_time_limit(cqm))`，使用者未給則 `min_time_limit`；規則、快取、例外包裝與 `LeapHybridBQMBackend` 完全相同（共用 §17.5 helper）。
- `solve`：`sampler.sample_cqm(cqm, time_limit=effective)`，`_resolved()` 強制解析。**不傳 `label`**（problem name 屬業務資料，不送到 Leap dashboard）。
- 不轉送 `num_reads` / `num_sweeps` / `seed` / `penalty_multiplier`。

### 17.3 SampleSet → RawSolverResult

- CQM sampleset 的 `record.sample` dtype 依 sampler 而定（`ExactCQMSolver` 實測 `int64`，Leap 回傳可能為 float）。**在 `LeapHybridCQMBackend.solve()` 內、呼叫共用的 `sampleset_to_arrays` 之前**先斷言 `np.all(np.isin(sampleset.record.sample, (0, 1)))`，否則 `SolverExecutionError(code="REMOTE_SOLVER_ERROR")`；共用 helper `solvers/base.py::sampleset_to_arrays` 本身**不改**（四個既有 backend 的行為與測試不受影響）。3b 加整數變數時，改的是這個 backend 內的斷言與 `RawSolverResult` 的 dtype（§32.1）。
- **不得**以 `record.is_feasible` 過濾或排序（原則 2）；全部 sample 原序回傳，service 的 independent validator 重驗。
- `metadata.sampler_reported_feasible = int(record.is_feasible.sum())`（§22），純供診斷：validator 與 sampler 對 feasible 的判定若不一致，這個數字讓人看得出來。
- `sampleset.info` 經 `sanitize_sampleset_info`（whitelist：hybrid 的 `run_time`、`charge_time`、`qpu_access_time` 已在 whitelist 內）；`effective_time_limit_seconds` 記入。

### 17.4 例外分類

沿用 `leap_hybrid_bqm` 的兩張表（§17.5 移到共用模組）：`SolverAuthenticationError → REMOTE_AUTH_FAILED`、`RequestTimeout → REMOTE_TIMEOUT`、sampler 建構階段的 `SolverNotFoundError` / `ConfigFileError` / `ValueError → DWAVE_CONFIG_INVALID`，其餘 `REMOTE_SOLVER_ERROR`。訊息一律 `redact()`。

### 17.5 `solvers/ocean.py`（三個 D-Wave backend 共用）

現在 `dwave_qpu.py` 與 `leap_hybrid_bqm.py` 各有一份 `_call_ocean`、`_resolved`、sampler 快取（factory + lock）與 `_SAMPLER_INIT_EXCEPTION_CODES`。第三個 backend 進來就是三份，因此抽到 `solvers/ocean.py`：

```python
SAMPLER_INIT_EXCEPTION_CODES: dict[str, str]      # 三者相同
HYBRID_SAMPLE_EXCEPTION_CODES: dict[str, str]     # leap_hybrid_bqm / cqm 相同；QPU 自己保留含 EmbeddingError 的表
def call_ocean(what: str, codes: dict[str, str], fn: Callable[[], T]) -> T
def resolved(sampleset: Any) -> Any
class LazySampler:                                 # factory + lock + 只快取成功建構
    def __init__(self, factory: Callable[[], Any] | None, default_factory: Callable[[], Any]) -> None
    def get(self) -> Any
```

`dwave_qpu.py` / `leap_hybrid_bqm.py` 改用；行為不變，既有 mock 測試不改期望值。`ocean.py` 不得 module 層 import `dwave.*`。

### 17.6 Leap 官方上限

查證（2026-09-02，D-Wave 文件 `solver_cqm_properties`）：`maximum_number_of_variables=5,000,000`、`maximum_number_of_constraints=100,000`、`maximum_number_of_biases=750,000,000`、`minimum_time_limit_s=5`、`maximum_time_limit_hrs=24`。這些**不放進** `BackendCapability.limits`（該欄位依 Phase 2 §22 定義是「本伺服器 policy 的上限」，混入 solver 原生上限會讓 Agent 無法分辨哪個是 operator 可調的），只寫進 `description` 與 README；validator 不檢查（超過的問題早就超出本專案任何路徑的合理範圍）。solver 拒絕時走 `REMOTE_SOLVER_ERROR`。

### 17.7 Registry

`SolverRegistry.default()` 註冊第五個 backend `leap_hybrid_cqm`。**註冊順序固定**：`exact`、`simulated_annealing`、`dwave_qpu`、`leap_hybrid_bqm`、`leap_hybrid_cqm`（前四個是 Phase 2 既有順序；capabilities 列表、CLI 表格與 routing 的最後 tie-break 都依此順序）。CI 的 minimal-install job 印出的 `names()` 應含它，且不需 `dwave-system`。

既有測試更新：`tests/mcp/test_capabilities.py`（backend 名單 `sorted(...) == [四個]`）與 `tests/unit/test_cli.py`（`capabilities` 表格列集合）改為五個。

---

## 18. `SolverPreferences` 擴充（向後相容）

```python
class LeapHybridCQMOptions(BaseModel):
    time_limit_seconds: float | None = _FINITE

class SolverPreferences(BaseModel):
    backend: Literal["simulated_annealing", "exact", "dwave_qpu", "leap_hybrid_bqm", "leap_hybrid_cqm"] = "simulated_annealing"
    ...（其餘不變）
    leap_hybrid_cqm: LeapHybridCQMOptions | None = None
```

- `OptimizationProblem.version` 仍 `Literal["1.0"]`。
- `test_phase1_compat.py` 加一條：三個 example 的 `solver.leap_hybrid_cqm is None`；example JSON 不改一字。
- `models/__init__.py` 匯出 `LeapHybridCQMOptions`、`SolverCapabilities`、`AvailabilityStatus`、`ParameterLimit`、`ModelType`。

---

## 19. 其他模型調整彙整

| 模型 | 變更 | 相容性 |
|---|---|---|
| `SolveAttempt.penalty` | `float` → `float | None` | 加寬，舊輸出仍合法 |
| `ConstraintTrace.penalty` | `float` → `float | None`；新增 `native: bool = False` | 加寬 + 新預設欄位 |
| `CompiledProblem` | 新增 `model_type`（預設 `"bqm"`）；`hard_penalty: float | None` | 加寬 |
| `ProblemValidationResult` | 新增 `model_type: ModelType | None = None` | 新預設欄位 |
| `SolverExecutionMetadata` | 新增 `model_type: ModelType | None = None`、`sampler_reported_feasible: int | None = None` | 新預設欄位 |
| `SolverBackend.is_available()` | 回傳 `AvailabilityStatus` | **Protocol 破壞性變更**，全部 backend、fakes、測試同步改 |

---

## 20. Error / warning codes 新增

| code | 類型 | status | 何時 |
|---|---|---|---|
| `BACKEND_CONFIG_INVALID` | error | `configuration_error` | availability category `config_invalid` 且 backend 未指定 `error_code` |
| `NO_COMPILER_FOR_MODEL_TYPE` | error | `configuration_error` | backend 宣告的 model types 沒有任何一個有 compiler |
| `UNKNOWN_BACKEND` | warning（新增 warning 用法） | — | `service.validate()` 對自訂 registry 找不到 backend；**共用**既有 catalog 條目與 `recommended_action`，訊息由 validator 產生，`test_error_catalog.py` 的 code 總數不因此 +1 |

`recommended_action`：

- `BACKEND_CONFIG_INVALID`：「The backend's configuration is present but invalid; fix or remove it on the server, or use a local backend.」
- `NO_COMPILER_FOR_MODEL_TYPE`：「The server has no compiler for the model types this backend accepts; this is a server configuration error—report it, or choose another backend.」

既有 warning code 全部保留；`PARAMETER_IGNORED`、`EXACT_NEAR_LIMIT`、`EXACT_OVER_LIMIT`、`DENSE_FOR_QPU` 的 `recommended_action` 文字改為能力描述（§13.2）。`test_error_catalog.py` 的覆蓋檢查擴到 routing reason codes（§23.4）以外的全部新 code；該檔對 code 總數的斷言（目前 `== 32`）**在每個新增 code 的開發步驟同步更新**，不是留到最後。

---

## 21. Validator 與 CQM 路徑

### 21.1 estimate

§9.2：`model_type == "cqm"` 時 `estimated_compiled_variables = len(variables)`，`LARGE_SLACK_RANGE` 仍會產（它描述的是 BQM 路徑的代價，對選 CQM 的人是「你選對了」的資訊；訊息文字改為 model-type 中立：「would need N slack bits on a BQM backend」）。

### 21.2 soft weight 語意一致性測試（必做）

`test_cqm_compiler.py`：對一條 soft constraint、取違反量 v ≥ 2 的 assignment（v = 1 分不出 linear 與 quadratic），斷言 `cqm.violations(sample)[constraint.id] == v`，且 `ExactCQMSolver` 回傳該 sample 的 `energy == sign × objective_value + constraint.weight * v * v`，而 `constraint.weight * v * v == validate_solution(...).evaluations[i].weighted_penalty`。實測（dimod 0.12.22，weight 10、v 2、objective −3）：`penalty="quadratic"` energy 37、`"linear"` energy 17，證明 quadratic 對應 `weight × v²`。這證明 solver 看到的偏好強度與 ranking 用的分數是同一件事（原則 3 在 CQM 路徑的體現）。

### 21.3 `NON_INTEGER_INEQUALITY` 維持為錯誤（刻意保守）

CQM 路徑其實不需要整數係數，但 `validate_problem` 的 errors 是 backend 無關的（Agent 換 backend 不該改變「問題有沒有寫錯」的判定），放寬它等於讓 errors 依 capabilities 而變。3b 的整數變數會再碰到同一個問題（整數變數的不等式係數規則），屆時一併決定；3a 在 README Limitations 明寫。

---

## 22. Metadata

`SolverExecutionMetadata` 新增：

- `model_type: ModelType | None`：service 在 `raw.metadata` 回來後填入（backend 不需要知道）；`raw.metadata is None` 時 service 不憑空建立 metadata（維持 Phase 2 行為：本機 backend 無 metadata）。
- `sampler_reported_feasible: int | None`：只有 CQM backend 填。

`sanitize_sampleset_info` 不變。

---

## 23. Solver routing（`orchestration/routing.py`，決策 4）

### 23.1 定位

**只建議、不代選。** `solve_optimization` 行為完全不變：使用者指定什麼就用什麼，不可用就回 `backend_unavailable`。routing 的輸出是給 Agent（或人）看的排序清單與理由；沒有任何程式路徑把推薦結果自動餵回 `solve`。

### 23.2 模型（`validation/recommendation.py`，純 Pydantic，供 interfaces 引用）

```python
class BackendRecommendation(BaseModel):
    rank: int
    backend: str
    usable: bool                          # enabled 且 available 且無 blocking limit
    model_type: ModelType | None          # 該 backend 會走的 compiler；無 compiler → None
    reasons: list[str]                    # §23.4 reason codes，依套用順序
    blocking: list[SolveError] = []       # 若現在 solve 會失敗，原因（REMOTE_DISABLED、QPU_READS_LIMIT…）
    warnings: list[SolveError] = []       # service.validate() 在該 backend 下的 warnings
    estimated_compiled_variables: int | None = None

class BackendRecommendationResult(BaseModel):
    valid: bool
    errors: list[SolveError] = []         # 問題本身的錯；有錯則 recommendations 為空
    recommendations: list[BackendRecommendation] = []
    advisory: str                         # 固定句：本結果僅供參考，solve_optimization 只用 solver.backend
```

### 23.3 演算法（deterministic、無網路、不 solve、不佔 slot）

```python
def recommend(problem, registry, policy, compilers: dict[ModelType, ModelCompiler]) -> BackendRecommendationResult
```

1. `validate_problem(problem)` 有錯 → `valid=False`，無推薦。
2. 對 registry 每個 backend（registry 順序）：
   - `caps = backend.capabilities`。
   - `blocking = gate_errors(name, backend, policy)` 的 errors（同 service，lazy 呼叫 `is_available()`，無網路）+ `preference_limit_errors(caps, problem.solver, policy)`（把使用者的 preferences 套到每個候選 backend 上，例如 `num_reads=5000` 對 QPU 是 blocking，對 SA 不是）。
   - `model_type`：與 service 同一個 `_select_model_type(caps)` 規則（依 `compilers` 取第一個有 compiler 的 type）；無 → blocking `NO_COMPILER_FOR_MODEL_TYPE`、`model_type=None`。
   - `validation = validate_problem_full(problem, capabilities=caps, max_compiled_variables=int(policy.limit("variables")), model_type=model_type)`；`caps.exhaustive` 且 `EXACT_OVER_LIMIT` 在 warnings → 轉為 blocking `EXACT_VARIABLE_LIMIT`。
   - `usable = not blocking`。
3. 排序 key（tuple，ascending）：
   1. `0 if usable else 1`
   2. `1 if "DENSE_FOR_QPU" in warning codes else 0`
   3. tier：
      - `caps.exhaustive` 且無 `EXACT_*` warning → 0（`R_EXACT_FITS`：可證最佳、免費）
      - `caps.exhaustive` 且 `EXACT_NEAR_LIMIT` → 1（`R_EXACT_NEAR_LIMIT`）
      - `caps.exhaustive` 且 `EXACT_OVER_LIMIT`（已轉 blocking） → 1（`R_EXACT_OVER_LIMIT`；反正 key 1 已把它排到 unusable 區）
      - 本機 heuristic → 2（`R_LOCAL_HEURISTIC`）
      - 遠端且 `model_type == "cqm"` 且 problem 有 ≥ 1 條 hard constraint → 3（`R_NATIVE_CONSTRAINTS`）
      - 其他遠端 → 4（`R_REMOTE`）；其中 `not caps.returns_multiple_samples` 且 `top_k > 1` 加 reason `R_SINGLE_SAMPLE`（不影響 tier）
   4. registry 順序
4. `rank` = 排序後位置（1 起）。`reasons` 依序記錄套用到的 code（含 `R_UNUSABLE` 與 `R_DENSE_FOR_QPU`）。

規則刻意簡單：能用 exact 證明最佳解就先用；免費的本機 heuristic 其次；有硬限制時 CQM 比 BQM 遠端優先（無 penalty/slack 失真）；embedding 會炸的 QPU 往後排。不做成本估算、不做 benchmark、不看歷史。

### 23.4 Reason codes（固定詞彙，`routing.py` 內 `REASON_DESCRIPTIONS` dict，測試檢查每個 code 都有說明）

| code | 說明 |
|---|---|
| `R_UNUSABLE` | 現在 solve 會失敗，見 `blocking` |
| `R_EXACT_FITS` | 編譯後變數在 exhaustive 上限內，可證最佳解與不可行性 |
| `R_EXACT_NEAR_LIMIT` | 在上限內但接近，執行時間與記憶體可觀 |
| `R_EXACT_OVER_LIMIT` | 編譯後變數超過 exhaustive 上限，solve 會被拒 |
| `R_LOCAL_HEURISTIC` | 本機 heuristic，免費、可重試、不保證最佳 |
| `R_NATIVE_CONSTRAINTS` | hard constraint 以模型原生表達，無 penalty / slack |
| `R_REMOTE` | 遠端，消耗 quota |
| `R_SINGLE_SAMPLE` | 通常只回一個 sample，top_k 實際 ≤ 1 |
| `R_DENSE_FOR_QPU` | 問題對 minor-embedding 而言可能過密 |

### 23.5 `OptimizationService.recommend(problem)`

一行委派 `recommend(problem, self._registry, self._policy, self._compilers)`。

---

## 24. MCP tool — `recommend_backend`

```python
@mcp.tool()
async def recommend_backend(problem: OptimizationProblem) -> BackendRecommendationResult:
    """Rank the solver backends of this server for a given problem, without solving it.

    Advisory only: solve_optimization always uses problem.solver.backend exactly as
    given and never substitutes a backend. Each entry reports whether the backend is
    usable right now (installed, credentialed, permitted by policy, within limits),
    the model type it would compile to, deterministic reason codes, blocking errors,
    and the same warnings validate_optimization_problem would give for that backend.
    Based only on the problem structure, backend capabilities and server policy —
    no cost estimation, no network requests, no quota consumed.
    """
    state = get_state()
    return state.service.recommend(problem)
```

`tools/list` 變成四個 tool；`solve_optimization` docstring 補一句「Use recommend_backend to compare backends; the choice remains yours.」

---

## 25. CLI 擴充

```bash
annealbridge validate problem.json [--backend X] [--json]
annealbridge recommend problem.json [--json]
```

`recommend` 人類可讀輸出：

```text
Problem:   knapsack
Advisory:  recommendations only; `solve` uses solver.backend as given

Rank  Backend              Usable  Model  Reasons
1     exact                yes     bqm    R_EXACT_FITS
2     simulated_annealing  yes     bqm    R_LOCAL_HEURISTIC
3     leap_hybrid_cqm      no      cqm    R_UNUSABLE, R_NATIVE_CONSTRAINTS   [REMOTE_DISABLED]
4     dwave_qpu            no      bqm    R_UNUSABLE, R_REMOTE   [REMOTE_DISABLED]
5     leap_hybrid_bqm      no      bqm    R_UNUSABLE, R_REMOTE, R_SINGLE_SAMPLE   [REMOTE_DISABLED]
```

（knapsack 不 DENSE；`dwave_qpu` 與 `leap_hybrid_bqm` 同為 unusable、tier 4，依 §17.7 的 registry 順序 `dwave_qpu` 在前。）

exit code：問題無效 → 1；否則 0（推薦清單本身沒有失敗）。CLI 與 MCP 共用同一個 `service.recommend()`；CLI 不含任何排序邏輯。

---

## 26. Tests

### 26.1 Unit

- `test_capabilities_model.py`：`ParameterLimit` / `AvailabilityStatus.available` / `preferred_model_type`；`supported_model_types` 空 → ValidationError。
- `test_availability.py`：`dwave_availability()` 四種 category（monkeypatch `find_spec` 與 `load_config`），`detail` 不含 config 值。
- `test_policy.py`：`limit()` 四個相容 key 讀舊欄位；`limits={"iterations": 5}` 可讀；`limits={"reads": 5}` → ValidationError（與相容 key 衝突）；`limits` 值 ≤ 0 / 非有限 → ValidationError；`limits_for()` 對四個 Phase 2 backend 的輸出逐 key 等於 Phase 2 期望，對 `leap_hybrid_cqm` 為 `{"max_time_seconds": 300}`。
- `test_settings.py`：`ANNEALBRIDGE_LIMITS='{"iterations": 100}'` 解析；非 JSON → 專案 `SettingsError`（來源是 `pydantic_settings.SettingsError`，§11.2）；舊 env 名全部仍生效。
- `test_limits.py`：`read_preference` 點路徑、block 未填回 None、路徑不存在 raise；`preference_limit_errors` 收齊多個超限、policy 缺 key → ValueError；`gate_errors` 三個 gate 順序，且前兩關擋下時 `is_available()` **未被呼叫**（spy），回報名稱規則（disabled-by-policy 用 registry key、其餘用 `caps.name`）。
- `test_problem_validator_full.py`：全部改成傳 `capabilities=`；§9.3 表每一列一個測試；`capabilities=None` 不產 backend warnings；exact + seed → `SEED_IGNORED`（漂移 2）；填錯 block（backend=exact 但給 `dwave_qpu` options）→ `PARAMETER_IGNORED`。
- `test_service_validate.py`：`service.validate()` 與直接呼叫 `validate_problem_full(capabilities=registry caps, max=policy)` 結果相等；自訂 registry 缺 backend → `UNKNOWN_BACKEND` warning。
- `test_cqm_compiler.py`：變數 / objective 符號（maximize 取負）/ hard 為 `weight=None` / soft 為 `weight, penalty="quadratic"` / label = id / `internal_variables == set()` / `hard_penalty is None` / 常數 constraint 的 redundant 與 CompilationError 兩支 / determinism / §21.2 語意一致性 / **交叉驗證**：對 knapsack 用 `dimod.ExactCQMSolver` 列舉全部 assignment，`is_feasible` 與 `validate_solution(...).feasible` 對每一筆完全一致（這裡是唯一允許讀 `is_feasible` 的地方，目的是驗 compiler）。實測（dimod 0.12.22）：`is_feasible` 只看 hard constraint，soft 違反的 sample 仍為 True，正是我們要的語意；**不要**用 `cqm.check_feasible()`，它把 soft constraint 也算進去。
- `test_service_cqm_flow.py`：`tests/fakes/local_cqm_backend.py` 的 `FakeLocalCQMBackend`（`name="fake_local_cqm"`、`supported_model_types=["cqm"]`、`remote=False`、`heuristic=False`、**`exhaustive=True`**（它確實列舉全部 assignment）、包 `dimod.ExactCQMSolver().sample_cqm`，回傳全部 sample、不過濾）。斷言：選到 `CQMCompiler`；`attempts == 1`、`penalty is None`；success 且 objective 與 exact 路徑相同；hard constraint 無 penalty（`constraint_trace[*].native`）；fake 回傳全部標 `is_feasible=True` 的 sample 中含違反 hard constraint 者 → service 仍過濾掉（不信任 sampler）；一個矛盾 constraint 的問題 → `infeasible` 且 `infeasibility_proven=True`（走 §16.2 步驟 15，因 exhaustive）；service 的 `compilers=[BQMCompiler()]` 時 → `configuration_error` / `NO_COMPILER_FOR_MODEL_TYPE`。
- `test_routing.py`：預設 policy（remote 關）→ exact 第一、SA 第二、三個遠端 `usable=False` 且 `blocking` 含 `REMOTE_DISABLED`；`allow_remote=True` + fake available 遠端 → CQM 排在 BQM hybrid 前（有 hard constraint）、無 constraint 時 hybrid_bqm 在 CQM 前；DENSE 問題 QPU 最後；`num_reads=5000` → QPU blocking `QPU_READS_LIMIT` 而 SA 不受影響；invalid problem → 無推薦；同一輸入兩次結果相等；每個 reason code 都在 `REASON_DESCRIPTIONS`。
- `test_cli.py`：`validate` 三種 exit code、`--json` 為合法 `ProblemValidationResult`；`recommend` 表格與 `--json`。
- `test_solvers.py` / `test_registry.py`：`is_available()` 回 `AvailabilityStatus`；registry 五個名稱。

### 26.2 Architecture

§13.1–§13.3 三個測試。

### 26.3 Scenario

`test_knapsack_cqm.py`：knapsack JSON → `OptimizationService(registry=含 FakeLocalCQMBackend)` → success、objective == 17、無 `__` 變數、`attempts[0].penalty is None`。

### 26.4 MCP（in-memory `Client`）

- `tools/list` 恰四個；`recommend_backend` output schema 對應 `BackendRecommendationResult`。
- `recommend_backend` 對 knapsack：`structured_content["recommendations"][0]["backend"] == "exact"`，三個遠端 `usable == False`。
- `recommend_backend` 不觸發任何 backend `solve()`（spy），不觸發 `resolve_time_limit()`。
- `validate_optimization_problem` 對 `backend=exact` + `seed` → warnings 含 `SEED_IGNORED`。
- `solve_optimization` 指定 `leap_hybrid_cqm` 且 policy 未開 → `backend_unavailable` / `REMOTE_DISABLED`、`solutions == []`。

### 26.5 Remote mock

`test_leap_hybrid_cqm_mock.py`（`FakeCQMSampler` 提供 `sample_cqm(cqm, time_limit=)`、`min_time_limit(cqm)`、`properties`，回 `dimod.SampleSet` 含 `is_feasible` 欄位，`info` 為 hybrid 型）：

- 傳入的是 `ConstrainedQuadraticModel`；hard constraint 的 `weight is None`。
- `time_limit` 轉送 == `resolve_time_limit`；使用者值低於 `min_time_limit` 時取 min 並記 `effective_time_limit_seconds`；高於 policy → `resource_limit_exceeded` / `REMOTE_TIME_LIMIT`（未 clamp）。
- 不傳 `label`、不傳 `num_reads`。
- fake 回傳 `is_feasible=True` 但違反 hard constraint 的 sample → 不進 solutions；`metadata.sampler_reported_feasible` 記 fake 的計數。
- 回傳非 {0,1} 值 → `solver_error` / `REMOTE_SOLVER_ERROR`。
- 例外分類四種；`allow_remote_retries` 開關對 CQM 路徑都只有 1 attempt 且**無** `REMOTE_RETRIES_DISABLED` warning。
- `test_credential_leak.py` 參數化涵蓋 `leap_hybrid_cqm`。

### 26.6 Remote live（opt-in，`@pytest.mark.remote`）

`test_leap_hybrid_cqm_live.py`：最小 knapsack、`time_limit_seconds=5`；斷言 success、無 token 外洩。conftest skip 規則與 Phase 2 §28 相同（允許且註解）。

---

## 27. README

- Architecture 圖加 `leap_hybrid_cqm` 與 compilers{bqm,cqm}。
- Solver Backends 表加一列；說明 CQM 路徑：hard constraint 原生、無 penalty / retry、solver feasible 旗標不採信、Leap 官方上限、`time_limit_seconds` 最小 5 s。
- MCP Server：四個 tool，`recommend_backend` 的定位（advisory only）。
- CLI：`validate`、`recommend`。
- Environment variables：加 `ANNEALBRIDGE_LIMITS`（JSON；3a 內建 backend 不需要）；舊變數全部保留。
- Limitations：加「CQM path still requires integer inequality coefficients（§21.3）」、「Integer variables: Phase 3b」。
- Testing：`pytest -m remote` 現含 CQM live 測試。

---

## 28. Logging 與 Security

- 每次 solve 記錄 `model_type`；CQM 路徑記 `hard_penalty=None`。
- routing 不記錄 problem 內容，只記 backend 數與第一名。
- Token 不得出現在任何輸出：CQM backend 走與其他遠端 backend 同一條 `redact()` 路徑；`test_credential_leak.py` 涵蓋。
- `sample_cqm` 不帶 `label`（§17.2）。

---

## 29. 不要做的事情（3a 特別注意）

- 在 service / validator / interfaces 寫 `if backend == "..."`（§13.2 會抓）
- 為了讓 CQM 路徑「也能 retry」而發明假的 penalty
- 用 `sampleset.filter(lambda d: d.is_feasible)` 或依 `is_feasible` 排序
- 用 `dimod.cqm_to_bqm` 在 production 路徑降級
- 把 Leap 官方上限塞進 `BackendCapability.limits`
- 新增 backend 命名的 `ExecutionPolicy` 欄位或 env 變數
- 讓 `recommend_backend` 的結果自動改寫 `problem.solver.backend`
- 讓 routing 打網路、呼叫 `resolve_time_limit()`、建立 sampler
- 動 `OptimizationProblem.version`、`Variable.type`（3b 的事）
- 為 3b 預先加 `type: "integer"` 分支、`FujitsuDAOptions` 等空殼（原則 7）
- 新增第四個 Protocol

---

## 30. Development Order

每步跑完相關測試（含 Phase 1/2 全部）再進下一步。

1. **Step 0a-1（capabilities 與 availability）**：`models/capabilities.py`（`SolverCapabilities` 搬家 + 新欄位含預設值、`AvailabilityStatus`、`ParameterLimit`、`ModelType`）；`solvers/base.py` re-export；SA 宣告 `supports_num_sweeps=True`、QPU 宣告 `requires_embedding=True`；`is_available()` 改回傳 `AvailabilityStatus`，必改位置：四個 backend、`dwave_availability()`、`interfaces/capabilities.py`（tuple 解構）、`tests/remote_mock/conftest.py::make_remote_available`、`tests/unit/test_capabilities_view.py`、`tests/unit/test_solvers.py`、`tests/remote_mock/test_*_mock.py` 的 `is_available()` 斷言、`test_service_remote_flow.py` 的 config-invalid / unknown-reason 案例（§8.2）；service `_AVAILABILITY_MAP` 改依 category；catalog 加 `BACKEND_CONFIG_INVALID` 並同步 `test_error_catalog.py` 總數。順手刪除 `interfaces/mcp/server.py` 三份重複的 `reset_state(build_state(settings))` 與 `interfaces/cli/main.py` 三份重複的 `_build_state` 定義。全綠。
2. **Step 0b-1（policy 通用 limits）**：`ExecutionPolicy.limit()` / `limits_for()` / `limits`；`ServerSettings.limits` + `load_settings` 捕捉 `pydantic_settings.SettingsError`；`orchestration/limits.py` 三個純函式；service `_gate_backend` / `_preference_limit_errors` 改用；四個既有 backend 宣告 `parameter_limits`；`interfaces/capabilities.py` 刪 `_policy_limits`；service `__init__` 的宣告/policy 一致性檢查。全綠，capabilities 測試期望值不變。（此步排在 validator 之前，因為 §10 的 `service.validate()` 需要 `policy.limit()`。）
3. **Step 0a-2（validator 注入）**：validator 改 `capabilities=` / `model_type=` 注入、§9.3 整張表（含 `model_type == "cqm"` 兩列，此時只有 fake 能觸發）、§9.2 estimate、§9.4 反射、刪名稱常數、§13.2 要求的訊息文字改寫；`service.validate()`（暫以 `caps.preferred_model_type` 當 `model_type`，步驟 5 再接 `_select_model_type`）；MCP validate tool 改一行；CLI `validate`。更新 `test_problem_validator_full.py` / `test_validate.py` 期望值（漂移 2 修正）。全綠。
4. **Step 0b-2（架構測試）**：`tests/fakes/declared_backend.py` + `test_fifth_backend.py`（此時尚無 `recommend()`，該斷言留到步驟 9 補）+ `test_no_backend_names.py` + import boundary 新規則。全綠。
5. `CompiledProblem.model_type` / `hard_penalty: None` / `ConstraintTrace.native`；`ModelCompiler` 擴充；`BQMCompiler` 宣告；`compiler/objective.py` 抽共用；service 改 `compilers=`（**此步預設暫為 `[BQMCompiler()]`**，步驟 6 再加 `CQMCompiler()`）、`_select_model_type` / `_select_compiler`、`service.validate()` 改用 `_select_model_type`；`SolveAttempt.penalty: None`；`_max_attempts` 加 `uses_hard_penalty`；catalog 加 `NO_COMPILER_FOR_MODEL_TYPE` 並同步總數；`FakeFailingCompiler` 加兩個屬性。`test_bqm_compiler.py` 不改仍綠。
6. `CQMCompiler` + `test_cqm_compiler.py`（含 ExactCQMSolver 交叉驗證、§21.2）；service 預設 compilers 加入 `CQMCompiler()`。
7. `tests/fakes/local_cqm_backend.py` + `test_service_cqm_flow.py` + `test_knapsack_cqm.py`（第一次有真實的 CQM 路徑走過 §9.2 estimate 與 §9.3 的 cqm 列，補對應斷言）。
8. `solvers/ocean.py` 抽共用（既有 mock 測試不改仍綠）→ `LeapHybridCQMOptions` + Literal → `LeapHybridCQMBackend` → registry（§17.7 順序；同步 `test_registry.py`、`tests/mcp/test_capabilities.py`、`tests/unit/test_cli.py` 的名單）→ `test_leap_hybrid_cqm_mock.py` → credential leak 參數化 → `test_phase1_compat.py` 加 CQM block 斷言 → metadata 新欄位。
9. `validation/recommendation.py` + `orchestration/routing.py` + `service.recommend()` + `test_routing.py` → MCP `recommend_backend` + `test_recommend.py` + `test_tools_list.py` 四個 → CLI `recommend` → 補 `test_fifth_backend.py` 的 `recommend()` 斷言。
10. error catalog 補齊 + `test_error_catalog.py` 覆蓋；README；`remote_live` CQM 測試；CI 不需改（minimal-install 印 names 會自動含新 backend）；acceptance 逐條核對。

---

## 31. Acceptance Criteria

- [ ] Phase 1、2 全部測試不變仍通過（除 §9.3 明列的 warning 期望值更新與 `is_available()` 型別更新）；三個 example JSON 不改一字仍可 parse；`version` 仍 `"1.0"`
- [ ] `grep` 測試：`orchestration/**`、`validation/**`、`interfaces/capabilities.py`、`interfaces/mcp/tools.py`、`interfaces/cli/main.py` 無 backend 名稱字串
- [ ] 假第五 backend 測試通過：capabilities `limits` 與 service 限制同源、validator 依 capabilities 產 warning，零改動 `optimizer.py` / `capabilities.py` / `problem_validator.py` / `policy.py`
- [ ] `exact` + `seed` 產 `SEED_IGNORED`（漂移 2 修正）
- [ ] 舊 env 變數名全部仍生效；`ANNEALBRIDGE_LIMITS` 可用；`limits` 與相容 key 衝突被拒
- [ ] backend 宣告 policy 沒有的 limit key → 建構時失敗，不會靜默無上限
- [ ] `is_available()` 回傳 `AvailabilityStatus`；service 依 category 對應 status，不比對 D-Wave 字串
- [ ] `service.validate()` 存在；MCP validate tool 與 CLI `validate` 皆為一行委派
- [ ] `CompiledProblem.model_type`；service 依 `capabilities.supported_model_types` 選 compiler；無 compiler → `NO_COMPILER_FOR_MODEL_TYPE`
- [ ] CQM 路徑：hard constraint `weight=None`、無 slack、`internal_variables == set()`、`hard_penalty is None`、`attempts == 1`、`SolveAttempt.penalty is None`、無 `REMOTE_RETRIES_DISABLED`、solutions 仍經 independent validator、`is_feasible` 不採信
- [ ] soft constraint：CQM `weight × violation²` 與 validator `weighted_penalty` 同公式（測試證明）
- [ ] `ExactCQMSolver` 交叉驗證：`is_feasible` 與 `validate_solution` 對全部 assignment 一致
- [ ] `LeapHybridCQMBackend`：lazy import、`time_limit` 轉送、min floor、policy 上限不 clamp、不傳 label、非 {0,1} 值報錯、例外分類、redaction；`SolverRegistry.default()` 五個 backend 且無 `dwave-system` 仍可 import
- [ ] `sampler_reported_feasible` 與 `model_type` 進 metadata；whitelist 不變
- [ ] `recommend_backend`：deterministic、無網路、不 solve、不佔 slot；`solve_optimization` 行為不變；CLI `recommend` 與 MCP 同源
- [ ] 四個 MCP tool；`tools/list` 測試更新；`structured_content` 為 dict
- [ ] 每個新 code 有 `recommended_action`；每個 reason code 有說明
- [ ] token 不進任何輸出；credential leak 測試涵蓋 CQM backend
- [ ] README 更新；`pytest` 全過，無 skip / xfail（除 remote_live 註解 skip）

---

## 32. 3b 接點（本文件只描述，不實作）

### 32.1 整數變數（IR 1.1）

3a 為它留下的位置，3b spec 必須逐一接上：

- `Variable.type` 從 `Literal["binary"]` 擴為 `Literal["binary", "integer"]` + `lower_bound` / `upper_bound`；`OptimizationProblem.version` → `Literal["1.0", "1.1"]`；1.0 JSON 不改一字。
- `CQMCompiler` §15.1 步驟 1 加 `cqm.add_variable("INTEGER", name, lower_bound=, upper_bound=)` 分支；soft constraint 的 `penalty="quadratic"` 只限 binary，含整數變數的 soft constraint 要改 `"linear"` 或另訂規則，並重做 §21.2 的一致性證明。
- `BQMCompiler` 加整數 → 二進位編碼（建議 binary expansion，編碼位元屬 internal `__` 前綴）與 decode；`estimate_compiled_variables` 納入編碼位元；`INTEGER_QUADRATIC_BLOWUP` warning。
- `sampleset_to_arrays` / `RawSolverResult.samples` 由 `int8` 放寬（§17.3 的 {0,1} 斷言要改為「在 bounds 內的整數」）；`_pack_rows` 去重要改為對整數列的通用 key；`evaluate_objective_batch` / `validate_batch` 已是算術，需測試證明對整數值正確。
- `NON_INTEGER_INEQUALITY` 是否依 model type 放寬（§21.3）在此一併決定。
- `SolverCapabilities` 可能需要 `supports_integer_variables: bool`；routing tier 加「有整數變數 + 二次項 → CQM 優先」規則。

### 32.2 Fujitsu Digital Annealer

- 查證（2026-09-02）：服務仍營運，API 文件 2026-07 更新，QUBO API 有 V3c / V4，另有 Storage API；3b spec 撰寫前仍須查證 SDK / Web API 版本、認證方式（API key 形式，用於 redaction regex）、`number_iterations` / `number_replicas` / `time_limit` 等參數，以及是否原生支援不等式 / 整數（若支援，是 §16.1 model-type 分派的第三個消費者）。
- 接點：`FujitsuDAOptions` block + backend Literal；`parameter_limits` 宣告新的 limit key（如 `iterations`），值只能來自 `ExecutionPolicy.limits` / `ANNEALBRIDGE_LIMITS`（§11），**不得**新增 `max_da_*` 欄位；availability 走 `AvailabilityStatus`（`not_installed` / `credentials_missing`）；`solvers/metadata.py` 的 redaction regex 加 Fujitsu key 形式；例外 → code 表自訂；`fujitsu` extra、mock 測試比照 `remote_mock`、live 比照 `remote_live`。
- 若查證後需付費帳號才能取得 API key 而不申請：只做介面 + mock，live 測試留空（決策 3）。

---

## 33. 偏離清單（明列）

### 33.1 對 Phase 2 spec 的偏離

| Phase 2 spec | 3a 變更 | 理由 |
|---|---|---|
| §37「Phase 3 不動 IR 與 interfaces」 | IR 不動；interfaces **加**一個 tool、兩個 CLI 指令（純新增，既有三個 tool 簽名不變） | routing 是 Agent-facing 功能，只能在 interfaces 露出；核心邏輯仍在 orchestration |
| §10 `is_available() -> tuple[bool, str | None]` | → `AvailabilityStatus` | 消除 service 對 D-Wave reason 字串的依賴（原則 4） |
| §20 `validate_problem_full(problem, *, exact_max_variables)` | → `capabilities=`、`max_compiled_variables=` | 消除 validator 內的 backend 名稱（原則 4） |
| §20 warning 表以 backend 名稱為條件 | 改以 capabilities 欄位為條件；`exact` 新增 `SEED_IGNORED` / `PARAMETER_IGNORED` | 修正漂移 2 |
| §22 `limits` 由 `_policy_limits` if-chain 產生 | 由 `policy.limits_for(caps)` 產生，輸出逐 key 相同 | 修正漂移 1 |
| §8 `ExecutionPolicy` | 欄位全保留，加 `limits` dict 與 `limit()` / `limits_for()` | 決策 2 |
| §14 step 9 以 `caps.remote and caps.supports_num_reads` 特判 | 改宣告驅動；exhaustive 變數上限與 effective time limit 兩項維持 flag 驅動 | 見 §12.2 |
| `OptimizationService(compiler=...)` | → `compilers=[...]` | 第二個 compiler |
| `SolveAttempt.penalty: float` | → `float | None` | CQM 無 penalty |
| §3「不做 CQM」 | 3a 做 CQM | Phase 3 範圍 |

### 33.2 對 outline（`annealbridge_phase3_spec_outline.md`）的偏離

| outline | 3a 變更 | 理由 |
|---|---|---|
| §4「`orchestration` 不得 import 任何具體 compiler 或 backend」 | backend 照禁；具體 compiler 只允許在 `optimizer.py` 建預設 `compilers` 清單那一處 import（§4、§13.3） | compiler 是核心元件；全禁就得再蓋 compiler registry，違反原則 7 |
| §2.2「Leap CQM 官方上限查證後列入 capabilities `limits`」 | 不進 `limits`，寫進 `description` 與 README（§17.6） | Phase 2 §22 定義 `limits` 是 operator 可調的 policy 上限；混入 solver 原生上限會讓 Agent 分不清哪個能請 operator 調 |
| §2.2「整數變數在 CQM 路徑直通」 | 3a 不做（§3、§32.1） | 決策 1：整數變數歸 3b |
| §1.2「`validate_problem_full(problem, *, capabilities, exact_max_variables)`」 | 參數改名 `max_compiled_variables` 並加 `model_type`（§9.1） | 名稱不該提 exact；model type 由 service 決定 |
