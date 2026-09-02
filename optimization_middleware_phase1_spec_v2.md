# Optimization Tool Middleware — Phase 1 開發規格 (v2)

> v2 變更摘要（相對於 v1）
> - 明訂 inequality constraint 的係數 / rhs 必須為整數（配合 slack encoding）
> - 明訂 soft constraint 的 penalty 形式與 weight 單位
> - Ranking 改為同時考慮 objective 與 soft violation（`ranking_score`）
> - Exact backend 跳過 retry，並可宣告 infeasibility 已被證明
> - ExactSolver 加入變數數量硬上限
> - 定義去重後 energy 歸屬、objective_value 語意
> - 補齊 Problem Validator 檢查項目，`version` 改為 Literal
> - 統一 constraint schema 為 `terms`（移除 `expression.terms` 寫法）
> - `penalty_multiplier` 可由 SolverPreferences 覆寫
> - Hard penalty 量級改以 `penalty_scale = objective_scale + soft 項能量上界` 計算（§18）：修正 soft weight 遠大於 objective 時 SA 找不到 feasible 解的問題；`objective_scale` 語意不變（仍為 objective-only，供 Agent 判讀 soft weight）

---

## 1. 專案目標

開發一個可供未來 AI Agent / RAG / MCP 系統調用的 **組合優化中間件（Optimization Tool Middleware）**。

核心理念：

**LLM 負責理解人類需求與產生結構化 Optimization Problem；後端負責 deterministic 的數學模型編譯、求解、約束驗證與重試。**

Phase 1 不實作 LLM，專注建立可靠、可測試、可獨立運作的 Optimization Core。

```text
Natural Language / Knowledge Base
             │
             ▼
            LLM
             │
             ▼
   OptimizationProblem JSON
             │
             ▼
       Schema Validation
             │
             ▼
       Problem Compiler
             │
             ▼
             BQM
             │
             ▼
        Solver Backend
       ┌─────┴─────┐
  Simulated      Exact
  Annealing      Solver
       │
       ▼
 Candidate Samples
       │
       ▼
 Solution Validator
       │
       ├── feasible ──► Ranking / Top-K
       │
       └── infeasible ──► Penalty Strategy ──► retry (SA only)
                                                  │
                                                  ▼
                                             SolveResult
```

---

## 2. 核心設計原則

### 2.1 LLM 不直接產生 QUBO

LLM 未來只能輸出 Variables、Objective、Constraints、business preference weights、Solver preferences。

LLM **不得產生 QUBO matrix、BQM bias 或 slack variables**。

LLM 應輸出（注意：constraint 使用扁平的 `terms`，沒有 `expression` 包裝層）：

```json
{
  "id": "daily_shift_limit",
  "type": "hard",
  "terms": [
    { "variable": "alice_morning", "coefficient": 1 },
    { "variable": "alice_night", "coefficient": 1 }
  ],
  "operator": "<=",
  "rhs": 1
}
```

QUBO / BQM 必須由 deterministic Python compiler 建立。

### 2.2 Feasibility 與 objective 一律由原始 problem 判定

BQM energy 只用於 solver 內部搜尋。是否滿足 constraint、business objective 是多少，一律回到原始 `OptimizationProblem` 重新計算。

### 2.3 Solver 是可替換元件

Domain model 不得與任何特定 sampler 硬綁。

---

## 3. Phase 1 Scope

必須完成：

1. Optimization Problem IR / Pydantic Schema
2. Problem Schema Validator
3. BQM Compiler（含自動 slack encoding）
4. Hard / Soft Constraint 支援
5. Automatic Penalty Strategy
6. Simulated Annealing Solver
7. Exact Solver
8. Solution Validator
9. Automatic Retry（僅 SA）
10. Top-K feasible solution ranking
11. CLI（solve / export-schema）
12. Unit Tests
13. Scenario Tests
14. Knapsack example
15. Assignment example
16. Compile trace / explainability metadata

---

## 4. Phase 1 明確不做

- LLM API / Prompt engineering
- MCP Server / FastMCP dependency
- D-Wave Leap API / 真實 QPU
- CQM Solver / OR-Tools
- Web UI / Database / Authentication
- RAG / Knowledge Base
- Integer / Real variables
- 非線性 constraint
- 自動選擇最佳 solver
- 自動正規化 soft weight（見 §10.3，Phase 1 只在 README 說明）
- 分散式運算 / Production deployment

Architecture 必須讓以上功能未來可加入，而不需重寫 core domain model。

---

## 5. 技術要求

```text
Python >= 3.11
Pydantic v2
dimod
dwave-samplers
Typer 或 argparse
pytest
```

```python
from dwave.samplers import SimulatedAnnealingSampler
import dimod
dimod.ExactSolver()
```

核心 domain layer（models / validation / compiler / penalty / orchestration）不得 import：FastMCP、D-Wave Cloud Client、LLM SDK、Web Framework。

---

## 6. 專案目錄

```text
optimization-tool/
├── pyproject.toml
├── README.md
├── src/optimizer/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── problem.py
│   │   ├── variable.py
│   │   ├── objective.py
│   │   ├── constraint.py
│   │   ├── solution.py
│   │   └── compiled.py
│   ├── compiler/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── slack.py          # slack encoding helper
│   │   └── bqm.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── simulated_annealing.py
│   │   └── exact.py
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── problem_validator.py
│   │   └── solution_validator.py
│   ├── penalty/
│   │   ├── __init__.py
│   │   └── strategy.py
│   ├── orchestration/
│   │   ├── __init__.py
│   │   └── optimizer.py
│   ├── exceptions.py
│   └── cli.py
├── examples/
│   ├── knapsack.json
│   └── assignment.json
└── tests/
    ├── unit/
    │   ├── test_models.py
    │   ├── test_problem_validator.py
    │   ├── test_slack.py
    │   ├── test_bqm_compiler.py
    │   ├── test_solution_validator.py
    │   ├── test_penalty_strategy.py
    │   └── test_solvers.py
    └── scenarios/
        ├── test_knapsack.py
        ├── test_assignment.py
        └── test_retry.py
```

避免建立不必要的 abstraction、repository layer 或 factory layer。

---

## 7. Variable

Phase 1 僅支援 binary variables。

```python
class Variable(BaseModel):
    name: str
    type: Literal["binary"] = "binary"
    description: str | None = None
```

變數名稱必須唯一，且不得以 `__` 開頭（保留給 internal variables）。

---

## 8. Objective

```python
class LinearTerm(BaseModel):
    variable: str
    coefficient: float

class QuadraticTerm(BaseModel):
    variable1: str
    variable2: str
    coefficient: float

class Objective(BaseModel):
    direction: Literal["minimize", "maximize"]
    linear_terms: list[LinearTerm]
    quadratic_terms: list[QuadraticTerm] = []
    constant: float = 0
```

規則：

- `QuadraticTerm` 的 `variable1 != variable2`。Binary 變數下 x² = x，若 LLM 需要表達應放進 linear term；validator 拒絕 `variable1 == variable2`。
- 同一變數（或同一變數對）在 term list 中重複出現時，compiler 將係數相加，validator 不拒絕但記錄 warning。
- Compiler 統一轉換成 minimization energy model：maximize 時所有 objective 係數乘以 -1。

### 8.1 objective_value 語意

`Solution.objective_value` 一律是 **原始 business 值**（用原始係數與方向前的定義計算），不是 minimization 後的值。方向由 `objective.direction` 表明，Agent 消費結果時依此判讀。

---

## 9. Constraint

```python
class Constraint(BaseModel):
    id: str
    description: str | None = None
    type: Literal["hard", "soft"]
    terms: list[LinearTerm]
    operator: Literal["==", "<=", ">="]
    rhs: float
    weight: float | None = None
```

### 9.1 係數限制（重要）

因為 BQM 的 inequality 需要以 binary slack variables 做整數 encoding：

- `operator` 為 `<=` 或 `>=` 的 constraint，**所有 `coefficient` 與 `rhs` 必須是整數值**（型別仍可為 float，但 `value.is_integer()` 必須為 True）。違反時 validator 回 `NON_INTEGER_INEQUALITY`。
- `operator` 為 `==` 的 constraint 不受此限制。

Phase 1 不做自動 scaling。未來若要支援小數係數，在 compiler 加 scaling 層即可，IR 不變。

---

## 10. Hard 與 Soft Constraint

### 10.1 Hard constraint

必須滿足。Penalty strength **由 PenaltyStrategy 決定**，LLM / 呼叫端不得提供。Hard constraint 不得帶 `weight`（validator 拒絕）。

### 10.2 Soft constraint

允許違反，`weight` 表示 business preference importance，必須 `> 0` 且 finite。

Soft constraint 在 BQM 中的 penalty 形式（compiler 必須依此實作）：

| operator | penalty term |
|---|---|
| `==` | `weight × (Σ aᵢxᵢ − rhs)²` |
| `<=` | `weight × (Σ aᵢxᵢ + s − rhs)²`，s 為 compiler 建立的 slack |
| `>=` | `weight × (Σ aᵢxᵢ − s − rhs)²`，s 為 compiler 建立的 slack |

Hard constraint 使用相同形式，但係數改為 `hard_penalty`。

### 10.3 weight 的單位

`weight` 的單位是 **objective 的單位**。也就是說「違反一單位的平方 ≈ 損失 weight 單位的 objective」。若 objective 係數在千位而 weight 為 5，該 soft constraint 幾乎沒有作用。Phase 1 不自動正規化，但 README 必須明確說明，且 constraint trace 需輸出 `objective_scale` 讓 Agent 判斷。`objective_scale` 只看 objective（不含任何 soft weight），否則 Agent 用它判讀 weight 會變成循環參照。

### 10.4 Hard penalty 與 soft weight 不得混用

兩者在程式碼中須使用不同的欄位與不同的來源；不得把 `weight` 當成 hard penalty 的 fallback。

PenaltyStrategy 在決定 hard penalty 的**量級**時，會把 soft 項可能貢獻的能量範圍納入上界（§18 `penalty_scale`）。這不是混用：hard penalty 仍由程式從問題結構算出，weight 從未被當成 penalty 的值或備用值使用，只是 penalty 必須壓過的能量地形多了 soft 項這一塊。

---

## 11. Problem Schema

```python
class SolverPreferences(BaseModel):
    backend: Literal["simulated_annealing", "exact"] = "simulated_annealing"
    num_reads: int = 100
    num_sweeps: int = 1000
    seed: int | None = None
    top_k: int = 5
    max_retries: int = 3
    penalty_multiplier: float = 2.0     # 見 §18，可覆寫以便調試

class OptimizationProblem(BaseModel):
    version: Literal["1.0"] = "1.0"
    name: str
    description: str | None = None
    variables: list[Variable]
    objective: Objective
    constraints: list[Constraint]
    solver: SolverPreferences = SolverPreferences()
```

`version` 使用 `Literal["1.0"]`，未來 incompatible change 時擴充為 `Literal["1.0", "2.0"]` 並由 compiler 分派。

---

## 12. Problem Validator

在 compile 前執行。Pydantic 型別檢查之外，至少檢查：

| code | 條件 |
|---|---|
| `DUPLICATE_VARIABLE` | variable name 重複 |
| `RESERVED_VARIABLE_NAME` | variable name 以 `__` 開頭 |
| `DUPLICATE_CONSTRAINT_ID` | constraint id 重複 |
| `UNKNOWN_VARIABLE` | objective linear / quadratic / constraint terms 引用不存在的變數 |
| `SELF_QUADRATIC_TERM` | quadratic term `variable1 == variable2` |
| `NON_FINITE_COEFFICIENT` | 任何 coefficient / rhs / constant / weight 非 finite |
| `EMPTY_CONSTRAINT` | constraint terms 為空 |
| `NON_INTEGER_INEQUALITY` | `<=` / `>=` constraint 有非整數係數或 rhs |
| `HARD_CONSTRAINT_HAS_WEIGHT` | hard constraint 帶 weight |
| `SOFT_CONSTRAINT_MISSING_WEIGHT` | soft constraint 無 weight 或 weight <= 0 |
| `INVALID_SOLVER_PREFERENCE` | top_k <= 0、num_reads <= 0、num_sweeps <= 0、max_retries < 0、penalty_multiplier <= 0 |
| `TRIVIALLY_INFEASIBLE` | hard constraint 在所有 binary 組合下都不可能滿足（用 min/max 可達值判斷，例如全正係數且 `>= rhs` 而 rhs > Σ aᵢ） |

Validation 失敗：**禁止送入 solver**，回傳 structured error：

```json
{
  "status": "invalid_problem",
  "errors": [
    {
      "code": "UNKNOWN_VARIABLE",
      "path": "constraints[1].terms[0]",
      "message": "Variable employee_xyz does not exist"
    }
  ]
}
```

`path` 盡量精確到 term 層級。同一次 validation 應收集所有錯誤後一次回傳，不要遇到第一個就停。

---

## 13. Compiler Interface

```python
class ModelCompiler(Protocol):
    def compile(
        self,
        problem: OptimizationProblem,
        hard_penalty: float,
    ) -> CompiledProblem: ...
```

Phase 1 實作 `BQMCompiler`。未來可加 `CQMCompiler`、`MILPCompiler`，Phase 1 不實作。

---

## 14. Slack Encoding（`compiler/slack.py`）

自行實作 slack encoding，不依賴 dimod 的 `add_linear_inequality_constraint`。理由：完整掌控 internal variable 命名、trace、與未來 scaling 擴充。

對 `Σ aᵢxᵢ <= rhs`（係數與 rhs 皆為整數）：

1. 計算 `lhs_min = Σ min(aᵢ, 0)`，`lhs_max = Σ max(aᵢ, 0)`。
2. 若 `lhs_max <= rhs`，constraint 永遠成立，compiler 不產生任何 term（trace 標記 `redundant=True`）。
3. 否則 slack 範圍 `S = rhs − lhs_min`（S >= 0，否則 validator 已回 `TRIVIALLY_INFEASIBLE`）。
4. 用 binary expansion 建 slack：`s = Σ 2ᵏ · sₖ` 加上一個補足項，使 s 可精確表達 `0..S`。位元數 `⌈log₂(S+1)⌉`。
5. 加入 penalty `λ (Σ aᵢxᵢ + s − rhs)²`。

對 `>=`：兩邊乘以 -1 化為 `<=` 再套用同一流程（trace 仍記錄原始 operator）。

Slack 變數命名：`__slack_{constraint_id}_{k}`。

Unit test 必須驗證：對所有 `0..S` 的整數 v，存在一組 slack bits 使 `s == v`，且不存在 `s > S` 的組合。

---

## 15. BQM Compiler

負責 `OptimizationProblem → dimod.BinaryQuadraticModel`。

- Objective：linear + quadratic；maximize 時係數乘以 -1；constant 進 offset。
- Equality：`λ (Σ aᵢxᵢ − rhs)²` 展開。Binary 下 xᵢ² = xᵢ，展開時對角項要折進 linear。
- Inequality：透過 §14 slack encoding。
- Hard：λ = `hard_penalty`；Soft：λ = `weight`。
- 相同變數 / 變數對的係數要累加，不得覆蓋。

Compiler 不得對 problem 做任何 mutation。

---

## 16. Internal Variables

Compiler 建立的變數（slack）：

- 不屬於原始 OptimizationProblem
- 不出現在 final business solution
- 全部記錄在 `CompiledProblem.internal_variables`
- 命名一律 `__` 開頭

---

## 17. CompiledProblem 與 Constraint Trace

```python
class ConstraintTrace(BaseModel):
    constraint_id: str
    constraint_type: Literal["hard", "soft"]
    operator: Literal["==", "<=", ">="]
    source_description: str | None
    generated_variables: list[str]
    penalty: float                 # hard_penalty 或 soft weight
    slack_range: int | None        # inequality 才有
    redundant: bool = False
    compiler: str                  # "BQMCompiler"

class CompiledProblem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Any                     # Phase 1: dimod.BinaryQuadraticModel
    original_problem: OptimizationProblem
    internal_variables: set[str]
    constraint_trace: list[ConstraintTrace]
    hard_penalty: float
    objective_scale: float         # objective-only scale（§18），供 Agent 判讀 soft weight；不含 soft 項
    num_variables: int             # 含 internal
```

上層 orchestration 不得依賴 `model` 的 concrete type。Phase 1 不要求 coefficient-level provenance。

---

## 18. Penalty Strategy

```python
class PenaltyStrategy(Protocol):
    def initial_penalty(self, problem: OptimizationProblem) -> float: ...
    def next_penalty(self, previous: float, attempt: int) -> float: ...
    def objective_scale(self, problem: OptimizationProblem) -> float: ...   # objective-only
    def penalty_scale(self, problem: OptimizationProblem) -> float: ...     # objective + soft 上界
```

Phase 1 實作 `ScaledPenaltyStrategy`：

```python
objective_scale = max(1.0, sum(|linear coeffs|) + sum(|quadratic coeffs|))
soft_bound      = Σ_{soft c} weight_c × D_c²
penalty_scale   = objective_scale + soft_bound
initial_penalty = penalty_scale * problem.solver.penalty_multiplier
next_penalty    = previous * 2
```

其中 `D_c` 是 soft constraint `c` 依 §10.2 編譯後、平方括號內線性式在**所有** binary assignment（含 compiler 為它產生的 slack bits）上的最大絕對值。lhs 上下界用累加後的係數算（同 §12 / §14 的 `lhs_min` / `lhs_max`）：

| 情形 | `D_c` |
|---|---|
| `==` | `max(abs(lhs_min − rhs), abs(lhs_max − rhs))` |
| inequality，正規化為 `<=` 後 slack range `S >= 0` | `max(abs(lhs_min − rhs), abs(lhs_max + S − rhs))`（slack 全開時剛好加 `S`） |
| inequality，redundant（`lhs_max <= rhs`） | `0`（compiler 不產生任何項） |
| soft inequality，trivially infeasible（`S < 0`） | 同 `==` 的公式（compiler 已 clamp 成 0 個 slack bit） |

單一 constraint 的 `weight × D²` 是該 soft 項能量的精確最大值；逐條相加得到的 `soft_bound` 是全部 soft 項總和的保守上界（不會低估）。整個計算是純算術（`validation/estimates.py`），不建 BQM，deterministic。

說明：對 binary 變數而言 `objective_scale` 是 objective 變動範圍的上界，`soft_bound` 是 soft 項總能量的上界，兩者相加就是 hard penalty 必須壓過的「非懲罰能量地形」。推導：任一違反 hard constraint 的 assignment，其違反量至少 1（inequality 係數為整數；equality 若係數與 rhs 為整數亦然），能量 `>= objective_min + λ`；最佳 feasible assignment 的能量 `<= objective_max + soft_bound`。因此 `λ > penalty_scale >= (objective_max − objective_min) + soft_bound` 即保證 BQM 的 global minimum 是 feasible；`multiplier = 1` 時保證沒有 infeasible assignment **嚴格**優於最佳 feasible 解，`multiplier = 2` 是保險，retry 是二次保險。若 equality constraint 有非整數係數，最小違反量可能小於 1，此保證不成立（與 v1 相同，不另處理）。

沒有 soft constraint 時 `soft_bound = 0`，`penalty_scale == objective_scale`，Phase 1 既有問題（knapsack / assignment / retry）的 penalty 數值完全不變。

`objective_scale` 保持 objective-only：它是 Agent（與 Phase 2 validator 的 `SOFT_WEIGHT_SMALL` warning）判讀 soft weight 相對 objective 大小的基準，若把 weight 混進去會變成循環參照。`penalty_scale` 只用來決定 hard penalty 的量級（§10.4）。

注意 penalty 過大會讓 SA 的 energy landscape 過於崎嶇而降低找到 feasible 解的機率。因此：

- 不要把預設 multiplier 設得很大
- retry 上限由 `max_retries` 控制，不會無限放大
- `penalty_multiplier` 開放覆寫，供調試

必須是 deterministic：相同 problem 產生相同 penalty。不宣稱 mathematically optimal。

---

## 19. Retry Strategy

僅適用於 **simulated_annealing** backend。

```text
attempt 1: λ = initial
attempt 2: λ = λ × 2
...
最多 1 + max_retries 次
```

每次 retry 必須重新 compile、重新 solve、重新 validate。不得修改既有結果。

**Exact backend 不 retry**：ExactSolver 枚舉全部狀態，只要 slack encoding 正確，任何 feasible 解一定出現在 samples 中，與 λ 無關。Service 對 exact backend 只跑一次 attempt。

---

## 20. Solver Backend Interface

```python
class RawSolverResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    samples: list[dict[str, int]]   # 含 internal variables
    energies: list[float]
    backend: str

class SolverBackend(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def is_exhaustive(self) -> bool: ...     # exact=True, SA=False
    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult: ...
```

`is_exhaustive` 供 orchestration 決定是否 retry 與 infeasibility 語意（§30）。

---

## 21. Simulated Annealing Backend

底層 `dwave.samplers.SimulatedAnnealingSampler`。至少傳入 `num_reads`、`num_sweeps`、`seed`。

保留所有 samples，**不要只取 `.first`**。所有 candidate 進 Solution Validator。

---

## 22. Exact Solver Backend

底層 `dimod.ExactSolver()`。用途：小型 scenario、compiler correctness、SA benchmark、global optimum。

**硬上限**：`compiled_problem.num_variables > 24` 時不執行，raise `SolverExecutionError`（service 轉為 `solver_error`，message 說明變數數與上限）。上限以常數定義，可在 backend 建構子覆寫。

README 必須說明它主要是 testing / debugging backend。

---

## 23. Solution Validator

直接對 **原始 OptimizationProblem + candidate solution（已移除 internal variables）** 驗證。**不得依靠 BQM energy 判斷合法性。**

```python
def validate(problem: OptimizationProblem, sample: dict[str, int]) -> ValidationResult
```

```python
class ConstraintEvaluation(BaseModel):
    constraint_id: str
    constraint_type: Literal["hard", "soft"]
    satisfied: bool
    actual_value: float
    operator: str
    expected_value: float
    violation_amount: float          # >= 0；satisfied 時為 0
    weighted_penalty: float | None   # soft 才有：weight × violation²

class ValidationResult(BaseModel):
    feasible: bool                   # 所有 hard 滿足
    evaluations: list[ConstraintEvaluation]
    hard_violations: list[ConstraintEvaluation]
    soft_violations: list[ConstraintEvaluation]
    soft_violation_score: float      # Σ weighted_penalty
```

### 23.1 Floating point tolerance

```python
EPSILON = 1e-8
==  : abs(actual - rhs) <= EPSILON
<=  : actual <= rhs + EPSILON
>=  : actual >= rhs - EPSILON
```

`violation_amount`：`==` 為 `abs(actual − rhs)`，`<=` 為 `max(0, actual − rhs)`，`>=` 為 `max(0, rhs − actual)`。

---

## 24. Objective Evaluation

`objective_value` 從原始 problem 重新計算（§8.1 語意）。不得把 BQM energy 當 business objective。`energy` 另外保留供 debug。

---

## 25. Candidate 處理與 Ranking

```text
samples
  ↓ 移除 internal variables
  ↓ 去重（以 business variables 為 key；同一 key 保留最小 energy）
  ↓ Solution Validator
  ↓ feasible only
  ↓ 計算 objective_value、soft_violation_score、ranking_score
  ↓ 依 ranking_score 排序
  ↓ Top-K
```

### 25.1 ranking_score

```text
minimize : ranking_score = objective_value + soft_violation_score   （ascending）
maximize : ranking_score = objective_value − soft_violation_score   （descending）
```

理由：soft constraint 若只影響 sampling 而不影響最終排序，會導致兩個 objective 相同的解中違反較多 soft 的反而排前面。`objective_value` 與 `soft_violation_score` 皆保留輸出，Agent 可自行重新排序。

Tie-break：相同 ranking_score 時依 `objective_value`（同方向），再依變數名稱字典序後的 assignment tuple，確保排序 deterministic。

不得使用 sampler energy 做最終排序。

---

## 26. SolveResult

```python
class Solution(BaseModel):
    rank: int
    variables: dict[str, int]
    objective_value: float
    soft_violation_score: float
    ranking_score: float
    energy: float | None
    hard_constraints_satisfied: bool
    constraint_evaluations: list[ConstraintEvaluation]

class SolveAttempt(BaseModel):
    attempt: int
    penalty: float
    samples_received: int
    unique_samples: int
    feasible_samples: int

class ProblemError(BaseModel):
    code: str
    path: str | None
    message: str

class SolveResult(BaseModel):
    status: Literal["success", "infeasible", "invalid_problem", "solver_error"]
    backend: str | None
    objective_direction: Literal["minimize", "maximize"] | None
    solutions: list[Solution]
    attempts: list[SolveAttempt]
    infeasibility_proven: bool = False
    errors: list[ProblemError] = []
    message: str | None = None
```

---

## 27. Status 語意

- **success**：至少一個解滿足所有 hard constraints。
- **infeasible**：attempts 耗盡仍無 feasible 解。
  - SA backend：`infeasibility_proven = False`，只表示「在目前 solver configuration 與 retry strategy 下沒找到」。
  - Exact backend：`infeasibility_proven = True`（在 §9.1 整數係數 + §14 精確 slack encoding 前提下，全枚舉找不到即為證明）。
- **invalid_problem**：validation 失敗，`errors` 填入。
- **solver_error**：底層 exception，轉為 structured result，`message` 填入摘要。

---

## 28. Orchestration Service

```python
class OptimizationService:
    def __init__(
        self,
        compiler: ModelCompiler | None = None,
        penalty_strategy: PenaltyStrategy | None = None,
        backends: dict[str, SolverBackend] | None = None,
    ): ...

    def solve(self, problem: OptimizationProblem) -> SolveResult: ...
```

流程：

```text
validate problem → invalid_problem 直接回傳
select backend
λ = initial_penalty
loop:
    compile(problem, λ)
    solve
    process candidates (§25)
    if feasible: rank, return success
    if backend.is_exhaustive: return infeasible (proven=True)
    if attempt > max_retries: return infeasible (proven=False)
    λ = next_penalty(λ, attempt)
任何 domain exception → solver_error
```

未來 MCP Server 只需呼叫 `service.solve(problem)`。

---

## 29. CLI

```bash
python -m optimizer.cli solve examples/knapsack.json
python -m optimizer.cli solve examples/knapsack.json --backend exact
python -m optimizer.cli solve examples/knapsack.json --json
python -m optimizer.cli export-schema
```

`--backend` 覆寫 JSON 內的 `solver.backend`。`--json` 輸出完整 `SolveResult.model_dump_json(indent=2)`。

Human-readable 輸出範例：

```text
Problem:   Simple Knapsack
Backend:   exact
Status:    success
Attempts:  1

Best solution (rank 1)
  objective (maximize):  17
  soft violation score:  0
  item_a = 1
  item_b = 0
  item_c = 1
  item_d = 0

Hard constraints: 1 / 1 satisfied
Soft constraints: 0 violations
```

CLI 只做格式化與參數解析，不得含 optimization logic。

---

## 30. Example 1 — Knapsack

`examples/knapsack.json`：capacity 10，maximize value，hard `Σ weight ≤ 10`。

| item | weight | value |
|---|---|---|
| A | 6 | 10 |
| B | 5 | 8 |
| C | 4 | 7 |
| D | 3 | 6 |

Global optimum：{A, C} = 17（weight 10）。Scenario test 中必須把預期值寫成常數並註解推導。

Scenario test：

- ExactSolver：status success、hard satisfied、`objective_value == 17`、`infeasibility_proven` 不適用。
- SA 固定 seed、num_reads=100：必須找到 feasible；可額外斷言找到 17，但不得設計得對隨機 sampler 過度 brittle（若要斷言 optimum，用較大 num_reads）。

---

## 31. Example 2 — Assignment

3 workers × 3 tasks，變數 `{worker}_{task}`，每個 worker 恰好一個 task、每個 task 恰好一個 worker（6 條 hard equality），minimize total cost。

驗證：多個 equality constraints、one-hot、quadratic penalty 展開、validator、ranking。ExactSolver 必須得到手算的最小 cost。

---

## 32. Example 3（測試用）— Retry

`tests/scenarios/test_retry.py`：用一個 `penalty_multiplier` 刻意設很小（例如 0.01）的 knapsack，SA 第一輪應無 feasible，retry 後應成功。斷言 `len(result.attempts) > 1` 且 penalty 遞增。同一 problem 用 exact backend 時 `len(attempts) == 1`。

---

## 33. Unit Tests

Schema：valid parse、duplicate variable、unknown variable、reserved name、self quadratic、invalid soft weight、hard with weight、non-integer inequality、invalid solver preference、version 不是 "1.0" 被拒。

Slack：範圍完整性（§14）、`>=` 轉換、redundant constraint 不產 term。

Compiler：minimize、maximize、linear、quadratic、equality 展開、`<=`、`>=`、internal variables 記錄、hard penalty 套用、soft weight 套用、係數累加、problem 未被 mutate。

Validator：valid、hard violation、soft violation 與 weighted_penalty、三種 operator 的 tolerance、violation_amount 計算。

Penalty：deterministic、retry 遞增、multiplier 覆寫。

Solver：SA seed reproducibility、Exact 回傳已知 optimum、Exact 超過變數上限 raise。

Ranking：去重（不同 slack 同 business 解）、energy 取最小、soft violation 影響排序、tie-break deterministic。

---

## 34. Scenario Tests

必須走完整 JSON → Pydantic → Validator → Compiler → Solver → Solution Validator → SolveResult，不得 bypass service layer。

---

## 35. Logging

標準 `logging`，不用 `print()`。至少記錄：problem name、backend、attempt、hard penalty、compiled variable count、sample count、unique count、feasible count、best ranking_score。CLI 負責 console 格式。

---

## 36. Error Handling

`exceptions.py`：

```text
OptimizerError (base)
├── ProblemValidationError
├── CompilationError
└── SolverExecutionError
```

Domain code 不得把 dimod / dwave exception 直接暴露。OptimizationService 最終轉為 SolveResult。

---

## 37. Determinism

seed 必須向下傳遞。Library code 不得呼叫 `random.seed()` 或修改 global random state。排序須 deterministic（§25.1 tie-break）。

---

## 38. JSON Schema

`OptimizationProblem.model_json_schema()` 由 CLI `export-schema` 輸出。此 schema 是重要 public interface，未來供 LLM structured output 與 MCP tool input schema 使用。

---

## 39. API Stability

`OptimizationProblem JSON` 視為公開 API，`version: "1.0"`。Incompatible change 升版，不得偷偷改變既有 field 語意。

---

## 40. README

至少包含：Architecture、Installation、Quick Start、JSON Input（含 §9.1 整數限制與 §10.3 weight 單位說明）、CLI、Examples、Tests、Current Limitations。

Limitations 必須明寫：

```text
Phase 1 currently supports binary-variable optimization problems only.
Inequality constraints require integer coefficients and right-hand sides.
Soft constraint weights are expressed in objective units and are not normalized.
The simulated annealing backend is heuristic and does not guarantee a global optimum.
The exact backend is for testing/debugging and is limited to small problems.
```

---

## 41. Coding Style

Type hints、Pydantic v2、小模組、public method 有 docstring、內部實作簡單。需要 abstraction 的只有 Compiler、SolverBackend、PenaltyStrategy。不要為「未來可能需要」建空 class。

---

## 42. Architecture Rule

```text
models ← validation ← compiler ← solvers ← orchestration ← CLI / future MCP
```

Core 不得依賴 CLI。Phase 2 加 MCP 時只需：

```python
@mcp.tool()
def optimize(problem: OptimizationProblem) -> SolveResult:
    return service.solve(problem)
```

若還需重寫 compiler / solver / validator / retry，代表 Phase 1 architecture 不合格。

---

## 43. Acceptance Criteria

- [ ] `pip install -e .` 成功
- [ ] CLI 從 JSON 載入 problem
- [ ] Pydantic + Problem Validator 正常，錯誤一次收集回傳
- [ ] 非整數 inequality 係數被拒
- [ ] 建立 BQM；maximize / minimize；`==` / `<=` / `>=`；hard / soft
- [ ] slack 由 compiler 管理，encoding 範圍正確
- [ ] soft penalty 形式符合 §10.2
- [ ] SA 與 Exact backend 皆可求解
- [ ] Exact backend 有變數上限
- [ ] 所有 candidate 經 Solution Validator
- [ ] objective 從原始 problem 重算
- [ ] ranking 使用 ranking_score，tie-break deterministic
- [ ] 去重正確，energy 取最小
- [ ] internal variables 不出現在結果
- [ ] SA retry 有效且有上限；exact 不 retry
- [ ] `infeasibility_proven` 語意正確
- [ ] CLI human-readable 與 `--json`
- [ ] export-schema
- [ ] Knapsack / Assignment / Retry scenario 通過
- [ ] pytest 全過，無 skip / xfail
- [ ] core 無 FastMCP、D-Wave cloud dependency
- [ ] README 完整

---

## 44. 開發順序

嚴格依序，每步跑完相關測試再進下一步：

1. pyproject、models、schema、model tests
2. Problem Validator、Solution Validator + tests（無 solver）
3. slack.py + tests；BQMCompiler：objective → `==` → `<=` → `>=` → soft，每功能先寫 test
4. ExactSolverBackend，Knapsack 用 exact 跑通
5. OptimizationService：compile / solve / dedup / validate / rank
6. SimulatedAnnealingBackend（num_reads / num_sweeps / seed）
7. PenaltyStrategy + retry；建立 test_retry.py
8. Assignment scenario
9. CLI、JSON output、export-schema
10. README、typing、error handling、logging 整理

最後 `pytest` 全過。不得用 skip / xfail 迴避 bug。

---

## 45. 不要做的事情

- 讓 LLM 產 QUBO
- 把 QUBO matrix 當 public API
- 把 BQM energy 當 business objective 或 feasibility 依據
- 只取 `sampler.first`
- 把 hard penalty 和 soft weight 混用
- 只按 objective 排序而忽略 soft violation
- 對 exact backend 做 retry
- 在 solver class 解析 JSON
- 在 CLI 實作 optimization logic
- Phase 1 加 MCP / Web API / database
- 寫 speculative abstraction

---

## 46. 最核心驗收原則

Phase 1 成功與否不是看能不能呼叫 SimulatedAnnealingSampler，而是這條 pipeline 是否可靠：

```text
Structured Optimization Problem
        ↓
Deterministic Compilation
        ↓
Solver (replaceable)
        ↓
Independent Constraint Validation
        ↓
Ranked Feasible Business Solutions
```
