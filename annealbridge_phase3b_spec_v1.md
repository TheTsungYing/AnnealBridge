# AnnealBridge — Phase 3b 開發規格 (v1)

> v1 修訂紀錄（2026-09-03）：初稿完成後經兩個獨立審查（架構原則 / 程式可行性）修正 34 處，主要：decode 對無整數變數的問題保留 int8（原稿一律 int64 會讓 exact 路徑記憶體 ×8 且與 §24 矛盾）；bounds 改限制絕對值 ±(2³¹−1)（原稿只限 range，decode 會溢位）；§8 / §14 的 bounds / forms 改為 keyword-optional，既有 7 個測試檔不改仍綠；`process_candidates` / `deduplicate_samples` 的 `internal_variables` 改可選、`_pack_rows` 名稱保留；值域斷言允許 float 型別的整數值（Leap CQM 回 float）；DA 缺索引改報錯不補 0；DA 400 預設 `REMOTE_SOLVER_ERROR`、只有 header 錯才 `BACKEND_CONFIG_INVALID`；urllib 例外分類改依 `TimeoutError` / 其他 `OSError`；移除 v3c 切換與獨立 http 模組；golden 錄製腳本與固定清單進 repo；補 batch 算術、環境變數隔離、DA code 表登記、MCP capabilities 斷言在步驟 1 更新等測試清單；decode 輸出順序統一為 problem 順序；§29 偏離表補五列。

> 本規格承接 `annealbridge_phase3a_spec_v1.md`（3a 已於 2026-09-03 完成，commit `6486719..20ba640`，`pytest` 1311 passed），依 3a §32 留下的接點與 `annealbridge_phase3_spec_outline.md` §2.1、§2.3 的待答問題撰寫。體例沿用 Phase 2 / 3a。
> 撰寫時遵守 `annealbridge_PROJECT_OVERVIEW.md` §五的核心原則，特別是原則 1（AI 只給要什麼）、原則 2（答案只信原始 JSON 重驗）、原則 4（核心不認識特定求解器）、原則 7（不蓋空殼）。
>
> **3b 範圍（決策 1）：整數變數（IR 1.1）+ Fujitsu Digital Annealer backend。**
>
> 3b 撰寫前的使用者決策（2026-09-03）：
> | 問題 | 決策 |
> |---|---|
> | Fujitsu DA 接到多深 | **走既有 BQM 路徑**：compiler 產出的 QUBO 整個送 DA，與 SA / QPU 同一路徑；整數變數經 BQM 編碼自動支援；DA 原生不等式與懲罰係數自動調整留給之後（§30.2） |
> | Fujitsu DA 認證 | **讀環境變數 `FUJITSU_DA_API_KEY`**（`X-Api-Key`），只在 backend 內 lazy 讀；不支援 OAuth access token；用標準庫發 HTTP，不加依賴 |
> | IR 版本 | **整數變數必須寫 `version: "1.1"`**；`"1.0"` 帶整數變數是驗證錯誤；1.0 永遠等於「只有 0/1」 |
>
> 撰寫前實測（dimod 0.12.22、numpy 2.5.2、Fujitsu 官方 OpenAPI YAML `da-qubo-v4-en.yaml` 1408 行、`ommx-da4-adapter` 原始碼）的事實散見各節，標「實測」者可直接引用，不必再驗。

---

## 0. 現況（實作者必讀）

- 3a 完成後的擴充點：`ModelCompiler`（`model_type` / `uses_hard_penalty` / `compile`）、`SolverBackend`（`capabilities` / `is_available` / `resolve_time_limit` / `solve`）、`PenaltyStrategy`；`SolverCapabilities` 在 `models/capabilities.py`，`ModelType = Literal["bqm", "cqm"]`；`ExecutionPolicy.limits` 通用通道；`orchestration/limits.py` 的 `gate_errors` / `preference_limit_errors` / `select_model_type`；routing 純函式；架構測試 `test_no_backend_names.py`（AST 整值比對五個 backend 名）、`test_import_boundaries.py`、`test_fifth_backend.py`。
- 五個 backend：`exact`、`simulated_annealing`、`dwave_qpu`、`leap_hybrid_bqm`、`leap_hybrid_cqm`（registry 順序固定）。error catalog 34 個 code（`test_error_catalog.py` 斷言總數）。
- 3b 會碰到的每個「只認 0/1」的位置（全部列出，實作時逐一處理）：
  - `models/variable.py`：`type: Literal["binary"]`，無 bounds。
  - `validation/estimates.py`：`lhs_bounds` 假設 0/1；`compute_objective_scale` 用 Σ|係數|；`analyze_inequality` / `count_slack_bits` / `compute_soft_energy_bound` / `estimate_compiled_variables` 全部建立在上述之上。
  - `validation/problem_validator.py`：`SELF_QUADRATIC_TERM`（x·x 對 binary 無意義）、`_check_trivially_infeasible`（用 `lhs_bounds`）。
  - `compiler/bqm.py`：變數直接 `bqm.add_variable(name)`；`compiler/objective.py::build_objective_bqm` 直接用變數名；`compiler/slack.py` 只做不等式 slack。
  - `compiler/cqm.py`：`cqm.add_variable("BINARY", name)`；soft constraint 用 `penalty="quadratic"`（**實測：含整數變數時 dimod 拋 `ValueError: quadratic penalty only allowed if the constraint has binary variables`，且拋錯前已把該 constraint 當 hard 加進 cqm**——所以 penalty 型別必須在呼叫前決定，不能 try/except）。
  - `solvers/base.py`：`RawSolverResult.samples` 強制 `int8`；`sampleset_to_arrays` 轉 `int8`。
  - `solvers/leap_hybrid_cqm.py::_assert_binary_samples`：斷言值 ∈ {0, 1}。`tests/fakes/local_cqm_backend.py` 同。
  - `orchestration/optimizer.py`：`_pack_rows`（`np.packbits`，只對 0/1 正確）用於去重與 tie-break；`deduplicate_samples(raw, internal_variables)` 只會「剔除」internal 欄位，不會把編碼位元合成整數；`evaluate_objective_batch` / `validate_batch` 是純算術，對整數值本來就正確（§26 要測試證明）。
  - `interfaces/capabilities.py`：`schema_version = get_args(version.annotation)[0]`（Literal 變成兩個值後會回 `"1.0"`）；`supported_variable_types` 由 `Variable.type` 的 Literal 自動產生。
  - `solvers/metadata.py`：`_env_token()` 只讀 `DWAVE_API_TOKEN`；redaction pattern 只有 D-Wave 形式。
- 沒有安裝 `dwave-system`；沒有 Fujitsu 帳號。DA 只能 mock，live 測試留 opt-in（決策 3）。

---

## 1. Phase 3b 目標

> **讓問題可以宣告有上下界的整數變數，兩條 compiler 路徑都正確處理（BQM 編碼、CQM 原生），1.0 問題行為位元一致；並以 Fujitsu Digital Annealer 作第一個非 D-Wave 的遠端 backend，證明 3a 的擴充點對「沒有 SDK、只有 HTTP」的廠商同樣成立。**

```text
OptimizationProblem 1.1
  variables: binary | integer[lower, upper]
          │
   ┌──────┴───────┐
   ▼              ▼
BQMCompiler    CQMCompiler
 整數→位元編碼    整數→dimod INTEGER
 slack 位元      hard 原生 / soft→objective
   │              │
   ▼              ▼
 exact / SA / dwave_qpu / leap_hybrid_bqm / fujitsu_da (new)   leap_hybrid_cqm
   │              │
   └──── compiler.decode()：位元→整數、去 internal ────┘
                       │
             process_candidates（不變：去重、獨立驗證、排名）
```

---

## 2. Phase 3b Scope

1. IR 1.1：`Variable.type` 加 `"integer"` + `lower_bound` / `upper_bound`；`version` 加 `"1.1"`；1.0 JSON 不改一字且行為不變
2. Validator：整數變數規則（5 個新 error code）、`SELF_QUADRATIC_TERM` 只對 binary、bounds-aware 的 trivially-infeasible 判斷、`LARGE_INTEGER_RANGE` / `INTEGER_QUADRATIC_BLOWUP` warnings、estimate 納入編碼位元
3. `validation/estimates.py` 全部改為 bounds-aware（binary 退化為現有公式，位元一致）
4. `ModelCompiler.decode()`：編碼位元 → 整數值、剔除 internal，service 改在 `process_candidates` 前呼叫
5. `BQMCompiler`：整數 → 二進位編碼（binary expansion，位元屬 `__` internal）；objective / constraint 以仿射代換展開；decode
6. `CQMCompiler`：整數 → `dimod INTEGER`；objective / lhs 改用 `QuadraticModel`；含整數的 soft constraint 改為「objective 加 weight × 偏差²」（不等式配整數 slack）；binary-only soft 維持 3a 原生寫法
7. 陣列層：`RawSolverResult.samples` 放寬為任意整數 dtype；去重 / tie-break key 通用化；CQM backend 的值域斷言改為「在 bounds 內的整數」
8. Routing：整數相關 reason code 與 blowup 降序
9. Fujitsu DA backend（BQM 路徑）：`fujitsu_da` options、HTTP client（標準庫）、非同步 job 流程、例外分類、redaction、mock / live 測試
10. `examples/integer_knapsack.json`（1.1 範例）、scenario 測試跨三條路徑、error catalog、README / OVERVIEW、acceptance

---

## 3. Phase 3b 明確不做

實數變數、無界整數、one-hot / unary 整數編碼選項（只做 binary expansion，無 compiler option）、放寬 `NON_INTEGER_INEQUALITY`（§9.5）、DA 原生 `inequalities` / `penalty_binary_polynomial` / one-hot groups（§30.2）、DA OAuth access token、DA Azure Blob 大問題路徑、`FujitsuDAOptions` 以外的 DA 參數（`gs_level`、`target_energy`、`guidance_config`…）、DA `frequency` 展開成重複 sample、自動 cost estimation、job queue、persistence、auth、Web UI、LLM 呼叫、IR 1.1 以外的任何 schema 變更（constraint 型式、非線性 constraint 仍不做）。

---

## 4. Architecture 與 Dependency Direction

不變：`models ← validation ← compiler ← solvers ← orchestration ← interfaces`，`config` 只有 interfaces 可 import。3b 新增規則：

- **整數編碼是 compiler 的事，不是 IR 的事**：JSON 只有 `type: "integer"` 與 bounds，沒有位元、沒有編碼方式。任何 backend 看到的都是 compiler 產出的模型，backend 不做編碼、不做 decode。
- `ModelCompiler` Protocol 新增 `decode(compiled, raw) -> RawSolverResult`（§13）。service 的 `process_candidates` 只處理「業務變數、整數值」的結果；`deduplicate_samples` 不再接 `internal_variables`。
- 新增第六個 backend，**backend 本體**只改：`solvers/fujitsu_da.py`（新）、`solvers/registry.py`、`models/problem.py`（Literal + option block）。另有三項**一次性的通用機制改動**，與 fujitsu 無關、之後任何廠商都受益：`solvers/metadata.py` 的 credential 清單與 redaction pattern 通用化（§20.5）（2026-09-09 F-10 後：backend 只需宣告，`metadata.py` 零改動）、`exceptions.py` 的 `SolverExecutionError.status` + `optimizer.py` 一行（§20.8）、`models/error_catalog.py` 兩個新 remote code（§19）。`test_fifth_backend.py` 的「零改動」清單不變；`test_no_backend_names.py` 的名單加 `"fujitsu_da"`。
- Fujitsu backend 不得 import `requests` / `httpx`（不加依賴）；HTTP 走標準庫 `urllib.request`，透過可注入的 transport 測試（§20.4）。
- 依賴方向新增檢查：`compiler/integer_encoding.py`（§14）在套件內只能 import `annealbridge.models` 與 `annealbridge.validation.estimates`（第三方 `dimod` / `numpy` 不受此限）。

---

## 5. 目錄調整

```text
src/annealbridge/
├── models/
│   ├── variable.py            (type Literal + lower_bound / upper_bound + bounds())
│   ├── problem.py             (version Literal["1.0","1.1"]；FujitsuDAOptions；backend Literal + "fujitsu_da")
│   ├── compiled.py            (IntegerEncoding；CompiledProblem.integer_encodings)
│   ├── metadata.py            (不變)
│   └── error_catalog.py       (+ 7 codes)
├── validation/
│   ├── estimates.py           (bounds-aware：variable_bounds、lhs_bounds(coefficients, bounds)…；integer_encoding_bits)
│   ├── problem_validator.py   (整數規則、版本規則、新 warnings)
│   └── recommendation.py      (不變)
├── compiler/
│   ├── base.py                (ModelCompiler + decode)
│   ├── integer_encoding.py    (new: AffineForm、encode_integer_variables、substitute_linear / substitute_quadratic、expand_square_qm)
│   ├── objective.py           (build_objective_bqm(objective, forms=None)；build_objective_qm(objective, bounds))
│   ├── slack.py               (encode_slack(constraint, bounds=None))
│   ├── bqm.py                 (整數編碼 + decode)
│   └── cqm.py                 (INTEGER 變數、QuadraticModel、mixed soft → objective、decode)
├── solvers/
│   ├── base.py                (RawSolverResult 任意整數 dtype；sampleset_to_arrays(dtype=)；assert_samples_within_bounds)
│   ├── metadata.py            (credential env 清單通用化、redaction pattern + X-Api-Key / X-Access-Token、TIMING_WHITELIST + solve_time / total_elapsed_time)
│   ├── leap_hybrid_cqm.py     (值域斷言改 bounds、int64)
│   ├── fujitsu_da.py          (new；含 HttpTransport Protocol 與 UrllibTransport——只有一個消費者，不另立模組)
│   └── registry.py            (+ fujitsu_da)
├── exceptions.py              (SolverExecutionError.status)
├── orchestration/
│   ├── optimizer.py           (row_keys 通用化；decode 接線；process_candidates 簽名)
│   └── routing.py             (+ 3 reason codes)
└── interfaces/
    ├── capabilities.py        (schema_version 取最新；+ schema_versions)
    ├── mcp/tools.py           (docstring：integer variables、version 1.1)
    └── cli/main.py            (不變；整數值印法本來就對)

examples/
└── integer_knapsack.json      (new, version 1.1)

tests/
├── golden/
│   ├── record_phase3a_golden.py       (new: 錄製腳本，固定清單，commit 進 repo)
│   └── phase3a_compile.json           (new: 步驟 1 前用 20ba640 的程式碼錄下)
├── conftest.py                        (autouse 清 FUJITSU_DA_* 環境變數)
├── unit/
│   ├── test_models.py                 (+ Variable bounds、version 1.1、bool 拒絕)
│   ├── test_golden_phase3a.py         (new: 位元一致)
│   ├── test_estimates_bounds.py       (new: bounds-aware 公式 vs 暴力枚舉；binary 退化位元一致)
│   ├── test_problem_validator.py      (+ 整數規則、版本規則)
│   ├── test_problem_validator_full.py (+ LARGE_INTEGER_RANGE、INTEGER_QUADRATIC_BLOWUP、estimate)
│   ├── test_integer_encoding.py       (new)
│   ├── test_bqm_compiler.py / test_slack.py / test_penalty_strategy.py (不改，仍全綠——靠 §8 的可選參數)
│   ├── test_bqm_compiler_integer.py   (new)
│   ├── test_cqm_compiler.py           (不改，仍全綠；若有斷言 objective 物件型別者改為斷言係數，回報)
│   ├── test_cqm_compiler_integer.py   (new)
│   ├── test_candidate_arrays.py / test_ranking.py (不改，仍全綠——靠 §11 的可選參數與保留的 _pack_rows) + 整數列案例
│   ├── test_batch_arithmetic_integer.py (new: validate_batch / evaluate_objective_batch 對整數矩陣與逐列結果逐元素相等)
│   ├── test_service_decode.py         (new: decode 接線、internal 不外洩、順序 = problem.variables)
│   ├── test_service_validation_consistency.py (FakeFailingCompiler 補 decode)
│   ├── test_routing.py                (+ 整數 reason)
│   ├── test_metadata.py               (+ Fujitsu redaction、timing whitelist)
│   └── test_phase1_compat.py          (+ 1.1 範例可 parse、fujitsu_da block None)
├── scenarios/
│   └── test_integer_knapsack.py       (new: exact / SA / FakeLocalCQM 三路徑同一最佳值)
├── fakes/
│   ├── local_cqm_backend.py           (bounds 斷言、int64)
│   └── da_transport.py                (new: FakeDATransport，腳本化回應)
├── mcp/
│   └── test_capabilities.py           (schema_version "1.1"、schema_versions、variable types、六個 backend——步驟 1 就改)
├── remote_mock/
│   ├── test_leap_hybrid_cqm_mock.py   (dtype 斷言 int8 → int64；值域錯誤訊息子字串改 "outside its bounds")
│   ├── test_fujitsu_da_mock.py        (new)
│   └── test_credential_leak.py        (+ fujitsu_da)
└── remote_live/
    ├── conftest.py                    (skip 條件改為「該測試宣告需要的 env」)
    └── test_fujitsu_da_live.py        (new, opt-in，需 FUJITSU_DA_API_KEY)
```

---

## 6. Dependencies

不新增套件。Fujitsu HTTP 用 `urllib.request`（標準庫）；JSON 用 `json`。`pyproject.toml` 不加 extra（沒有可選依賴可裝）。README 的 Installation 段說明 `fujitsu_da` 只需 API key。

---

## 7. IR 1.1：`Variable` 與 `version`

```python
class Variable(BaseModel):
    name: str
    type: Literal["binary", "integer"] = "binary"
    lower_bound: int | None = None      # integer 必填；binary 必須為 None
    upper_bound: int | None = None
    description: str | None = None

    def bounds(self) -> tuple[int, int]:
        """binary → (0, 1)；integer → (lower_bound, upper_bound)。只供 validator / compiler 用。"""
```

- schema 層：`lower_bound` / `upper_bound` 用 `models/quantities.py` 的 `Count`（`Annotated[int, BeforeValidator(...)]`）。bounds 是數量，不是旗標、也不是文字，所以 `True` 與 `"3"` 都拒絕（實測 pydantic lax mode 會把 `True` coerce 成 `1`、把 `"3"` coerce 成 `3`）；`1.5` 仍拒絕，整數值 float（`2.0` → `2`）仍接受。`BeforeValidator` 不影響發布的 JSON Schema（仍是 `{"type": "integer"}`）。
  - 2026-09-09 review F-11：同一規則套用到全部 IR 數值欄位（`coefficient`、`constant`、`rhs`、`weight`、`SolverPreferences` 與各 option block 的數值欄位、`seed`），共用 `Quantity`（實數）與 `Count`（整數）兩個型別。
- `OptimizationProblem.version: Literal["1.0", "1.1"] = "1.0"`。
- `Variable.bounds()`：binary → `(0, 1)`；integer 且兩個 bound 皆非 None → `(lower, upper)`；否則 `raise ValueError`。**只允許在 validator 的 error pass 通過後呼叫**（compiler、estimates、`validate_problem_full` 的 warning 層都在 error pass 之後）；validator 的 error pass 內對 bounds 不合法的變數視為「未知」（§9.2）。
- **語意規則（validator，§9.1）**：
  - `type == "integer"` 必須同時給 `lower_bound` 與 `upper_bound`，且 `upper_bound > lower_bound`（相等是常數，不是變數）。
  - `type == "binary"` 不得給任何 bound。
  - `|lower_bound| <= 2**31 - 1` 且 `|upper_bound| <= 2**31 - 1`（§9.1 `INTEGER_RANGE_TOO_LARGE`，自然蘊含 range 上限）。理由：值域在 ±2³¹ 內時，decode 的 `lower + Σ coef·bit` 與二次項 `x·y`（≤ 2⁶²）都在 int64 內不溢位，且所有值精確落在 float64 的 2⁵³ 整數精確區內，`evaluate_objective_batch` / `validate_batch` 的 float64 算術對整數值仍精確。這也是「無界整數不做」的具體界線。
  - 負下界允許（編碼時位移，§14）。
  - 問題含任何 integer 變數 → `version` 必須是 `"1.1"`；`"1.0"` → `INTEGER_REQUIRES_VERSION_1_1`。`"1.1"` 只有 binary 變數 → 合法（1.1 是 1.0 的超集）。
- **位元一致保證**：三個既有 example（`version: "1.0"`）與所有 Phase 1/2/3a 測試的 BQM / CQM / 估算 / penalty 數值完全不變。§26.1 以 golden 測試證明（§25 開工前錄製）。
- 整數值在 `Solution.variables: dict[str, int]` 內仍是 `int`（型別不變）。

---

## 8. `validation/estimates.py` bounds-aware

所有公式改為接受 **可選的** `bounds: Mapping[str, tuple[int, int]] | None = None`（由 `variable_bounds(problem)` 一次算出；binary → `(0, 1)`；`None` 表示「全部視為 binary」，即 3a 行為）。**簽名以 keyword-optional 方式擴充**，既有呼叫者（`tests/unit/test_slack.py`、`test_penalty_strategy.py`、`test_bqm_compiler.py`、`test_cqm_compiler.py`、`test_problem_validator_full.py`、`test_service_compilers.py` 對 `compute_objective_scale(objective)`、`encode_slack(constraint)`、`count_slack_bits(c)`、`compute_soft_energy_bound(c)`、`analyze_inequality(c)`、`build_objective_bqm(objective)` 的單參數呼叫）**不改仍全綠**。production 呼叫端（compiler、validator、penalty）一律傳 bounds。**對全 binary 問題每個函式回傳值與 3a 完全相同**（§26.1 golden 測試；審查時對 2000 個隨機全 binary 問題實測 `lhs_bounds` 與 `compute_objective_scale` 零差異）。

| 函式 | 3b 定義 |
|---|---|
| `variable_bounds(problem) -> dict[str, tuple[int, int]]` | 新 |
| `lhs_bounds(coefficients, bounds)` | `min = Σ min(c·lo, c·hi)`、`max = Σ max(c·lo, c·hi)` |
| `compute_objective_scale(objective, bounds)` | ~~`max(1, Σ|c_i|·M_i + Σ|c_ij|·M_i·M_j)`，`M = max(|lo|, |hi|)`~~ → **2026-09-09 review F-06 改為範圍公式（§8.1）**：`max(1, Σ|c_i|·(hi_i − lo_i) + Σ|c_ij|·range(x_i·x_j))`；binary 時每項寬度為 1 → 現有公式位元一致 |
| `analyze_inequality(constraint, bounds)` | 同 3a，`lhs_bounds` 換 bounds 版；`slack_range = int(round(rhs - lhs_min))` |
| `count_slack_bits(constraint, bounds)` | 同 3a |
| `compute_soft_energy_bound(constraint, bounds)` | 同 3a 結構，`_max_abs_affine` 用 bounds |
| `compute_penalty_scale(problem)` | 內部呼叫 `variable_bounds` |
| `integer_encoding_bits(lower, upper) -> int` | 新：`(upper - lower).bit_length()`；`compute_slack_coefficients(upper - lower)` 給位元係數（**與 slack 共用同一函式**，§14） |
| `estimate_compiled_variables(problem)` | `binary 個數 + Σ integer_encoding_bits + Σ count_slack_bits`（BQM 路徑）；CQM 路徑另計（§9.4） |
| `estimate_encoded_interactions(problem) -> int` | 新：BQM 路徑二次交互作用數的上界，供 `INTEGER_QUADRATIC_BLOWUP`（§9.3）：objective 每個 `c·x·y` 貢獻 `bits(x)·bits(y)`（`x·x` 貢獻 `bits(x)·(bits(x)-1)/2`）；每條 constraint 貢獻 `B·(B-1)/2`，`B` = 該 constraint 涉及的位元數（含 slack 位元） |

`PenaltyStrategy` Protocol 簽名不變（收 `problem`），`ScaledPenaltyStrategy` 內部呼叫 bounds 版；Phase 1 §18 公式對整數變數的解讀寫進 docstring：`objective_scale` 是「objective 變動範圍（`max − min`）的上界」（原文「最大絕對值上界」經 §8.1 修正）。

### 8.1 2026-09-09 review 修正（F-06）：`compute_objective_scale` 改用範圍公式

**問題**：Phase 1 §18 的推導需要 `penalty_scale ≥ (objective_max − objective_min) + soft_bound`，即 objective 的**範圍**。本節原公式 `Σ|c|·M`（`M = max(|lo|, |hi|)`）是 `max|objective|` 的上界；binary 時兩者相等，但整數變數 `lo < 0` 時範圍可達其 2 倍，`lo > 0`（區間不含 0）時又比需要的大。反例：`x ∈ [-4, 4]`、`y` binary、minimize `x`、hard `x + 9y == 4`（唯一可行解 `x=4, y=0`）：舊 `objective_scale = 4`，真實範圍 8；`multiplier = 1.5` 時 `λ = 6`，不可行的 `x=-4, y=1`（違反量 1）能量 `−4 + 6 = 2 < 4`，是全域最小值；預設 `multiplier = 2.0` 時 `λ = 8` 剛好平手。`exact` 因窮舉仍找到可行解，但非窮舉 backend 在少量 reads 下會誤報 infeasible。

**新公式**：`objective_scale = max(1, Σ_k |c_k|·width(t_k))`，對 objective 的 raw term list 逐項相加（重複項不先合併，與原公式讀法相同），`width` 是該項在 box 上的精確 `max − min`：

| 項 | `width` | 理由 |
|---|---|---|
| `c·x`，`x ∈ [lo, hi]` | `hi − lo` | 線性，端點取極值 |
| `c·x·y`，`x ≠ y` | `max P − min P`，`P = {lo_x·lo_y, lo_x·hi_y, hi_x·lo_y, hi_x·hi_y}` | 雙線性：固定 `y` 對 `x` 線性 ⇒ 極值在 `x` 端點，對 `y` 同理 ⇒ 四個角落（皆為整數點，可取得） |
| `c·x·x` | `M² − m²`，`M = max(|lo|, |hi|)`；`lo ≤ 0 ≤ hi` 時 `m = 0`，否則 `m = min(|lo|, |hi|)` | `x²` 的 max 是 `M²`，min 是 0（區間含 0）或較小端點的平方。**不能用四角公式**：`[-4, 4]` 四角給 32，真實範圍 16 |

**定理**：對任意 bounds，`objective_scale ≥ objective_max − objective_min`。
**證明**：任取 box 內兩點 `a, b`，`f(a) − f(b) = Σ_k (t_k(a) − t_k(b)) ≤ Σ_k (max t_k − min t_k) = Σ_k width(t_k)`；取 `a = argmax f`、`b = argmin f` 即得。逐項相加是次可加性，不需要任何獨立性假設，所以重複項與共用變數都成立。∎

**§18 保證恢復**：`λ = multiplier × penalty_scale`，`multiplier > 1` ⇒ `λ > penalty_scale ≥ range + soft_bound`；任一違反 hard constraint 的指派（違反量 ≥ 1、平方 ≥ 1）能量 `≥ objective_min + λ > objective_max + soft_bound ≥ 最佳可行解能量`，故編譯後模型的全域最小值必可行。仍存在的例外與 §18 原註記相同：equality 係數或 rhs 非整數時最小違反量可能 < 1；保證的是全域最小值，非窮舉 backend 不一定找到它（retry 加倍 penalty 是為此）。反例驗算：新 `objective_scale = 8`，`multiplier = 1.5` ⇒ `λ = 12`，`x=-4, y=1` 能量 `8 > 4`，`x=4, y=0` 唯一最低。

**binary 位元一致**：`[0, 1]` 的線性寬度 `1 − 0 = 1 = M`，乘積角落 `{0, 0, 0, 1}` 寬度 `1 = M_i·M_j`，程式上每項都是 `abs(c) * 1`，加總順序不變 ⇒ 全 binary 問題與原公式**位元一致**（§26.1 golden 不變；`test_estimates_bounds.py` 另以複製的舊公式作 reference 對 250 個隨機全 binary 問題比對 `repr`）。

**語意副作用**：`objective_scale` 從「絕對值上界」變成「範圍上界」，兩者互不蘊含（`[-4, 4]`：8 vs 4；`[2, 3]`、`3x`：3 vs 9）。`SOFT_WEIGHT_SMALL` 門檻（`objective_scale × 0.01`）對區間不含 0 的整數問題會變小，但 soft weight 本來就該與「objective 變動幅度」比較（常數平移不該計入）。整數測試的期望值同步更新：`test_problem_validator_full.py` 的 `n ∈ [-5, 3]`、`x1` binary、`2n + x1 + n·x1` 由 16（`2·5 + 1 + 5`）改為 25（`2·8 + 1 + 8`）；`test_estimates_bounds.py` 與 `test_bqm_compiler_integer.py` 的暴力枚舉斷言由 `scale ≥ max|objective − constant| + soft` 改為 `scale ≥ (max − min) + soft`。新增 `tests/unit/test_penalty_dominance_integer.py` 以上述反例走 `BQMCompiler` + `ScaledPenaltyStrategy` 窮舉全部位元指派，驗證 `multiplier ∈ {1.5, 2.0}` 下最低能量指派唯一且可行。

`lhs_bounds` / `_max_abs_affine` / slack range / `compute_soft_energy_bound` 是 constraint 值域界，語意本來就是範圍或絕對值，**不在本次修正範圍**。

---

## 9. Validator

### 9.1 新 error code（進 `VALIDATOR_ERROR_CODES` 與 catalog）

| code | 條件 | path |
|---|---|---|
| `INTEGER_BOUNDS_MISSING` | integer 變數缺 `lower_bound` 或 `upper_bound` | `variables[i]` |
| `INTEGER_BOUNDS_INVALID` | `upper_bound <= lower_bound` | `variables[i]` |
| `BOUNDS_ON_BINARY` | binary 變數帶任一 bound | `variables[i]` |
| `INTEGER_RANGE_TOO_LARGE` | `|lower_bound| > 2**31 - 1` 或 `|upper_bound| > 2**31 - 1` | `variables[i]` |
| `INTEGER_REQUIRES_VERSION_1_1` | 有 integer 變數但 `version == "1.0"` | `version` |
| `INEQUALITY_MAGNITUDE_TOO_LARGE`（2026-09-09 review F-24） | inequality 的 `Σ|c|·max(|lower|,|upper|) + |rhs| > 2**53`（binary bound 視為 1；係數、rhs 非有限或非整數、或任一變數 bounds 不合法時跳過；**總和必須以 Python 整數累加**，float 累加會把 2^53+1 捨入回 2^53 而放行） | `constraints[i]` |

`recommended_action` 文字（固定）：
- `INTEGER_BOUNDS_MISSING`：「An integer variable needs both lower_bound and upper_bound; add them, or make the variable binary.」
- `INTEGER_BOUNDS_INVALID`：「upper_bound must be greater than lower_bound; a variable with equal bounds is a constant—fold it into the objective and constraints instead.」
- `BOUNDS_ON_BINARY`：「Binary variables are 0/1 and take no bounds; remove lower_bound/upper_bound, or set type to integer.」
- `INTEGER_RANGE_TOO_LARGE`：「Integer bounds must lie within ±(2^31-1); tighten the bounds or rescale the variable's unit.」
- `INTEGER_REQUIRES_VERSION_1_1`：「Integer variables require schema version 1.1; set version to \"1.1\".」
- `INEQUALITY_MAGNITUDE_TOO_LARGE`（2026-09-09 review F-24）：「An inequality's coefficients times its variables' bounds are too large for the slack range to be computed exactly, so the constraint could be encoded wrongly; rescale the unit of the coefficients or the variables, or tighten the bounds.」

### 9.2 既有規則的整數版

- `SELF_QUADRATIC_TERM`：只對 **binary** 變數報錯（`x·x = x`）；integer 的 `x·x` 合法（實測 dimod `QuadraticModel.add_quadratic("x","x")` 對 INTEGER 允許、對 BINARY 拒絕）。訊息不變。
- `TRIVIALLY_INFEASIBLE`：`lhs_bounds` 用 bounds 版；訊息的區間改為含整數 bounds 的區間。constraint 若引用任何 bounds 不合法（缺、顛倒、超界）的變數，**跳過**此判斷（該變數已有自己的 error；不得呼叫 `Variable.bounds()`）。
- `_check_variables` 對 bounds 的檢查在 `DUPLICATE_VARIABLE` / `RESERVED_VARIABLE_NAME` 之後、同一迴圈內；error pass 內任何需要 bounds 的地方都用「可能為 None」的安全取法，`Variable.bounds()` 只在 error pass 之後使用（§7）。
- `NON_INTEGER_INEQUALITY`：**維持**（§9.5）。
- `UNKNOWN_VARIABLE` 等不變。

### 9.3 新 warnings（進 `_WARNING_RECOMMENDED_ACTIONS`）

| code | 條件 | 說明 |
|---|---|---|
| `LARGE_INTEGER_RANGE` | 某 integer 變數 `integer_encoding_bits > 10` 且 `model_type == "bqm"` | 比照 `LARGE_SLACK_RANGE`；CQM 路徑不產（原生整數無代價） |
| `INTEGER_QUADRATIC_BLOWUP` | 問題含 integer 變數、`model_type == "bqm"` 且 `estimate_encoded_interactions(problem) > 2000` | 建議改用 cqm 路徑的 backend；門檻為模組常數 |

`DENSE_FOR_QPU` 的第一條件沿用 `estimated_compiled_variables`（已含編碼位元）；第二條件「最大 constraint 涉及變數數 > 30」改為以**該 constraint 涉及的位元數**計（業務 binary 各 1、integer 各 `integer_encoding_bits`、加該 constraint 的 slack 位元）——squared penalty 的 clique 是在位元上形成的。全 binary 問題兩種算法相同。

### 9.4 estimate 依 model type

- `"bqm"`：§8 的 `estimate_compiled_variables`。
- `"cqm"`：`len(problem.variables) + Σ(§15.3 會產生整數 slack 的 soft 不等式數)`，條件與 §15.3 完全相同：constraint 含 ≥ 1 個 integer 變數、非 redundant、`slack_range > 0`。估算值必須等於 `len(cqm.variables)`（§26.1 測試）。

### 9.5 `NON_INTEGER_INEQUALITY` 維持為錯誤（3a §21.3 的延續，此處定案）

理由不變：errors 必須 backend 無關。整數變數不改變這個結論：BQM 路徑的 slack 仍需整數係數。README Limitations 保留該條並補「applies to integer variables as well」。

### 9.6 `INVALID_SOLVER_PREFERENCE` 的 §9.4（3a）反射規則

`FujitsuDAOptions` 的欄位（§20.2）都是 `int | None` 且合法值 ≥ 1，落在 3a §9.4「> 0」規則內；範圍上限用 pydantic `Field(ge=, le=)` 在 schema 層擋（3a §9.4 已預告此做法）。

---

## 10. 版本、schema 與 capabilities

- `interfaces/capabilities.py`：`schema_version` 改為**最新版**（`get_args(...)[-1]` → `"1.1"`），新增 `schema_versions: list[str] = ["1.0", "1.1"]`；`supported_variable_types` 由 Literal 自動變成 `["binary", "integer"]`；`problem_json_schema` 自動含 bounds。
- MCP `solve_optimization` / `validate_optimization_problem` / `recommend_backend` docstring：把 "binary" 改為 "binary or bounded integer"，加一句 "Integer variables require version \"1.1\" and explicit lower_bound/upper_bound."。
- `export-schema` CLI 自動反映。

---

## 11. `RawSolverResult` 與陣列層

```python
class RawSolverResult(BaseModel):
    variables: list[str]
    samples: np.ndarray        # 任意 numpy 整數 dtype；BQM backend 仍給 int8（位元），CQM backend 給 int64
    energies: np.ndarray
    backend: str
    metadata: SolverExecutionMetadata | None = None
```

- `_coerce_arrays`：**先處理空輸入**（`samples == []` 或長度 0 → `np.empty((0, len(variables)), dtype=np.int8)`，維持 `test_empty_result_keeps_variable_count` 行為；實測 `np.asarray([])` 是 float64，不能先做 dtype 檢查），再 `np.asarray(samples)`；dtype 非整數（float / bool / object）→ `ValueError`；**不再強制轉 int8**。`from_dicts(..., dtype=np.int8)` 加關鍵字，**預設仍 int8**（既有斷言 `test_candidate_arrays.py` 的 int8 不變）。
- `sampleset_to_arrays(sampleset, *, dtype=np.int8)`：新增 `dtype` 關鍵字；四個 BQM backend 不傳（行為不變），CQM backend 傳 `np.int64`。
- 新增 `assert_samples_within_bounds(sampleset, bounds: Mapping[str, tuple[int,int]], *, code: str)`（`solvers/base.py`）：對 `record.sample` 的每個值檢查 **整數值**（`np.equal(x, np.floor(x))`，**型別可以是 float**——3a §17.3 已記錄 Leap 可能回 float）且在該變數的 `[lo, hi]` 內，否則 `SolverExecutionError("... returned a value outside its bounds ...", code=code)`。`leap_hybrid_cqm.py::_assert_binary_samples` 改呼叫它（bounds 從 `compiled.model` 的 `cqm.lower_bound(v)` / `upper_bound(v)` 取——實測對 BINARY 回 `0.0` / `1.0`），code 仍 `REMOTE_SOLVER_ERROR`，訊息含子字串 `"outside its bounds"`（`test_leap_hybrid_cqm_mock.py` 的 `"outside {0, 1}"` 斷言與 `dtype == int8` 斷言同步改）；`FakeLocalCQMBackend` 同。
- `orchestration/optimizer.py`：
  - 新增 `_row_keys(matrix) -> list[np.ndarray]`：若 `matrix.dtype == np.int8` 且 `matrix.min() >= 0 and matrix.max() <= 1`（實測 16M×24：min/max 0.04 s，`np.isin` 5 s——**不用 isin**；且 `np.packbits` 對 2 / −1 一律視為 1，所以這個檢查不可省）→ 現有 `_pack_rows` 的 packbits words；否則回傳各欄位的 `int64` 視圖（`np.lexsort` 的 key）。`_pack_rows` / `_words` / `_lexsort` **名稱保留**（`test_candidate_arrays.py` 直接 import 它們）。兩種 key 對「比較兩列」的結果完全相同（都是逐欄字典序），所以去重的代表元選擇與 tie-break 順序與 Phase 1 §25 定義一致。`deduplicate_samples` 與 `process_candidates` 的 tie-break 都改用 `_row_keys`。
  - `CandidateSet.samples` 註解改為「整數矩陣」。
  - `deduplicate_samples(raw, internal_variables=frozenset())` / `process_candidates(problem, raw, internal_variables=frozenset(), top_k=5)`：`internal_variables` 改為 **可選**（decode 後 service 傳空集合；既有 `test_candidate_arrays.py` / `test_ranking.py` 的位置參數呼叫**不改仍全綠**）。`process_candidates` 的 `top_k` 維持位置參數順序 `(problem, raw, internal_variables, top_k)`。

實測：`np.lexsort((energies, col1, col0))` 對 int64 欄位順序確定；`np.unique(axis=0)` 不用（無法保留 min-energy / 首見順序語意）；`int8 @ int64` 結果 int64（decode 向量化可行）。

---

## 12. `CompiledProblem` 擴充

```python
class IntegerEncoding(BaseModel):
    """一個整數變數在 BQM 路徑的二進位編碼：value = lower + Σ coefficient_k · bit_k。"""
    variable: str
    lower: int
    bits: list[str]              # internal 名稱，如 "__int_x_0"
    coefficients: list[int]      # compute_slack_coefficients(upper - lower)

class CompiledProblem(BaseModel):
    ...（3a 欄位不變）
    integer_encodings: dict[str, IntegerEncoding] = {}   # BQM 路徑填；CQM 路徑空
```

`ConstraintTrace` 不變（3a 已有 `native`、`generated_variables`）。`internal_variables` 對 BQM 路徑含編碼位元與 slack 位元；對 CQM 路徑含 §15.3 的整數 slack。

---

## 13. `ModelCompiler.decode()`

```python
class ModelCompiler(Protocol):
    @property
    def model_type(self) -> ModelType: ...
    @property
    def uses_hard_penalty(self) -> bool: ...
    def compile(self, problem, hard_penalty: float | None) -> CompiledProblem: ...
    def decode(self, compiled: CompiledProblem, raw: RawSolverResult) -> RawSolverResult:
        """把 backend 回傳的模型變數矩陣轉成「業務變數 × 整數值」矩陣：
        剔除 internal 欄位、把編碼位元合成整數；energies 與列順序不變；
        回傳的 variables 順序 = compiled.original_problem.variables 的順序（兩條路徑一致；
        實測 ExactCQMSolver 會把 binary 排在 integer 前面，所以不能沿用 raw 順序）。
        dtype：問題沒有 integer 變數時保留輸入 dtype（BQM backend 的 int8 位元路徑不變）；
        有 integer 變數時輸出 int64。metadata 原樣帶過。"""
```

- `BQMCompiler.decode`：`integer_encodings` 為空 → 只做欄位選取與重排（保留 int8）；否則對每個整數變數 `value = lower + samples[:, bit_cols] @ coefficients`（向量化，int64），binary 業務變數直接取欄，slack 與位元欄丟棄。
- `CQMCompiler.decode`：剔除 `internal_variables` 欄位、依 problem 順序重排；dtype 依上述規則（backend 已給 int64；無 integer 時仍可為 int64——CQM 路徑本來就不走 packbits，去重用欄位 key）。
- service `_run_attempts`：`raw = backend.solve(compiled, prefs)` → `decoded = compiler.decode(compiled, raw)` → `process_candidates(problem, decoded, frozenset(), top_k)`；`SolveAttempt.samples_received` 仍記 `raw.num_samples`；`metadata` 取自 `raw`。
- `tests/unit/test_service_validation_consistency.py::FakeFailingCompiler` 與 `tests/unit/test_service_compilers.py` 內任何 fake compiler 必須補 `decode`（步驟 2）。
- **1.0 問題的可觀察行為不變**：`Solution.variables` 的鍵序在 3a 是 raw 順序剔除 internal，對 BQM backend 等於 problem 順序（compiler 依 problem 順序 add_variable）；3a 的 CQM 路徑（ExactCQMSolver 重排）在 3b 改為 problem 順序——dict 相等的既有測試不受影響，若有測試比較 `list(solution.variables)` 或 JSON 文字，更新並在回報中列出。
- 不變量（測試）：decode 後 `variables` 列表 == `[v.name for v in problem.variables]`、無 `__` 前綴、每個整數值在 bounds 內（BQM 路徑由編碼保證：`compute_slack_coefficients` 的任何位元組合不超過 range）。

---

## 14. `compiler/integer_encoding.py`（BQM 路徑的整數編碼）

```python
@dataclass(frozen=True)
class AffineForm:
    """一個變數在位元上的仿射表示：constant + Σ coefficients[bit] · bit。"""
    constant: float
    coefficients: dict[str, float]    # 有序（dict 保序）

def encode_integer_variables(problem) -> tuple[dict[str, AffineForm], dict[str, IntegerEncoding]]:
    """binary → AffineForm(0, {name: 1})；integer → 位元 "__int_{name}_{k}"，係數 compute_slack_coefficients(upper - lower)，constant = lower。"""

def substitute_linear(coefficients: Mapping[str, float], forms) -> tuple[dict[str, float], float]:
    """Σ c_v · v → 位元係數 + 常數。"""

def substitute_quadratic(objective_or_terms, forms) -> tuple[dict[str, float], dict[frozenset[str], float], float]:
    """Σ c · u · v → 展開成位元的線性 / 二次 / 常數；u == v（整數自乘）時 bit·bit = bit 折進線性。"""

def expand_square_qm(qm: dimod.QuadraticModel, coefficients: Mapping[str, float], constant: float, weight: float) -> None:
    """CQM 路徑（§15.3）：把 weight · (Σ a_v · v + constant)² 累加進 QuadraticModel；
    INTEGER 變數保留 v·v 二次項，BINARY 變數 v·v 折線性。與 substitute_quadratic 共用同一個展開核心。"""
```

- 編碼方式**只有 binary expansion**（決策：不做 one-hot / unary、不提供 option，原則 7）；係數與 slack 共用 `compute_slack_coefficients`，所以「任何位元組合都落在 `[lower, upper]` 內」由同一個既有測試保證，decode 不需要 clamp。
- 位元命名 `__int_{name}_{k}`；`__` 前綴受 `RESERVED_VARIABLE_NAME` 保護，不會與使用者變數衝突；與 `__slack_{id}_{k}` 不重疊。
- 位元順序：依 `problem.variables` 順序、每個變數的 k 升冪，決定性。

### 14.1 `BQMCompiler.compile` 的變更

1. `forms, encodings = encode_integer_variables(problem)`；對 `forms` 的每個位元 `bqm.add_variable`（binary 業務變數名不變）。
2. objective：`build_objective_bqm(objective, forms)`（§14.2）。
3. 每條 constraint：`accumulate_terms` → `substitute_linear` 得位元係數 + 常數 `k`；`==`：`_add_squared_penalty(bqm, bit_coefficients, k - rhs, lam)`；不等式：`encode_slack(constraint, bounds)` 改為對「已代換的位元係數 + 常數」做分析（§14.3）。
4. `internal_variables` = 位元 ∪ slack；`integer_encodings` 填入；`num_variables = bqm.num_variables`。
5. **全 binary 問題**：`forms` 全是恆等，每一步輸出與 3a 位元一致（golden 測試）。

### 14.2 `build_objective_bqm(objective, forms=None)`

`compiler/objective.py` 的 BQM 版加可選 `forms`（`None` → 恆等，3a 行為，既有單參數呼叫不改）：線性項 `c·x` → `substitute_linear`；二次項 `c·x·y` → `substitute_quadratic`；`sign` 與 offset 規則不變；累加語意（`add_*`）不變。全 binary 時與 3a 輸出位元一致。

### 14.3 `encode_slack(constraint, bounds=None)` / `analyze_inequality(constraint, bounds=None)`

`analyze_inequality(constraint, bounds)` 的 `lhs_min` / `lhs_max` 用 bounds 版 `lhs_bounds`；`slack_range = rhs − lhs_min`（整數，因係數與 bounds 皆整數）。位元代換後的 lhs 係數（`c·coef_k`）與常數（`Σ c·lower`）餵給 `_add_squared_penalty`：`constant = Σ c·lower − rhs_normalized`。`count_slack_bits` 與 compiler 用同一個 `analyze_inequality` → estimate 不漂移。

`__slack_` 命名、redundant、soft clamp 規則全部不變。

### 14.4 2026-09-09 review 修正（F-13a / F-13e）：兩個展開核心

實作與 §14 / §15.3 的「共用展開核心」文字有兩處出入，審查後定案如下：

- **平方展開**：`_add_squared_penalty`（BQM）與 `expand_square_qm`（CQM）原本是兩份實作，且浮點運算順序不同（`lam·(a² + 2·c0·a)` / `2·lam·a_i·a_j` vs `expand_product(form, form)` 的兩個半乘積相加；整數資料相同，非二進位小數資料約 8 成位元不同）。golden 鎖定的是 BQM 那份順序，所以新增 `expand_square(coefficients, constant, weight, *, fold_square)` 採 **BQM 的運算順序**，兩邊改為薄 adapter。CQM 含整數的 soft（3a golden 不經過）在非二進位小數權重下可能差最後幾個 ulp，數學相同；整數 snapshot（71 題、52 條含整數 soft）比對 0 差異。
- **二次項代換**：`substitute_quadratic` 生產零呼叫，`build_objective_bqm` 自己對每個 term 做 `expand_product` 後直接 `add_*`；若改走 `substitute_quadratic` 會把 `(L+q1)+q2` 變成 `L+(q1+q2)`。**刪除 `substitute_quadratic`**，其測試改為直接驗證 `build_objective_bqm`（生產路徑）。§14 簽名清單與 §14.2「二次項 → `substitute_quadratic`」以此為準。
- `accumulate_terms` + 丟零係數的 dict comprehension 原本逐字出現 4 次（兩個 compiler、`estimates.py` 兩處），抽成 `validation/estimates.py::nonzero_coefficients(terms)`（F-13b）。

---

## 15. `CQMCompiler` 整數版

### 15.1 變數與 objective

- integer → `cqm.add_variable("INTEGER", name, lower_bound=lo, upper_bound=hi)`；binary 不變。
- objective 改用 `build_objective_qm(objective, bounds) -> dimod.QuadraticModel`（`compiler/objective.py` 新函式）：對每個變數 `qm.add_variable(vartype, name, lower_bound=, upper_bound=)`，線性 / 二次 / offset 規則與 BQM 版相同（含 `sign`），`x·x` 對 INTEGER 允許（實測）。`cqm.set_objective(qm)`。**全 binary 時**：仍改用 `QuadraticModel`（dimod 對 binary-only QM 與 BQM 在 `set_objective` 後語意相同）；3a 的 `test_cqm_compiler.py` 斷言的是 `cqm.objective` 的係數與能量，不是物件型別，所以不改仍綠——若有測試斷言了 `BinaryQuadraticModel` 型別，改成斷言係數（在回報中列出）。

### 15.2 hard constraint

lhs 改用 `QuadraticModel`（各變數帶 vartype / bounds），`add_constraint_from_model(lhs, sense, rhs, label, weight=None)`。整數變數直通，無 slack、無編碼。`rhs` 非整數在 CQM 合法，但 §9.5 的 validator 規則仍會擋在前面。

### 15.3 soft constraint：兩種寫法

| constraint 涉及的變數 | 寫法 | 理由 |
|---|---|---|
| 全 binary | **3a 原生**：`weight=w, penalty="quadratic"`（能量 `w·v²`） | 3a 已證明與 validator 同公式 |
| 含 ≥ 1 個 integer | **objective 形式**：`==` → 把 `w·(lhs − rhs)²` 直接加進 objective；不等式先正規化為 `<=` 並 `analyze_inequality(constraint, bounds)`：**redundant → 什麼都不加**（trace `redundant=True`，與 `encode_slack` 相同）；`slack_range > 0` → 新增 internal INTEGER slack `__slack_{id}`（bounds `[0, S]`），加 `w·(lhs + s − rhs)²`；`slack_range <= 0`（含 soft 的 trivially-infeasible clamp，log warning 同 `encode_slack`）→ **不加 slack 變數**，只加 `w·(lhs − rhs)²` | dimod 對含整數的 soft 只允許 `penalty="linear"`（`w·|v|`），與 validator 的 `w·v²` 不同公式（原則 3）；objective 形式讓 solver 看到的偏好強度與 ranking 一致 |

- **penalty 型別必須在呼叫 `add_constraint_from_model` 前決定**（實測：dimod 拋錯前已把 constraint 加進去，事後 try/except 會留下一條假的 hard constraint）。
- objective 形式的展開：`(Σ a_i v_i + k)²` 在 `QuadraticModel` 上展開為線性 / 二次（含 `v·v`）/ offset；對 binary 變數 `v·v` 折線性（QM 對 BINARY 不允許自乘）。用 `compiler/integer_encoding.py::expand_square_qm`（§14），與 BQM 路徑共用展開核心，避免兩份邏輯。
- 審查時實測（50 組含整數 slack 的小 CQM，含 S<0 clamp）：`ExactCQMSolver` 對每個業務 assignment 在所有 slack 值上的最小能量 `== sign·objective + w·max(0, v)²`。
- trace：objective 形式 `native=False`、`generated_variables=[slack]`（若有）、`penalty=w`、`slack_range=S`；原生 `native=True`。
- `internal_variables` 含 `__slack_*`（CQM 路徑第一次有 internal 變數）；`CQMCompiler.decode` 剔除。
- §21.2 一致性證明（3a）擴充：對含整數的 soft constraint，`ExactCQMSolver` 對每個業務 assignment 在所有 slack 值上的**最小** energy `== sign·objective + w·max(0, v)²`，且等於 `validate_solution(...).weighted_penalty` 加 objective（§26.1）。

### 15.4 `CQMCompiler.decode`

剔除 `internal_variables`、依 problem 順序重排（§13）。

### 15.5 `estimated_compiled_variables`（cqm）

§9.4。

---

## 16. Service 流程變更

3a §16.2 只改兩處：

```text
12. raw = backend.solve(compiled, prefs)
12a. decoded = compiler.decode(compiled, raw)          ← 新
13. process_candidates(problem, decoded, frozenset(), top_k)   ← internal 已由 decode 剔除
```

其餘（gate、limits、compiler 選擇、penalty / retry、infeasible 判定、metadata）不變。`compiled.internal_variables` 仍記錄於 `CompiledProblem`（trace / 測試用），service 不再讀它。

---

## 17. Routing 整數規則

`orchestration/routing.py` 新增三個 reason code（`REASON_DESCRIPTIONS` 同步）：

| code | 條件 | 影響 |
|---|---|---|
| `R_INTEGER_NATIVE` | 問題含 integer 變數且該 backend `model_type == "cqm"` | 只加 reason，不改 tier |
| `R_INTEGER_ENCODED` | 問題含 integer 變數且 `model_type == "bqm"` | 只加 reason；描述「integers are binary-encoded; compiled size grows with the range」 |
| `R_INTEGER_BLOWUP` | 該 backend 的 validation warnings 含 `INTEGER_QUADRATIC_BLOWUP` | 排序 key 第 2 層與 `DENSE_FOR_QPU` 同層（`1 if dense or blowup else 0`）：往後排 |

不新增「整數 → 遠端 CQM 優先於本機 SA」的 tier 規則：遠端消耗 quota，free 的本機 heuristic 仍應排前；blowup 時再往後排，由 reason 說明。deterministic、無網路不變。

---

## 18. `examples/integer_knapsack.json`

`version: "1.1"`，4 個整數變數（各 `[0, 3]`，代表每種物品拿幾個）、1 條 `<=` 容量 hard constraint、1 條 soft constraint（含整數變數，觸發 §15.3 的 objective 形式）、`solver.backend: "exact"`。編碼後變數數必須 ≤ 24（例如 4 × 2 位元 + slack ≤ 24）。**最佳解必須唯一**（§26.3 三條路徑要比較第一名的 `variables`；設計係數時用暴力枚舉確認，測試也以暴力枚舉斷言唯一）。三個既有 example 不改一字。

---

## 19. Error / warning codes 新增

| code | 類型 | status | 何時 |
|---|---|---|---|
| `INTEGER_BOUNDS_MISSING` / `INTEGER_BOUNDS_INVALID` / `BOUNDS_ON_BINARY` / `INTEGER_RANGE_TOO_LARGE` / `INTEGER_REQUIRES_VERSION_1_1` | error | `invalid_problem` | §9.1 |
| `INEQUALITY_MAGNITUDE_TOO_LARGE`（2026-09-09 review F-24） | error | `invalid_problem` | §9.1 |
| `REMOTE_QUOTA_EXCEEDED` | error | `solver_error` | Fujitsu 400 "Monthly usage exceeds specified metering limit."（§20.5） |
| `REMOTE_BUSY` | error（`retryable=True`） | `solver_error` | Fujitsu 429 "Exceed limit of number of request."（帳號 16 個 job 上限） |
| `LARGE_INTEGER_RANGE` / `INTEGER_QUADRATIC_BLOWUP` | warning | — | §9.3 |

`recommended_action`：
- `REMOTE_QUOTA_EXCEEDED`：「The remote solver's usage quota for this billing period is exhausted; wait for the next period or use a local backend.」
- `REMOTE_BUSY`：「The remote solver has too many pending jobs for this account; retry later, or delete finished job results on the vendor portal.」
- `LARGE_INTEGER_RANGE`：「An integer variable needs more than 10 encoding bits on a BQM backend; tighten its bounds, rescale its unit, or use a backend that accepts integer variables natively.」
- `INTEGER_QUADRATIC_BLOWUP`：「Binary-encoding the integer variables produces many quadratic interactions on a BQM backend; prefer a backend that accepts integer variables natively (model type cqm), or reduce the ranges.」

計數分開看：**catalog（`RECOMMENDED_ACTIONS`）34 → 41**（5 個 validator error + 2 個 remote code），2026-09-09 review F-24 再 +1（`INEQUALITY_MAGNITUDE_TOO_LARGE`）→ **42**；**warning 表（`_WARNING_RECOMMENDED_ACTIONS`）9 → 11**，warning 不進 catalog 計數。`test_error_catalog.py` 每步同步；DA 的 `_HTTP_STATUS_CODES` 表（§20.8）與 Ocean 兩張表一樣要登記到該測試的 `declared_error_codes()`，否則 AST 掃描看不到表格內的 code。`RETRYABLE_CODES` 加 `REMOTE_BUSY`。

---

## 20. Fujitsu Digital Annealer backend（`solvers/fujitsu_da.py`）

### 20.1 查證結果（2026-09-03，官方 OpenAPI YAML 與 adapter 原始碼）

| 項目 | 事實 |
|---|---|
| Base URL | `https://api.aispf.global.fujitsu.com/da` |
| 版本 | 官方提供兩份 YAML（`da-qubo-v4-en.yaml`、`da-qubo-v3c-en.yaml`，各 1408 行）；`diff` 只有路徑前綴與 operationId 不同，schema 完全相同，solver 區塊都叫 `fujitsuDA3`。**3b 只實作 v4**（§30.2） |
| 認證 | header `X-Api-Key`（永久）或 `X-Access-Token`（OAuth 2.0，一小時）；二擇一，同時給是 400 |
| 端點 | `POST /v4/async/qubo/solve` → `{"job_id"}`；`GET /v4/async/jobs/result/{Job_ID}` → `{"status": Waiting|Running|Done|Error|Canceled, "qubo_solution": {...}}`；`DELETE /v4/async/jobs/result/{Job_ID}`；`POST /v4/async/jobs/cancel {"job_id"}`；`GET /v4/async/jobs`（最多列 16 個） |
| 請求 | `{"fujitsuDA3": {...}, "binary_polynomial": {"terms": [{"coefficient": c, "polynomials": [i] | [i, j] | []}]}}`；變數是 `uint64` 索引；係數 double（建議整數且 |c| ≤ 2^52） |
| solver 參數（用到的） | `time_limit_sec` int 1..3600（預設 10）；`num_run` 1..1024（預設 16）；`num_group` 1..16（預設 1）；`num_output_solution` 1..1024（預設 5） |
| 回應 | `qubo_solution.solutions[]: {configuration: {"i": bool}, energy, penalty_energy, frequency}`（`frequency` 是 double）；`timing: {solve_time: "ms", total_elapsed_time: "ms"}`（字串）；`progress[]` |
| 限制 | 100,000 bits；request body 2 GB；job 數超過帳號上限 → 429 "Exceed limit of number of request."（YAML 只寫要刪掉不需要的結果；User's Guide 說上限 16）；月用量超過 → 400 "Monthly usage exceeds specified metering limit." |
| SDK | 無官方 Python SDK；社群 `ommx-da4-adapter` 用 `requests` + `X-Api-Key` 直打 |

### 20.2 `FujitsuDAOptions` 與 Literal

```python
class FujitsuDAOptions(BaseModel):
    time_limit_seconds: int | None = Field(default=None, ge=1, le=3600)
    num_run: int | None = Field(default=None, ge=1, le=1024)
    num_group: int | None = Field(default=None, ge=1, le=16)
    num_output_solution: int | None = Field(default=None, ge=1, le=1024)

class SolverPreferences(BaseModel):
    backend: Literal[..., "leap_hybrid_cqm", "fujitsu_da"]
    ...
    fujitsu_da: FujitsuDAOptions | None = None     # 欄位名 == backend name（3a §9.3 命名契約）
```

`None` → 不送該欄位，DA 用自己的預設。不支援其他 DA 參數（§3）。

### 20.3 Capabilities

```python
SolverCapabilities(
    name="fujitsu_da", remote=True, heuristic=True, exhaustive=False,
    supports_seed=False, supports_num_reads=False, supports_time_limit=True,
    supported_model_types=["bqm"], returns_multiple_samples=True,
    parameter_limits=[ParameterLimit(preference="fujitsu_da.time_limit_seconds", limit="time_seconds", error_code="REMOTE_TIME_LIMIT")],
    description=(
        "Fujitsu Digital Annealer (QUBO API V4, 3rd/4th-generation solver) accessed over HTTPS with an API key. "
        "Receives the compiled QUBO (objective, hard-constraint penalties and slack bits) as one binary polynomial; "
        "its native inequality and one-hot features are not used. Returns up to num_output_solution × num_group "
        "solutions; the solver's energy and penalty_energy are ignored — every solution is re-validated. "
        "Problem size limit 100,000 bits and at most 16 pending jobs per account are enforced by the vendor."
    ),
)
```

`num_reads` / `num_sweeps` / `seed` 不轉送（validator 依 3a §9.3 產 `PARAMETER_IGNORED` / `SEED_IGNORED`）。只用既有 limit key `time_seconds`，不新增 policy key（`num_run` 等的範圍由 schema `Field(le=)` 擋，計費主要依 `time_limit_sec`）。

### 20.4 HTTP（都在 `solvers/fujitsu_da.py` 內）

```python
class HttpTransport(Protocol):
    def request(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]: ...

class UrllibTransport:
    """標準庫 urllib.request。HTTPError（是 URLError 的子類，先 catch）→ 回 (status, body)；
    其餘 OSError（URLError、TimeoutError、http.client.RemoteDisconnected…）原樣拋出，由 backend 分類（§20.8）。"""
```

- `FujitsuDABackend(transport: HttpTransport | None = None, *, poll_interval_seconds: float = 2.0, request_timeout_seconds: float = 30.0, clock=time.monotonic, sleep=time.sleep)`：transport 是唯一的網路接縫；`clock` / `sleep` 讓輪詢測試不用真等。
- 模組層不得 import 任何第三方 HTTP 套件。只有一個消費者，所以 Protocol 與實作都放在 backend 檔內，不另立模組（原則 7）。

### 20.5 設定、availability、redaction

- 環境變數（**只在 backend 內、每次呼叫時讀，不快取，不進 `ServerSettings`**）：`FUJITSU_DA_API_KEY`（必要）、`FUJITSU_DA_URL`（預設 §20.1 base URL）。API 版本固定 `v4`，不提供切換（沒有消費者；v3c 見 §30.2）。
- `is_available()`：key 缺或空 → `AvailabilityStatus(category="credentials_missing", detail="Fujitsu DA API key not configured")`；URL 不是 `https://` 開頭 → `config_invalid`（`error_code="BACKEND_CONFIG_INVALID"`，detail 不含值）；否則 available。無網路 I/O。
- 測試隔離：`tests/conftest.py` 的 autouse fixture 目前只 `delenv("DWAVE_API_TOKEN")`，改為同時清除 `FUJITSU_DA_API_KEY` / `FUJITSU_DA_URL`（否則開發機設了 key 會改變 capabilities / recommend / CLI 表的輸出）；`tests/remote_live/conftest.py` 同。
- `solvers/metadata.py` 通用化：`_CREDENTIAL_ENV_VARS = ("DWAVE_API_TOKEN", "FUJITSU_DA_API_KEY")`，`redact()` 對每個非空值做替換；`_REDACTION_PATTERNS` 加 `(r"X-Api-Key: [^\n]+", "X-Api-Key: ***")`、`(r"X-Access-Token: [^\n]+", "X-Access-Token: ***")`、`(r'"X-Api-Key":\s*"[^"]*"', '"X-Api-Key": "***"')`。D-Wave 的 Ocean config 讀取邏輯不變。所有進 `SolveError` / log / metadata 的字串（含 HTTP 回應 body 摘要）必經 `redact()`。**2026-09-09 review F-10 修正**：`_CREDENTIAL_ENV_VARS` / `_REDACTION_PATTERNS` 已移除；改由各 backend 在 `SolverCapabilities.credentials`（`CredentialDeclaration`：`env_vars` / `header_names` / `value_patterns`）宣告，`SolverRegistry` 註冊時餵給 `metadata.declare_credentials`；Ocean config 檔 token 由 `solvers/ocean.py` 以 `register_secret_source` 提供；`metadata.py` 不再出現任何廠商名。
- 不得把 key 放進 URL query、log、metadata、exception `__cause__`（沿用 3a `call_ocean` 的「在 except 外 raise」手法，抽成 `solvers/ocean.py` 的通用 `guarded_call`，或在 `fujitsu_da.py` 內同樣寫法——擇一，不複製第三份）。

### 20.6 `resolve_time_limit`

`preferences.fujitsu_da.time_limit_seconds` 若給則用之，否則 DA 預設 `10`；回 `float`。無網路。service 以 `time_seconds` policy 比對（預設 300）。

### 20.7 `solve()` 流程

1. `bqm = compiled.model`；變數索引：依 `bqm.variables` 順序 0..n-1（`index_of: dict[str, int]`）。
2. 請求 body：`fujitsuDA3 = {"time_limit_sec": int(effective)}` + 使用者給的 `num_run` / `num_group` / `num_output_solution`；`binary_polynomial.terms`：每個 linear bias `{"coefficient": h_i, "polynomials": [i]}`、每個 quadratic `{"coefficient": J_ij, "polynomials": [i, j]}`、offset `{"coefficient": offset, "polynomials": []}`（offset 為 0 也送，讓 energy 可比對）。**不送** `penalty_binary_polynomial`、`inequalities`、one-hot、`guidance_config`、`fixed_config`；**不送**問題名稱或任何業務字串。
3. `POST {url}/v4/async/qubo/solve`，headers `X-Api-Key`、`Content-Type: application/json`、`Accept: application/json`；timeout `request_timeout_seconds`。
4. 輪詢 `GET .../jobs/result/{job_id}`：間隔 `poll_interval_seconds`，總等待上限 `effective_time_limit + 60`；`status` 為 `Done` → 取結果；`Error` → `REMOTE_SOLVER_ERROR`（訊息含 `message`，經 redact）；超時 → 嘗試 `POST .../jobs/cancel`（best effort，失敗只 log）後 `REMOTE_TIMEOUT`。（**2026-09-09 review F-23**：deadline 在每次 GET 前後各檢查一次——GET 前已逾時就不再發請求；超時後改為 cancel 再 DELETE（皆 best-effort）；實際最壞總等待為 `effective_time_limit + 60` 再加 submit、最後一次 GET、cancel、DELETE 各至多一個 `request_timeout_seconds`；`urllib` 的 timeout 是每個 socket 操作的逾時，單次 GET 若對端持續慢速回應可超過該值，屬已知限制。）
5. 取得結果後 `DELETE .../jobs/result/{job_id}`（best effort；失敗 log warning，不影響結果）——帳號的 job 位子有限。
6. 轉換：每個 solution → 一列（`configuration["i"]` → 0/1）；**任何一個變數索引在 `configuration` 缺席、或值不是 bool → `REMOTE_SOLVER_ERROR`**（不補 0：替求解器捏造一個位元是偷偷做決定，原則 5）；`energies = energy`（不加 `penalty_energy`：我們沒送 penalty polynomial，DA 回 0）；`frequency` 不展開（§3）。`RawSolverResult(samples=int8)`。
7. metadata：`sanitize_sampleset_info({"timing": {...}}, backend, remote=True)` 前先把 DA 的 `solve_time` / `total_elapsed_time`（毫秒字串）轉成 µs 浮點；`TIMING_WHITELIST` 加 `solve_time`、`total_elapsed_time`；`solver_id = "fujitsuDA3/v4"`；`num_reads_requested = num_output_solution × num_group`，未給者以 DA 預設（5、1）代入，所以永遠有值；`effective_time_limit_seconds`。

**2026-09-09 review 修正（F-09）：submit 之後的任何失敗都 best-effort 釋放 job。** 原步驟 4 / 5 只在 `Done` 成功後 DELETE、`Error` 後 DELETE、超時後 cancel；其餘終止（`Done` 但無 `qubo_solution`、`Canceled` / 非預期 status、payload 非 dict、輪詢中 HTTP 非 2xx、transport 例外、`KeyboardInterrupt`）直接 raise，job 留在帳號佔用 slot（原則 5：不留下使用者看不見的付費副作用）。修正後 `_await_result` 的所有 raise 出口統一經過一次 best-effort 清理，依「已知多少」決定送什麼（Fujitsu OpenAPI YAML 語意：`POST .../jobs/cancel` 只對 Waiting 的 job 有效、`DELETE .../jobs/result/{id}` 只刪已完成的 job，其餘情況皆回 200 帶目前狀態，是無害 no-op）：

| 出口 | 清理 |
|---|---|
| 已讀到終止 status（`Done` 缺 `qubo_solution`、`Error`、`Canceled`、其他非預期字串） | `DELETE` 一次 |
| 狀態未知（輪詢請求本身失敗：HTTP 非 2xx / transport 例外 / 非 JSON；payload 非 dict；`KeyboardInterrupt` 落在 request 或 sleep） | `POST cancel` 再 `DELETE` |
| 輪詢超時 | cancel 再 DELETE（2026-09-09 review F-23 修正，與「狀態未知」列一致） |
| `Done` 成功 | 維持步驟 5 由 `solve()` DELETE |

清理失敗只留 WARNING（經 redact，job id 可出現、key 不可），訊息註明「the job may still occupy a job slot on the vendor side」；原本要 raise 的例外與 code 完全不變。`BaseException` 也清理（slot 是付費資源），但清理本身只 catch `Exception`，第二次中斷會直接穿出、不被吞掉。

### 20.8 例外 → code（`_HTTP_STATUS_CODES` 表 + 訊息比對）

| 情況 | code |
|---|---|
| HTTP 401 / 403 | `REMOTE_AUTH_FAILED` |
| HTTP 400 且 body 含 `Monthly usage exceeds` | `REMOTE_QUOTA_EXCEEDED` |
| HTTP 400 且 body 含 `X-Access-Token` / `X-Api-Key` / `Invalid request header`（YAML 列出的三種 header 錯） | `BACKEND_CONFIG_INVALID`（`configuration_error`）——我們的請求組錯 |
| HTTP 400 其他（含 "the number of variables exceeds the limit"、JSON 格式錯） | `REMOTE_SOLVER_ERROR`（DA 對問題層級的拒絕也是 400，不能一律當設定錯） |
| HTTP 413 | `REMOTE_SOLVER_ERROR`（訊息「payload too large」） |
| HTTP 429 | `REMOTE_BUSY` |
| HTTP 5xx | `REMOTE_SOLVER_ERROR` |
| `TimeoutError`（含 `URLError` 其 `.reason` 為 `TimeoutError`；實測：連線後 read timeout 拋的是裸 `TimeoutError`） | `REMOTE_TIMEOUT` |
| 其他 `OSError`（DNS、拒連、`RemoteDisconnected`…） | `REMOTE_SOLVER_ERROR`（retryable 已是 True） |
| 輪詢超時 | `REMOTE_TIMEOUT` |
| job `status == "Error"` | `REMOTE_SOLVER_ERROR` |
| 回應 JSON 缺欄位 / 型別錯 / configuration 缺索引 | `REMOTE_SOLVER_ERROR` |

例外包裝沿用 3a `call_ocean` 的「在 except 外 raise、不帶 `__cause__`」手法：把 `solvers/ocean.py::call_ocean` 的核心抽成不含 Ocean 字樣的 `solvers/metadata.py::guarded_call(what, classify, fn)`（`call_ocean` 改為它的薄包裝，既有測試不改），DA 用同一個。

`configuration_error` 的 status：service 對 `OptimizerError` 目前一律回 `solver_error`。3b 加一條通用機制：`SolverExecutionError.__init__(message, *, code=None, status=None)`，`status: str | None`，backend 可指定 `"configuration_error"`；service `_run_attempts` 的 `except OptimizerError` 取 `getattr(exc, "status", None) or "solver_error"`。3b 只有 `BACKEND_CONFIG_INVALID` 用它；既有 D-Wave 的 `DWAVE_CONFIG_INVALID` 在 sampler 建構階段的行為**不變**（仍 `solver_error`，避免改既有測試期望；統一留給之後）。

### 20.9 Registry

第六個，順序 `exact, simulated_annealing, dwave_qpu, leap_hybrid_bqm, leap_hybrid_cqm, fujitsu_da`。名單測試（`test_registry`、`tests/mcp/test_capabilities`、`test_cli`）同步為六個；`test_no_backend_names.py` 名單加 `"fujitsu_da"`。

### 20.10 Live 測試

`tests/remote_live/test_fujitsu_da_live.py`：`pytestmark = pytest.mark.remote`；無 `FUJITSU_DA_API_KEY` → skip（註解說明，比照 D-Wave；`remote_live/conftest.py` 的 skip 條件改為「該測試需要的 key」而不是一律看 `DWAVE_API_TOKEN`）。最小 knapsack、`time_limit_seconds=1`。沒有帳號就永遠 skip（決策 3）。

---

## 21. Metadata

`SolverExecutionMetadata` 不加欄位。`TIMING_WHITELIST` 加 `solve_time`、`total_elapsed_time`（DA）；`sanitize_sampleset_info` 的數值規則不變（呼叫端先轉 float µs）。

### 21.1 2026-09-09 review 修正（F-18）

- **刪除** Phase 2 §17 的 `logical_variables` / `logical_interactions`：六個 backend 沒有任何一個寫入，全庫零讀者，卻在 MCP 的 `solve_optimization` 輸出 schema 中永遠是 `null`。MCP 輸出 schema 因此少兩個欄位（README 未列過這兩個欄位）。
- `sanitize_sampleset_info(info, backend)` 回到 Phase 2 §17 的兩參數簽名：`remote=` 關鍵字生產端無人傳（本機 backend 不產 metadata，`SolveResult.metadata` 恆 `None`），刪除；測試用的本機 CQM fake 以 `model_copy(update={"remote": False})` 覆寫。
- `ExecutionPolicy.enabled_backends` 補環境變數入口 `ANNEALBRIDGE_ENABLED_BACKENDS`（逗號分隔的 registry 名稱；未設或空 = 全部），讓既有的 `BACKEND_DISABLED_BY_POLICY` 在 CLI / MCP 部署中可達（Phase 2 §9 的 env 清單原本沒有它）。

---

## 22. Logging 與 Security

- DA：log 記 backend、job_id（不是秘密）、變數數、solutions 數、solve_time；不記 key、不記 URL 的 query、不記 body。所有訊息經 `redact()`。
- `test_credential_leak.py` 參數化加 `fujitsu_da`：transport 拋含 key 的例外 / 回含 key 的 body；斷言 result JSON、caplog、所有 message 不含 key。
- README Security 段加 Fujitsu：key 只從環境變數讀、不進任何輸出、live 測試 opt-in。

---

## 23. README / OVERVIEW

- JSON Input Format：加 integer 變數段（bounds、version 1.1、負下界、範圍上限）、`integer_knapsack.json` 範例片段。
- Solver Backends 表加 `fujitsu_da` 列；說明 BQM 路徑、API key、100K bits、16 jobs。
- Environment variables：`FUJITSU_DA_API_KEY` / `FUJITSU_DA_URL`。
- Limitations：整數變數在 BQM 路徑的編碼代價（LARGE_INTEGER_RANGE / INTEGER_QUADRATIC_BLOWUP）、`NON_INTEGER_INEQUALITY` 仍適用、DA 原生限制未用。
- OVERVIEW §三名詞表「Variable」改為「0/1 或有上下界的整數（Phase 3b）」；§四加 Phase 3b 白話段；§七狀態表。§五原則不動。

---

## 24. 不要做的事情（3b 特別注意）

- 在 JSON 裡出現位元、編碼方式、slack（原則 1）
- 在 backend 內做整數編碼或 decode（是 compiler 的事）
- CQM 路徑對含整數的 soft constraint 用 `penalty="linear"`（與 validator 公式不一致）或事後 try/except
- 用 `np.unique(axis=0)` 去重（會丟掉 min-energy / 首見順序語意）
- 把 `RawSolverResult.samples` 一律轉 int64（exact 16M 列的記憶體會 ×8；位元路徑保持 int8）
- 讀 `FUJITSU_DA_API_KEY` 進 `ServerSettings` 或任何模型欄位
- 把 API key 放 URL、log、metadata、exception chain
- 送問題名稱、description 或任何業務字串給 DA
- 用 DA 的 `energy` / `penalty_energy` 判可行或排名
- 展開 `frequency` 成重複列
- 為 DA 加 `requests` / `httpx` 依賴
- 新增 backend 命名的 policy 欄位或 env 變數（`ANNEALBRIDGE_MAX_DA_*`）
- 為 one-hot / unary 編碼預留 option 或分支（原則 7）

---

## 25. Development Order

每步跑完全部測試再進下一步。**步驟 0（開工前）**錄 golden：`tests/golden/record_phase3a_golden.py`（commit 進 repo）對**固定清單**——三個 example、`tests/scenarios/` 用到的全部問題 JSON / 建構函式、`tests/unit/test_bqm_compiler.py` 與 `test_cqm_compiler.py` 內的具名 fixture 問題（腳本內以路徑或 fixture 名逐一列出）——記錄 3a 程式碼（`20ba640`）的：`BQMCompiler` 輸出（變數順序列表、linear dict、排序後的 `[u, v, bias]` 二次列表、offset）、`CQMCompiler` 輸出（變數順序、objective 同形式、每條 constraint 的 lhs 同形式 + sense + rhs + soft weight / penalty）、`estimate_compiled_variables`、`compute_penalty_scale`、`validate_problem_full(problem, capabilities=<exact caps>)` 的完整輸出（warnings、estimate、objective_scale）。自訂 JSON 格式（不用 `bqm.to_serializable()`，它不是穩定契約）；float 經 json 往返可精確相等（實測）。存成 `tests/golden/phase3a_compile.json`，`tests/unit/test_golden_phase3a.py` 逐項 `==` 比對，從步驟 1 起每步都跑。

1. **IR 1.1 與估算**：`Variable` bounds（含 bool 拒絕）+ `version` Literal + `bounds()`；`variable_bounds` 與 §8 全部 bounds-aware 函式（**keyword-optional**，既有呼叫不改）；production 呼叫端（`slack.py`、`penalty/strategy.py`、validator）傳 bounds；§9.1 五個 error、§9.2、§9.3 兩個 warning（含 `DENSE_FOR_QPU` 第二條件改位元數）、§9.4 estimate；catalog +5 error（總數 39，warning 表 +2 另計）；`interfaces/capabilities.py` §10 **與 `tests/mcp/test_capabilities.py` 的 `schema_version` / `supported_variable_types` 斷言同步**；golden 測試 + `test_estimates_bounds.py`（bounds 版公式對小問題與暴力枚舉一致）。三個 example 不變。
2. **陣列層與 decode 接線**：`RawSolverResult` dtype 放寬（空輸入先處理）、`from_dicts(dtype=)`、`sampleset_to_arrays(dtype=)`、`assert_samples_within_bounds`（`leap_hybrid_cqm.py` 與 `FakeLocalCQMBackend` 改用；`test_leap_hybrid_cqm_mock.py` 的 dtype / 訊息斷言同步）、`_row_keys`（保留 `_pack_rows`）、`ModelCompiler.decode` + `IntegerEncoding` / `CompiledProblem.integer_encodings`、兩個 compiler 的 decode（此時 BQM 版只做剔除 internal + 重排，保留 int8）、`FakeFailingCompiler` 等 fake 補 `decode`、service 接線、`process_candidates` / `deduplicate_samples` 的 `internal_variables` 改可選；`test_candidate_arrays.py` 加整數列案例（與逐列參考實作一致）、`test_batch_arithmetic_integer.py`、`test_service_decode.py`。位元一致仍綠。
3. **BQM 整數編碼**：`compiler/integer_encoding.py`、`build_objective_bqm(objective, forms)`、`encode_slack(constraint, bounds)`、`BQMCompiler.compile` / `decode`；`test_integer_encoding.py`、`test_bqm_compiler_integer.py`（含：exact 對 `integer_knapsack.json` 的解 == 對整數值的暴力枚舉最佳解；負下界；`x·x`；decode 值在 bounds 內；`__int_` 不外洩；`estimate_compiled_variables == compiled.num_variables`）；`test_bqm_compiler.py` 不改仍綠。
4. **CQM 整數版**：`build_objective_qm`、hard lhs QM、§15.3 兩種 soft、`expand_square_qm`、decode；`FakeLocalCQMBackend` 與 `leap_hybrid_cqm.py` 改 bounds 斷言 + int64；`test_cqm_compiler_integer.py`（§15.3 一致性證明、ExactCQMSolver 交叉驗證含整數）；`test_cqm_compiler.py` 不改仍綠（或只改型別斷言，回報）。
5. **範例、scenario、routing、文件字串**：`examples/integer_knapsack.json`；`tests/scenarios/test_integer_knapsack.py`（exact / SA seed 固定 / FakeLocalCQM 三路徑同一最佳 objective，且解相同）；routing §17 + `test_routing.py`；MCP docstring；`test_phase1_compat.py` 加 1.1 範例可 parse、1.0 範例位元一致引用 golden。
6. **Fujitsu DA**：`solvers/metadata.py` 通用化（credential env 清單、pattern、`guarded_call`；§20.5、§20.8）+ 既有 D-Wave 測試不改仍綠 → `tests/conftest.py` / `remote_live/conftest.py` 環境變數隔離 → `FujitsuDAOptions` + Literal + `models/__init__` → `SolverExecutionError.status` + service 一行 → catalog +2（總數 41）並登記 DA 表到 `declared_error_codes()` → `solvers/fujitsu_da.py`（含 transport）→ registry 六個 + 名單測試 + `test_no_backend_names.py` 名單 → `tests/fakes/da_transport.py`（腳本化：依序回 job_id、Waiting、Running、Done 結果；可設定任一步的 HTTP status / body / 例外）→ `tests/remote_mock/test_fujitsu_da_mock.py`（§26.5）→ `test_credential_leak.py` 參數化 → live 測試。
7. **收尾**：README / OVERVIEW、`annealbridge_phase3b_acceptance.md`（比照 3a：§27 逐條 ✅/❌ + 證據），CI 不需改。

---

## 26. Tests

### 26.1 Unit（重點）

- **golden 位元一致**（步驟 1 起每步都跑）：`tests/golden/phase3a_compile.json` vs 現行 compile / estimate / penalty_scale / `validate_problem_full` 對固定清單的 1.0 問題**完全相等**（§25 步驟 0）。
- `test_batch_arithmetic_integer.py`：隨機整數矩陣（含負值、bounds 內）上 `validate_batch` 的 `feasible` / `soft_violation_score` 與逐列 `validate_solution`、`evaluate_objective_batch` 與逐列 `evaluate_objective` 逐元素相等（3a §32.1 要求的證明）。
- `test_models.py`：bounds 型別、version Literal、`Variable.bounds()`。
- `test_problem_validator.py`：§9.1 五個 code 各至少一例；`SELF_QUADRATIC_TERM` 對 integer 不報、對 binary 報；`TRIVIALLY_INFEASIBLE` 用 bounds（例：`x ∈ [0,3]`，`2x <= 7` 可行、`2x >= 7` 不可行）；1.1 純 binary 合法。
- `test_estimates_bounds.py`：對隨機小問題（≤ 3 個變數、範圍 ≤ 4）暴力枚舉驗證 `lhs_bounds` 的 min/max 可達、`compute_objective_scale ≥ objective_max − objective_min`（§8.1，2026-09-09 F-06 前為 `≥ max|objective|`）、`compute_soft_energy_bound ≥` 任何 assignment 的 soft 能量。
- `test_integer_encoding.py`：`encode_integer_variables` 位元數 == `integer_encoding_bits`；任何位元組合的值在 bounds 內且每個整數值都可達；`substitute_quadratic` 對 `x·x`、`x·y`、`x·b` 的展開與數值代入一致（隨機 assignment）。
- `test_bqm_compiler_integer.py`、`test_cqm_compiler_integer.py`：§25 步驟 3、4 所列。
- `test_candidate_arrays.py`：int64 列的去重（min-energy、首見順序）與 tie-break 對逐列參考實作一致；int8 位元路徑輸出與 3a 完全相同（同一測試資料）。
- `test_service_decode.py`：service 對 fake BQM backend 回傳的位元矩陣，`Solution.variables` 是整數值、無 `__`、在 bounds 內；`samples_received` 記原始列數。
- `test_routing.py`：三個 reason；blowup 問題的 BQM backend 排在 CQM backend 後（在同 usable 層內）。
- `test_metadata.py`：redaction 對 `FUJITSU_DA_API_KEY` 值與三個新 pattern；`solve_time` / `total_elapsed_time` 進 whitelist。

### 26.2 Architecture

- `test_no_backend_names.py` 名單 + `"fujitsu_da"`。
- `test_import_boundaries.py`：`compiler/integer_encoding.py` 在套件內只 import `models` / `validation.estimates`；`solvers/fujitsu_da.py` 不 import `requests` / `httpx`（掃描）。
- `test_fifth_backend.py`：不改仍綠（六個內建 + fake）。

### 26.3 Scenario

`test_integer_knapsack.py`：JSON → Service → exact success、objective 等於暴力枚舉最佳值；SA（固定 seed）success 且 objective 相同；FakeLocalCQM success 且相同；三者第一名的 `variables` 相同；無 `__`。

### 26.4 MCP

- `validate_optimization_problem` 對 1.0 + integer → `INTEGER_REQUIRES_VERSION_1_1`；對 1.1 integer knapsack → valid，`estimated_compiled_variables` 含編碼位元。
- `solve_optimization` 對 `integer_knapsack.json` exact → success、整數值。
- `get_optimization_capabilities`：`schema_version == "1.1"`、`schema_versions == ["1.0","1.1"]`、`supported_variable_types == ["binary","integer"]`、六個 backend。
- `tools/list` 仍四個。

### 26.5 Remote mock（`test_fujitsu_da_mock.py`，`FakeDATransport`）

- 請求：URL（`FUJITSU_DA_URL` 覆寫生效、路徑 `/v4/async/qubo/solve`）；headers 恰含 `X-Api-Key` / `Content-Type` / `Accept`，無 `X-Access-Token`；body 的 `binary_polynomial` 與 `bqm` 逐項一致（含 offset 項）；`fujitsuDA3.time_limit_sec == resolve_time_limit`；未給的 options 不出現；不含問題名稱字串。
- 流程：Waiting → Running → Done 輪詢（用注入的 clock / sleep，不真等）；Done 後 DELETE 被呼叫一次；DELETE 失敗不影響結果。
- 轉換：`configuration` → int8 列、順序保留；缺索引 / 非 bool 值 → `solver_error` / `REMOTE_SOLVER_ERROR`；`energies == energy`；`frequency` 不展開；metadata `solve_time` / `total_elapsed_time`（µs）、`solver_id`、`effective_time_limit_seconds`。
- 上限：`time_limit_seconds=400` 且 policy 300 → `resource_limit_exceeded` / `REMOTE_TIME_LIMIT`，未 clamp；`num_run=2000` → schema 拒絕（pydantic ValidationError，MCP 為 tool error）。
- 例外 §20.8 每列一個測試（transport 拋 `TimeoutError`、`URLError(reason=TimeoutError())`、`URLError(reason=ConnectionRefusedError())`、`http.client.RemoteDisconnected` 各一）；`BACKEND_CONFIG_INVALID` → `configuration_error`；400 問題層級拒絕 → `solver_error`；`REMOTE_BUSY.retryable is True`。
- 輪詢超時 → cancel 被呼叫 → `REMOTE_TIMEOUT`（**2026-09-09 review F-23**：deadline 在每次 GET 前後各檢查一次，序列 clock 驗證「GET 前已逾時就不發 GET」；超時後 cancel 再 DELETE 各一次、cancel 在前；訊息含 budget 70 與 request timeout 30，不含 key）。
- 2026-09-09 F-09（`TestFailedJobCleanup`）：終止 status（`Done` 缺 `qubo_solution` / `Error` / `Canceled`）→ DELETE 恰一次、不 cancel；狀態未知（輪詢 HTTP 500 / transport 例外 / payload 非 dict / `KeyboardInterrupt`）→ cancel 再 DELETE 各一次、請求序列固定；清理失敗（DELETE 500、兩者皆拋例外、body 含 key）→ 原例外與 code 不變、WARNING 含 job id 與「may still occupy a job slot」且無 key；清理中第二次 `KeyboardInterrupt` 穿出、不送 DELETE；經 service 仍 `solver_error`；成功與超時路徑呼叫次數不變。
- `is_available()` 兩種失敗 category（無 key、http URL）與 available；無網路（transport spy 未被呼叫）。
- 整數問題經 DA mock：fake 回傳的位元列經 decode 成整數值、在 bounds 內。
- `allow_remote_retries=False` → 1 attempt + `REMOTE_RETRIES_DISABLED`；`True` → 可到 `1 + max_retries`。
- credential leak：key 出現在 transport 例外文字與 400 body → 全部遮罩。

### 26.6 Remote live（opt-in）

`test_fujitsu_da_live.py`（需 key）、既有 D-Wave 三個不變。

---

## 27. Acceptance Criteria

- [ ] Phase 1/2/3a 全部測試不變仍通過；三個 1.0 example 不改一字；golden 位元一致測試通過（BQM / CQM 輸出、estimate、penalty_scale）
- [ ] `version` 接受 `"1.0"` / `"1.1"`；1.0 + integer → `INTEGER_REQUIRES_VERSION_1_1`；1.1 純 binary 合法
- [ ] §9.1 五個 error、§9.3 兩個 warning 皆有測試；`SELF_QUADRATIC_TERM` 只對 binary
- [ ] estimates 全部 bounds-aware，暴力枚舉測試通過；`estimate_compiled_variables == compiled.num_variables`（BQM 路徑，含編碼位元與 slack）
- [ ] `ModelCompiler.decode()` 存在；service 在 `process_candidates` 前呼叫；`__int_` / `__slack_` 不出現在任何 `Solution`
- [ ] BQM 路徑：整數 binary expansion、負下界、`x·x`、decode 值在 bounds 內；exact 對 `integer_knapsack.json` == 暴力枚舉最佳解
- [ ] CQM 路徑：INTEGER 變數直通；含整數 soft constraint 走 objective 形式且與 validator 同公式（測試證明）；binary-only soft 維持 3a 原生；penalty 型別在呼叫前決定
- [ ] `RawSolverResult` 接受 int8 / int64；位元路徑仍 int8；去重與 tie-break 對整數列正確且對位元列位元一致
- [ ] CQM backend（Leap 與 fake）值域斷言改為 bounds
- [ ] Routing 三個 reason；blowup 往後排；deterministic、無網路
- [ ] `fujitsu_da`：只讀 `FUJITSU_DA_API_KEY` / `FUJITSU_DA_URL` 兩個 env、不進 ServerSettings；測試環境隔離這兩個變數；`X-Api-Key` header；標準庫 HTTP、無新依賴；非同步流程含 DELETE 與 cancel；例外 → code 表；`BACKEND_CONFIG_INVALID` → `configuration_error`；`REMOTE_BUSY` retryable；redaction 涵蓋 key 與 header；不送業務字串
- [ ] registry 六個、名單測試、`test_no_backend_names.py` 名單、`test_fifth_backend.py` 不改仍綠
- [ ] catalog 41 個 code 皆有 `recommended_action`；warning 覆蓋測試通過；reason 覆蓋測試通過
- [ ] `get_optimization_capabilities` 回 `schema_version "1.1"`、`schema_versions`、`supported_variable_types` 含 integer
- [ ] `examples/integer_knapsack.json` 存在，三條路徑（exact / SA / fake CQM）scenario 同一最佳值
- [ ] README / OVERVIEW 更新；`pytest` 全過，無 skip / xfail（除 remote_live 註解 skip）；DA live 測試在無 key 時 skip、有 key 時可跑
- [ ] `annealbridge_phase3b_acceptance.md` 逐條核對

---

## 28. 3a spec §32 接點的落實對照

| 3a §32 條目 | 3b 落點 |
|---|---|
| `Variable.type` 擴充 + bounds；`version` 1.1；1.0 不改 | §7 |
| CQM 加 INTEGER 分支；soft `"quadratic"` 只限 binary，含整數要另訂規則並重做一致性證明 | §15.1、§15.3 |
| BQM 整數 → 二進位編碼（建議 binary expansion）；位元 internal；`estimate_compiled_variables` 納入；`INTEGER_QUADRATIC_BLOWUP` | §14、§8、§9.3 |
| `sampleset_to_arrays` / `RawSolverResult` 由 int8 放寬；{0,1} 斷言改 bounds；`_pack_rows` 通用化；batch 算術測試證明 | §11、§26.1 |
| `NON_INTEGER_INEQUALITY` 是否放寬 | §9.5：維持 |
| `supports_integer_variables` flag；routing 整數規則 | **不加 flag**（BQM 路徑靠編碼、CQM 路徑原生，每個 backend 都能解，flag 沒有消費者）；routing 見 §17 |
| Fujitsu：查證、`FujitsuDAOptions`、Literal、`parameter_limits`（可能新 key）、availability、redaction、例外表、extra、mock / live | §20；**不新增 limit key**（只用 `time_seconds`），**不加 extra**（標準庫） |

---

## 29. 對既有規格的偏離（明列）

| 來源 | 3b 變更 | 理由 |
|---|---|---|
| outline §2.1「編碼策略允許 compiler option 切換」 | 只做 binary expansion，無 option | 原則 7；沒有第二個消費者 |
| outline §2.3「`parameter_limits` 宣告新的 limit key（如 iterations）」 | DA 只用 `time_seconds` | DA 沒有 iterations 參數；`num_run` 等只影響平行度不影響計費，範圍由 schema 擋 |
| outline §2.3「`fujitsu` extra」 | 無 extra | 標準庫 HTTP，沒有可選依賴 |
| 3a §32.1「`supports_integer_variables`」 | 不加 | 見 §28 |
| 3a §17.3「`sampleset_to_arrays` 本身不改」 | 加 `dtype=` 關鍵字（預設不變） | 放寬 int8 的最小改動 |
| Phase 2 §14「任何 backend exception → `solver_error`」 | `SolverExecutionError.status` 可指定 `configuration_error`（只有 `BACKEND_CONFIG_INVALID` 用） | DA 的 400 header 錯是伺服器設定問題，回 `solver_error` 會誤導 Agent 去改問題 |
| Phase 1 §12「`SELF_QUADRATIC_TERM`」 | 只對 binary | 整數的 x² 有意義 |
| Phase 2 §19 redaction 只有 D-Wave | 通用化為 credential env 清單 + 廠商 header pattern；`call_ocean` 核心抽成 `guarded_call` | 第二個廠商 |
| Phase 2 §22 `schema_version` 是唯一版本 | 改為「最新版本」，另加 `schema_versions` 列表 | IR 有兩個版本 |
| 3a §32.1 / outline §2.4「有整數變數且二次項 → CQM 優先」 | 不加 tier 規則，只加 reason 與 blowup 降序（§17） | 遠端消耗 quota，免費本機 heuristic 仍應排前；blowup 才是該往後排的訊號 |
| 3a `ModelCompiler` Protocol | 加 `decode`；既有 fake compiler 補方法 | 整數編碼的 decode 是 compiler 的事 |
| Phase 2 §14 / 3a `process_candidates(problem, raw, internal_variables, top_k)` | `internal_variables` 改可選（預設空），service 傳空集合 | decode 已剔除 internal；保留參數讓既有測試不改 |
| 3b §14 / §14.2 `substitute_quadratic`；§15.3「與 `substitute_quadratic` 共用展開核心」 | 刪除 `substitute_quadratic`；平方展開改共用 `expand_square`（BQM 運算順序），`expand_product` 只服務 objective 的二次項 | 2026-09-09 review F-13a / F-13e，見 §14.4 |
| Phase 2 §17 `logical_variables` / `logical_interactions` | 刪除 | review F-18：零寫零讀，MCP schema 永遠 null；見 §21.1 |
| Phase 2 §17 `sanitize_sampleset_info(info, backend)` 的 `remote=` 擴充（3a 加） | 移除，回到 spec 簽名 | review F-18：生產無人傳；見 §21.1 |
| Phase 2 §9 env 清單 | 加 `ANNEALBRIDGE_ENABLED_BACKENDS` | review F-18：`enabled_backends` 原本無 env 入口，`BACKEND_DISABLED_BY_POLICY` 不可達 |
| Phase 1 §17 `CompiledProblem.constraint_trace`（含 `ConstraintTrace.source_description` / `native`）與 `CompiledProblem.objective_scale` | 保留，但**目前無生產讀者**：兩個 compiler 寫入，只有測試與 golden 讀；CLI 顯示的是 `ProblemValidationResult.objective_scale` | review F-18：spec 明定的 explainability 欄位，依原則 7 不新增消費者，只記錄現況 |
| Phase 2 §21 `build_service(settings)` | 保留（`build_state(settings).service` 的單行 wrapper），生產零呼叫 | review F-18：CLI / MCP 用 `build_state` 因還需要 registry |
| 3b §11 `RawSolverResult.from_dicts` / `as_dicts`、`CandidateSet.as_pairs`、`ProblemError` | 保留，docstring 標為 test-facing | review F-18：只有測試使用 |

---

## 30. 之後（不在 3b）

### 30.1 整數變數
- 實數變數（Leap CQM 支援 REAL；需要新的 validator 語意與 BQM 路徑的「不支援」判定）。
- `NON_INTEGER_INEQUALITY` 依 model type 放寬（需先決定 errors 是否可以 backend 相關）。

### 30.2 Fujitsu DA 原生能力
- 第三種 model type（例如 `"qubo_constrained"`）：整數編碼成位元、但線性 constraint 保留原生（不等式 → `inequalities`、等式 → `penalty_binary_polynomial` 交 DA `penalty_auto_mode` 自動調係數、one-hot 偵測）。需要第三個 compiler 與新的 `PenaltyStrategy` 不適用規則；3b 的 §20.7 請求組裝已把 `binary_polynomial` 獨立成函式，屆時只加 `penalty_binary_polynomial` / `inequalities` 兩段。
- OAuth access token、v3c 端點切換、Azure Blob 大問題、`guidance_config` 暖啟動。
- `DWAVE_CONFIG_INVALID` 在 sampler 建構階段改走 `configuration_error`（與 §20.8 的機制統一；3b 刻意不動既有期望）。
