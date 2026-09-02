# AnnealBridge — Phase 3 規格骨架（給 spec 撰寫者）

> 這不是規格，是「Phase 3 規格必須回答的問題清單」。請依 Phase 2 spec v2 的體例（目標 / scope / 明確不做 / 架構規則 / 每個元件的介面 / 測試 / 開發順序 / acceptance criteria）寫成 `annealbridge_phase3_spec_v1.md`。
> 寫作時必須遵守 `annealbridge_PROJECT_OVERVIEW.md` §五的核心原則，尤其是原則 4（核心不認識特定求解器）、原則 5（不偷偷幫使用者做決定）、原則 7（不蓋空殼）。
> Phase 2 spec §37 的承諾：**Phase 3 只應新增 compiler 與 backend，不動 IR 與 interfaces。** 整數變數勢必會動 IR，所以 spec 必須明確說明「動哪裡、為什麼非動不可、如何保持 1.0 JSON 完全相容」。

---

## 0. 現況（spec 撰寫者必讀）

- Phase 1 + 2 完成，`pytest` 892 passed；套件 `annealbridge`；git main 最新 commit `8d56366`。
- 2026-09-02 全庫 code review 後已修正 5 批問題。兩項 **刻意留給 Phase 3** 的架構債（見 §1），因為它們的正確解法會決定新 backend 的介面。
- 現有擴充點只有三個 Protocol：`ModelCompiler`、`SolverBackend`、`PenaltyStrategy`。其他地方都是具體程式，Phase 3 不應新增第四個抽象層，除非 spec 能證明兩個以上的實作會共用它。
- 目前 `CompiledProblem` 只承載 BQM；`SolverCapabilities.supported_model_types` 已預留 `["bqm"]` 這個欄位，是 CQM 的接點。

---

## 1. Step 0：先償還兩項架構債（必做，在任何新功能之前）

### 1.1 Backend 一次宣告自己的參數與上限

問題：現在「哪個 backend 吃哪些參數、受哪個 policy 上限管」寫在四個地方——`orchestration/policy.py`（backend 命名的欄位如 `max_qpu_reads`）、`config/settings.py`、`orchestration/optimizer.py` `_preference_limit_errors`（用 capabilities flag 分派但讀 backend 命名的 option 子模型）、`interfaces/capabilities.py` `_policy_limits`（用 backend 名稱 if-chain）。Review 已重現：第五個 backend 若 `supports_num_reads=True`，capabilities 會說 `limits={}`、solve 卻回 `QPU_READS_LIMIT`。

spec 要回答：
- `SolverCapabilities` 是否新增一個宣告（例如 `limited_parameters: list[LimitedParameter]`，每項含 preference 路徑、policy key、error code）讓 service 用一個通用迴圈檢查、capabilities 讀同一份宣告？
- `ExecutionPolicy` 的 backend 命名欄位（`max_qpu_reads` 等）是保留（Phase 2 spec §8 明定、env 變數名已公開）還是改成 `backend_limits: dict[str, dict[str, float]]`？建議：保留既有欄位與 env 名作為相容層，內部轉成通用結構。
- 新 backend 加入時，除了 `registry.default()` 與 `SolverPreferences.backend` Literal（spec §12 明定）之外，不得再需要改 `optimizer.py`、`capabilities.py`。

### 1.2 Validator 以 capabilities 而非名稱判斷

問題：`problem_validator.py` 用 `SEED_IGNORING_BACKENDS = {"dwave_qpu","leap_hybrid_bqm"}`、`backend == "exact"` 等硬寫名稱產生 `SEED_IGNORED` / `PARAMETER_IGNORED` / `EXACT_*` / `DENSE_FOR_QPU`。`exact` 其實已宣告 `supports_seed=False` 卻不在集合內（今天就漂移）。原因是依賴方向 `validation` 不得 import `solvers`。

spec 要回答：
- 把 `SolverCapabilities`（純 Pydantic）搬到 `models/`（先例：`SolverExecutionMetadata` 已因同樣理由搬過），`validate_problem_full(problem, *, capabilities: SolverCapabilities | None, exact_max_variables)` 注入；MCP / CLI 由 registry 查到 capabilities 後傳入。
- 順勢新增 `OptimizationService.validate(problem) -> ProblemValidationResult`，讓 MCP tool 與（新增的）CLI `validate` 指令都變成一行，policy → validator 的接線不再只存在於 MCP 層（Phase 2 spec §37 的「任何 Python 程式都能不經 MCP 得到同樣行為」）。
- `DENSE_FOR_QPU` 改由 capability flag（例如 `requires_embedding`）觸發。
- Availability reason 改為結構化：`is_available() -> AvailabilityStatus(category: Literal[...], detail: str | None)`，service 以 category 對應 status/code，不再比對 D-Wave 專屬字串（修正 5 已把字串收成常數，但 category 化才是終點）。

---

## 2. Phase 3 目標與範圍（spec 要拍板的四個功能）

### 2.1 整數變數（IR 變更，影響最大，建議最先做）

spec 要回答：
- IR 版本：`OptimizationProblem.version` 從 `"1.0"` 變成 `Literal["1.0","1.1"]`？1.0 JSON 不改一字必須仍可 parse 且行為不變（Phase 1 example 是回歸基準）。
- `Variable` 如何表達整數：`type: Literal["binary","integer"]` + `lower_bound` / `upper_bound`（整數、有限、upper > lower）。是否允許負數下界？（建議允許，編碼時位移。）
- 編碼策略歸誰：**compiler**，不是 IR。BQM compiler 的整數 → 二進位編碼（binary expansion / one-hot / unary，建議 binary expansion 並允許 compiler option 切換）、編碼位元屬 internal variable（`__` 前綴），回傳解時 compiler 負責 decode 回整數值。
- `Solution.variables: dict[str, int]` 型別不變（整數值仍是 int），但 `validate_solution` / `evaluate_objective` 要能對整數值計算（目前已是算術，應可直接適用，spec 要求測試證明）。
- `estimated_compiled_variables` 要把整數編碼位元算進去；`DENSE_FOR_QPU` / `EXACT_*` 門檻同樣以編譯後變數數計。
- 二次項含整數變數時 BQM 會爆長（位元兩兩相乘）：validator 要有 warning（例如 `INTEGER_QUADRATIC_BLOWUP`）；CQM backend 天生支援整數，是這類問題的正解（見 2.2）。
- 不做：實數變數、無界整數。

### 2.2 CQM compiler + Leap Hybrid CQM backend

spec 要回答：
- `ModelCompiler` 第二個實作 `CQMCompiler`：constraints 直接進 `dimod.ConstrainedQuadraticModel`，**hard constraint 不需要 penalty、不需要 slack**；soft constraint 如何表達（dimod CQM 有 `weight` / `penalty` 參數的 soft constraint，spec 要查證版本並決定是否使用，或仍以 objective 加權表達）。
- `CompiledProblem` 如何同時承載 BQM 與 CQM：建議 `model_type: Literal["bqm","cqm"]` + `model: Any`，service 依 `backend.capabilities.supported_model_types` 選 compiler——**這是 service 少數可以依 capabilities 分派的地方，不得依 backend 名稱分派**。
- Penalty / retry 對 CQM 無意義：`PenaltyStrategy` 對 CQM 路徑要有明確的「不適用」行為（attempt 恆為 1？`SolveAttempt.penalty` 記 None？）。
- `LeapHybridCQMBackend`：`dwave.system.LeapHybridCQMSampler`，lazy import、與 Phase 2 兩個 remote backend 共用修正 5 抽出的 helper（`dwave_availability`、`classify_exception`、redact、sanitize）。CQM sampler 回傳含 `is_feasible` 欄位——**仍須走 independent validator 重驗，不得信任 sampler 的 feasible 旗標**（原則 2）。
- 整數變數在 CQM 路徑不需編碼，直接是 `dimod.Integer`。
- 上限：`time_limit`（沿用 `max_remote_time_seconds`）、變數/constraint 數量上限（Leap CQM 有官方限制，spec 查證後列入 capabilities `limits`）。

### 2.3 Fujitsu Digital Annealer backend

spec 要回答（撰寫前請查證 Fujitsu DA 目前的 SDK / Web API 版本與認證方式，不要憑印象寫）：
- 模型型別：DA 吃 QUBO（BQM 路徑），是否支援不等式/整數原生表達（若支援，是第二個 CQM-like 消費者，會影響 2.2 的介面設計）。
- 認證：與 D-Wave 同原則——AnnealBridge 不管理 token，交由廠商 SDK 原生 config / env；`ServerSettings` 不得出現 token 欄位；redaction regex 要加 Fujitsu token 形式。
- 參數：DA 的 `number_iterations` / `number_replicas` / `time_limit` 等映射到哪些 `SolverPreferences` 欄位？需要新的 option 子模型（`fujitsu_da: FujitsuDAOptions | None`，比照 `dwave_qpu`）。policy 上限透過 §1.1 的通用宣告，不新增 backend 命名的 policy 欄位。
- 可選依賴 extra：`fujitsu = [...]`；未安裝時 registry 仍可 import（lazy）。
- Mock 測試比照 `remote_mock`；live 測試 opt-in 比照 `remote_live`（新增 marker 或共用 `remote`）。
- 若查證後發現 DA 服務狀態不明（停售 / API 關閉），spec 應明說並建議改列為「擴充點驗證用的第二個遠端 backend」或延後。

### 2.4 Solver routing（與原則 5 最容易衝突，要寫得最保守）

spec 要回答：
- 定位：**只建議、不代選。** 新 MCP tool（例如 `recommend_backend`）輸入 problem，輸出「候選 backend 排序 + 每個的理由 + 預期限制/警告」；`solve_optimization` 行為不變——使用者指定什麼就用什麼，不可用就回 `backend_unavailable`。
- 建議依據只能來自已有的結構化資訊：`validate_problem_full` 的 estimate 與 warnings、capabilities（available / enabled / limits / supports_*）、policy。不做成本估算、不打網路、不做 benchmark。
- 規則要 deterministic 且可測試（例如：有整數變數且二次項 → CQM 優先；編譯後變數 ≤ exact 上限 → exact 可作驗證用；DENSE_FOR_QPU → 不建議 QPU）。
- 是否讓 CLI 也有 `annealbridge recommend problem.json`。

---

## 3. 明確不做（沿用 Phase 2 §3 並延伸）

實數變數、無界整數、自動 cost estimation、自動選 backend 後直接求解、OR-Tools / Gurobi 等精確 MILP、job queue、persistence、auth、多租戶、Web UI、LLM 呼叫。

---

## 4. 架構規則（必須逐條寫進 spec）

- 依賴方向不變：`models ← validation ← compiler ← solvers ← orchestration ← interfaces`；`config` 只有 interfaces 可 import。
- 新增第二個 compiler 後，`tests/architecture` 要加測試：`orchestration` 不得 import 任何具體 compiler 或 backend 的模組（只能經 registry / protocol）。
- service 內不得出現 `if backend == "..."`；允許依 `capabilities` 欄位分派。
- 所有遠端 backend 共用 `solvers/metadata.py` 的 availability / classify / redact / sanitize helper；exception → code 表可各自定義。
- 每個新 error code 都要進 `error_catalog` 且有 `recommended_action`。
- 測試紅線不變：MCP 測試走 Client、無 skip/xfail（live 除外）、Phase 1/2 example JSON 不改一字。

---

## 5. 需要使用者決策的問題（spec 撰寫者把它們列在 spec 開頭，由使用者拍板後再細寫）

1. 整數變數是否進 Phase 3，或拆成 Phase 3a（Step 0 + CQM + routing）與 3b（整數 + DA）？——影響 IR 版本號與相容測試的工作量。
2. `ExecutionPolicy` 的 backend 命名欄位是相容保留還是一次改成通用結構（env 變數名會變）。
3. Fujitsu DA 查證結果若不理想，是否以「第二個非 D-Wave 遠端 backend」的角色改選其他標的（純粹為了證明擴充點可用）。
4. routing 是否納入 Phase 3，或先只做 `validate` 的 warnings 強化。

---

## 6. 開發順序建議（供 spec §35 參考）

1. Step 0a：capabilities 搬 models/ + validator 注入 + `service.validate()` + CLI `validate`（Phase 2 測試全綠）。
2. Step 0b：backend 參數/上限宣告 + availability category 化；capabilities 與 service 讀同一份宣告（新增「假第五個 backend」的架構測試，證明零改動 optimizer/capabilities）。
3. `CompiledProblem` 多型別 + `CQMCompiler`（先用 exact-like 本機 CQM sampler 或 `dimod` 的 CQM → BQM 轉換做本機測試）。
4. `LeapHybridCQMBackend` + mock tests。
5. 整數變數：IR 1.1 + BQM compiler 編碼/解碼 + validator estimate + CQM 路徑直通。
6. Fujitsu DA backend + mock tests（依查證結果）。
7. `recommend_backend` tool + CLI。
8. README / spec 文件 / live tests / acceptance。

---

## 7. Acceptance criteria 草稿

- Phase 1、2 全部測試不變仍通過；1.0 example JSON 不改一字。
- 新增 backend 只改 registry + Literal + 自身模組（架構測試證明）。
- service 無 backend 名稱字串比對（grep 測試）。
- CQM 路徑：hard constraint 無 penalty、無 slack、attempt == 1、solutions 仍經 independent validator。
- 整數變數：BQM 路徑編碼位元不外洩、decode 正確；CQM 路徑直通；1.0 問題行為位元一致。
- routing 從不代選；`solve_optimization` 行為不變。
- token（D-Wave、Fujitsu）不進任何輸出；credential leak 測試涵蓋新 backend。

---

## 8. 使用者決策（2026-09-02 已拍板，spec 撰寫者依此細寫，不得再向使用者重問）

| # | 問題 | 決策 |
|---|------|------|
| 1 | 整數變數是否進 Phase 3 | **拆成 3a / 3b**。3a = Step 0 技術債 + CQM compiler/backend + routing（不動 IR）。3b = 整數變數（IR 1.1）+ Fujitsu DA。先寫 `annealbridge_phase3a_spec_v1.md`；3b 待 3a 完成後另寫。 |
| 2 | `ExecutionPolicy` backend 命名欄位 | **保留既有欄位與 env 變數名作相容層**，內部轉成通用結構（如 `backend_limits`）。新 backend 的上限只走通用宣告，不再新增 backend 命名欄位。 |
| 3 | Fujitsu Digital Annealer | **照做（歸 3b）**。2026-09-02 查證：服務仍營運，API 文件 2026-07 更新，QUBO API 有 V3c / V4。spec 撰寫前仍須查證 SDK / Web API 版本與認證方式；若需付費帳號才能取得 API key，退而只做介面 + mock 測試，live 測試留空。 |
| 4 | Solver routing | **納入 3a，排在開發順序最後**。只建議、不代選；只讀 `validate_problem_full` 結果與 capabilities，不打網路、不估成本。含 MCP tool `recommend_backend` 與 CLI `recommend`。 |

3a 開發順序即 §6 的 1、2、3、4、7、8；3b 為 §6 的 5、6 加各自的文件與 acceptance。
