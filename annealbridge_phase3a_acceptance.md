# AnnealBridge Phase 3a 驗收報告

> 對照 `annealbridge_phase3a_spec_v1.md` §31 Acceptance Criteria 逐條核對。
> 核對日期：2026-09-03。commit 範圍：`6486719..HEAD`（步驟 1–9 已 commit；步驟 10 為本次工作樹變更）。
> 執行環境：Windows 11、`.venv\Scripts\python.exe`（**未安裝** `dwave-system`，`pip show dwave-system` → not found）。

---

## 1. 總結

| 項目 | 結果 |
|---|---|
| `pytest`（預設 `-m "not remote"`） | **1311 passed, 3 deselected**，0 failed、0 skipped、0 xfail（步驟 10 前為 1266 passed；新增 45 個測試） |
| `pytest -rs` skip 清單 | 預設執行**沒有任何 skip**；`pytest tests/remote_live -m remote -rs`（無 token）→ 3 skipped，全部來自 `tests/remote_live/conftest.py::_require_live_opt_in` 的註解 skip |
| `pytest tests/architecture` | 27 passed |
| §31 checkbox | **19 條全部 ✅**，0 條 ❌ |
| §33 偏離清單 | 33.1 十列、33.2 四列，全部已按規格落實（見第 4 節） |

---

## 2. §31 逐條核對

### 2.1 ✅ Phase 1、2 全部測試不變仍通過；三個 example JSON 不改一字仍可 parse；`version` 仍 `"1.0"`

- `pytest` → 1311 passed（含 Phase 1/2 全部既有測試；§9.3 明列的 warning 期望值與 `is_available()` 型別更新已在步驟 1、3 同步）。
- `git diff --stat 6486719~1..HEAD -- examples/` → 空（三個 example 未改動）。
- `tests/unit/test_phase1_compat.py::TestPhase1ExamplesStillParse::test_example_parses_unchanged`（三個 example 參數化）、`::test_new_backend_options_default_to_none`、`::test_cqm_backend_options_default_to_none`。
- 指令 `python -c "from annealbridge.models import OptimizationProblem; print(OptimizationProblem.model_fields['version'].annotation)"` → `typing.Literal['1.0']`；`Variable.type` 仍為 `typing.Literal['binary']`。
- `annealbridge solve examples/{knapsack,assignment,tsp}.json` → status success、objective **17 / 8 / 8**、exit 0。
- Phase 2 的 MCP solve 測試（`tests/mcp/test_solve_exact.py`、`test_solve_sa.py`、`test_solve_errors.py`）、`tests/scenarios/test_knapsack.py`、`test_assignment.py`、`tests/unit/test_bqm_compiler.py` 自 `6486719` 起 `git diff --stat` 為空，未改一字仍通過。

### 2.2 ✅ `grep` 測試：`orchestration/**`、`validation/**`、`interfaces/capabilities.py`、`interfaces/mcp/tools.py`、`interfaces/cli/main.py` 無 backend 名稱字串

- `tests/architecture/test_no_backend_names.py::test_core_files_contain_no_backend_name_constants`（AST 掃描五個 backend 名稱常數，docstring 豁免）＋ `TestScannerItself` 四個自證測試。
- 手動 `grep -rn "backend ==\|== \"exact\"" src/annealbridge/{orchestration,validation,interfaces}` → 無結果。

### 2.3 ✅ 假第五 backend 測試通過：capabilities `limits` 與 service 限制同源、validator 依 capabilities 產 warning，零改動 `optimizer.py` / `capabilities.py` / `problem_validator.py` / `policy.py`

- `tests/architecture/test_fifth_backend.py`：`TestCapabilitiesView::test_limits_come_from_the_policy_custom_key`、`test_view_and_service_share_one_limit_source`、`TestGenericLimitEnforcement::test_over_limit_is_refused_under_the_declared_code`、`test_never_clamps`（`test_exactly_at_limit_is_allowed`）、`TestValidatorFollowsDeclaration::test_seed_is_reported_ignored_without_any_name_list`、`TestPolicyConsistency::test_service_refuses_to_build_without_the_key`、`TestRecommend::test_fake_is_listed_and_usable`、`test_core_sources_never_mention_the_fake`（掃描 `optimizer.py`、`limits.py`、`policy.py`、`routing.py`、`capabilities.py`、`problem_validator.py`）。
- 全檔 27 passed（與 `test_no_backend_names.py`、`test_import_boundaries.py` 合計）。

### 2.4 ✅ `exact` + `seed` 產 `SEED_IGNORED`（漂移 2 修正）

- `tests/unit/test_problem_validator_full.py::TestSeedIgnored::test_registry_backends_that_declare_no_seed_warn`（含 `exact`）、`::test_seed_alone_does_not_also_produce_parameter_ignored`。
- `tests/unit/test_service_validate.py::TestMatchesDirectValidatorCall::test_exact_with_seed_reports_seed_ignored`。
- `tests/mcp/test_validate.py::test_exact_with_seed_reports_seed_ignored`（經 in-memory MCP Client）。

### 2.5 ✅ 舊 env 變數名全部仍生效；`ANNEALBRIDGE_LIMITS` 可用；`limits` 與相容 key 衝突被拒

- `tests/unit/test_settings.py::TestServerSettingsFromEnvironment::test_env_vars_override_every_field`、`TestGenericLimits::test_old_env_names_still_drive_the_compatibility_keys`（舊變數）。
- `TestGenericLimits::test_json_object_is_parsed_into_floats`、`test_limits_reach_the_policy`（`ANNEALBRIDGE_LIMITS`）；`test_non_json_value_is_a_project_settings_error`。
- `TestGenericLimits::test_compatibility_key_is_rejected_naming_the_variable`；`tests/unit/test_policy.py::TestLimitsValidation::test_compatibility_key_in_limits_is_rejected`（衝突被拒）。

### 2.6 ✅ backend 宣告 policy 沒有的 limit key → 建構時失敗，不會靜默無上限

- `tests/unit/test_policy.py::TestServiceDeclaredLimitConsistency::test_missing_policy_value_raises_naming_backend_and_key`、`test_composition_root_reports_it_as_a_settings_error`。
- `tests/unit/test_limits.py::TestPreferenceLimitErrors::test_policy_without_the_declared_key_raises`。
- `tests/architecture/test_fifth_backend.py::TestPolicyConsistency::test_service_refuses_to_build_without_the_key`。

### 2.7 ✅ `is_available()` 回傳 `AvailabilityStatus`；service 依 category 對應 status，不比對 D-Wave 字串

- `tests/unit/test_solvers.py::TestExactSolverBackend::test_is_available`、`TestSimulatedAnnealingBackend::test_is_available`；`tests/unit/test_availability.py::TestCategories`（四種 category）、`TestDetailNeverCarriesConfigValues::test_detail_is_categorical`。
- `tests/unit/test_limits.py::TestGateOrder::test_category_maps_to_status_and_default_code`、`test_backend_supplied_error_code_wins_over_the_default`；`tests/remote_mock/test_service_remote_flow.py::TestAvailabilityClassification::test_reason_maps_to_status_and_code`。
- 來源：`orchestration/limits.py::AVAILABILITY_MAP` 以 category 為 key。

### 2.8 ✅ `service.validate()` 存在；MCP validate tool 與 CLI `validate` 皆為一行委派

- `tests/unit/test_service_validate.py::TestMatchesDirectValidatorCall::test_same_result_as_validate_problem_full`、`TestDoesNotSolve::test_no_backend_is_called_and_no_slot_is_taken`、`TestUnknownBackend::test_custom_registry_without_backend_warns`。
- 一行委派（來源）：`interfaces/mcp/tools.py:50` `return get_state().service.validate(problem)`；`interfaces/cli/main.py:242` `result = _build_state().service.validate(problem)`。
- `tests/unit/test_cli.py::TestValidate`（exit 0 / 1 / 2、`--json` 為合法 `ProblemValidationResult`）；`tests/mcp/test_validate.py`。
- 指令：`annealbridge validate examples/{knapsack,assignment,tsp}.json` → Valid: yes、estimated compiled variables 8 / 9 / 16、exit 0。

### 2.9 ✅ `CompiledProblem.model_type`；service 依 `capabilities.supported_model_types` 選 compiler；無 compiler → `NO_COMPILER_FOR_MODEL_TYPE`

- `tests/unit/test_service_compilers.py::TestBQMCompilerContract::test_compiled_problem_records_model_type`、`TestNoCompilerForModelType::test_cqm_only_backend_with_bqm_only_service`、`test_first_declared_type_with_a_compiler_wins`、`test_validate_reports_model_type_none_without_compiler`。
- `tests/unit/test_service_cqm_flow.py::TestCompilerSelection::test_declaration_selects_the_cqm_compiler`、`TestNoCqmCompiler::test_bqm_only_service_is_a_configuration_error`。
- 來源：`orchestration/limits.py::select_model_type`（solve / validate / recommend 共用）。

### 2.10 ✅ CQM 路徑：hard constraint `weight=None`、無 slack、`internal_variables == set()`、`hard_penalty is None`、`attempts == 1`、`SolveAttempt.penalty is None`、無 `REMOTE_RETRIES_DISABLED`、solutions 仍經 independent validator、`is_feasible` 不採信

- `tests/unit/test_cqm_compiler.py::TestConstraints::test_hard_constraint_is_native_with_no_weight`、`TestDeclaration::test_hard_penalty_is_rejected`。
- `tests/unit/test_service_cqm_flow.py::TestCompilerSelection::test_hard_constraints_are_native_without_penalty_or_slack`、`TestSingleAttemptWithoutPenalty::test_exactly_one_attempt_with_penalty_none`、`test_max_retries_does_not_add_attempts`、`TestKnapsackMatchesTheExactPath::test_no_internal_variable_leaks`、`TestSamplerVerdictIsNotTrusted::test_tampered_feasible_flag_does_not_reach_solutions`。
- `tests/remote_mock/test_leap_hybrid_cqm_mock.py::TestRetryPolicyDoesNotApply::test_infeasible_run_makes_one_attempt_and_no_warning`、`test_successful_run_makes_one_attempt_and_no_warning`（`allow_remote_retries` 兩種值皆無 `REMOTE_RETRIES_DISABLED`）、`TestSamplerVerdictIsNotTrusted::test_violator_is_excluded_from_solutions`、`TestSampleSetConversion::test_infeasible_flagged_samples_are_not_filtered`、`test_samples_are_not_reordered_by_feasibility`。
- `tests/scenarios/test_knapsack_cqm.py::TestKnapsackCQM`（JSON → Service → objective 17、無 `__` 變數、`attempts[0].penalty is None`）。

### 2.11 ✅ soft constraint：CQM `weight × violation²` 與 validator `weighted_penalty` 同公式（測試證明）

- `tests/unit/test_cqm_compiler.py::TestSoftWeightSemantics::test_soft_energy_is_weight_times_violation_squared`（對每個 assignment 比對 CQM soft 能量與 `validate_solution` 的 `weighted_penalty`）。

### 2.12 ✅ `ExactCQMSolver` 交叉驗證：`is_feasible` 與 `validate_solution` 對全部 assignment 一致

- `tests/unit/test_cqm_compiler.py::TestCrossValidationWithExactCQMSolver::test_knapsack_example`、`test_mixed_ge_eq_hard_constraints_with_a_soft_one`、`test_lowest_energy_feasible_sample_is_the_business_optimum`。

### 2.13 ✅ `LeapHybridCQMBackend`：lazy import、`time_limit` 轉送、min floor、policy 上限不 clamp、不傳 label、非 {0,1} 值報錯、例外分類、redaction；`SolverRegistry.default()` 五個 backend 且無 `dwave-system` 仍可 import

- `tests/remote_mock/test_leap_hybrid_cqm_mock.py`：`TestLazyImport::test_construction_does_not_import_dwave`、`test_module_has_no_dwave_import_at_module_level`；`TestTimeLimitPolicy::test_forwarded_time_limit_equals_resolve_time_limit`、`test_below_minimum_is_raised_and_recorded`、`test_user_value_above_policy_is_refused_not_clamped`、`test_effective_value_above_policy_is_refused_not_clamped`；`TestSuccessfulSolve::test_no_label_and_no_num_reads_reach_the_sampler`、`TestModelForwarding::test_only_time_limit_is_forwarded`；`TestSampleSetConversion::test_non_binary_value_is_a_remote_solver_error`、`TestNonBinarySampleThroughService::test_non_binary_value_is_a_structured_solver_error`；`TestExceptionClassification::test_sample_exceptions_map_to_codes_and_are_redacted`、`TestSamplerInitExceptionClassification::test_factory_exceptions_map_to_codes_and_are_redacted`。
- `tests/unit/test_registry.py::TestDefaultRegistry::test_default_registers_exactly_the_expected_backends_in_order`、`test_default_registry_needs_no_dwave_system`。
- 指令（本機無 `dwave-system`）：`python -c "from annealbridge.solvers import SolverRegistry; print(SolverRegistry.default().names())"` → `['exact', 'simulated_annealing', 'dwave_qpu', 'leap_hybrid_bqm', 'leap_hybrid_cqm']`。

### 2.14 ✅ `sampler_reported_feasible` 與 `model_type` 進 metadata；whitelist 不變

- `tests/remote_mock/test_leap_hybrid_cqm_mock.py::TestSampleSetConversion::test_sampler_reported_feasible_counts_the_flag_verbatim`、`test_missing_is_feasible_field_reports_none`、`TestSuccessfulSolve::test_metadata_model_type_is_cqm`、`TestSamplerVerdictIsNotTrusted::test_sampler_count_is_reported_as_given`；`tests/unit/test_service_compilers.py::TestMetadataModelType`（三支）。
- whitelist 不變：`git diff 6486719~1..HEAD -- src/annealbridge/solvers/metadata.py` 對 `TIMING_WHITELIST` 無變更；`tests/unit/test_metadata.py::TestSanitizeSamplesetInfo::test_only_whitelist_keys_survive_and_are_floats`、`test_leap_hybrid_cqm_mock.py::TestMetadataSanitization::test_only_whitelisted_timing_keys_survive`。`models/metadata.py` 只新增 `model_type` 與 `sampler_reported_feasible` 兩欄。

### 2.15 ✅ `recommend_backend`：deterministic、無網路、不 solve、不佔 slot；`solve_optimization` 行為不變；CLI `recommend` 與 MCP 同源

- `tests/unit/test_routing.py::TestDeterminism::test_same_input_gives_equal_results`、`TestNothingIsSolved::test_no_backend_solve_or_time_limit_is_called`（spy）；`tests/mcp/test_recommend.py::test_nothing_is_solved`、`test_knapsack_ranks_exact_first`、`test_remote_backends_are_unusable_under_the_default_policy`。
- 不佔 slot / 無網路（來源）：`orchestration/routing.py` 不引用 `_solve_slots`、不建立 sampler；`gate_errors` 只讀本機 config；`OptimizationService.recommend()` 為一行委派。
- `solve_optimization` 不變：`tests/mcp/test_solve_exact.py`、`test_solve_sa.py`、`test_solve_errors.py` 自 `6486719` 起未改動仍通過；`tests/mcp/test_solve_errors.py` 涵蓋指定 `leap_hybrid_cqm` 且 policy 未開 → `backend_unavailable` / `REMOTE_DISABLED`。
- CLI 與 MCP 同源（來源）：`interfaces/mcp/tools.py:65` 與 `interfaces/cli/main.py:318` 皆為 `service.recommend(problem)`；`tests/unit/test_cli.py::TestRecommend::test_knapsack_table_matches_the_spec_example`、`test_json_output_is_a_recommendation_result`。
- 指令：`annealbridge recommend examples/{knapsack,assignment,tsp}.json` → 五個 backend 皆列出，exact 第一、SA 第二、三個遠端 `usable=no` 且 blocking `REMOTE_DISABLED`；exit 0。

### 2.16 ✅ 四個 MCP tool；`tools/list` 測試更新；`structured_content` 為 dict

- `tests/mcp/test_tools_list.py::test_exactly_four_tools`、`test_recommend_output_schema_matches_recommendation_result`、`test_solve_docstring_points_at_recommend_backend`。
- `tests/mcp/test_recommend.py::test_structured_content_parses_as_the_result_model`；`tests/mcp/test_capabilities.py::test_structured_content_is_dict`、`test_all_five_backends_listed`。

### 2.17 ✅ 每個新 code 有 `recommended_action`；每個 reason code 有說明

- `tests/unit/test_error_catalog.py`（步驟 10 擴充）：`TestRecommendedActions`（34 個 code；`BACKEND_CONFIG_INVALID`、`NO_COMPILER_FOR_MODEL_TYPE` 在列，`UNKNOWN_BACKEND` 共用既有條目，總數不因 validate() 的 warning 用法 +1）；`TestEmittedErrorCodesAreCatalogued`（AST 收集 `orchestration/**`、`validation/**`、`interfaces/**`、`solvers/**` 內 `catalog_error(...)` / `_error(code=...)` / `SolverExecutionError(code=...)` 的字面 code，加上 `AVAILABILITY_MAP` 預設 code、五個 backend 宣告的 `ParameterLimit.error_code`、`ocean.py` 兩張例外分類表與 `REMOTE_ERROR_FALLBACK_CODE`，逐一斷言在 catalog 內）；`TestEmittedWarningCodesAreCatalogued`（`_warning("...")` 字面 code 與 `_WARNING_RECOMMENDED_ACTIONS` 集合相等）；`TestGuidanceTextCarriesNoConfigValues`（所有 action / warning 文字不含數字，即不含 config 值；3a 兩條文字與 spec §20 一致）。
- `tests/unit/test_routing.py::TestReasonCatalog::test_every_reason_code_has_a_description`、`test_every_emitted_reason_is_described`（執行時）；步驟 10 新增 `TestReasonCatalogMatchesTheSource::test_every_source_reason_code_is_described`、`test_no_other_r_constant_hides_in_the_module`（AST 收集 `routing.py` 內 `R_` 開頭常數：`reasons.append(...)` 發出集合 == dict key 集合 == `REASON_DESCRIPTIONS`，九個 code）。
- 核對結果：無遺漏，不需補文字。

### 2.18 ✅ token 不進任何輸出；credential leak 測試涵蓋 CQM backend

- `tests/remote_mock/test_credential_leak.py`：`REMOTE_KINDS` 參數化 fixture `remote_kind` 涵蓋 `dwave_qpu` 與 `leap_hybrid_cqm`（`TestTokenNeverLeaves` 六支、`TestBareTokenIsMasked`、`TestLazyResolveFailureIsRedacted` 皆對兩種 backend 執行）；`TestExceptionChainCarriesNoToken::test_cqm_sample_failure`、`test_cqm_sampler_construction_failure`、`test_cqm_time_limit_resolution_failure`。
- `tests/remote_live/test_leap_hybrid_cqm_live.py` 亦斷言 token 不在 result JSON 與 log 中（opt-in）。
- README 內無真實 token（僅 `DEV-xxxxxxxxxxxxxxxxxxxx` 佔位符）。

### 2.19 ✅ README 更新；`pytest` 全過，無 skip / xfail（除 remote_live 註解 skip）

- README（§27 全部項目）：Architecture 圖含 `compiler {bqm, cqm}` 與五個 backend；Solver Backends 表加 `leap_hybrid_cqm` 列與 CQM 路徑說明（原生 hard constraint、無 penalty / retry、feasible 旗標不採信、Leap 官方上限 5,000,000 變數 / 100,000 constraints / `time_limit_seconds` 最小 5 s）；MCP Server 段四個 tool 並說明 `recommend_backend` advisory only；CLI 段加 `validate`、`recommend`；Environment variables 加 `ANNEALBRIDGE_LIMITS` 且舊變數全部保留；Limitations 加 CQM 整數係數限制（§21.3）與「Integer variables: Phase 3b」；Testing 段註明 `pytest -m remote` 含 CQM live 測試。
- `annealbridge_PROJECT_OVERVIEW.md` §四加 Phase 3a 白話說明、§七狀態表更新（Phase 3a 規格 / 實作完成；Phase 3b 只留擴充點）。
- `.github/workflows/ci.yml`：既有 minimal-install job 的 `SolverRegistry.default().names()` 會印五個 backend；另加一步 `annealbridge validate` / `annealbridge recommend examples/knapsack.json` 作 smoke（無 extra 環境）。
- `pytest -rs` → 1311 passed, 3 deselected；預設執行 0 skipped、0 xfail。`grep -rn "pytest.skip\|pytest.mark.skip\|xfail\|importorskip" tests` 只命中 `tests/remote_live/conftest.py`（註解說明的唯一允許 skip）。
- `pytest tests/remote_live -m remote -rs`（無 `DWAVE_API_TOKEN`）→ 3 skipped，訊息皆為 `DWAVE_API_TOKEN is not set; cannot reach D-Wave Leap`；CQM live 測試 `test_leap_hybrid_cqm_live.py::test_leap_hybrid_cqm_solves_minimal_knapsack` 已在 collect 清單中（本機無 token，未實際打 Leap）。

---

## 3. 步驟 10 本次變更

| 檔案 | 變更 |
|---|---|
| `tests/unit/test_error_catalog.py` | 新增 AST 收集的 emitted-code 覆蓋測試（error、warning、宣告驅動 code、文字不含 config 值、3a 兩條文字比對）；共 +43 測試 |
| `tests/unit/test_routing.py` | 新增 `TestReasonCatalogMatchesTheSource`（AST 收集 `R_` 常數）；+2 測試 |
| `README.md` | §27 全部項目 |
| `annealbridge_PROJECT_OVERVIEW.md` | §四 Phase 3a 白話說明、§七狀態表、規格檔清單；§五原則未動 |
| `.github/workflows/ci.yml` | minimal-install job 加 `validate` / `recommend` smoke |
| `annealbridge_phase3a_acceptance.md` | 本檔 |

程式碼（`src/`）本步驟**無變更**：catalog 已含 3a 全部新 code、`REASON_DESCRIPTIONS` 已含全部九個 reason code，核對未發現遺漏。

---

## 4. §33 偏離清單落實核對

### 4.1 對 Phase 2 spec 的偏離（§33.1）

| 3a 變更 | 已落實 | 證據 |
|---|---|---|
| interfaces 加一個 tool、兩個 CLI 指令；既有三個 tool 簽名不變 | ✅ | `tests/mcp/test_tools_list.py::test_exactly_four_tools`、`test_solve_input_schema_wraps_problem`；`tests/unit/test_cli.py::TestValidate` / `TestRecommend` |
| `is_available()` → `AvailabilityStatus` | ✅ | 2.7 |
| `validate_problem_full(problem, *, capabilities=, max_compiled_variables=, model_type=)` | ✅ | `validation/problem_validator.py:236`；`tests/unit/test_problem_validator_full.py` 全部改用 `capabilities=` |
| warning 表以 capabilities 欄位為條件；`exact` 新增 `SEED_IGNORED` / `PARAMETER_IGNORED` | ✅ | 2.4；`TestParameterIgnored` |
| `limits` 由 `policy.limits_for(caps)` 產生，逐 key 相同 | ✅ | `tests/unit/test_policy.py::TestLimitsFor::test_shipped_backends_match_the_phase2_output_key_for_key`；`tests/architecture/test_fifth_backend.py::test_view_and_service_share_one_limit_source` |
| `ExecutionPolicy` 欄位全保留，加 `limits` / `limit()` / `limits_for()` | ✅ | `tests/unit/test_policy.py::TestLimitLookup`、`TestLimitsValidation` |
| §14 step 9 改宣告驅動；exhaustive 上限與 effective time limit 維持 flag 驅動 | ✅ | `orchestration/limits.py::preference_limit_errors`；`tests/unit/test_limits.py::TestPreferenceLimitErrors`；`tests/remote_mock/test_service_remote_flow.py::TestEffectiveHybridTimeLimit` |
| `OptimizationService(compilers=[...])` | ✅ | `orchestration/optimizer.py:285`；`tests/unit/test_service_compilers.py::TestConstruction` |
| `SolveAttempt.penalty: float \| None` | ✅ | `models/solution.py:51`；2.10 |
| 3a 做 CQM | ✅ | 2.9–2.13 |

### 4.2 對 outline 的偏離（§33.2）

| 3a 變更 | 已落實 | 證據 |
|---|---|---|
| 具體 compiler 只允許在 `optimizer.py` 建預設清單處 import | ✅ | `tests/architecture/test_import_boundaries.py::test_concrete_compilers_only_imported_by_optimizer`、`test_orchestration_does_not_import_concrete_backends` |
| Leap 官方上限不進 `limits`，寫進 `description` 與 README | ✅ | `policy.limits_for(leap_hybrid_cqm caps)` → `{"max_time_seconds": 300}`（`test_policy.py::TestLimitsFor::test_hybrid_cqm_style_declaration_yields_one_time_limit`）；`test_leap_hybrid_cqm_mock.py::TestCapabilities::test_description_states_leap_limits_and_ignored_flag`；README Solver Backends 段 |
| 整數變數 3a 不做 | ✅ | `Variable.type` 仍 `Literal['binary']`；`version` 仍 `Literal['1.0']` |
| 參數改名 `max_compiled_variables` 並加 `model_type` | ✅ | `validation/problem_validator.py:236-242` |

---

## 5. ❌ 項目

無。
