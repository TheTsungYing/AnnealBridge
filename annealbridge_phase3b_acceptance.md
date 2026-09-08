# AnnealBridge Phase 3b 驗收報告

> 對照 `annealbridge_phase3b_spec_v1.md` §27 Acceptance Criteria 逐條核對，並核對 §28 接點對照與 §29 偏離清單。
> 核對日期：2026-09-08。commit 範圍：`02d821d..c0de356`（步驟 0–6b，共 8 個 commit），加上本步驟（步驟 7）的文件 commit。
> 執行環境：Windows 11、`.venv\Scripts\python.exe`（**未安裝** `dwave-system`；**沒有** Fujitsu DA 帳號，`FUJITSU_DA_API_KEY` 未設）。
> 本報告所有數字與指令輸出皆於核對日重跑取得，非引用 spec 或步驟回報。

---

## 1. 總結

| 項目 | 結果 |
|---|---|
| `pytest -q -rs`（預設 `-m "not remote"`） | **2900 passed, 4 deselected**，0 failed、0 skipped、0 xfail（3a 結束時 1311 passed；3b 新增 1589 個測試） |
| 各目錄 | `tests/unit` 2422、`tests/scenarios` 30、`tests/remote_mock` 385、`tests/mcp` 35、`tests/architecture` 28（3a 為 27，新增 `test_no_third_party_http_package_is_imported_anywhere`）；合計 2900 |
| `pytest tests/remote_live -m remote -rs`（無任何 key） | 4 skipped：三個 D-Wave 測試 `DWAVE_API_TOKEN is not set; cannot reach the remote solver`，`test_fujitsu_da_live.py:69` 為 `FUJITSU_DA_API_KEY is not set; cannot reach the remote solver` |
| `grep -rn "pytest.skip\|pytest.mark.skip\|xfail\|importorskip" tests` | 真正的 `pytest.skip` 只有 `tests/remote_live/conftest.py:41,44` 兩處（註解說明的唯一允許 skip）；另兩處命中是 docstring 文字「不 skip、不 xfail」 |
| registry | `['exact', 'simulated_annealing', 'dwave_qpu', 'leap_hybrid_bqm', 'leap_hybrid_cqm', 'fujitsu_da']` |
| error catalog | `RECOMMENDED_ACTIONS` **41**（3a 為 34：+5 validator error、+2 remote code）；`_WARNING_RECOMMENDED_ACTIONS` **11**（3a 為 9）；`RETRYABLE_CODES` 加 `REMOTE_BUSY` |
| §27 checkbox | **17 條全部 ✅**，0 條 ❌ |
| §28 接點 | 七列全部落實（第 4 節） |
| §29 偏離清單 | 十二列全部與實作一致；另發現 **1 項 spec 未記載的文字層級偏離**（`LARGE_INTEGER_RANGE` 的 guidance 文字，第 3.2 節） |
| 三個 1.0 example | `git diff --stat 02d821d~1..HEAD -- examples/` → 只有 `examples/integer_knapsack.json | 50 +`；`knapsack.json` / `assignment.json` / `tsp.json` 未改一字 |

與 3a 結束時的差異：+1 backend（`fujitsu_da`）、IR 1.0 → 1.0 + 1.1、catalog 34 → 41、warning 表 9 → 11、architecture 測試 27 → 28、`pytest` 1311 → 2900。

---

## 2. §27 逐條核對

### 2.1 ✅ Phase 1/2/3a 全部測試不變仍通過；三個 1.0 example 不改一字；golden 位元一致測試通過（BQM / CQM 輸出、estimate、penalty_scale）

- `pytest -q` → 2900 passed（含 Phase 1/2/3a 全部既有測試）。
- 三個 1.0 example 未改：`git diff --stat 02d821d~1..HEAD -- examples/knapsack.json examples/assignment.json examples/tsp.json` → 空輸出；`git log --oneline 02d821d~1..HEAD -- examples/` 只有 `4bbd9b1`（新增 `integer_knapsack.json`）。
- 3a 核心編譯測試檔零改動：`git diff --numstat 02d821d~1..HEAD -- tests/` 中 `tests/unit/test_bqm_compiler.py`（16 passed）、`test_cqm_compiler.py`（21 passed）、`tests/scenarios/test_knapsack.py`、`test_assignment.py`、`test_knapsack_cqm.py`、`tests/mcp/test_solve_sa.py`、`test_solve_errors.py` 皆未出現。
- golden 位元一致：`tests/unit/test_golden_phase3a.py`（42 passed）——`test_recording_provenance`（錄自 3a commit `20ba640`）、`test_problem_list_matches_recording`、`test_output_is_bit_identical_to_phase3a[<name>]` 參數化 **40 個 1.0 問題**（三個 example、5 個 scenario 問題、`bqm_test_*` 16 個、`cqm_test_*` 16 個）。快照內容（`tests/golden/record_phase3a_golden.py::snapshot_problem`）：BQM（`hard_penalty`、variables、linear / quadratic / offset、`num_variables`、`internal_variables`、`objective_scale`、`constraint_trace`）、CQM（variables、objective、每條 constraint 的 lhs / sense / rhs / weight / penalty、`num_variables`、`internal_variables`、`objective_scale`、`constraint_trace`）、`estimate_compiled_variables`、`compute_penalty_scale`、`validate_problem_full`（exact capabilities 與 `model_type="cqm"` 兩份完整輸出）。
- 相容性：`tests/unit/test_phase1_compat.py::TestPhase1ExamplesStillParse::test_example_parses_unchanged`、`::test_phase1_examples_compile_bit_identical_to_golden`、`::test_fujitsu_da_options_default_to_none`；`::TestPhase3bExampleParses::test_integer_knapsack_parses`（15 passed）。
- 指令：`annealbridge solve examples/{knapsack,assignment,tsp}.json` → success、objective 17 / 8 / 8（與 3a 報告相同）。

### 2.2 ✅ `version` 接受 `"1.0"` / `"1.1"`；1.0 + integer → `INTEGER_REQUIRES_VERSION_1_1`；1.1 純 binary 合法

- 指令：`OptimizationProblem.model_fields['version'].annotation` → `typing.Literal['1.0', '1.1']`（預設 `"1.0"`）；`Variable.model_fields['type'].annotation` → `typing.Literal['binary', 'integer']`。
- `tests/unit/test_models.py::TestVersion::test_version_defaults_to_1_0`、`::test_version_1_1_accepted`、`::test_other_versions_rejected[2.0|1.2|1|""]`。
- `tests/unit/test_problem_validator.py::TestIntegerBoundsErrors::test_integer_requires_version_1_1`（`path == "version"`、recommended_action 逐字）、`::test_version_error_is_collected_with_bounds_errors`、`::test_version_1_1_with_only_binary_variables_is_legal`、`::test_legal_integer_problem_passes`。
- `tests/mcp/test_validate.py::test_integer_variable_on_version_1_0_is_an_error`（經 in-memory MCP Client）。
- 指令：把 `integer_knapsack.json` 複製到暫存目錄、`version` 改 `"1.0"` 後 `annealbridge validate` → `Valid: no`，唯一 error `[INTEGER_REQUIRES_VERSION_1_1] version: Problem declares integer variables (item_a, item_b, item_c, item_d) but version is '1.0'; integer variables require version "1.1"`，exit 1。

### 2.3 ✅ §9.1 五個 error、§9.3 兩個 warning 皆有測試；`SELF_QUADRATIC_TERM` 只對 binary

- 五個 error（`tests/unit/test_problem_validator.py::TestIntegerBoundsErrors`）：`test_integer_bounds_missing`、`test_integer_bounds_invalid`、`test_bounds_on_binary`（皆參數化）、`test_integer_range_too_large`（含 `2**31`）、`test_bounds_exactly_at_the_limit_are_legal`、`test_integer_requires_version_1_1`、`test_invalid_and_too_large_are_reported_together`。
- catalog 覆蓋：`tests/unit/test_error_catalog.py::TestValidatorCodesCovered::test_validator_error_codes_match_expected_set`、`TestEmittedErrorCodesAreCatalogued::test_validator_error_codes_are_exactly_its_literal_codes`、`TestGuidanceTextCarriesNoConfigValues::test_contract_constant_texts_match_the_spec_wording`（`±(2^31-1)`、`"1.1"`、`0/1` 逐字）。
- 兩個 warning（`tests/unit/test_problem_validator_full.py`）：`TestLargeIntegerRange::test_eleven_bits_warns`、`::test_ten_bits_does_not_warn`、`::test_cqm_path_does_not_warn`、`::test_negative_lower_bound_counts_the_span`；`TestIntegerQuadraticBlowup::test_wide_integers_in_one_constraint_warn`（2556 interactions）、`::test_small_integers_do_not_warn`、`::test_cqm_path_does_not_warn`、`::test_all_binary_problem_never_warns_however_dense`、`::test_integer_warning_texts_name_no_backend_and_no_number`；`TestWarningPayload::test_every_warning_code_is_covered`。
- 門檻常數：`validation/problem_validator.py` `LARGE_INTEGER_BITS_THRESHOLD = 10`、`INTEGER_QUADRATIC_BLOWUP_THRESHOLD = 2000`。
- `SELF_QUADRATIC_TERM`（`tests/unit/test_problem_validator.py::TestSelfQuadraticTermByType`）：`test_binary_square_is_rejected`、`test_integer_square_is_legal`、`test_unknown_variable_square_keeps_the_binary_verdict`；來源 `validation/problem_validator.py:806-813`（`variable_types.get(name, "binary") == "binary"`）。

### 2.4 ✅ estimates 全部 bounds-aware，暴力枚舉測試通過；`estimate_compiled_variables == compiled.num_variables`（BQM 路徑，含編碼位元與 slack）

- `validation/estimates.py` 的 `lhs_bounds`、`compute_objective_scale`、`_max_abs_affine`、`compute_soft_energy_bound`、`analyze_inequality`、`count_slack_bits`、`constraint_bit_count` 皆有 keyword-optional `bounds=None`；`integer_encoding_bits(lower, upper) == (upper - lower).bit_length()`。
- 暴力枚舉（`tests/unit/test_estimates_bounds.py`，634 passed）：`TestBruteForce::test_lhs_bounds_are_attained`、`::test_objective_scale_bounds_the_objective`、`::test_soft_energy_bound_is_the_exact_maximum`、`::test_soft_penalty_scale_dominates_every_soft_energy`、`::test_slack_range_covers_every_feasible_slack`；`TestIntegerEncodingBits::test_bit_count`、`::test_expansion_reaches_every_value_and_nothing_more`；`TestBinaryDegeneracy::test_every_function_agrees_with_and_without_bounds`（全 binary 時有無 bounds 結果連 `repr` 都相同）。
- `estimate == num_variables`：`tests/unit/test_bqm_compiler_integer.py::TestCompiledVariableEstimate::test_hand_written`、`::test_random`、`::test_random_corpus_actually_contains_integers`；`::TestIntegerKnapsackExample::test_estimate_matches_compiled_count`（15 = 4×2 編碼位元 + 5 capacity slack + 2 soft slack）；`tests/unit/test_problem_validator_full.py::TestValidateCompileConsistency::test_validator_and_compiler_agree`；`tests/mcp/test_validate.py::test_integer_problem_estimate_counts_encoding_bits`（MCP 層 `estimated_compiled_variables == 15`）。CQM 路徑另有 `tests/unit/test_cqm_compiler_integer.py::TestEstimateMatchesCompiledCount`。
- 指令：`annealbridge validate examples/integer_knapsack.json` → `Estimated compiled variables: 15`、`Objective scale: 96`。

### 2.5 ✅ `ModelCompiler.decode()` 存在；service 在 `process_candidates` 前呼叫；`__int_` / `__slack_` 不出現在任何 `Solution`

- Protocol：`compiler/base.py:48` `def decode(self, compiled, raw) -> RawSolverResult`；實作 `compiler/bqm.py:142`、`compiler/cqm.py:154`。
- service：`orchestration/optimizer.py:684` `decoded = compiler.decode(compiled, raw)`，緊接 `:685` `process_candidates(problem, decoded, frozenset(), problem.solver.top_k)`；`process_candidates` 在該檔只有 `:235` 定義與 `:685` 唯一呼叫點。
- 內部名不外洩：`tests/unit/test_service_decode.py`（31 passed）——`TestServiceBQMPath::test_no_internal_variable_reaches_a_solution`、`::test_the_backend_really_did_shuffle_and_include_internal_columns`、`TestServiceCQMPath::test_no_internal_variable_and_correct_optimum`、`TestServiceIntegerBQMPath::test_no_internal_variable_reaches_a_solution`、`::test_values_are_python_ints_inside_the_declared_bounds`、`TestBQMCompilerDecodeIntegers::test_a_missing_encoding_bit_is_a_contract_violation`；`tests/unit/test_bqm_compiler_integer.py::TestInternalNamesNeverLeak::test_service_solutions_have_no_internal_keys`、`::test_internal_variables_are_prefixed`（internal 皆 `__int_` / `__slack_` 前綴）；`tests/scenarios/test_integer_knapsack.py::TestIntegerKnapsackAcrossPaths::test_every_solution_is_a_clean_integer_assignment`。

### 2.6 ✅ BQM 路徑：整數 binary expansion、負下界、`x·x`、decode 值在 bounds 內；exact 對 `integer_knapsack.json` == 暴力枚舉最佳解

- binary expansion：`tests/unit/test_integer_encoding.py`（38 passed）——`TestEncodeIntegerVariables::test_bit_count_naming_and_coefficients_follow_the_estimates`、`::test_every_bit_pattern_is_in_range_and_every_value_is_reachable`、`::test_form_keys_follow_the_declaration_order`；`TestSubstituteLinear`、`TestSubstituteQuadratic`、`TestExpandProduct`。
- 負下界：`tests/unit/test_bqm_compiler_integer.py::TestDecodeNegativeLowerBounds::test_decode_folds_bits_back_into_integers`（窮舉位元列、每個 bounds 內的值都被取到）、`::test_encoding_shape`（下界 −3 → 3 個位元）。
- `x·x`：`::TestEnergyIdentityWithSquares::test_energy_matches_objective_plus_penalties[minimize|maximize]`、`::test_pure_square_without_constraints`；`tests/unit/test_integer_encoding.py::TestExpandProduct::test_both_folding_modes_agree_on_every_bit_pattern`。
- decode 在 bounds 內：`tests/unit/test_service_decode.py::TestBQMCompilerDecodeIntegers::test_every_decoded_value_is_inside_the_declared_bounds`、`::test_int8_bits_widen_to_int64_values`。
- exact == 暴力枚舉：`tests/unit/test_bqm_compiler_integer.py::TestIntegerKnapsackExample::test_service_optimum_equals_brute_force`、`::test_brute_force_runner_up_is_strictly_worse`（34.0 vs 32.0）；`tests/scenarios/test_integer_knapsack.py::TestIntegerKnapsackOracle::test_brute_force_oracle_confirms_a_unique_optimum`（256 組枚舉）、`TestIntegerKnapsackExact::test_exact_backend_finds_the_optimum`。
- 指令：`annealbridge solve examples/integer_knapsack.json` → success、objective 34、`item_a=0, item_b=1, item_c=1, item_d=3`、hard 1/1、soft 0 violations。
- 3a 位元路徑不變：`::TestAllBinaryUnchanged::test_no_encodings_and_int8_is_preserved`。

### 2.7 ✅ CQM 路徑：INTEGER 變數直通；含整數 soft constraint 走 objective 形式且與 validator 同公式（測試證明）；binary-only soft 維持 3a 原生；penalty 型別在呼叫前決定

- 直通：`tests/unit/test_cqm_compiler_integer.py::TestHardConstraintsPassThrough::test_declared_variables_keep_their_vartype_and_bounds`（`dimod.INTEGER` + bounds）、`::test_lhs_keeps_integer_vartypes_and_declared_bounds[eq|le|ge]`、`::test_no_slack_and_no_internal_variables`。
- objective 形式與 validator 同公式：`::TestSoftEnergyIdentity::test_hand_written`、`::test_random`（≥ 30 個隨機問題）；斷言（`test_cqm_compiler_integer.py:394 assert_energy_identity`）對每個業務 assignment 同時比對 `min_s energy == sign·objective + Σ w·max(0, v)²` 與 `== objective + validate_solution(...).soft_violation_score`；`::test_corpus_covers_every_objective_form_case`（`==` / `<=` / `>=` / clamp / redundant 五分支各至少一例）、`::test_objective_form_traces_are_marked`；`TestExpandSquareQM::test_energy_equals_direct_square`、`::test_squares_stay_quadratic_for_integer_only`。
- binary-only 維持原生：`::TestBinaryOnlySoftStaysNative::test_soft_constraint_is_a_native_weighted_constraint`（1.0 與 1.1 參數化）、`::test_mixed_problem_splits_the_two_writings`；3a 的 `tests/unit/test_cqm_compiler.py::TestSoftWeightSemantics::test_soft_energy_is_weight_times_violation_squared`、`TestCrossValidationWithExactCQMSolver`（三支）未改仍綠。
- penalty 型別在呼叫前決定：`compiler/cqm.py:196-202`（`involves_integer = any(...)` 於任何東西碰到 cqm 之前分支）；全檔 `grep -n "try:\|except"` 零命中（唯一 `except` 字樣是 import `CompilationError`）。
- 端到端：`::TestIntegerKnapsackOnFakeCQM::test_optimum_equals_the_exact_bqm_path`、`::test_single_attempt_without_penalty`；`TestNegativeLowerBoundsOnFakeCQM`（6 支，含整數 slack 生成與 decode 剔除）。

### 2.8 ✅ `RawSolverResult` 接受 int8 / int64；位元路徑仍 int8；去重與 tie-break 對整數列正確且對位元列位元一致

- dtype：`solvers/base.py:29 _coerce_samples`（非整數 dtype `raise ValueError`，:43-47；只有非 ndarray 輸入才窄化為 int8，:48-51）、`RawSolverResult.from_dicts(..., dtype=np.int8)`（:136）、`sampleset_to_arrays(sampleset, *, dtype=np.int8)`（:224-228）。`tests/unit/test_candidate_arrays.py::TestRawSolverResultDtypes::test_ndarray_dtype_is_preserved`、`::test_from_dicts_dtype_keyword_round_trips_integer_values`、`::test_non_integer_dtypes_are_rejected`、`::test_empty_input_is_an_int8_matrix_with_the_variable_count`。
- 位元路徑仍 int8：四個 BQM backend 皆以預設 dtype 呼叫（`exact.py:83`、`simulated_annealing.py:93`、`dwave_qpu.py:210`、`leap_hybrid_bqm.py:155`）；只有 CQM backend 明寫 `dtype=np.int64`。測試：`tests/remote_mock/test_fujitsu_da_mock.py::TestResultConversion::test_samples_are_int8_rows_in_solution_order`、`tests/unit/test_bqm_compiler_integer.py::TestAllBinaryUnchanged::test_no_encodings_and_int8_is_preserved`、`tests/unit/test_service_decode.py::TestBQMCompilerDecode::test_int8_stays_int8`。
- 整數列去重 / tie-break：`tests/unit/test_candidate_arrays.py::TestIntegerRowsMatchReference::test_int64_rows_match_the_row_by_row_reference`、`::test_heavily_duplicated_int64_rows_keep_min_energy_and_first_seen`、`::test_int8_rows_holding_2_and_minus_1_do_not_take_the_bit_path`、`::test_more_than_64_integer_columns`；`TestIntegerProcessCandidatesMatchesRowByRow::test_same_solutions_same_order_same_numbers`。
- 位元列位元一致：`::TestRowKeys::test_binary_int8_still_takes_the_packbits_path`；`TestDeduplicationMatchesReference::test_same_candidates_min_energy_and_first_seen_order`、`::test_energy_ties_keep_the_earliest_read`；`tests/unit/test_batch_arithmetic_integer.py::TestDtypeIndependence::test_int8_and_int64_agree_bit_for_bit`、`TestIntegerBatchMatchesRowByRow::test_every_candidate_agrees_with_the_scalar_validator`（3a §32.1 要求的 batch 算術證明）。
- 註：`tests/unit/test_solvers.py` 本身沒有 dtype 斷言；位元路徑 int8 的直接斷言在上列檔案。

### 2.9 ✅ CQM backend（Leap 與 fake）值域斷言改為 bounds

- 共用 helper `solvers/base.py:251 assert_samples_within_bounds(sampleset, bounds, *, code)`（逐變數 `(lower, upper)`；允許 float 但須整數值；在 `sampleset_to_arrays` 之前執行）。
- Leap：`solvers/leap_hybrid_cqm.py:200-208`（bounds 取自 `cqm.lower_bound / upper_bound` → 斷言 → `dtype=np.int64`）。`tests/remote_mock/test_leap_hybrid_cqm_mock.py::TestSampleSetConversion::test_non_binary_value_is_a_remote_solver_error[two|half|minus-one]`（訊息含 `outside its bounds`）、`::test_float_zero_one_samples_are_accepted`、`TestNonBinarySampleThroughService::test_non_binary_value_is_a_structured_solver_error`。
- fake：`tests/fakes/local_cqm_backend.py:132-148` 同一斷言（`code="SOLVER_ERROR"`）；由 bounds 判定的正例：`tests/unit/test_cqm_compiler_integer.py::TestIntegerKnapsackOnFakeCQM::test_optimum_matches_the_hand_checked_expectation`（`item_d = 3`）、`TestNegativeLowerBoundsOnFakeCQM::test_every_decoded_value_is_an_int_inside_its_bounds`。
- 註：fake backend 的越界拒絕沒有專屬負例測試（越界拒絕由 Leap 的三個參數化案例覆蓋，兩者呼叫同一個 helper）。

### 2.10 ✅ Routing 三個 reason；blowup 往後排；deterministic、無網路

- `orchestration/routing.py:68-81` `REASON_DESCRIPTIONS` 三條：`R_INTEGER_NATIVE`、`R_INTEGER_ENCODED`（純資訊，不改 tier）、`R_INTEGER_BLOWUP`（進排序 key 第二欄，與 dense 同層）；`_tier()`（:93-119）不看整數。
- `tests/unit/test_routing.py::TestIntegerReasons::test_bqm_backends_report_encoded`、`::test_cqm_backend_reports_native`、`::test_no_tier_rule_keeps_the_binary_order`、`::test_blowup_reason_and_warning`、`::test_blowup_ranks_bqm_backends_after_cqm`（順序變為 `leap_hybrid_cqm, exact, simulated_annealing, dwave_qpu, leap_hybrid_bqm, fujitsu_da`）、`::test_blowup_still_sorts_after_usability`、`::test_pure_binary_problem_emits_no_integer_reason`。
- deterministic / 無網路：`::TestDeterminism::test_same_input_gives_equal_results`、`::TestNothingIsSolved::test_no_backend_solve_or_time_limit_is_called`、`::test_disallowed_remotes_are_not_even_asked_for_availability`；`tests/mcp/test_recommend.py::test_nothing_is_solved`。
- 指令：`annealbridge recommend examples/integer_knapsack.json` → 六列；exact / SA / dwave_qpu / leap_hybrid_bqm / fujitsu_da 帶 `R_INTEGER_ENCODED`，`leap_hybrid_cqm` 帶 `R_INTEGER_NATIVE`；此問題未觸發 blowup。
- 註：整數 reason 的斷言全在 `tests/unit/test_routing.py`；`tests/mcp/test_recommend.py` 無整數專屬測試（MCP 層 `recommend_backend` 為一行委派，3a 已證明同源）。

### 2.11 ✅ `fujitsu_da`：只讀兩個 env、不進 ServerSettings；測試環境隔離；`X-Api-Key` header；標準庫 HTTP、無新依賴；非同步流程含 DELETE 與 cancel；例外 → code 表；`BACKEND_CONFIG_INVALID` → `configuration_error`；`REMOTE_BUSY` retryable；redaction 涵蓋 key 與 header；不送業務字串

- **只讀兩個 env**：`grep -rn "FUJITSU" src/` 只命中 `solvers/fujitsu_da.py:11,12,60,61`（docstring 與 `API_KEY_ENV` / `URL_ENV`）與 `solvers/metadata.py:68`（redaction 清單）；`src/annealbridge/config/` 零命中；`ServerSettings.model_fields` 十個欄位無 Fujitsu。`_read_settings()`（`fujitsu_da.py:229`）每次呼叫 live 讀、不快取。`tests/remote_mock/test_fujitsu_da_mock.py::TestIsAvailable::test_missing_key`、`::test_empty_key_counts_as_missing`、`::test_detail_never_contains_the_url_or_the_key`、`TestRequestShape::test_url_env_overrides_the_base_and_loses_its_trailing_slash`。
- **測試環境隔離**：`tests/conftest.py:38` `_CREDENTIAL_ENV_VARS = ("DWAVE_API_TOKEN", "FUJITSU_DA_API_KEY", "FUJITSU_DA_URL")`，autouse fixture `_clear_credential_env` 逐一 `delenv`；`tests/remote_live/conftest.py:21-30` 以同名 no-op fixture 覆寫（live 需真憑證），`:33-44 _require_live_opt_in` 依模組 `REQUIRED_ENV` skip。
- **`X-Api-Key` header**：`fujitsu_da.py:455-459` headers 恰為 `X-Api-Key` / `Content-Type` / `Accept`。`TestRequestShape::test_headers_are_exactly_the_three_declared_ones`、`::test_every_request_carries_the_same_headers`。
- **標準庫 HTTP、無新依賴**：`fujitsu_da.py:22-23` 只 import `urllib.error` / `urllib.request`；`grep -rn "^import requests\|^import httpx\|^from requests\|^from httpx" src/` 零命中；`pyproject.toml` optional-dependencies 只有 `mcp` / `dwave` / `all` / `dev`。`tests/architecture/test_import_boundaries.py::test_no_third_party_http_package_is_imported_anywhere`（`BANNED_HTTP_PACKAGES = {"requests", "httpx", "aiohttp"}`）、`tests/remote_mock/test_fujitsu_da_mock.py::TestCapabilities::test_no_third_party_http_package_is_imported`、`tests/unit/test_registry.py::TestDefaultRegistry::test_default_registry_needs_no_third_party_http_package`。
- **非同步流程含 DELETE 與 cancel**：`_submit`（:548，`POST /v4/async/qubo/solve`）→ `_await_result`（:562，輪詢 `GET /v4/async/jobs/result/{job_id}`）→ 成功後 DELETE（:463-469）；`Error` 狀態也 DELETE 釋放 slot（:587-593）；逾時 `POST /v4/async/jobs/cancel`（:605-611）；`_best_effort`（:620）失敗只 log。`TestPolling::test_default_script_polls_three_times_and_sleeps_twice`、`::test_result_is_deleted_exactly_once_after_done`、`::test_delete_http_failure_is_only_a_warning`、`::test_delete_exception_is_only_a_warning`；`TestPollTimeout::test_timeout_cancels_the_job`、`::test_cancel_failure_does_not_hide_the_timeout`、`::test_no_result_is_deleted_on_timeout`；`TestMalformedResponses::test_job_status_error_reports_the_message_and_frees_the_slot`。
- **例外 → code 表**：`fujitsu_da.py:85-90` `_HTTP_STATUS_CODES = {401: REMOTE_AUTH_FAILED, 403: REMOTE_AUTH_FAILED, 413: REMOTE_SOLVER_ERROR, 429: REMOTE_BUSY}`；`:98-103` `_BAD_REQUEST_MESSAGE_CODES`（`Monthly usage exceeds` → `REMOTE_QUOTA_EXCEEDED`；`X-Access-Token` / `X-Api-Key` / `Invalid request header` → `BACKEND_CONFIG_INVALID`）；其他 status（含 5xx、其餘 400）→ `REMOTE_SOLVER_ERROR`；`_classify_transport_exception`（:179）`TimeoutError` 與 `URLError(reason=TimeoutError)` → `REMOTE_TIMEOUT`，其他 `OSError` → `REMOTE_SOLVER_ERROR`。`TestHttpStatusClassification::test_status_maps_to_a_code`（10 個 id：401 / 403 / 400-quota / 400-access-token-header / 400-api-key-header / 400-problem-level / 413 / 429 / 500 / 503）、`::test_413_message_names_the_payload_size_not_the_body`、`::test_status_is_also_classified_while_polling`；`TestTransportExceptionClassification::test_submit_exception_maps_to_a_code`（timeout / url-timeout / refused / disconnected / oserror）、`::test_poll_exception_is_classified_too`；`TestMalformedResponses::test_non_json_2xx_is_a_solver_error`、`::test_missing_job_id_is_a_solver_error`、`::test_done_without_qubo_solution_is_a_solver_error`、`::test_unexpected_job_status_is_a_solver_error`；`TestResultConversion`（缺索引 / 非 bool → `REMOTE_SOLVER_ERROR`，不補 0）。
- **`BACKEND_CONFIG_INVALID` → `configuration_error`**：`fujitsu_da.py:217-218,225`（400 header 錯）與 `:506-511`（非 https URL）設 `status="configuration_error"`；`exceptions.py:20-43` `SolverExecutionError.status`；service `orchestration/optimizer.py:808` `getattr(exc, "status", None) or "solver_error"`。`TestServiceStatusMapping::test_header_rejection_is_a_configuration_error`、`::test_problem_level_rejection_is_a_solver_error`、`::test_auth_failure_is_a_solver_error`；`TestIsAvailable::test_non_https_url_is_a_configuration_error`。指令：`FUJITSU_DA_API_KEY=FAKEKEY-abc123 ANNEALBRIDGE_ALLOW_REMOTE=true FUJITSU_DA_URL=http://localhost:1 annealbridge solve examples/knapsack.json --backend fujitsu_da --json` → `"status": "configuration_error"`、`"code": "BACKEND_CONFIG_INVALID"`，輸出不含 `FAKEKEY`。
- **`REMOTE_BUSY` retryable**：`RETRYABLE_CODES == {CONCURRENCY_LIMIT, REMOTE_BUSY, REMOTE_SOLVER_ERROR, REMOTE_TIMEOUT}`。`TestHttpStatusClassification::test_429_is_retryable_in_the_catalog`、`TestServiceStatusMapping::test_busy_is_reported_as_retryable`。
- **redaction 涵蓋 key 與 header**：`solvers/metadata.py:68 _CREDENTIAL_ENV_VARS`、`:73-80 _REDACTION_PATTERNS`（加 `X-Api-Key: …`、`X-Access-Token: …`、`"X-Api-Key": "…"` 三條）。`tests/unit/test_metadata.py::TestRedact::test_fujitsu_env_key_is_masked`、`::test_both_vendor_env_credentials_are_masked_together`、`::test_api_key_header_is_masked`、`::test_access_token_header_is_masked`、`::test_api_key_in_json_headers_is_masked`；`tests/remote_mock/test_credential_leak.py` 的 `RemoteKind` 參數化表（:97）加入 `fujitsu_da`，`TestTokenNeverLeaves`（六支）、`TestBareTokenIsMasked`、`TestLazyResolveFailureIsRedacted` 皆對 DA 執行；DA 專屬 `TestFujitsuResponseBodyEchoingTheKeyIsMasked`（三支）、`TestExceptionChainCarriesNoToken::test_da_transport_failure`、`::test_da_poll_failure_after_submission`、`::test_da_error_body_echoing_the_key`、`::test_da_invalid_json_body_echoing_the_key`；`test_fujitsu_da_mock.py::TestRedaction::test_key_in_a_transport_exception_is_masked`。指令：設 `FUJITSU_DA_API_KEY=FAKEKEY-abc123` 跑 `capabilities` / `recommend` / 上述 `solve --json` → 三者輸出 `grep -c FAKEKEY` 皆 0。
- **不送業務字串**：`build_request_body`（:266-285）body 只有 `fujitsuDA3` 與 `binary_polynomial`；`TestRequestShape::test_no_business_strings_are_sent`、`::test_body_is_exactly_the_solver_block_and_the_polynomial`、`::test_binary_polynomial_matches_the_bqm_term_by_term`、`::test_unset_options_are_omitted_and_only_the_default_time_limit_is_sent`、`::test_given_options_are_all_forwarded`。
- 上限：`TestPolicyLimits`（`time_limit_seconds=400` 且 policy 300 → `resource_limit_exceeded` / `REMOTE_TIME_LIMIT`，未 clamp；`num_run=2000` → pydantic 拒絕）；retries：`TestRemoteRetries`（`allow_remote_retries` 兩種值的 attempt 數）；整數問題經 DA mock decode 成整數值（`TestIntegerProblemThroughTheService`）。

### 2.12 ✅ registry 六個、名單測試、`test_no_backend_names.py` 名單、`test_fifth_backend.py` 不改仍綠

- 指令：`SolverRegistry.default().names()` → `['exact', 'simulated_annealing', 'dwave_qpu', 'leap_hybrid_bqm', 'leap_hybrid_cqm', 'fujitsu_da']`。
- 名單測試：`tests/unit/test_registry.py::TestDefaultRegistry::test_default_registers_exactly_the_expected_backends_in_order`；`tests/mcp/test_capabilities.py::test_all_backends_listed`、`::test_limits_come_from_policy`（`fujitsu_da` → `{"max_time_seconds": 300}`）；`tests/unit/test_cli.py::TestCapabilities::test_table_with_default_policy`、`::test_remote_backends_enabled_when_policy_allows`、`TestSolveErrors::test_unknown_backend_lists_all_known_backends`、`TestRecommend::test_knapsack_table_matches_the_spec_example`（六列）。
- `tests/architecture/test_no_backend_names.py:22-29` `BACKEND_NAMES` 含 `"fujitsu_da"`。
- `git diff --stat 02d821d~1..HEAD -- tests/architecture/test_fifth_backend.py` → 空；`pytest tests/architecture -q` → 28 passed。

### 2.13 ✅ catalog 41 個 code 皆有 `recommended_action`；warning 覆蓋測試通過；reason 覆蓋測試通過

- 指令：`len(RECOMMENDED_ACTIONS)` = 41，空字串條目 0；`len(_WARNING_RECOMMENDED_ACTIONS)` = 11。
- `tests/unit/test_error_catalog.py`（116 passed）：`TestRecommendedActions::test_code_present_with_non_empty_text`（參數化 41）、`::test_no_extra_codes`；`TestEmittedErrorCodesAreCatalogued::test_collector_sees_the_fujitsu_da_http_codes`（`declared_error_codes()` 於 :249-274 直接 import `_HTTP_STATUS_CODES` 與 `_BAD_REQUEST_MESSAGE_CODES` 併入掃描）、`::test_every_literal_code_has_a_recommended_action`、`::test_every_declared_code_has_a_recommended_action`；`TestRetryableCodes::test_retryable_codes_are_exactly_the_transient_ones`。
- warning：`TestEmittedWarningCodesAreCatalogued::test_every_emitted_warning_has_guidance_and_none_is_stale`；`TestGuidanceTextCarriesNoConfigValues::test_warning_guidance_has_no_numbers`、`::test_recommended_action_has_no_numbers`。
- reason：`tests/unit/test_routing.py::TestReasonCatalog::test_every_reason_code_has_a_description`、`::test_every_emitted_reason_is_described`、`TestReasonCatalogMatchesTheSource::test_every_source_reason_code_is_described`、`::test_no_other_r_constant_hides_in_the_module`（AST 收集 `R_` 常數，十二個 code）。

### 2.14 ✅ `get_optimization_capabilities` 回 `schema_version "1.1"`、`schema_versions`、`supported_variable_types` 含 integer

- `interfaces/capabilities.py:91-99`：`schema_versions = list(get_args(OptimizationProblem.model_fields["version"].annotation))`、`schema_version = schema_versions[-1]`、`supported_variable_types` 由 `Variable.type` 的 Literal 推導（無硬編碼）。
- `tests/mcp/test_capabilities.py::test_schema_metadata`（`"1.1"`、`["1.0", "1.1"]`、`["binary", "integer"]`）、`::test_all_backends_listed`（六個）、`::test_capabilities_never_calls_solve`；`tests/mcp/test_tools_list.py` 仍四個 tool。
- 指令：`build_capabilities(SolverRegistry.default(), ExecutionPolicy())` → `schema_version 1.1`、`schema_versions ['1.0', '1.1']`、`supported_variable_types ['binary', 'integer']`、六個 backend。

### 2.15 ✅ `examples/integer_knapsack.json` 存在，三條路徑（exact / SA / fake CQM）scenario 同一最佳值

- 檔案：`version "1.1"`、四個 `[0, 3]` integer、hard `<= 18`、soft `b + c <= 2`（weight 3）、`solver.backend = "exact"`。
- `tests/scenarios/test_integer_knapsack.py`（8 passed）：`TestIntegerKnapsackOracle::test_brute_force_oracle_confirms_a_unique_optimum`（256 組枚舉，34.0 / `{item_a:0, item_b:1, item_c:1, item_d:3}` 唯一）、`TestIntegerKnapsackExact::test_exact_backend_finds_the_optimum`、`TestIntegerKnapsackSimulatedAnnealing::test_sa_with_fixed_seed_finds_the_optimum`（`seed=1234, num_reads=100`）、`TestIntegerKnapsackCQM::test_cqm_backend_finds_the_optimum`（`FakeLocalCQMBackend`，`metadata.model_type == "cqm"`）、`TestIntegerKnapsackAcrossPaths::test_three_paths_agree_on_the_first_solution`、`::test_every_solution_is_a_clean_integer_assignment[exact|simulated_annealing|fake_local_cqm]`。
- 指令：`annealbridge solve examples/integer_knapsack.json` → 34 / `0,1,1,3`。**注意**：CLI 的 `--backend simulated_annealing` 沒有 seed 選項且 example 未設 `seed`，核對時兩次執行分別得到 32（`0,0,2,3`，可行但非最佳）與 34；這是啟發式求解器的正常行為，scenario 以固定 seed 斷言，README 已註明。

### 2.16 ✅ README / OVERVIEW 更新；`pytest` 全過，無 skip / xfail（除 remote_live 註解 skip）；DA live 測試在無 key 時 skip、有 key 時可跑

- README（spec §23 全部項目，本步驟）：簡介與 Architecture 圖加 `fujitsu_da` 與整數編碼；JSON Input Format 新增「Integer variables (version 1.1)」小節（bounds 規則、五個 error code、`integer_knapsack.json` 片段、BQM 編碼位元數 / CQM 原生、`x·x`、三個 reason code）；`NON_INTEGER_INEQUALITY` 補「applies to integer variables as well」；`objective_scale` 公式改 bounds-aware；CQM 含整數 soft 的 objective 形式；CLI 加 `integer_knapsack.json` 與 reason 說明；Solver Backends 表與條列加 `fujitsu_da`（V4、BQM 路徑、API key、100,000 bits、16 jobs、四個 option 範圍與預設、只轉送四欄位、`PARAMETER_IGNORED` / `SEED_IGNORED`、不 clamp、DELETE / cancel、error code 對照、metadata）；MCP 段 `schema_version` / `schema_versions` / `supported_variable_types`；新增 Fujitsu Setup 小節；Environment variables 加 `FUJITSU_DA_API_KEY` / `FUJITSU_DA_URL` 表；Security 加 Fujitsu 條；Examples 加 `integer_knapsack.json`；Testing 段更新測試數、golden 測試意義、DA live / mock 說明；Limitations 改寫。
- OVERVIEW：規格清單四份；§三名詞表 Variable / Compiler / Slack / Solver 四列；§四新增 Phase 3b 白話段；§七狀態表加 3b 規格與實作列、技術選型補 DA；**§五原則逐字未動**（`git show HEAD:annealbridge_PROJECT_OVERVIEW.md` 與工作樹各抽出 §五至 §六之間文字 `diff` → 相同）。
- `pytest -q -rs` → 2900 passed、4 deselected、0 skipped、0 xfail；`grep -rn "pytest.skip\|pytest.mark.skip\|xfail\|importorskip" tests` 只命中 `tests/remote_live/conftest.py:41,44` 兩處真正的 skip。
- DA live：`tests/remote_live/test_fujitsu_da_live.py`（`pytestmark = pytest.mark.remote`、`REQUIRED_ENV = "FUJITSU_DA_API_KEY"`、`time_limit_seconds=1`）；`pytest tests/remote_live -m remote -rs`（無 key）→ 該測試 skip 訊息 `FUJITSU_DA_API_KEY is not set; cannot reach the remote solver`，其餘三個 D-Wave 測試為 `DWAVE_API_TOKEN is not set`。本機無 Fujitsu 帳號，「有 key 時可跑」未實際執行（決策 3：沒有帳號就永遠 skip）；可跑性由 `test_fujitsu_da_mock.py` 對同一程式路徑（`UrllibTransport` 以外的全部）的 mock 覆蓋支撐，`UrllibTransport` 本身經 `HttpTransport` Protocol 隔離。

### 2.17 ✅ `annealbridge_phase3b_acceptance.md` 逐條核對

- 本檔。

---

## 3. §29 偏離清單落實核對

### 3.1 spec §29 十二列

| 來源 → 3b 變更 | 一致 | 證據 |
|---|---|---|
| outline §2.1「編碼策略允許 compiler option 切換」→ 只做 binary expansion、無 option | ✅ | `compiler/integer_encoding.py:23-24`「Only binary expansion is implemented; there is no encoding option (3b §29)」；`ModelCompiler.compile(problem, hard_penalty)` 簽名（`compiler/base.py:35-45`）無編碼參數；`BQMCompiler` 無可調參數 |
| outline §2.3「`parameter_limits` 宣告新 limit key」→ DA 只用 `time_seconds` | ✅ | `solvers/fujitsu_da.py:115-121` 唯一 `ParameterLimit(limit="time_seconds")`；`test_fujitsu_da_mock.py::TestCapabilities::test_declared_parameter_limits`；`tests/mcp/test_capabilities.py::test_limits_come_from_policy` |
| outline §2.3「`fujitsu` extra」→ 無 extra | ✅ | `pyproject.toml` optional-dependencies 只有 `mcp` / `dwave` / `all` / `dev`；`test_registry.py::test_default_registry_needs_no_third_party_http_package` |
| 3a §32.1「`supports_integer_variables`」→ 不加 | ✅ | `grep -rn "supports_integer_variables" src/ tests/` 零命中 |
| 3a §17.3「`sampleset_to_arrays` 本身不改」→ 加 `dtype=`（預設不變） | ✅ | `solvers/base.py:224-228` keyword-only `dtype: Any = np.int8`；使用點 `leap_hybrid_cqm.py:208`（`np.int64`） |
| Phase 2 §14「任何 backend exception → `solver_error`」→ `SolverExecutionError.status`（只有 `BACKEND_CONFIG_INVALID` 用） | ✅ | `exceptions.py:20-43`；`grep -rn "status=" src/annealbridge/solvers/*.py`（排除 `AvailabilityStatus`）只有 `fujitsu_da.py:225`、`:510` 兩處，皆綁 `BACKEND_CONFIG_INVALID`；`optimizer.py:808`；`TestTransportExceptionClassification::test_submit_exception_maps_to_a_code` 斷言其他例外 `status is None` |
| Phase 1 §12「`SELF_QUADRATIC_TERM`」→ 只對 binary | ✅ | 2.3 |
| Phase 2 §19 redaction 只有 D-Wave → credential env 清單 + 廠商 header pattern；`call_ocean` 抽成 `guarded_call` | ✅ | `solvers/metadata.py:68,73-80,267-292`；`solvers/ocean.py:62-72` `call_ocean` 函式體只有一行 `return guarded_call(...)`；DA 於 `fujitsu_da.py:529-535` 直接用 `guarded_call`；`tests/unit/test_metadata.py::TestGuardedCall`（六支，含 `test_original_exception_is_not_chained`、`test_call_ocean_still_classifies_by_class_name`）；既有 D-Wave mock 測試未改仍綠 |
| Phase 2 §22 `schema_version` 唯一版本 → 最新版本 + `schema_versions` | ✅ | 2.14 |
| 3a §32.1 / outline §2.4「整數 + 二次項 → CQM 優先」→ 不加 tier 規則 | ✅ | 2.10（`_tier()` 不看整數；`test_no_tier_rule_keeps_the_binary_order`） |
| 3a `ModelCompiler` Protocol → 加 `decode`；fake compiler 補方法 | ✅ | 2.5；`tests/unit/test_service_compilers.py` 的 fake compiler 有 `decode` |
| Phase 2 §14 / 3a `process_candidates(problem, raw, internal_variables, top_k)` → `internal_variables` 可選 | ✅ | `orchestration/optimizer.py:235-240` `internal_variables: Collection[str] = frozenset()`；service 傳 `frozenset()`（:685）；3a 的 `test_candidate_arrays.py` 既有呼叫未改 |

### 3.2 spec 未記載的新偏離（實作與 spec 文字不同，文字層級）

| 項目 | spec | 實作 | 判斷 |
|---|---|---|---|
| `LARGE_INTEGER_RANGE` 的 `recommended_action`（§19） | 「An integer variable needs **more than 10** encoding bits on a BQM backend; …」 | 「An integer variable needs **more encoding bits on a BQM backend than is comfortable**; tighten its bounds, rescale its unit, or use a backend that accepts integer variables natively.」 | 因 3a 起的規則「guidance 文字不含數字（即不含 config 值）」由 `test_error_catalog.py::TestGuidanceTextCarriesNoConfigValues::test_warning_guidance_has_no_numbers` 強制；門檻 10 改放在 warning 的 **message**（`… needs {bits} encoding bits on a BQM backend (more than 10) …`）。語意不變、Agent 仍能看到門檻值。`INTEGER_QUADRATIC_BLOWUP` 與五個 error 的文字則與 spec 逐字相同。**不改程式，僅記錄。** |

其餘實作與 §29 十二列完全一致，未發現其他新偏離。

---

## 4. §28 接點對照核對

| 3a §32 條目 → 3b 落點 | 落實 | 證據 |
|---|---|---|
| `Variable.type` 擴充 + bounds；`version` 1.1；1.0 不改（§7） | ✅ | 2.1、2.2；`Variable` 欄位 `name / type / lower_bound / upper_bound / description`，`_reject_bool` field validator，`bounds()` 方法 |
| CQM 加 INTEGER 分支；soft `"quadratic"` 只限 binary，含整數另訂規則並重做一致性證明（§15.1、§15.3） | ✅ | 2.7 |
| BQM 整數 → binary expansion；位元 internal；`estimate_compiled_variables` 納入；`INTEGER_QUADRATIC_BLOWUP`（§14、§8、§9.3） | ✅ | 2.3、2.4、2.6；`compiler/integer_encoding.py`（`AffineForm`、`encode_integer_variables`、`substitute_linear` / `substitute_quadratic`、`expand_square_qm`）；`validation/estimates.py:341-358 estimate_compiled_variables` |
| `sampleset_to_arrays` / `RawSolverResult` 放寬；{0,1} 斷言改 bounds；`_pack_rows` 通用化；batch 算術測試證明（§11、§26.1） | ✅ | 2.8、2.9；`optimizer.py:101-116 _pack_rows`（保留）、`:133-146 _row_keys`（int8 0/1 走 packbits，其餘一欄一 int64 key） |
| `NON_INTEGER_INEQUALITY` 是否放寬 → §9.5 維持 | ✅ | `problem_validator.py:88,1037` 規則不依 model type 分支；`tests/unit/test_problem_validator.py::TestNonIntegerInequality`；README Limitations 補「applies to integer variables as well」 |
| `supports_integer_variables` flag → 不加；routing 見 §17 | ✅ | 3.1 第 4 列；2.10 |
| Fujitsu：查證、`FujitsuDAOptions`、Literal、`parameter_limits`（不新增 key）、availability、redaction、例外表、不加 extra、mock / live | ✅ | `models/problem.py:45-55 FujitsuDAOptions`（1–3600 / 1–1024 / 1–16 / 1–1024）、`:68-75 backend Literal` 含 `"fujitsu_da"`、`:85 fujitsu_da: FujitsuDAOptions | None`；2.11；`tests/remote_live/test_fujitsu_da_live.py` |

---

## 5. 已知限制、發現的不一致與 §30 之後事項

### 5.1 程式與文件的文字面不一致（本步驟只回報，未改程式；均不影響 §27 任何一條）

| 位置 | 內容 | 建議 |
|---|---|---|
| `models/error_catalog.py` `REMOTE_AUTH_FAILED` | `recommended_action` 仍寫「verify the **D-Wave** credentials on the server」；3b 起 Fujitsu 的 HTTP 401 / 403 也對應此 code（核對時以假 key 真打 Fujitsu 端點得 401，回的就是這段文字） | 之後改為廠商中性用語（例如「verify the remote solver's credentials」）；需同步 `test_error_catalog.py::test_3a_texts_match_the_spec_wording` 若有引用 |
| `pyproject.toml` `markers` | `remote` marker 說明「live D-Wave tests (…; require DWAVE_API_TOKEN)」，未提 `FUJITSU_DA_API_KEY`；`addopts` 上方註解亦只提 D-Wave / Leap quota | 之後改為「live remote tests (opt-in; consume vendor quota; require the vendor credential)」 |
| `.github/workflows/ci.yml` | minimal-install job 註解「recommend lists all **five** backends」，實際六個（只是註解，執行內容不受影響） | 之後改 six |
| `.github/workflows/remote-live.yml` | 只注入 `DWAVE_API_TOKEN`，未注入 `FUJITSU_DA_API_KEY`，所以手動觸發時 DA live 測試在 CI 上永遠 skip | 有帳號時加 secret |
| commit `4f50903` 標題 | 開頭多一個殘留的 `"` 字元（`"refactor(solvers): …`） | 歷史紀錄，不動 |

### 5.2 測試覆蓋的小缺口（如實標註，均有替代證據）

- `tests/unit/test_solvers.py` 沒有 dtype 斷言；位元路徑 int8 的斷言在 `test_fujitsu_da_mock.py`、`test_bqm_compiler_integer.py`、`test_service_decode.py`（2.8）。
- fake CQM backend 的 bounds 斷言只有正例（整數值與負下界通過），越界拒絕的負例只在 Leap mock（兩者共用同一個 `assert_samples_within_bounds`）（2.9）。
- MCP 層 `tests/mcp/test_recommend.py` 無整數 reason 專屬測試；三個 reason 的斷言全在 `tests/unit/test_routing.py`（2.10）。

### 5.3 已知限制（與 README Limitations 一致）

- 整數只有 binary expansion；無 one-hot / unary option。BQM 路徑上寬範圍整數的代價由 `LARGE_INTEGER_RANGE`（> 10 位元）與 `INTEGER_QUADRATIC_BLOWUP`（> 2000 個二次交互）警告。
- `NON_INTEGER_INEQUALITY` 對整數變數仍適用。
- DA 只支援 V4 端點與 API key；原生 `inequalities` / `penalty_binary_polynomial` / one-hot 未用；`frequency` 不展開；無 OAuth、無 Azure Blob 大問題路徑。
- 沒有 Fujitsu 帳號，DA live 測試從未實際執行過。

### 5.4 §30 之後（不在 3b）

- 實數變數；`NON_INTEGER_INEQUALITY` 依 model type 放寬。
- DA 原生能力（第三種 model type、`penalty_auto_mode`）、OAuth、v3c、Azure Blob、`guidance_config`。
- `DWAVE_CONFIG_INVALID` 在 sampler 建構階段改走 `configuration_error`（與 §20.8 機制統一；3b 刻意不動）。
- 5.1 的四項文字修正。

---

## 6. 步驟 7 本次變更

| 檔案 | 變更 |
|---|---|
| `README.md` | spec §23 全部項目（見 2.16） |
| `annealbridge_PROJECT_OVERVIEW.md` | 規格清單、§三名詞表四列、§四 Phase 3b 白話段、§七狀態表與技術選型；§五原則未動 |
| `annealbridge_phase3b_acceptance.md` | 本檔 |

程式碼（`src/`）、測試、examples、spec、`pyproject.toml`、CI 本步驟**無變更**。

---

## 7. ❌ 項目

無。
