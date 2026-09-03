# AnnealBridge — 專案概覽與開發指引

> 這份文件是給「人」和「AI coding agent」一起看的。
> 目的是說清楚：我們到底在做什麼、為什麼這樣做、哪些事絕對不能偏。
> 細節規格在另外三份文件：`optimization_middleware_phase1_spec_v2.md`、`annealbridge_phase2_spec_v2.md`、`annealbridge_phase3a_spec_v1.md`。
> 規格若與這份文件衝突，以規格為準；但如果你發現自己正在做的事違反了本文件的「核心原則」，先停下來問。

---

## 一、我們在做什麼（一句話）

**做一個「翻譯機 + 求解器」：AI 把人話翻成一份結構化的「最佳化問題」，我們的程式負責把它算出來、檢查答案對不對、再把可信的答案交回去。**

### 用生活例子說

老闆說：「三個員工、三件工作，每人只做一件，總成本最低。」

- **AI（不是我們寫的）** 把這句話翻成一份 JSON：有哪些變數（alice 做 A？bob 做 B？…）、目標（成本最低）、限制（每人一件、每件一人）。
- **AnnealBridge（我們寫的）** 收到這份 JSON，把它變成數學模型、丟給求解器、拿到一堆候選答案、逐一檢查有沒有違規、把合格的排好名次交回去。
- **AI** 再把結果講成人話給老闆聽。

我們**不做**「聽懂人話」這件事，也**不做**「講成人話」這件事。我們只做中間那段：**結構化問題 → 可信答案**。

---

## 二、為什麼要這樣分工

因為 AI 擅長理解語意，但不擅長做精確數學。如果讓 AI 直接產出數學模型（QUBO 矩陣、penalty 係數、slack 變數），它會：

- 算錯係數
- 忘記某個限制
- 每次產出不一樣
- 出錯時沒人知道錯在哪

所以我們的分界線是：

```
AI 負責「說清楚要什麼」    →    程式負責「精確地算出來並驗證」
     （語意、需求）                  （數學、確定性、可測試）
```

這條線是整個專案的靈魂。**任何讓 AI 越過這條線、或讓程式越過這條線的設計，都是偏掉。**

---

## 三、系統長什麼樣

```
        人話
         │
         ▼
    ┌─────────┐
    │   AI    │  ← 不在我們的專案裡（Claude / GPT / 其他 Agent）
    └────┬────┘
         │  結構化問題 JSON
         ▼
╔══════════════════════════════════════════╗
║            AnnealBridge（我們）           ║
║                                          ║
║  1. 檢查問題寫得對不對（Validator）        ║
║  2. 翻成數學模型（Compiler）              ║
║  3. 丟給求解器算（Solver）                ║
║  4. 逐一檢查每個候選答案（Validator）      ║
║  5. 合格的排名、取前幾名（Ranking）        ║
║  6. 不合格就調整參數重算（Retry）          ║
╚══════════════════╤═══════════════════════╝
                   │  結構化答案 JSON
                   ▼
              ┌─────────┐
              │   AI    │
              └────┬────┘
                   │
                   ▼
                  人話
```

### 幾個名詞白話解釋

| 名詞 | 白話 |
|---|---|
| Optimization Problem / IR | AI 交給我們的那份 JSON。裡面只有「變數、目標、限制」，沒有任何數學公式 |
| Variable（變數） | 一個是/否的選擇，例如「alice 要不要做 A」。Phase 1、2 只支援 0/1 |
| Objective（目標） | 想要最大或最小的那個數，例如總成本 |
| Hard constraint（硬限制） | 絕對不能違反的規則。違反 = 答案無效 |
| Soft constraint（軟限制） | 希望盡量滿足，但可以妥協的偏好。有 weight 表示多重要 |
| Compiler（編譯器） | 把 JSON 翻成求解器看得懂的數學模型（BQM / QUBO）。這是**我們的程式**做，不是 AI。Phase 3a 起有兩個：BQM compiler（penalty + slack）與 CQM compiler（限制原生表達） |
| Penalty（懲罰係數） | 編譯時為了讓求解器「不敢違反硬限制」加上的數學重量。**由程式自動算**，AI 不用給、也不該給 |
| Slack variable | 編譯器為了表達「小於等於」自己加的內部輔助變數。AI 看不到、最後答案也不會出現 |
| Solver / Backend（求解器） | 真正算答案的引擎。Phase 1 有本機的模擬退火（SA）和窮舉（Exact）；Phase 2 加 D-Wave 量子退火與雲端混合求解器；Phase 3a 加 D-Wave Leap hybrid CQM 求解器 |
| Solution Validator（答案驗證器） | 拿求解器的每個候選答案，回到**原始 JSON** 逐條檢查限制有沒有違反。這一步不信任求解器，一律重查 |
| MCP | 讓 AI 能標準化呼叫外部工具的協定。Phase 2 用它把 AnnealBridge 包成 AI 可以呼叫的工具 |

---

## 四、各階段做什麼

### Phase 1：把核心引擎做出來（不碰 AI、不碰 MCP、不碰雲端）

做出一個純 Python 套件，能：

1. 讀一份問題 JSON
2. 檢查寫得對不對
3. 翻成數學模型
4. 用本機求解器算
5. 逐一驗證候選答案
6. 排名輸出

用命令列就能跑，用 pytest 就能測。**做完後，不需要任何 AI 或網路，它自己就是一個完整能用的最佳化工具。**

### Phase 2：把引擎包給 AI 用、接上真正的量子硬體

1. 用 MCP 把 Phase 1 引擎包成三個工具：查能力、驗證問題、求解
2. 加 D-Wave 的兩個遠端求解器
3. 加安全機制：遠端預設關閉、不外洩 token、有用量上限、不偷偷降級

**Phase 2 不改 Phase 1 的核心邏輯。** 如果做 Phase 2 時發現要重寫 compiler、validator、retry，代表 Phase 1 架構有問題，要回頭修 Phase 1，而不是在 Phase 2 另寫一套。

### Phase 3a：加 CQM 路徑、加推薦工具、清掉「認名字」的架構債

1. **加一條 CQM 路徑。** 硬限制直接交給求解器原生處理，不用 penalty、不用 slack、也不用重試（只會有一次嘗試）。求解器自己說「這個答案可行」我們也不信，一樣把每個候選答案拿回原始 JSON 重查一遍；它的說法只被當成統計數字記下來。
2. **加一個「推薦工具」**（MCP 的 `recommend_backend`、CLI 的 `annealbridge recommend`）。它只把所有求解器排個名、講清楚哪個能用、為什麼不能用，**只建議、不代替使用者選**：真的要算的時候，永遠用使用者自己指定的那個 backend。
3. **清掉架構債。** service、validator、介面層不再「認名字」（不再寫 `if backend == "..."` 這種特判），改成看每個求解器自己宣告的能力。並且有測試證明：新增第五個求解器，核心一行都不用改。
4. **不動 IR。** 那份 JSON 的格式完全沒變（`version` 仍是 `"1.0"`、仍然只有 0/1 變數）。整數變數和 Fujitsu 留給 Phase 3b。

**Phase 3a 不改 Phase 1／2 的核心邏輯。** 新東西一律以插件和宣告的方式加上去，不是回頭改既有流程。

---

## 五、核心原則（最容易偏掉的地方）

以下是我最在意的幾條。AI agent 執行時如果覺得「這樣做比較方便」而想繞過，**請先停下來問我**。

### 1. AI 只給「要什麼」，不給「怎麼算」

- ✅ JSON 裡有 variables、objective、constraints、soft weight
- ❌ JSON 裡出現 Q 矩陣、penalty λ、slack 變數、BQM bias

如果測試或範例需要 QUBO，那是 compiler 產出來的，不是輸入。

### 2. 答案對不對，只信原始 JSON 的重新檢查，不信求解器的能量值

求解器回傳的 energy 是內部數字，混了懲罰項、slack、偏移量。**不能拿它判斷答案合法，也不能拿它當 business 目標值排名。** 每個候選答案都要回到原始 JSON 重算目標值、逐條檢查限制。

常見偏法：「energy 最低那個一定是合法的，直接取 `.first` 就好」。**不行。** 全部候選都要驗。

### 3. Hard penalty 和 soft weight 是兩件事

- Hard penalty：程式為了「硬限制不能違反」自動算的數學重量。AI 不能給。
- Soft weight：AI 給的「這個偏好多重要」。單位是目標值的單位。

兩者不能互相推導、不能共用欄位、不能一個當另一個的備用值。Phase 2 還多了 chain strength（量子硬體的物理參數），同樣獨立。

### 4. 求解器是可換的零件，核心不能認識任何一個特定求解器

核心模型、compiler、validator 只認識抽象介面 `SolverBackend`。SA、Exact、D-Wave 都是插上去的插件。未來換成 Fujitsu 或 OR-Tools，只加插件，不動核心。

常見偏法：在 orchestration 裡寫 `if backend == "simulated_annealing": ...` 一堆特判。

### 5. 不偷偷幫使用者做決定

- 指定 D-Wave 但不可用 → 回「不可用」，**不偷偷改用 SA**
- 要求 100 萬次 reads 但上限 1000 → 回「超過上限」，**不偷偷砍成 1000**
- 遠端求解失敗 → 回結構化錯誤，**不自動重送四次燒 quota**

理由：AI agent 會根據結果做下一步判斷。偷偷降級會讓它誤以為拿到的是量子解。

### 6. 每一層只做自己的事

```
CLI / MCP  → 只做「收請求、轉格式、回結果」，不含任何最佳化邏輯
Service    → 串流程：驗證 → 編譯 → 求解 → 驗證 → 排名 → 重試
Compiler   → 只翻譯，不求解
Solver     → 只求解，不解析 JSON、不驗證答案
Validator  → 只檢查，不修改
```

常見偏法：在 CLI 裡寫排名邏輯、在 MCP tool 裡呼叫 compiler、在 solver 裡讀 JSON。

### 7. 不為「未來可能需要」先蓋空殼

只有三個地方需要抽象介面：Compiler、SolverBackend、PenaltyStrategy。其他地方直接寫具體程式。不要 factory、不要 repository layer、不要 plugin loader。

### 8. 測試不准繞

- 不准用 `skip` / `xfail` 掩蓋真的 bug
- Scenario test 必須走完整流程（JSON → Service → 結果），不准直接呼叫內部函式充當整合測試
- MCP 測試必須真的透過 MCP Client 呼叫，不准直接呼叫 Python function 就說測過了

---

## 六、AI agent 執行時的自我檢查

每完成一個 Step，對照這幾個問題。任何一個答「是」就是偏了：

- 我有沒有讓輸入 JSON 出現 QUBO / penalty / slack？
- 我有沒有用 energy 判斷答案合不合法，或用 energy 排名？
- 我有沒有只取 `.first` 或只取最低能量的一個？
- 我有沒有把 hard penalty 和 soft weight 寫在同一個欄位、或互相推導？
- 我有沒有在 orchestration / compiler / validator 裡 import 特定求解器的套件？
- 我有沒有在 CLI 或 MCP 層寫最佳化邏輯？
- 我有沒有在求解器不可用時偷偷換成另一個？
- 我有沒有偷偷 clamp 使用者的參數？
- 我有沒有為了讓測試過而 skip / xfail？
- 我有沒有在 Phase 1 加 MCP / D-Wave 雲端 / 資料庫 / Web？
- 我有沒有在 Phase 2 重寫 Phase 1 的核心邏輯？
- 我有沒有讓 token 出現在任何輸出、log、錯誤訊息、測試檔？

另一個實用的檢查：**如果把 MCP 和 CLI 兩個資料夾整個刪掉，剩下的核心能不能 `service.solve(problem)` 照常運作？** 答案必須是能。

---

## 七、目前狀態

| 項目 | 狀態 |
|---|---|
| Phase 1 規格 | 完成（v2） |
| Phase 2 規格 | 完成（v2） |
| Phase 3a 規格 | 完成（v1） |
| Phase 1 實作 | 完成（2026-08，`optimizer` 套件，後於 Phase 2 Step 1 改名 `annealbridge`） |
| Phase 2 實作 | 完成（2026-09-02，spec §35 全部 16 步；`pytest` 510 passed，remote_live 為 opt-in） |
| Phase 3a 實作 | 完成（2026-09-02，spec §30 全部 10 步；`pytest` 1311 passed，remote_live 為 opt-in） |
| 版本控制 / CI | 2026-09-02 建立 git repo；CI workflow 已就緒，推上 GitHub 後生效 |
| Phase 3b（整數變數、Fujitsu DA） | 只留擴充點（3a spec §32），開工前先寫 3b spec |

技術選型（已確定，不要換）：

- Python ≥ 3.11、Pydantic v2
- 求解：dimod、dwave-samplers（本機）、dwave-system（Phase 2 / 3a 遠端）
- MCP：官方 Python SDK v2（`mcp>=2,<3`，`MCPServer`，2026-07 已穩定釋出）
- 測試：pytest
- 套件名：`annealbridge`

---

## 八、給 AI agent 的開工方式

1. 先讀這份文件，再讀對應 Phase 的規格。
2. 嚴格照規格的「開發順序」一步一步做，**每步跑測試全綠才進下一步**。
3. 規格沒寫到的細節，用「第五節的原則」判斷；判斷不了就問，不要猜。
4. 發現規格內部矛盾、或規格與原則矛盾，停下來指出，不要自己選一邊默默做。
5. 每個 Step 結束時，簡短回報：做了什麼、測試結果、有沒有偏離規格的地方及原因。

---

## 九、一句話總結

**AI 說要什麼，程式精確算出來並自己驗證，求解器只是可以換的引擎。**

守住這句話，其他都是細節。
