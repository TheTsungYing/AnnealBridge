# AnnealBridge

[![CI](https://github.com/TheTsungYing/AnnealBridge/actions/workflows/ci.yml/badge.svg)](https://github.com/TheTsungYing/AnnealBridge/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | 繁體中文

**給 AI agent 使用的組合最佳化中介軟體。** agent 用結構化 JSON 描述
*要最佳化什麼*；AnnealBridge 決定*怎麼編碼與求解*，把每個答案拿回原始問題
重新檢查，再透過 [MCP](https://modelcontextprotocol.io)、CLI 或純 Python
回傳排名過、已驗證的解。

## 安裝

需要 Python 3.11 以上。給 MCP host 用的話，只要裝好
[uv](https://docs.astral.sh/uv/)：把 server 加進 `claude_desktop_config.json`
（或你的 host 對應的設定檔），然後重啟 host。host 第一次啟動 server 時，
`uvx` 會把套件抓進它自己的快取環境。

```json
{
  "mcpServers": {
    "annealbridge": {
      "command": "uvx",
      "args": ["--from", "annealbridge[mcp]", "annealbridge-mcp"]
    }
  }
}
```

Claude Code 一行就能註冊：

```bash
claude mcp add annealbridge -- uvx --from "annealbridge[mcp]" annealbridge-mcp
```

要用命令列或 Python API，把套件裝進任何環境即可：

```bash
pip install annealbridge              # core: local backends + CLI
pip install "annealbridge[mcp]"       # + MCP server (annealbridge-mcp)
pip install "annealbridge[dwave]"     # + D-Wave cloud backends
pip install "annealbridge[all]"       # everything
```

`pipx`、用 pip 裝好後以絕對路徑指定 server，以及從 checkout 安裝開發版本，
見[深入安裝](#深入安裝)。

## 一段對話

server 註冊好之後，最佳化就是一段普通的聊天。agent 把需求轉成一份小小的
JSON 文件；AnnealBridge 求解，並在回傳前把每個答案拿回那份文件重新檢查。

> **你：** 我最多能背 10 公斤。物品 A 價值 10、重 6，B 價值 8、重 5，C 價值
> 7、重 4，D 價值 6、重 3。我該帶哪些？

在回覆背後，agent 依序呼叫 server 的三個工具：

1. `get_optimization_capabilities`：可以用哪些變數型別與運算子、現在哪些
   backend 可用、上限是多少。
2. `validate_optimization_problem`：它草擬的問題（四個 binary 變數、一個
   maximize 目標、一條 `<= 10` 的 hard constraint）一次拿回所有錯誤，或者
   確認沒問題。此時還沒求解，也沒花任何成本。
3. `solve_optimization`：排名過的解，每一個都對照原始約束重新驗證過；因為
   窮舉的 `exact` backend 列舉了所有組合，所以帶著 `optimality_proven: true`。

> **Agent：** 帶 A 和 C：價值 17，剛好 10 公斤。次佳是 A 加 D（16，9 公斤）
> 與 B 加 C（15，9 公斤）。這是已證明的最佳解，所有組合都列舉過了。

措辭是 agent 的，數字來自工具結果。第四個工具 `recommend_backend` 會針對
問題把 backend 排名，僅供參考。任何支援 stdio 的 MCP host 都是同樣的用法，
另外也有 streamable-http transport；見 [docs/mcp.md](docs/mcp.md)。agent 送出
的那份文件就是[問題 JSON 一覽](#問題-json-一覽)。

## 命令列與 Python

同一個問題，從終端機或腳本求解。

### 命令列

先把[下方的問題 JSON](#問題-json-一覽)存成 `knapsack.json`，然後：

```bash
annealbridge solve knapsack.json
```

```text
Problem:   knapsack
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

加上 `--json` 可以拿到完整的 `SolveResult`，`--backend simulated_annealing`
可以覆寫 backend，也可以試試 `validate`、`recommend`、`capabilities` 與
`export-schema`。見 [docs/cli.md](docs/cli.md)。

### Python

```python
import json

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

with open("knapsack.json", encoding="utf-8") as f:
    problem = OptimizationProblem.model_validate(json.load(f))

result = OptimizationService().solve(problem)
print(result.status)                          # "success"
print(result.solutions[0].variables)          # {"item_a": 1, "item_b": 0, ...}
print(result.solutions[0].objective_value)    # 17.0
```

領域上的失敗一律以結果回傳，不會丟例外：`result.status` 會是 `success`、
`infeasible`、`invalid_problem`、`resource_limit_exceeded`、
`backend_unavailable`、`configuration_error` 或 `solver_error` 其中之一。
見 [docs/output-format.md](docs/output-format.md)。

## 問題 JSON 一覽

這就是上面兩個範例背後的文件：一個容量為 10 的 0/1 背包問題，也是
repository 中 [examples/knapsack.json](examples/knapsack.json) 的精簡版。
存成 `knapsack.json` 放在任何位置都可以。

```json
{
  "version": "1.0",
  "name": "knapsack",
  "variables": [
    {"name": "item_a", "type": "binary"},
    {"name": "item_b", "type": "binary"},
    {"name": "item_c", "type": "binary"},
    {"name": "item_d", "type": "binary"}
  ],
  "objective": {
    "direction": "maximize",
    "linear_terms": [
      {"variable": "item_a", "coefficient": 10},
      {"variable": "item_b", "coefficient": 8},
      {"variable": "item_c", "coefficient": 7},
      {"variable": "item_d", "coefficient": 6}
    ]
  },
  "constraints": [
    {
      "id": "capacity",
      "type": "hard",
      "terms": [
        {"variable": "item_a", "coefficient": 6},
        {"variable": "item_b", "coefficient": 5},
        {"variable": "item_c", "coefficient": 4},
        {"variable": "item_d", "coefficient": 3}
      ],
      "operator": "<=",
      "rhs": 10
    }
  ],
  "solver": {"backend": "exact"}
}
```

整數變數（`"type": "integer"` 加上 bounds、`"version": "1.1"`）、二次目標
項、帶權重的 soft constraint，以及各 backend 的 solver 偏好設定，都寫在
[docs/problem-format.md](docs/problem-format.md)。`annealbridge
export-schema` 會印出 JSON Schema，agent 可以拿它來做結構化輸出。

repository 裡有四個可直接執行的範例：[背包問題](examples/knapsack.json)、
[指派問題](examples/assignment.json)、[TSP](examples/tsp.json) 與
[整數背包問題](examples/integer_knapsack.json)。安裝後的 wheel 不含這些檔案，
請從 checkout 或 GitHub 取得。

## 深入安裝

核心安裝不需要 `mcp`，也不需要 `dwave-system`：registry 照樣載入，只是把
遠端 backend 回報為不可用。Fujitsu backend 完全不需要任何 extra；它用標準
函式庫直接跟廠商的 HTTPS API 溝通，只等一個 `FUJITSU_DA_API_KEY`。

上面的 `uvx` 會在需要時從獨立的快取環境執行 MCP server。
[pipx](https://pipx.pypa.io/) 則是把 `annealbridge-mcp` 永久裝到 `PATH` 上的
對應做法：

```bash
pipx install "annealbridge[mcp]"
```

用 pip 裝進虛擬環境的 server 不在 host 的 `PATH` 上：這時不用 `uvx`，改把
`"command"` 指向 `annealbridge-mcp` 的絕對路徑（Windows 上是
`Scripts\annealbridge-mcp.exe`）。設定值透過 host 的 `env` 區塊傳給
server；見 [docs/mcp.md](docs/mcp.md)。

要安裝 checkout 上的開發版本：

```bash
pip install "annealbridge[all] @ git+https://github.com/TheTsungYing/AnnealBridge.git"
```

## 運作方式

agent 產生一個 `OptimizationProblem`：binary 或有界整數變數、線性或二次的
目標函式，以及 hard 或 soft 的線性約束。就只有這些。接著 AnnealBridge 會
確定性地：

1. **驗證**問題，一次收集所有錯誤；
2. **編譯**成 Binary Quadratic Model (BQM) 或 Constrained Quadratic Model
   (CQM)，penalty、slack 與整數編碼都由它自己算出來；
3. 在本地或遠端 backend 上**求解**；
4. 把每個候選解拿去對照*原始*的 JSON **重新驗證**，絕不相信 solver 自己
   報的 energy；
5. 把可行解**排名**，回傳前 *K* 名，並附上每條約束的評估結果。

agent 永遠不必寫 QUBO 矩陣、penalty 權重、slack 變數或整數編碼。這些全部
由 Python 核心負責，而且每一步都能在沒有 AI、沒有網路、沒有廠商帳號的情況
下測試。

六個 backend 藏在同一套協定之後：免費的窮舉 `exact` 與本地
`simulated_annealing`，加上可做遠端執行的 D-Wave QPU、Leap hybrid BQM、
Leap hybrid CQM 與 Fujitsu Digital Annealer。

```text
  自然語言 ──► Agent ──► OptimizationProblem JSON
                         (只有 variables / objective / constraints)
                                 │
                                 ▼
          ┌──────────── AnnealBridge ────────────┐
          │ 驗證 → 編譯 → 求解 →                 │
          │ 對照原始問題重新驗證 →               │
          │ 排名                                 │
          └──────────────────┬───────────────────┘
                             │
               SolveResult：排名過、已驗證的解
                             │
                             ▼
                           Agent ──► 自然語言答案
```

## 特色

- **兩個方向都是商業層級的合約。** 輸入是變數、目標函式與約束；輸出是排名
  過的解，附帶目標值與每條約束的評估。兩個方向都不會洩漏 solver 內部細節。
- **兩條編譯路徑。** BQM（自動的 hard constraint penalty、binary slack、
  二進位編碼的整數）給退火機用，CQM（原生約束與整數）給 Leap hybrid CQM
  solver 用。走哪一條由 backend 宣告自己支援什麼來決定。
- **有界整數變數**（`"version": "1.1"`），編碼方式對 agent 完全隱藏；
  `1.0` 的問題行為分毫不變，由 golden test 釘住。
- **結構化的失敗，不丟例外。** 每個結果都是 `SolveResult`，帶有 `status`；
  每個失敗都帶同一份目錄裡的穩定錯誤碼，以及每個錯誤對應的
  `recommended_action`。（`infeasible` 是答案而不是失敗：它帶的是
  `infeasibility_proven` 與一段說明，而不是錯誤碼；窮舉 backend 的
  `success` 則帶 `optimality_proven`。）
- **不做沉默的決定。** backend 不可用就如實回報，絕不偷偷換成本地的。參數
  超過上限就直接拒絕，絕不自動夾到範圍內。schema 沒宣告的欄位一律拒絕，
  絕不無聲丟棄。solve 的結果帶有與 `validate` 相同的 warning，所以被忽略的
  seed 或過寬的整數範圍不會藏在 `success` 後面。
- **預設就安全。** 遠端執行與遠端重試在啟用前都是關閉的；每一項資源上限都
  是環境變數；廠商憑證會從結果、log 與錯誤訊息中遮蔽，並有一整套憑證外洩
  測試佐證。
- **架構是被強制的。** import 邊界、「orchestration、validation 與介面層
  不出現 backend 名稱」、「新增 backend 不必動到求解流程」這些都是測試，
  不是慣例。

## 求解器 backend

| Backend               | 類型   | 路徑 | 說明                                                         |
| --------------------- | ------ | ---- | ------------------------------------------------------------ |
| `exact`               | 本地   | BQM  | 窮舉所有指派；編譯後變數預設上限 24 個（可設定）             |
| `simulated_annealing` | 本地   | BQM  | 啟發式；支援 `num_reads`、`num_sweeps`、`seed`               |
| `dwave_qpu`           | 遠端   | BQM  | 透過 `EmbeddingComposite` 使用 D-Wave 量子退火機             |
| `leap_hybrid_bqm`     | 遠端   | BQM  | D-Wave Leap hybrid BQM solver                                |
| `leap_hybrid_cqm`     | 遠端   | CQM  | D-Wave Leap hybrid CQM solver；原生約束                      |
| `fujitsu_da`          | 遠端   | BQM  | Fujitsu Digital Annealer，QUBO API V4 走 HTTPS，不需 SDK     |

遠端 backend 需要對應的廠商憑證**以及**
`ANNEALBRIDGE_ALLOW_REMOTE=true`；少了任何一個都會回報
`backend_unavailable`。`annealbridge recommend` 會針對給定的問題把 backend
排名而不求解，而且絕不會改掉你指定的那一個。設定步驟與各 backend 的行為見
[docs/backends.md](docs/backends.md)。

## 文件

以下頁面都在 [docs/](docs/README.md) 之下。

| 頁面                                             | 內容                                                                  |
| ------------------------------------------------ | --------------------------------------------------------------------- |
| [docs/problem-format.md](docs/problem-format.md) | 輸入 JSON：變數、目標函式、約束、solver 偏好設定                      |
| [docs/output-format.md](docs/output-format.md)   | `SolveResult` 以及它帶的每一個欄位                                    |
| [docs/errors.md](docs/errors.md)                 | 錯誤目錄、warning code、reason code、exit code                        |
| [docs/cli.md](docs/cli.md)                       | `annealbridge` 命令列                                                 |
| [docs/mcp.md](docs/mcp.md)                       | MCP server、工具、host 設定、Inspector                                |
| [docs/backends.md](docs/backends.md)             | 六個 backend、D-Wave 與 Fujitsu 設定、如何新增 backend                |
| [docs/configuration.md](docs/configuration.md)   | 每一個 `ANNEALBRIDGE_*` 變數與廠商憑證                                |
| [docs/architecture.md](docs/architecture.md)     | 分層、套件結構、設計原則                                              |
| [docs/security.md](docs/security.md)             | 預設值、上限、憑證遮蔽、有哪些資料會送到廠商端                        |
| [docs/testing.md](docs/testing.md)               | 測試結構、golden test、live test、CI                                  |
| [docs/limitations.md](docs/limitations.md)       | 已知限制與哪些東西不在範圍內                                          |

## 安全性一段話

遠端執行預設是**關閉**的，遠端重試預設也是**關閉**的，而且每一項上限
（`ANNEALBRIDGE_MAX_*`）都是以錯誤方式強制執行，而不是把值夾住，所以一個
請求絕不會悄悄變成好幾筆計費送單。streamable-http transport 綁定
`127.0.0.1` 而且**沒有任何認證**；請把它放在反向代理或私有網路後面。憑證
不會出現在結果、log 或錯誤訊息中。細節見
[docs/security.md](docs/security.md)；通報方式見 [SECURITY.md](SECURITY.md)。

## 開發

```bash
git clone https://github.com/TheTsungYing/AnnealBridge.git
cd AnnealBridge
pip install -e ".[all,dev]"
pytest
```

`pytest` 會跑完整套測試，沒有 skip、沒有 xfail，也完全不碰網路；對廠商的
live 測試要自己指定才會跑（`pytest -m remote`）。架構規則、設計原則與
pull request 檢查清單都在 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 現況

版本 0.1.0。問題合約（`1.0` / `1.1`）、六個 backend、CLI 與 MCP 工具都已
完成並有測試涵蓋。目前刻意不支援：實數或無界變數、其他整數編碼方式
（one-hot、unary）、非線性約束、soft 權重自動正規化，以及 Fujitsu 退火機
原生的不等式 / one-hot 功能。見
[docs/limitations.md](docs/limitations.md) 與 [CHANGELOG.md](CHANGELOG.md)。

## 授權

[MIT](LICENSE)
