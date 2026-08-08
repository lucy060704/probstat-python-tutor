# G1–G3 本地复现与演示手册

## 1. 目的与安全边界

本手册只复现本地工程，不需要 API Key、ADP、教材 PDF 或外部网络。演示脚本使用团队原创知识卡、
合成数据、临时 SQLite 和匿名档案，不污染仓库内已有学习状态。任何命令失败都应停止并保存输出，
不能跳过失败后宣称通过。

绝对不要运行 `evals/blind/` 或任何 holdout 数据；G3.2 的正式独立评测已经冻结，本轮只允许公开
development 重放。

## 2. 一次性环境准备

在项目根目录执行。macOS：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

不要创建 `.env`；离线演示不需要真实密钥。

## 3. 推荐验收顺序

### 3.1 启动预检

macOS：

```bash
scripts/start_macos.command --check
```

Windows 11 实机：

```bat
scripts\start_windows.cmd --check
```

预期看到“启动检查通过”。仓库自动测试只能静态核对 Windows 脚本，正式材料前仍需在 Windows
11 机器实际执行本节以及 3.6 的完整页面旅程。

### 3.2 G1 离线诊断闭环

```bash
.venv/bin/python scripts/g1_offline_demo.py
```

预期看到确定性诊断、可观察 finding、推荐和下一题；不得调用在线模型。

### 3.3 G3.1 本地 RAG

```bash
.venv/bin/python scripts/g3_local_rag_demo.py \
  '均值 中位数 异常值' --concept mean_median
```

预期返回带来源、版本、切片、checksum 和摘录的命中，并显示 15 个来源、478 个切片。

### 3.4 G3.2 公开 development 评测

```bash
.venv/bin/python -m evals.rag_eval --split development
```

程序先验证冻结指纹、数据独立性和索引，再输出 21 项门禁；`all_gates_pass` 应为 `true`。不要把
本命令改为 blind 或 holdout。

### 3.5 G3.3 五次学生旅程与教师汇总

```bash
.venv/bin/python scripts/g3_product_demo.py \
  --output docs/competition/g3_3_demo_result.json
```

预期 5 个隔离档案都完成答错、四级提示、订正、推荐和下一题；教师汇总不含原始答案；注入模型
网络故障后仍返回本地确定性诊断；`passed=true`。

### 3.6 页面人工演示

```bash
scripts/start_macos.command
```

浏览器只访问终端显示的 `127.0.0.1` 地址。建议按固定路径演示：

1. 学生端开始“数据质量”题；
2. 提交错误答案，观察自动一级概念提示；
3. 逐级查看提示，确认第四级才出现完整解释；
4. 提交订正，核对确定性证据、本地引用、推荐和下一题；
5. 切换教师端，核对匿名汇总与 `k=3` 隐藏；
6. 清空当前匿名档案，确认不要求姓名、学号或联系方式。

截图基线位于 `docs/competition/assets/g3_3/`。它用于显示工程旅程，不代表真实学生试点。

### 3.7 G3.4 API 契约

```bash
.venv/bin/python scripts/g3_api_contract_demo.py \
  --output docs/competition/g3_4_api_contract_result.json
.venv/bin/python scripts/export_api_contract.py --check
```

演示使用进程内 ASGI，不开放网络端口；预期健康、推荐、四级提示、诊断、幂等和 422/503 路径均
通过，结果 `passed=true`。

### 3.8 G3.5 可靠性

```bash
.venv/bin/python scripts/g3_reliability_demo.py \
  --output docs/competition/g3_5_reliability_result.json
```

预期覆盖同键重放、同键改文 409、SQLite 回滚、客户端有界重试、模型超时/熔断/降级和 24 个
隔离诊断；`serious_fault_count=0`、`passed=true`。延迟只用于本机回归，不是线上 SLA。

### 3.9 G3.6 发布审计和全量质量门

```bash
.venv/bin/python scripts/g3_release_audit.py \
  --output docs/competition/g3_release_audit_result.json
.venv/bin/python scripts/run_release_tests.py
.venv/bin/python scripts/g3_verify_rebuilt_release.py \
  --output docs/competition/g3_rebuilt_release_result.json
.venv/bin/python -m ruff check .
git diff --check
.venv/bin/python -m pip check
```

`run_release_tests.py` 只运行不依赖失效 blind/holdout 的公开套件。重建验证器只按审计允许列表复制
文件，强制从临时 `src` 导入，然后执行 G1、RAG、公开 development、产品、API、可靠性、启动
预检、OpenAPI、公开 pytest、ruff、依赖和包内审计；`all_passed` 应为 `true`。维护者可以在含旧
隔离素材的原仓库另跑 `python -m pytest`，但不能把该命令写成公开包承诺或用于 blind 调参。

审计结果应为 `engineering_gate_passed=true`、`competition_submission_ready=false`。后者是诚实的
边界：本地 G1–G3 完成不等于比赛材料、真人证据和赛事平台已经完成。

## 4. 故障处理

- 提示找不到 `.venv`：返回第 2 节创建环境，不要改启动器绕过检查；
- Pydantic 报题库或知识卡错误：保留完整中文错误，检查对应 YAML，不手工跳过；
- development 冻结哈希错误：停止，不修改哈希或评测数据来“让结果变绿”；
- 端口占用：停止旧 Streamlit 进程后重试，不改为公网监听；
- 任何演示 JSON 的 `passed=false`：先修复并重跑聚焦测试，再进行全量测试；
- 重建验证导入路径不是临时 `src`：停止，不能用原工作树替代交付包；
- Windows 实机失败：记录系统版本、Python 版本、完整终端输出和截图，不能只用 macOS 结果替代。
