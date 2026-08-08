# G3.4 平台无关 API 契约阶段门最终报告

## 1. 主 Agent 结论

截至 2026-08-08，G3.4 最终阶段门：**PASS**。独立评委 Agent 最终确认 P0=0、P1=0，允许
进入 G3.5。两次中间复审曾如实判定 HOLD，主 Agent 关闭问题后才获得 PASS：

1. 推荐接口的存储读取故障最初落入 `500/internal_error`，与文档承诺的可重试 503 不一致；现以
   明确服务异常区分“策略无题”的 409 与“基础设施不可用”的 503，并有 SQLite 故障黑盒回归；
2. 运行时增加 503 后，OpenAPI 曾只声明 409；现代码、导出的 JSON 和精确状态码断言均同时覆盖
   409/503。

本轮建立了版本化 Pydantic/OpenAPI JSON 契约、四个 ASGI 路由、统一安全错误、回环服务器和
无网络模拟客户端。API 只适配已通过 G3.3 的 `LearningService`，没有复制或改变确定性判题、
RAG、掌握度、推荐和原子存储逻辑。

当前实现**不是公开 HTTPS 服务**：本地服务器固定绑定 `127.0.0.1`，没有公网鉴权、限流、TLS
证书或密钥托管，OpenAPI 明确标记 `x-public-deployment-ready=false`。本轮没有配置、调用或
发布 ADP，没有上传教材 PDF，没有运行旧 blind/holdout，也没有修改冻结 G3.2 检索器。

## 2. 路由与契约

| 方法 | 路径 | Pydantic 成功响应 | 核心边界 |
| --- | --- | --- | --- |
| GET | `/health` | `HealthResponse` | 不返回 Key、模型名、路径、环境或账号 |
| POST | `/v1/recommend` | `RecommendResponse` | 公开题目无答案、rubric、grader 和提示合集 |
| POST | `/v1/hint` | `HintResponse` | 一次只返回指定层级；L1 无完整解释，L4 明确标记 |
| POST | `/v1/diagnose` | `DiagnoseResponse` | 复用 `LearningService.submit()` 和同一原子回执 |

所有 POST 请求必须带 `schema_version=1.0.0`、`request_id` 和 `idempotency_key`，并禁止额外
字段。诊断与推荐只接受固定格式的随机匿名档案标识；姓名、邮箱、学号和自由文本 ID 会被拒绝，
错误响应不会回显非法值。

机器可读 OpenAPI 位于 `docs/api/openapi.json`。`scripts/export_api_contract.py --check` 会把
Pydantic 重新生成结果与仓库文件逐字比较，防止文档和代码漂移。

## 3. 错误、安全与隐私边界

- 请求正文使用流式累计并限制为 32 KiB；仅接受 `application/json`；
- 400、404、405、409、413、415、422、500、503 均返回 `ApiErrorResponse`；
- 错误只含稳定错误码、简体中文安全文案、是否可重试和无值字段路径；
- 非法答案、邮箱、未知字段、内部异常消息和提供商细节不会进入错误响应；
- 所有响应带 `Cache-Control: no-store` 和 `X-Content-Type-Options: nosniff`；
- 推荐响应不含标准答案、受控别名、rubric、grader 规则或四级提示合集；
- API 不编译、不导入、不执行学习者 Python 文本；
- 本地启动器没有 `--host`，源码中没有 `0.0.0.0`，只能绑定回环地址。

## 4. 幂等和状态事实

`LearningSubmissionRequest` 新增受限 `idempotency_key`。诊断服务以“匿名档案 + 幂等键”生成
提交回执键：相同逻辑请求重试直接返回首次已提交报告，学习历史仍只有一条，模型或客户端超时
不会重复计分。

G3.4 的客户端合同要求一个幂等键只用于一个逻辑提交。若客户端错误地用同一键发送不同正文，
当前服务仍返回第一次报告以保护状态不重复更新；持久化请求指纹和显式 `409` 冲突属于 G3.5。
真正的 service/storage 失败返回可重试 503，且没有学习状态或回执半写入。

## 5. HTTPS 与部署边界

- 本地验证使用 `http://127.0.0.1:8765`，只在同机调试；
- 后期如获用户批准，外部必须由受控网关/反向代理终止 TLS，只暴露 HTTPS；
- 认证、授权、API Key 托管、限流、超时、重试、熔断和负载门槛尚未完成；
- 因此本轮只证明“HTTPS 可承载的应用层 JSON 契约”，不宣称已部署 HTTPS 或可公网使用。

## 6. 无网络模拟与真实回环检查

执行：

```bash
.venv/bin/python scripts/g3_api_contract_demo.py \
  --output docs/competition/g3_4_api_contract_result.json
```

结果：

- `transport=asgi_in_process_no_external_network`；
- health、推荐、L1、L4、正确诊断均通过；
- 相同幂等请求重放后 `idempotent_history_count=1`；
- 额外字段返回 422 且不回显值；
- 注入服务失败返回 503、`retryable=true`，内部异常不泄露；
- `passed=true`。

另用真实 Uvicorn 临时绑定 `127.0.0.1:8765`，`GET /health` 返回版本化 JSON 和 `status=ok`；
随后正常停止服务器。没有执行真实诊断 POST，避免污染默认学习数据库。

## 7. 依赖决定

- 新增直接生产依赖声明：`starlette>=1.0,<2`、`uvicorn>=0.30,<1`；
- 新增开发依赖声明：`httpx>=0.28,<1`；
- 三者此前已由现有环境的上游包安装，本轮把直接使用关系写入项目元数据，避免依赖传递偶然性；
- Starlette 只负责 ASGI 路由和响应，Uvicorn 只负责本地回环服务器，HTTPX 只用于合同测试；
- 没有引入 FastAPI，也没有新增模型、数据库或代码执行依赖。

## 8. 工程验证

- `ruff check .`：通过；
- `git diff --check`：通过；
- `pip check`：`No broken requirements found`；
- `pytest -q`：633/633 通过，用时 70.03 秒；
- API/OpenAPI/模拟/服务聚焦：33/33 通过；
- `scripts/run_local_api.py --check`：路由正确且未打开端口；
- `scripts/export_api_contract.py --check`：OpenAPI 与 Pydantic 一致；
- `scripts/g3_api_contract_demo.py`：无外网模拟通过；
- 真实回环 Uvicorn `/health`：通过。
- 独立评委前两次复审：HOLD，各发现 1 项 P1；逐项修复后最终 PASS，P0=0、P1=0；
- 评委保留 1 项非阻断 P2：OpenAPI 尚未描述可选 `X-Request-ID` 请求头，也未逐项核对所有
  路由方法与参数，已转入 G3.5 契约加固清单。

## 9. 主要变更文件

- `pyproject.toml`
- `src/probstat_tutor/api/__init__.py`
- `src/probstat_tutor/api/schemas.py`
- `src/probstat_tutor/api/adapter.py`
- `src/probstat_tutor/api/app.py`
- `src/probstat_tutor/api/openapi.py`
- `src/probstat_tutor/schemas.py`
- `src/probstat_tutor/service.py`
- `scripts/run_local_api.py`
- `scripts/g3_api_contract_demo.py`
- `scripts/export_api_contract.py`
- `tests/test_api_contract.py`
- `tests/test_g3_api_contract_demo.py`
- `docs/api_contract.md`
- `docs/api/openapi.json`
- `docs/competition/g3_4_api_contract_result.json`
- `README.md`
- `docs/product_spec.md`

## 10. 剩余风险与下一项

1. 尚无 TLS、鉴权、授权、限流和密钥托管，当前 API 绝不能公开部署。
2. 同幂等键不同正文尚未持久化比对请求指纹；G3.5 必须改为明确 409，不能长期只返回首次报告。
3. 超时、客户端重试策略、熔断、并发/负载与 P95 尚未实现或测量，全部属于 G3.5。
4. 当前模拟覆盖工程行为，不代表 ADP 真实 HTTPS 联调成功，也不代表真实教学效果。
5. OpenAPI 的可选 `X-Request-ID` 请求头和全路由方法/参数精确断言仍待 G3.5 补齐。
6. 下一项是 G3.5 可靠性与故障注入；仍不配置或发布 ADP。
