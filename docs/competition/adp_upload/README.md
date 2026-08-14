# 腾讯 ADP 无 API 适配上传包

## 知识库文件

`knowledge/` 中包含 15 份从正式本地 RAG 资料确定性导出的 Markdown 知识卡。上传到赛事
ADP 的 `StatPy原创课程知识库` 时，只选择该目录内的 15 个 `.md` 文件。

不要上传：

- 两本完整教材 PDF；
- `data/questions.yaml`；
- 评测用例、标准答案或评分规则；
- 本地 SQLite、日志、环境变量或任何学习者数据。

## 重新生成

当 `data/rag/sources/` 中的正式原创资料通过审核并发生变化后，运行：

```bash
.venv/bin/python scripts/export_adp_knowledge.py
```

导出脚本会先执行 manifest、checksum、资料 schema、授权和答案泄漏资格检查，再生成适合平台
切分的 Markdown。目录中如果存在无法对应当前 manifest 的过期 `.md` 文件，脚本会停止并要求
人工核对，不会静默混入知识库。

## 代码节点文件

`code/deterministic_grader.py` 是工作流“确定性判题”代码节点的可粘贴版本。它只依赖
Python 标准库，把学习者的 Python 文本作为 AST 进行静态检查，不会编译或执行学习者代码。
它保持答案正确性、理由证据和 Python 结构三条通道独立：理由不充分不会篡改正确答案，
但缺少题目强制要求的 Python 结构会阻止本轮完成。
