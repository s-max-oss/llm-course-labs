# 实验与失败记录

- `local_full.log`：初版本机正式全流程原始控制台输出。
- `local_full_reviewed.log`：完成缓存、量化和验真修复后重跑的正式全流程日志，对应最终报告。
- `quick_check.log`：小步数功能自检，只用于排错，不进入结果对比。
- HPC 作业 17476 首次启动失败。原因是 Slurm 从临时 spool 目录执行脚本，基于 `$0` 推导出的工作目录不可写；已改为显式 `$HOME/llm_course_lab08`。
- 初次提交还因分区把内存资源报告为 1 MB 而拒绝 `--mem=8G`；按照同课程既有作业方式移除显式内存请求后成功排队。
- HPC 作业 17477 于 gpu4 完成全流程复现，退出码 0，耗时 4 分 09 秒；作业 17478 完成基线与教师规模扫描，退出码 0，耗时 1 分 14 秒。修正后的脚本不再默认借用实验五环境；作业 17513 通过显式 `LAB08_DIR` 与 `LAB08_PYTHON` 完成教师级验证，退出码 0，耗时 1 分 54 秒。原始 Slurm 输出、日志与 JSON 保存在 `hpc_results/`。
- 本机 PyTorch 2.12 对 `torch.ao.quantization.quantize_dynamic` 给出弃用提醒。当前版本仍能正确量化；代码保留兼容入口并记录真实产物，不屏蔽该提醒。
- 空白报告模板的工具包渲染器缺少随包 LibreOffice。未调用用户桌面版 LibreOffice；模板和最终报告改用 Microsoft Word 只读导出 PDF，再用工具包自带 Poppler 转为 PNG 做逐页检查。
