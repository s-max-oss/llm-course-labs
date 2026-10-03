# 实验四报告

孙驰，2024311281，计算机10班。按更新课程样板填写，20页、七章、六道思考题、四图、十四表。模板原文件未修改；正式Word经独立隐藏Microsoft Word实例只读导出PDF，全部20页PNG检查通过。

- [Word报告](2024311281_孙驰_实验作业四.docx)
- [PDF报告](2024311281_孙驰_实验作业四.pdf)
- [人工忠实度评分清单](human_review.md)

保留本地BGE参考检索，追加61次真实DeepSeek官方API调用，累计18856 token。表7a/7b补充现场知识的检索结果和实际生成错误，表5使用九例真实DeepSeek答案及同模型AI初评。原请求/响应和结果在`../results/live_api/`，没有密钥。保留两条“设计”关系错误、无依据摘要、抽取遗漏，以及Wiki Q01的评审漏拆；未将API同模型评分当作人工确认。

**人工忠实度评分仍待确认**。提交前请阅读human_review.md九例完整答案及实际原文，填写或确认评分，并核查答案完整性。

reviewer_verification.json为原730条基线回放及92向量重编码记录；live_verification.json为480条现场结果、120汇总、123向量回放/重编码与61次请求/用量核对。report_api_review.json与当前report_content_review.json为20页报告内容复核；带_baseline后缀的文件保留旧18页报告检查记录。report_validation.json记录最终模板保留、排版和哈希。
