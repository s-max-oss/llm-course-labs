# 实验四报告

孙驰，2024311281，计算机10班。按更新课程样板填写，20页、七章、六道思考题、四图、十四表。模板原文件未修改；正式Word经独立隐藏Microsoft Word实例只读导出PDF，全部20页PNG检查通过。

- [Word报告](2024311281_孙驰_实验作业四.docx)
- [PDF报告](2024311281_孙驰_实验作业四.pdf)
- [忠实度评分确认记录](human_review.md)
- [九例确认分数](human_scores.json)

保留本地BGE参考检索，追加61次真实DeepSeek官方API调用，累计18856 token。表7a/7b补充现场知识的检索结果和实际生成错误，表5展示九例真实DeepSeek答案、同模型AI初评和本人确认分数。原请求/响应和结果在`../results/live_api/`，没有密钥。保留两条“设计”关系错误、无依据摘要、抽取遗漏，以及Wiki Q01的评审漏拆。

本人参考AI初评与理由后确认九例均为5分，评分方式为确认AI建议，未开展独立盲审。当前状态以human_scores.json为准；原API结果中的待确认字段是运行时历史状态。九例是Q01/Q06/Q11三道题分别由三范式回答的配对对照，符合“每范式抽3题”的数量要求；抽样结论只覆盖三题，忠实度与答题完整性分开记录。

reviewer_verification.json为原730条基线回放及92向量重编码记录；live_verification.json为480条现场结果、120汇总、123向量回放/重编码与61次请求/用量核对。report_api_review.json与当前report_content_review.json为20页报告内容复核；带_baseline后缀的文件保留旧18页报告检查记录。report_validation.json记录最终模板保留、排版和哈希。
