# Language Model Systems Atlas · 语言模型系统图谱

一个蓝白配色、中文为主的 CS224N / CS336 复习网站。内容围绕 NLP 表示、神经网络与反向传播、RNN 语言模型、Transformer、训练系统和评估展开。

## 本地预览公开站点

需要 Node.js 22+：

~~~powershell
npm ci
npx quartz build --serve
~~~

公式由 MathJax 在构建时生成 SVG；浏览器不需联网加载数学引擎。站点示意图是本地原创 SVG。

## 私人离线资料

维护者的个人离线副本保存在仓库外部的本机资料目录，不属于公开网站源码。仓库内预留的 `quartz/static/private-materials` 和 `content/98-Local-Study-Cache` 路径由 Git 忽略，避免误收录私人文件。第三方 CS224N 翻译笔记和 Stanford 课件仅用于维护者个人学习，不会进入 GitHub Pages。

## GitHub Pages

本站通过 GitHub Actions 构建 Quartz `public/` 并发布 Pages。Workflow 会检查私有资料目录没有进入发布树；Pages 来源设为 GitHub Actions。

项目仓库 Language-Model-Systems-Atlas 的目标地址配置为：

https://blog.linhk.top/Language-Model-Systems-Atlas/

GitHub Pages 的 `lin-hongkuan.github.io` 项目地址会重定向到上述账户域名。

更改仓库名或绑定自定义域名后，需要同步更新 quartz.config.yaml 的 configuration.baseUrl。

## 内容范围

- CS224N：L01-L14 逐讲中文讲解与练习，链接 Stanford 官方课件；L16、L19 等拓展材料保留官方入口。
- CS336：A1-A5 五篇中文概念导读与官方课程地图，覆盖 L01-L17。
- Hands-on LLM：收录 FRS2003 公开仓库的 MIT 授权中文讲义、Word、实验图和记录，固定到 2026-09-16 的提交；特殊数据和模型卡的许可例外见镜像说明页。
- Python / PyTorch：张量、自动微分、shape、训练一步与调试索引。
- 公式与图示：MathJax 构建生成的静态 SVG 和本地图示；图片带中文替代文字和说明。

本站原创讲解不是 Stanford 官方材料或翻译。作业需遵守原课程的独立完成与 AI 使用规范。Quartz 项目代码遵循根目录 LICENSE.txt；课程和引用资料仍按其自身来源与许可处理。
