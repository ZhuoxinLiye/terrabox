# N070 图谱查询接口

访问日：2026-11-01（北京时间）。实现只读已有标注 JSON，不启动原 Graph-Agent 服务、不导入文档检索或模型。工具由内置 `setup(registrar)` 自动发现，无共享注册文件改动，无新增依赖。

服务配置 `EARTHMIND_GRAPH_SOURCE` 指定原始 JSON；当前本机默认定位同级 `graph-agent/data/nnss_annotations_5.21.1_json_aligned_to_csv.json`。源文件仅读取，缓存按路径、mtime_ns、size刷新；每份实际读取内容计算 SHA-256。调用方不能传文件路径。部署其他位置时显式设置环境变量。

| 工具 | 参数 | 结果口径 |
| --- | --- | --- |
| `graph_query.get_item` | 必需 `item_id` | ID取详情、原标注字段、父链；area/facility带成员（首80条）与完整member_count；target是叶子 |
| `graph_query.query_nodes` | 可选 `partition_id`、`facility_id`、`node_type`、`target_type_contains`、`risk_level`、`limit` | count为过滤后完整记录数；nodes最多limit条，另给returned_count、truncated和完整类别分布 |
| `graph_query.search_graph` | 必需 `query`，可选 `source_types`、`limit` | 大小写不敏感的字段子串查找，返回候选ID、matched_fields、来源类型；count是返回候选数，total_matches为完整关键词匹配数，不当作目标总体计数 |

三工具均可传 `dataset_id` 与 `snapshot_id`，与上次响应provenance相匹配才继续，防止跨来源/版本混用；建议后续调用携带snapshot_id。输出含完整hash、源版本、只读标识和synthetic标识。原JSON没有datasetId时采用内容hash生成稳定ID。源文件改变后必须重新核对调用证据；缓存不是可写图数据库。

`node_type`只支持facility/target。`target_type_contains`沿用已确认接口字段名，但本适配器只接受完整类别码，不做自由子串筛选；允许类别由数据生成并在响应列出。类别由原store的标签规则推导，来源标明`original_graph_store_label_rules_v1`。`risk_level`只允许unknown，标注文件没有真实风险评估。未移植semantic_category、sibling_of、新闻、文档和时相工具；传入未支持参数明确失败，不静默忽略。搜索排序是原记录顺序，不声称保留原服务的评分排名或分词语义；派生边ID也不是原服务E序号。

错误以`success=false, ok=false, execution_blocked=true, reason_code, error`返回。非法/错误层级scope、跨区scope冲突、无效ID、未知类别、错误limit、来源错配均不返回count/nodes。合法范围无匹配则`success=true,count=0,nodes=[]`。ID字段不会把名称静默取首个；重名给候选，唯一名称也要求使用ID。设施归属按同一区域与名称联合解析，跨区同名设施互不混入；同区域同名父设施、同名区域和未解析设施父区域都阻断不确定范围。target详情会显示未解析父链警告。

计数单位是已有标注记录，不是影像检测出的物体数量；车辆区域不是一辆车。边只从既有层级关系派生，明确`derived=true`。工具输出不支持洪水成因、新淹没或现场条件主张。原标注含私有名称/几何，接口在已有本地认证范围内提供详情；提交的研究证据只保留必要ID、计数、字段与hash，原响应放本机临时目录。

CPU验证：

```bash
cd /data2/liyezhuoxin/terrabox-water-demo
/data2/liyezhuoxin/envs/terrabox-water-demo/bin/python -m pytest tests/test_graph_query.py -q
PYTHONPATH=src /data2/liyezhuoxin/envs/terrabox-water-demo/bin/python /data2/liyezhuoxin/Earthmind/research/n070-graph-tools/verify_direct.py
```

共享服务重启由主agent统一执行；重启Terrabox后刷新Terraflow目录并核查三个动态组件。原生Agent任务与TerraLink联合验证分别保留证据，不能用以上CPU测试代替。

## 2026-11-06 层级与业务类别说明

`node_type=target` 表示目标层级，包含该层级下的所有业务类别；`target_type_contains=target` 只表示其中一个业务类别。任务要求“设施内全部目标记录”时，只传设施ID与 `node_type=target`，不得自行追加业务类别或风险条件。`limit` 只限制返回列表，不改变完整 `count`。本次仅补充工具/schema说明及该历史失败的合成回归，不改变查询算法或原始图谱。
