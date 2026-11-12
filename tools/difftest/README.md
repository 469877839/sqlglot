# SQLGlot 优化器差分测试（SQLite 参考实现）

用随机生成的小型 SQLite `SELECT`，把**优化前后**两条 SQL 放到同一个内存库
（`sqlite3.connect(":memory:")`）里执行并比对结果。重点覆盖 NULL、重复行、
外连接与聚合的组合。

## 快速开始

```bash
# 默认套件：500 个种子 x 2 个数据画像 = 1000 个用例，约 3 秒
python -m tools.difftest run

# 指定规则集 / 种子范围 / 画像
python -m tools.difftest run --rules no-simplify --profile nullable \
    --seed-start 1000 --seeds 200 --shrink-budget 200

# 证明框架能抓错并缩减（约 0.5 秒），注入的错误规则见 rules.py
python -m tools.difftest selftest

# 重放反例文件（重现则退出码 1）
python -m tools.difftest replay counterexample-small-5.json
```

退出码：`0` 无差异 / 重放不再复现；`1` 发现 mismatch 或重放复现；`2` 用法错误。

## 可重放性

- **种子**：每个用例只由一个 `random.Random(seed)` 驱动，schema、数据、查询
  全部确定。`--seed-start/--seeds` 可精确重放任意区间。
- **规则集合**：`--rules` 选命名集合（`default`、`no-simplify`、`qualify-only`、
  `simplify-only` 以及三个 `bug-*` 集合）；反例文件里记录的是**规则名列表**，
  `replay` 经 `rules.py` 的注册表按名解析。
- **缩减预算**：`--shrink-budget N` 限制每个反例最多多少次“候选执行”评估，
  预算耗尽立即停止并保留当前最小结果；文件里记录实际花费和预算。
- 不下载任何语料，不依赖网络；3 张表、每列 5-7 个固定取值。

## 结果分类（严格区分“优化错误”与“无法比较”）

- `ok`：两边都能执行且结果一致。
- `mismatch`：两边都能执行、结果不同 —— 唯一算作优化错误的类别。
- `skipped`：解析 / 优化 / 任一侧执行报错，或命中已知 SQLite 方言分歧
  （如 GROUP BY 下 ORDER BY 非聚合列）。**永不计为优化错误**，原因会单列统计。

## 比较语义

- 无顶层 `ORDER BY`：按多重集合（`collections.Counter`）比较，忽略行序但
  保留重复行数；浮点与整数做数值归一化，`NaN`/`bytes`/`NULL` 有规范化处理
  （见 `compare.py`）。
- 有 `ORDER BY`：按序列逐元素比较，保留顺序语义。
- 生成器在分组查询里只产出聚合表达式或 GROUP BY 键，避免 SQLite “裸列任意取值”
  的非确定性污染结果。

## 缩减器（`shrink.py`）

失败后自动保留仍有差异的更小例子，预算内循环执行：

1. **行级 ddmin**：对每条 `INSERT ... VALUES (...), (...)` 的行做 delta debugging；
2. **语句级 ddmin**：整条 `INSERT` 的子集缩减；
3. **查询缩减**：AST 级移除 `DISTINCT / WHERE / GROUP BY / HAVING / ORDER BY /
   LIMIT`、多余 JOIN、多余 SELECT 项（删 GROUP BY 时若有裸列则跳过，防止非确定）。

实测三种注入错误都能在 30 次评估内缩到 1-2 行数据、单表查询。

## 注入的错误规则（仅用于自测，勿用于真实管线）

| 规则 | 错误语义 | 主要命中区域 |
| --- | --- | --- |
| `drop_outer_join_BUG` | 外连接改内连接，丢掉 NULL 补位行 | 外连接 / NULL |
| `drop_distinct_BUG` | 删除 DISTINCT，重新引入重复行 | 重复行 |
| `invert_is_null_BUG` | `IS NULL` 与 `IS NOT NULL` 互翻 | NULL 谓词 / 聚合 |

正常运行使用 `default`（即 `sqlglot.optimizer.optimizer.RULES` 全部 14 条规则），
不包含任何 bug 规则。

## 反例文件格式（JSON，见 `report.py`）

单对象 JSON，字段：`version, seed, profile, rules(规则名列表), schema, ddl,
inserts, query, optimized_sql, ordered, original_rows, optimized_rows,
shrink{evaluations, budget}`。可直接 `replay`，也可手工建库复现。

## 覆盖与盲区

默认 1000 例中的特征占比（两画像合计）：GROUP BY 46%、聚合 43%、IS [NOT] NULL
36%、ORDER BY 33%、DISTINCT 28%、左/右/全外连接 28%、HAVING 22%、OR/NOT 22%、
COALESCE 15%、IN 10%、BETWEEN 9%、CROSS JOIN 8%。

已知盲区（按设计取舍，扩展点在 `gen.py`）：

- **无相关子查询 / CTE / 窗口函数 / 集合运算（UNION 等）/ 自连接**。
- 无标量子查询、`CASE WHEN`、字符串/日期函数和类型转换；仅 INT/TEXT 两类型。
- 数据为固定小画像（含 NULL 与重复行），不含随机值，因此值域相关的边界问题
  （溢出、排序 collation、浮点误差）覆盖不到。
- 只对照 SQLite：与其他引擎语义不同的构造会在 `skipped` 单列，不算错误。
- 不比较 SQL 文本/执行计划，只比较结果集；性能类回归不可见。
- 缩减器是贪心 ddmin 而非穷尽最小化，预算内给出的是“已验证可复现的更小例子”，
  不保证全局最小。
