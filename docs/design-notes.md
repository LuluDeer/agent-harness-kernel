# 在没有 function calling 的上游上重建 tool-calling

> 本文讲一个具体的工程问题：当你拿到的大模型接口**没有 `tools` 参数、没有
> `tool_calls` 返回、也没有你能控制的上下文句柄**时，如何在其上重建一套完整、
> 可用的 Agent 运行时。
>
> 配套代码：[agent-harness-kernel](https://github.com/LuluDeer/agent-harness-kernel)

---

## 一、问题的形状

主流 Agent 框架都建立在一个隐含契约上：

```
messages[]  ──▶  model  ──▶  tool_calls[]  ──▶  执行  ──▶  结果回填  ──▶  循环
```

这个契约有两个前提：**模型接受 `tools` 声明**，以及**上下文由我掌控**。

但有一类接口两个前提都不成立：

- 它只给你一个**网页对话能力**——你发一段文本，它回一段文本；
- 会话由上游管理，你只拿到一个不透明的 **session id**；
- 你**无法传入历史消息**，也就无法自己维护上下文。

此时框架全线失效：模型回给你的是一段**描述它想做什么的自然语言**，没有任何结构
化字段可供分发。Agent 循环在第一轮就断了。

要补上这一层，需要解决三个独立的问题：

| 问题 | 本质 |
|---|---|
| 工具调用怎么表达出来？ | 协议设计 + 反向解析 |
| 会话怎么接回去？ | 无状态请求 → 有状态会话的映射 |
| 模型原地打转怎么办？ | 不收敛检测 |

下面逐一展开。这三个问题在代码里对应 `parsing.py`（913 行）与
`hash_chain.py`（653 行）两个模块。

---

## 二、工具协议：把 schema 降维成文本，再从文本还原结构

### 2.1 为什么用控制字符

做法本身很直接：把 OpenAI 的 `tools` schema 序列化成文本注入 prompt，模型按约定
格式输出，我们再把它解析回结构化的 `tool_calls`。

```
␞{"tool_calls": [{"name": "read_file", "arguments": {"path": "a.txt"}}]}␟
```

关键在边界标记的选择。`␞` 是 U+241E（Record Separator），`␟` 是 U+241F（Unit
Separator）。

不用 ` ```json ` 或 `<tool>`，原因是**这些标记模型本来就会正常使用**：它可能正在
写一篇讲 JSON 的文章，或输出一段含代码块的正常回答。用它们做边界，解析器就必须
区分"模型真的想调用工具"和"模型在举例"。控制字符几乎不可能被自然生成，这个歧义
直接消失。

### 2.2 三级降级解析

即便协议注入到位，模型输出仍会漂移——尤其对话变长之后。所以解析是主路径 + 两层
兜底：

```python
def parse_xml_tool_calls(text: str) -> list:
    # 主路径：␞{...}␟
    while True:
        s, s_len = find_tc_start(text, pos)
        ...
    if results:
        return results

    # 兜底1：```json 代码块
    # 兜底2：<tool>...</tool>
```

一个容易被忽略的细节是**流截断恢复**：当上游在流式输出中途断掉，结束标记还没
到达。此时不能丢弃整轮——那会让 Agent 静默失败。正确做法是把起始标记之后的
全部剩余文本拿去做 JSON 修复：

```python
e, e_len = find_tc_end(text, s + s_len)
if e < 0:
    # TC_END 缺失（流截断），取 TC_START 之后全部剩余文本尝试修复
    raw = text[s + s_len:].strip()
```

### 2.3 JSON 修复：为什么必须做

模型的 JSON 输出**比预期更常是非法**的。最典型的是 Windows 路径：

```json
{"path": "C:\Users\test\file.txt"}
```

`\U` 不是合法 JSON 转义，`json.loads` 直接抛异常。如果就此丢弃，一次工具调用就
没了，而 Agent 会继续基于缺失的信息往下推理——**错误的沉默传播，比报错危险得多**。

修复层分四条独立路径，每条对应一种实测到的畸形：

| 函数 | 目标失效 |
|---|---|
| `_fix_windows_path_escapes` | `\U` 等非法转义 |
| `_fix_unescaped_inner_quotes` | 字符串值内的裸引号 |
| `_fix_missing_mid_braces` | 对象字面量被截断 |
| `_escape_single_backslashes` | 路径中的孤立反斜杠 |

顺序有讲究：先补转义、再修引号、最后补括号。反过来做会在半成品结构上误判。

---

## 三、会话映射：上下文裁枝哈希

### 3.1 约束

上游不让你传上下文。它给你一个 session id，然后期望你"自然地"接回去。但你的调用
方是无状态 HTTP，甚至可能落到不同的 worker 进程。

**你只能从请求体本身，反推它属于哪个会话。**

### 3.2 算法

对消息列表做**逐级裁枝哈希**：先哈希完整上下文，再去掉最后一条哈希，再去掉最后
两条……

```
slice_index=1  ->  messages[:]      ->  round N
slice_index=2  ->  messages[:-1]    ->  round N-1
slice_index=3  ->  messages[:-2]    ->  round N-2
```

这产生一个关键不变量：

```python
round_n  = _msgs(4)
round_n1 = round_n + [tail...]

assert (_compute_message_hash(round_n,  1)     # 完整上下文
        == _compute_message_hash(round_n1, 3)) # 裁掉最后两条
```

**后一轮的裁枝切片，恰好等于前一轮的完整哈希。** 这正是能够沿链回溯的原因——链条
把每一轮串了起来，回溯就是在链上走。

两个实现细节值得单独说：

**`tool_calls` 必须参与哈希。** 否则两个仅工具调用不同的轮次会算出同一个摘要，
会话身份直接坍缩。

**内容需要截断参与哈希。** 单条消息取前 2000 字符、`tool_calls` 取前 500 字符。
超长内容（例如一次文件读取返回几万字符）不应该让整条链的哈希变得不稳定。

### 3.3 四级降级匹配

精确匹配在真实流量里的命中率**远低于直觉**：模型丢掉一个前缀、系统消息被注入、
输出格式随上下文增长而漂移——任何一种都会让完整哈希对不上。

所以匹配不"失败"，而是**降级**：

| 层级 | 策略 | 置信度 |
|---|---|---|
| `exact` | 完整上下文哈希命中 | 1.0 |
| `partial` | 沿链回溯到某个更早的匹配轮次 | 高，按 `round_diff` 衰减 |
| `branch` | 经 `create_branch()` 解析分叉会话 | 继承 |
| `fuzzy` | 字符级相似度 | `< HASH_CHAIN_TOLERANCE` |

```python
def find_matching_chatbot(...):
    if (m := self._try_exact_match(...)):   return m
    if (m := self._try_partial_match(...)): return m
    if (m := self._try_fuzzy_match(...)):   return m
    return None   # 全部未命中才新建会话
```

`MatchResult` 记录**是哪一层命中的**，所以降级是**可观测的**，不是静默的。这一点
在排查"为什么这个用户老是新建会话"时是决定性的。

### 3.4 缓存是可选依赖

Redis 缓存热点切片（前 10 个，6 小时 TTL），MySQL 存持久链。但**Redis 失败时
匹配仍可用**——降级为纯 DB 查询，而不是整体不可用：

```python
except Exception as e:
    # Redis 在这里只是缓存；没有它匹配依然成立。
    self.redis_client = None
    logger.warning(f"Redis connection failed: {e}")
```

一个把缓存当硬依赖的系统，会在缓存抖动时放大成故障。

---

## 四、循环检测：区分"重试"和"卡死"

Agent 的默认失效模式是**不收敛**。三重刹车：

```python
MAX_TOOL_ROUNDS_HARD    = 500   # 失控上限
MAX_TOOL_CALLS_PER_TURN = 8     # 单轮扇出上限
deduplicate_tool_calls()        # 折叠完全相同的 (name, arguments)
```

真正有意思的是第三层意义上的**原地打转检测**：

```python
same_tail    = _count_same_tail_rounds(messages, candidate_calls)
has_progress = _has_progress_vs_last(messages, candidate_calls)
if same_tail >= 4 and not has_progress:
    return True, f"no_progress_repeat:{same_tail}"
```

请注意那个 `and`。

**重复本身是合法的**——轮询任务状态、等待外部资源、失败重试，都会产生看起来一模
一样的连续调用。**重复且无进展**才是 bug。把这两者分开，是这个守卫不会在正常工作
上误伤的唯一原因。

`_has_progress_vs_last` 的判据是：本轮调用的参数集合，是否提供了上一轮没有的新
信息。参数在推进（比如 `page=1,2,3`）就放行；参数完全一样、结果也一样，才判定
为卡死。

顺带一提，`MAX_TOOL_ROUNDS_HARD = 500` 这个值**故意定得很高**。它是失控保护，不是
产品限制——正常情况下终止循环的是进展检测，不是这个硬上限。

---

## 五、三条设计原则

这个模块里，有几条决定是反复权衡后固定下来的。

### 5.1 导入期零副作用

```python
class HashChainManager:
    def __init__(self, autoconnect: bool = False):
        self.mysql_pool = None
        self.redis_client = None
        ...
```

原始实现的 `__init__` 直接建连接。这让模块**在任何没有实时凭据的环境里都无法
import**——包括单元测试、CI、以及任何只想用解析那一半的宿主。

改为惰性连接后，`parsing` 那一半的依赖是**零第三方库**（只用 `re`/`json`/
`uuid`/`typing`），可以独立采用。

### 5.2 失败要软化，不要抛出

```python
def store_hash_chain(...) -> bool:
    except Exception as e:
        logger.warning(f"存储哈希链失败: {e}")
        return False
```

存储失败返回 `False`，`cleanup_expired_sessions` 记录后继续。会话匹配基础设施
**应该让 Agent 降级，而不是让请求链断掉**。把可观测性做好，把中断性错误留给真正
不可恢复的场景。

### 5.3 已线上验证的解析路径不轻易改语义

修复逻辑的边界条件是从真实流量里踩出来的。新增能力应该走配置开关或**追加兜底
层**，而不是修改既有路径的语义——否则一个"小优化"可能悄悄破坏一类畸形输入的
处理，而这类输入可能几周才出现一次。

---

## 六、已知限制

诚实地列出来：

- **会话状态在内存里。** 进程内 store 要求单 worker 部署。哈希链让多 worker
  *成为可能*，但不等于自动成立。
- **`pymysql` / `redis` 是同步客户端。** 高并发下会阻塞事件循环。要扩容，先换
  异步客户端或连接池。
- **`fuzzy` 是字符级相似度，不是语义相似度。** 它是最后手段，置信度应当当作弱
  信号看待。
- **`MAX_TOOL_ROUNDS_HARD = 500` 偏高。** 见上文，这是刻意的。

---

## 七、小结

在一个能力残缺的上游之上重建 Agent 运行时，难点不在"调通接口"，而在三件容易被
低估的事：

1. **协议边界的选择**——决定了解析器是否能区分"真的调用"和"在举例"；
2. **失败必须降级而不是中断**——解析失败、缓存失败、匹配失败，各有各的退路；
3. **区分重试与卡死**——这是唯一能既管住死循环、又不误伤正常工作的方法。

这三件事都有配套测试：`pytest -q` 共 39 个用例，不需要任何外部服务。
