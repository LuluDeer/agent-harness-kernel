"""工具调用解析、JSON 修复、reasoning/final 分离。

这一层全部是纯函数，不持有任何状态。这些代码路径经过多轮线上验证，
修复逻辑的边界条件（畸形 JSON、流截断、模型输出格式漂移）都是在真实
流量里踩出来的，因此不轻易改动行为——新增能力请走配置开关或兜底层，
不要修改既有解析路径的语义。
"""
import re
import json
import uuid
from typing import Any, List

from .config import (
    TOOL_CALL_START_MARKERS, TOOL_CALL_END_MARKERS,
    MAX_TOOL_ROUNDS_HARD, MAX_TOOL_CALLS_PER_TURN,
)


# ===================== 边界标记查找助手 =====================
def find_tc_start(text: str, pos: int = 0) -> tuple:
    """在 text 中从 pos 起查找最早出现的任一开始符。
    返回 (位置, 标记长度)；找不到返回 (-1, 0)。
    位置最早优先；同位置时长标记优先（避免短前缀误吞）。"""
    best_pos, best_len = -1, 0
    for m in TOOL_CALL_START_MARKERS:
        i = text.find(m, pos)
        if i >= 0 and (best_pos < 0 or i < best_pos
                       or (i == best_pos and len(m) > best_len)):
            best_pos, best_len = i, len(m)
    return best_pos, best_len


def find_tc_end(text: str, pos: int = 0) -> tuple:
    """在 text 中从 pos 起查找最早出现的任一结束符。语义同 find_tc_start。"""
    best_pos, best_len = -1, 0
    for m in TOOL_CALL_END_MARKERS:
        i = text.find(m, pos)
        if i >= 0 and (best_pos < 0 or i < best_pos
                       or (i == best_pos and len(m) > best_len)):
            best_pos, best_len = i, len(m)
    return best_pos, best_len


def contains_tc_marker(text: str) -> bool:
    """text 是否含任一(开始或结束)标记。"""
    for m in TOOL_CALL_START_MARKERS + TOOL_CALL_END_MARKERS:
        if m and m in text:
            return True
    return False


def strip_trailing_end_marker(text: str) -> str:
    """若 text 以任一结束符结尾，去掉它。"""
    for m in TOOL_CALL_END_MARKERS:
        if m and text.endswith(m):
            return text[:-len(m)].rstrip()
    return text


# ===================== 基础工具 =====================
def _safe_json_obj(s: Any) -> dict:
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return {}
    t = s.strip()
    if not t:
        return {}
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    first = t.find("{")
    if first >= 0:
        depth = 0
        end = -1
        for i in range(first, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end > first:
            try:
                obj = json.loads(t[first:end + 1])
                return obj if isinstance(obj, dict) else {}
            except Exception:
                pass
    return {}


def _make_tool_call(name: str, args: dict) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def _fix_tool_calls_format(raw: str) -> str:
    """修复常见的工具调用格式错误：
    {"tool_calls": [{...}]} , {"name": ...}]}  →  {"tool_calls": [{...}, {"name": ...}]}
    """
    pattern = r'\]\}\s*,\s*(\{["\']name["\']\s*:)'
    replacement = ', \\1'
    fixed = re.sub(pattern, replacement, raw)
    try:
        json.loads(fixed)
        return fixed
    except Exception:
        pass
    if fixed.endswith("]}"):
        if len(fixed) >= 3 and fixed[-3] == "}":
            fixed = fixed[:-2] + "}"
    try:
        json.loads(fixed)
        return fixed
    except Exception:
        pass
    return fixed


def _fix_json(raw: str) -> str:
    """修复括号不平衡的 JSON；同时尝试修复 Windows 路径未转义反斜杠。"""
    try:
        json.loads(raw)
        return raw
    except Exception:
        pass

    candidate = _fix_windows_path_escapes(raw)
    if candidate != raw:
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    else:
        candidate = raw

    # 第三层:修复字符串值内未转义的 shell 双引号(如 git -C"path")
    candidate = _fix_unescaped_inner_quotes(candidate)
    if candidate != raw:
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    else:
        candidate = raw

    # 第四层:修复数组元素对象中途缺 } (如 tool_calls 第二个对象漏闭合)
    candidate = _fix_missing_mid_braces(candidate)
    if candidate != raw:
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    else:
        candidate = raw

    stack = []
    result = []
    in_string = False
    escape = False

    for ch in candidate:
        if escape:
            escape = False
            result.append(ch)
            continue
        if ch == "\\" and in_string:
            escape = True
            result.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            result.append(ch)
            continue
        if ch in ("{", "["):
            stack.append(ch)
            result.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
                result.append(ch)
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
                result.append(ch)
        else:
            result.append(ch)

    if in_string:
        result.append('"')
    for opener in reversed(stack):
        result.append("}" if opener == "{" else "]")
    return "".join(result)


def _escape_single_backslashes(s: str) -> str:
    """将字符串中"单个反斜杠"全部转义为 \\，保留已是合法双反斜杠 \\ 的部分。"""
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            if i + 1 < len(s) and s[i + 1] == "\\":
                out.append("\\\\")
                i += 2
            else:
                out.append("\\\\")
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _fix_windows_path_escapes(raw: str) -> str:
    """修复 JSON 文本中字符串字面量里因 Windows 路径导致的非法反斜杠转义。"""
    try:
        json.loads(raw)
        return raw
    except Exception:
        pass

    result = []
    i = 0
    n = len(raw)
    in_string = False
    str_buf = []

    def flush_string():
        nonlocal str_buf
        if not str_buf:
            return
        content = "".join(str_buf)
        if re.match(r"^[A-Za-z]:\\", content) or re.match(r"^\\\\", content):
            content = _escape_single_backslashes(content)
        result.append('"' + content + '"')
        str_buf = []

    while i < n:
        ch = raw[i]
        if not in_string:
            if ch == '"':
                in_string = True
                str_buf = []
                i += 1
            else:
                result.append(ch)
                i += 1
        else:
            if ch == "\\":
                if i + 1 < n and raw[i + 1] in ("\\", '"'):
                    str_buf.append(ch)
                    str_buf.append(raw[i + 1])
                    i += 2
                else:
                    str_buf.append(ch)
                    i += 1
            elif ch == '"':
                flush_string()
                in_string = False
                i += 1
            else:
                str_buf.append(ch)
                i += 1

    if in_string:
        flush_string()
    return "".join(result)


def _fix_unescaped_inner_quotes(raw: str) -> str:
    """识别 JSON 字符串值内未转义的双引号并补转义。

    典型场景:模型把 shell 命令里的 `git -C"path"` / `npm --prefix"path" run`
    直接塞进 JSON 字符串值,中间的 `"` 被 JSON 词法器当成字符串结束符,
    整段解析失败。本函数把这些"被非结构字符前后夹住的 `"`"识别为 shell
    引号而非 JSON 边界,前面补一个反斜杠;仅当修复后能被 json.loads 通过时
    才采纳,否则原样返回。

    算法:
    1. 先 json.loads 早退(合法 JSON 零开销返回)。
    2. 为每个未转义的 `"` 标注类型:start(前邻 {[,,:) / end(后邻 }],:)
       / other(被字母/路径字符前后夹住)。
    3. start...end 配对,把配对区间内的 'other' 标记为待转义。
    4. 重建字符串插入反斜杠;再过一次 json.loads 验证,失败回退原文本。
    """
    try:
        json.loads(raw)
        return raw
    except Exception:
        pass

    n = len(raw)
    # 结构性邻接字符:出现在这些字符旁边(忽略空白)的 " 视为合法 JSON 边界
    STRUCT_BEFORE = "{[,:"
    STRUCT_AFTER = "}],:"
    WS = " \t\r\n"

    # 第一遍:为每个未转义的 " 标注类型
    quotes = []  # [(index, type)]
    for i, ch in enumerate(raw):
        if ch != '"':
            continue
        # 统计前面连续反斜杠数,奇数表示已转义(如 \"),跳过
        backslashes = 0
        j = i - 1
        while j >= 0 and raw[j] == "\\":
            backslashes += 1
            j -= 1
        if backslashes % 2 == 1:
            continue
        # 前一个非空白字符
        p = i - 1
        while p >= 0 and raw[p] in WS:
            p -= 1
        prev_ch = raw[p] if p >= 0 else ""
        # 后一个非空白字符
        q = i + 1
        while q < n and raw[q] in WS:
            q += 1
        next_ch = raw[q] if q < n else ""
        if prev_ch in STRUCT_BEFORE:
            qtype = "start"
        elif next_ch in STRUCT_AFTER:
            qtype = "end"
        else:
            qtype = "other"
        quotes.append((i, qtype))

    # 第二遍:start...end 配对,将其间的 'other' 标记为待转义
    to_escape = set()
    idx = 0
    while idx < len(quotes):
        if quotes[idx][1] != "start":
            idx += 1
            continue
        end_idx = -1
        for j in range(idx + 1, len(quotes)):
            if quotes[j][1] == "end":
                end_idx = j
                break
        if end_idx < 0:
            break  # 没有配对的 end,放弃
        for j in range(idx + 1, end_idx):
            if quotes[j][1] == "other":
                to_escape.add(quotes[j][0])
        idx = end_idx + 1

    if not to_escape:
        return raw

    # 第三遍:重建字符串,在待转义位置插入反斜杠。
    # 这里不做最终 json.loads 验证:被标为 'other' 的 " 按定义绝不可能是
    # 合法 JSON 边界(合法边界必邻 {[,/}],: 之一),转义只会让 JSON 更合法、
    # 不会更差。若设验证门,会因 JSON 同时存在别处错误(如缺 ] })而拒绝
    # 本层正确的转义,导致与下游栈平衡修复无法组合(线上真实 case 就是
    # _fix_tool_calls_format 先删了 ] ,本层转义正确却被门挡回,整链崩)。
    # 合法输入已由函数入口的早退门保护,无需再在此处二次校验。
    result = []
    for i, ch in enumerate(raw):
        if i in to_escape:
            result.append("\\")
        result.append(ch)
    return "".join(result)


def _fix_missing_mid_braces(raw: str) -> str:
    """修复数组元素对象在中途缺少闭合 } 的 JSON。

    典型场景:模型在 tool_calls 数组里连续输出多个对象,某个对象的 arguments
    闭合了但对象本身的 } 漏掉,导致下一个 ,{ 被当成当前对象的成员、parser 报
    "Expecting property name"。例如:
        [{"name":"A","arguments":{...}}, {"name":"B",...}]
                          ↑这里漏了 }
    现有 _fix_json 末段的栈平衡只能补结尾缺的括号,补不了中间缺的(会产生
    歧义、栈态被污染),本层专门处理中间缺 } 的情况。

    算法:字符级状态机扫描(正确处理字符串内/转义),维护栈。在两个时机检测:
    1. 遇到 ] 关闭数组时,若栈顶是连续的 {(说明数组元素对象未闭合),在 ] 前
       补对应数量的 }。
    2. 遇到逗号且下一非空白字符是 { 时,若栈顶是 { 且其下是 [(数组元素对象
       未闭合就开新元素),在逗号前补一个 }。
    合法 JSON 早退;不设最终验证门(与下游栈平衡组合,理由同
    _fix_unescaped_inner_quotes)。
    """
    try:
        json.loads(raw)
        return raw
    except Exception:
        pass

    n = len(raw)
    WS = " \t\r\n"
    stack = []          # 栈元素: "{" 或 "["
    insertions = []     # (pos, "}") 在 pos 前插入
    in_string = False
    escape = False
    i = 0
    while i < n:
        ch = raw[i]
        if escape:
            escape = False
            i += 1
            continue
        if ch == "\\" and in_string:
            escape = True
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            i += 1
            continue
        if in_string:
            i += 1
            continue
        if ch in ("{", "["):
            stack.append(ch)
            i += 1
            continue
        if ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
            i += 1
            continue
        if ch == "]":
            # 数组关闭:栈顶连续的 { 都是未闭合的数组元素对象,补 }
            while stack and stack[-1] == "{":
                insertions.append((i, "}"))
                stack.pop()
            if stack and stack[-1] == "[":
                stack.pop()
            i += 1
            continue
        if ch == ",":
            # 逗号后若跟 { 且栈顶是 { 且其下是 [,说明数组元素对象未闭合
            j = i + 1
            while j < n and raw[j] in WS:
                j += 1
            if (j < n and raw[j] == "{"
                    and len(stack) >= 2
                    and stack[-1] == "{" and stack[-2] == "["):
                insertions.append((i, "}"))
                stack.pop()
            i += 1
            continue
        i += 1

    if not insertions:
        return raw

    # 应用插入(从后往前,避免位置偏移)
    result = []
    last = 0
    for pos, brace in insertions:
        result.append(raw[last:pos])
        result.append(brace)
        last = pos
    result.append(raw[last:])
    return "".join(result)


def _parse_tool_json(json_str: str, tool_calls_found: list):
    """解析一段工具 JSON，结果追加到 tool_calls_found。"""
    raw = json_str.strip()
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        fixed = _fix_json(raw)
        try:
            obj = json.loads(fixed)
        except Exception:
            pass
    if not isinstance(obj, dict):
        return
    for call in obj.get("tool_calls", []):
        if isinstance(call, dict) and call.get("name"):
            tool_calls_found.append(
                _make_tool_call(call["name"], call.get("arguments", {}))
            )


def _extract_code_blocks(text: str) -> list:
    """按行扫描提取 ```json 代码块。"""
    results = []
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        if not re.match(r'\s*```json\s*$', lines[i], re.IGNORECASE):
            i += 1
            continue
        i += 1
        block_lines = []
        while i < len(lines):
            if re.match(r'\s*```\s*$', lines[i]):
                i += 1
                break
            block_lines.append(lines[i])
            i += 1
        if block_lines:
            results.append("".join(block_lines).strip())
    return results


def _extract_calls_from_obj(obj: dict) -> list:
    """从 {tool_calls: [...]} 或单 {name, arguments} 提取调用列表。"""
    out = []
    calls = obj.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name", "")
            if not name:
                continue
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            out.append(_make_tool_call(name, arguments))
    else:
        name = obj.get("name", "")
        if name:
            arguments = obj.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            out.append(_make_tool_call(name, arguments))
    return out


def parse_xml_tool_calls(text: str) -> list:
    """解析模型输出中的工具调用，支持多种格式兜底。"""
    results = []

    # 主路径：␞{...}␟ 控制字符包裹的 JSON（支持备选符）
    pos = 0
    while True:
        s, s_len = find_tc_start(text, pos)
        if s < 0:
            break
        e, e_len = find_tc_end(text, s + s_len)
        if e < 0:
            # TC_END 缺失（流截断），取 TC_START 之后全部剩余文本尝试修复
            raw = text[s + s_len:].strip()
            if not raw:
                break
            raw = _fix_tool_calls_format(raw)
            fixed = _fix_json(raw)
            try:
                obj = json.loads(fixed)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                results.extend(_extract_calls_from_obj(obj))
            break
        raw = text[s + s_len:e].strip()
        pos = e + e_len
        obj = None
        try:
            obj = json.loads(raw)
        except Exception:
            raw = _fix_tool_calls_format(raw)
            fixed = _fix_json(raw)
            try:
                obj = json.loads(fixed)
            except Exception:
                pass
        if isinstance(obj, dict):
            results.extend(_extract_calls_from_obj(obj))

    if results:
        return results

    # 兜底1：```json 代码块
    for block_content in _extract_code_blocks(text):
        obj = None
        try:
            obj = json.loads(block_content)
        except Exception:
            try:
                obj = json.loads(block_content.replace("\\'", "'"))
            except Exception:
                pass
        if isinstance(obj, dict):
            results.extend(_extract_calls_from_obj(obj))
    if results:
        return results

    # 兜底2：<tool>...</tool> XML / JSON-in-XML
    for block in re.finditer(r'<tool>\s*([\s\S]*?)\s*</tool>', text, re.IGNORECASE):
        inner = block.group(1).strip()
        _parsed_json = False
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict):
                results.extend(_extract_calls_from_obj(obj))
                _parsed_json = True
        except Exception:
            pass
        if _parsed_json:
            continue
        try:
            fixed = _fix_json(inner)
            obj = json.loads(fixed)
            if isinstance(obj, dict):
                calls = obj.get("tool_calls")
                if isinstance(calls, list):
                    for call in calls:
                        if not isinstance(call, dict):
                            continue
                        name = call.get("name", "")
                        if not name:
                            continue
                        arguments = call.get("arguments", {})
                        if not isinstance(arguments, dict):
                            arguments = {}
                        results.append(_make_tool_call(name, arguments))
                    continue
        except Exception:
            pass
        # 最后尝试 XML 参数格式 <ToolName><param>val</param></ToolName>
        name_m = re.match(r'<(\w+)>([\s\S]*)</\1>', inner, re.DOTALL)
        if not name_m:
            continue
        name = name_m.group(1)
        params_str = name_m.group(2)
        args = {}
        for param in re.finditer(r'<(\w+)>([\s\S]*?)</\1>', params_str):
            args[param.group(1)] = param.group(2).strip()
        results.append(_make_tool_call(name, args))

    if results:
        return results

    # 兜底3：[tool_calls] / [TOOL_CALLS] 格式（Mistral 等）
    _TC_TAG = re.compile(r'\[tool_calls\]\s*', re.IGNORECASE)
    m = _TC_TAG.search(text)
    if m:
        rest = text[m.end():].strip()
        rest = strip_trailing_end_marker(rest)
        if rest.startswith('['):
            arr = None
            try:
                arr = json.loads(rest)
            except Exception:
                try:
                    arr = json.loads(_fix_json(rest))
                except Exception:
                    arr = None
            if isinstance(arr, list):
                for call in arr:
                    if not isinstance(call, dict):
                        continue
                    name = call.get('name', '')
                    if not name:
                        continue
                    arguments = call.get('arguments', {})
                    if not isinstance(arguments, dict):
                        try:
                            arguments = json.loads(arguments) if isinstance(arguments, str) else {}
                        except Exception:
                            arguments = {}
                    results.append(_make_tool_call(name, arguments))
                if results:
                    return results
        if rest.startswith('{'):
            obj = None
            try:
                obj = json.loads(rest)
            except Exception:
                try:
                    obj = json.loads(_fix_json(rest))
                except Exception:
                    obj = None
            if isinstance(obj, dict):
                inner_calls = obj.get('tool_calls', [])
                if isinstance(inner_calls, list):
                    for call in inner_calls:
                        if not isinstance(call, dict):
                            continue
                        name = call.get('name', '')
                        if not name:
                            continue
                        arguments = call.get('arguments', {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except Exception:
                                arguments = {}
                        if not isinstance(arguments, dict):
                            arguments = {}
                        results.append(_make_tool_call(name, arguments))
                    if results:
                        return results
    return results


# ===================== 工具调用去重 / 循环检测 =====================
def deduplicate_tool_calls(tool_calls: list) -> list:
    """同一轮内，相同 name+arguments 的调用只保留第一个。"""
    seen = set()
    result = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", "{}")
        try:
            args_normalized = json.dumps(
                json.loads(args), ensure_ascii=False, sort_keys=True
            )
        except Exception:
            args_normalized = args
        key = (name, args_normalized)
        if key not in seen:
            seen.add(key)
            result.append(tc)
    return result


def _tool_call_signature(tc: dict) -> tuple:
    fn = tc.get("function") or {}
    name = fn.get("name", "")
    args_obj = _safe_json_obj(fn.get("arguments", "{}"))
    return name, json.dumps(args_obj, ensure_ascii=False, sort_keys=True)


def _count_same_tail_rounds(messages: list, current_calls: list) -> int:
    if not current_calls:
        return 0
    cur_set = set(_tool_call_signature(tc) for tc in current_calls if isinstance(tc, dict))
    if not cur_set:
        return 0
    rounds = 0
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if not isinstance(tcs, list) or not tcs:
            continue
        s = set(_tool_call_signature(tc) for tc in tcs if isinstance(tc, dict))
        if s == cur_set:
            rounds += 1
        else:
            break
    return rounds


def _has_progress_vs_last(messages: list, current_calls: list) -> bool:
    if not current_calls:
        return False
    prev_calls = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            tcs = m.get("tool_calls") or []
            if isinstance(tcs, list) and tcs:
                prev_calls = tcs
                break
    if not prev_calls:
        return True
    prev_map = {}
    for tc in prev_calls:
        fn = tc.get("function") or {}
        prev_map[fn.get("name", "")] = _safe_json_obj(fn.get("arguments", "{}"))
    for tc in current_calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args = _safe_json_obj(fn.get("arguments", "{}"))
        if name not in prev_map:
            return True
        if json.dumps(args, sort_keys=True) != json.dumps(prev_map[name], sort_keys=True):
            return True
    return False


def should_block_tool_calls(messages: list, candidate_calls: list) -> tuple:
    rounds_hist = sum(
        1 for m in messages
        if m.get("role") == "assistant"
        and isinstance(m.get("tool_calls"), list)
        and m.get("tool_calls")
    )
    if rounds_hist >= MAX_TOOL_ROUNDS_HARD:
        return True, f"hard_round_limit:{rounds_hist}"
    if not candidate_calls:
        return False, "no_calls"
    same_tail = _count_same_tail_rounds(messages, candidate_calls)
    has_progress = _has_progress_vs_last(messages, candidate_calls)
    if same_tail >= 4 and not has_progress:
        return True, f"no_progress_repeat:{same_tail}"
    return False, f"ok(same_tail={same_tail},progress={has_progress})"


# ===================== reasoning / final 分离 =====================
def split_reasoning_and_final(text: str) -> tuple:
    if not text or not text.strip():
        return "", ""
    lines = text.splitlines()

    def is_quote_line(ln): return ln.lstrip().startswith(">")
    def is_blank(ln): return ln.strip() == ""

    types = ["B" if is_blank(ln) else "Q" if is_quote_line(ln) else "N" for ln in lines]
    end = len(lines) - 1
    while end >= 0 and types[end] == "B":
        end -= 1
    if end < 0:
        return "", ""
    start = end
    seen_normal = (types[end] == "N")
    while start - 1 >= 0 and types[start - 1] in ("N", "B"):
        start -= 1
        if types[start] == "N":
            seen_normal = True
    final_candidate = "\n".join(lines[start:end + 1]).strip() if seen_normal else ""
    reasoning_candidate = "\n".join(lines[:start]).strip()
    if any(t == "Q" for t in types[:start]) and final_candidate:
        return reasoning_candidate, final_candidate
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) >= 2:
        return "\n\n".join(blocks[:-1]).strip(), blocks[-1].strip()
    return "", text.strip()


class IncrementalReasoningSplitter:
    """流式 reasoning/final 分离：> 开头行视为 reasoning。"""
    UNKNOWN = 0
    REASONING = 1
    FINAL = 2

    def __init__(self, final_switch_chars: int = 24):
        self.state = self.UNKNOWN
        self.line_buf = ""
        self.pending_normal = ""
        self.final_switch_chars = final_switch_chars

    @staticmethod
    def _is_quote_line(line): return line.lstrip().startswith(">")

    @staticmethod
    def _is_blank_line(line): return line.strip() == ""

    @staticmethod
    def _strip_quote_prefix(line: str) -> str:
        stripped = line.lstrip()
        if stripped.startswith(">"):
            stripped = stripped[1:]
            stripped = stripped.lstrip(" \t")
        return stripped

    def feed(self, chunk: str) -> dict:
        out_r, out_f = [], []
        if not chunk:
            return {"reasoning": out_r, "final": out_f}
        self.line_buf += chunk
        while True:
            pos = self.line_buf.find("\n")
            if pos < 0:
                break
            line = self.line_buf[:pos + 1]
            self.line_buf = self.line_buf[pos + 1:]
            is_q = self._is_quote_line(line)
            is_b = self._is_blank_line(line)
            if self.state == self.UNKNOWN:
                if is_q:
                    self.state = self.REASONING
                    out_r.append(self._strip_quote_prefix(line))
                elif is_b:
                    out_r.append(self._strip_quote_prefix(line))
                else:
                    self.state = self.FINAL
                    out_f.append(line)
            elif self.state == self.REASONING:
                if is_q or is_b:
                    if self.pending_normal:
                        out_r.append(self._strip_quote_prefix(self.pending_normal))
                        self.pending_normal = ""
                    out_r.append(self._strip_quote_prefix(line))
                else:
                    self.pending_normal += line
                    if len(re.sub(r"\s+", "", self.pending_normal)) >= self.final_switch_chars:
                        self.state = self.FINAL
                        out_f.append(self.pending_normal)
                        self.pending_normal = ""
            else:
                out_f.append(line)
        return {"reasoning": out_r, "final": out_f}

    def flush(self) -> dict:
        out_r, out_f = [], []
        tail = self.line_buf
        self.line_buf = ""
        if self.pending_normal:
            out_f.append(self.pending_normal)
            self.pending_normal = ""
        if self.state in (self.REASONING, self.UNKNOWN):
            if tail:
                out_r.append(self._strip_quote_prefix(tail))
        else:
            if tail:
                out_f.append(tail)
        return {"reasoning": out_r, "final": out_f}
