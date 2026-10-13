"""
哈希链会话匹配系统 - 优化版
完全采用哈希链匹配，记录所有消息切片的哈希值
"""
import pymysql
import redis
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import asyncio

try:
    from .config import get_logger
except ImportError:  # pragma: no cover - direct script import fallback
    from config import get_logger

logger = get_logger("hash_chain")


def _env_int(name: str, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        value = int(str(os.environ.get(name, str(default))).strip())
    except (TypeError, ValueError):
        return default
    if minimum is not None and value < minimum:
        return default
    if maximum is not None and value > maximum:
        return default
    return value


# 数据库配置
MYSQL_CONFIG = {
    'host': os.environ.get('HASH_CHAIN_MYSQL_HOST') or os.environ.get('DB_HOST', '127.0.0.1'),
    'port': _env_int('HASH_CHAIN_MYSQL_PORT', _env_int('DB_PORT', 3306, 1, 65535), 1, 65535),
    'user': os.environ.get('HASH_CHAIN_MYSQL_USER') or os.environ.get('DB_USER', 'app_user'),
    'password': os.environ.get('HASH_CHAIN_MYSQL_PASSWORD') or os.environ.get('DB_PASSWORD', 'app_user'),
    'database': os.environ.get('HASH_CHAIN_MYSQL_DATABASE') or os.environ.get('DB_NAME', 'session_hash'),
    'charset': os.environ.get('HASH_CHAIN_MYSQL_CHARSET', 'utf8mb4'),
    'autocommit': True,
    'connect_timeout': _env_int('HASH_CHAIN_MYSQL_CONNECT_TIMEOUT', 5, 0),
    'read_timeout': _env_int('HASH_CHAIN_MYSQL_READ_TIMEOUT', 10, 0),
    'write_timeout': _env_int('HASH_CHAIN_MYSQL_WRITE_TIMEOUT', 10, 0),
}

REDIS_CONFIG = {
    'host': os.environ.get('HASH_CHAIN_REDIS_HOST') or os.environ.get('REDIS_HOST', '127.0.0.1'),
    'port': _env_int('HASH_CHAIN_REDIS_PORT', _env_int('REDIS_PORT', 6379, 1, 65535), 1, 65535),
    'password': os.environ.get('HASH_CHAIN_REDIS_PASSWORD') or os.environ.get('REDIS_PASSWORD') or None,
    'decode_responses': True,
    'db': _env_int('HASH_CHAIN_REDIS_DB', _env_int('REDIS_DB', 10, 0), 0),
    'socket_timeout': _env_int('HASH_CHAIN_REDIS_SOCKET_TIMEOUT', 5, 0),
    'socket_connect_timeout': _env_int('HASH_CHAIN_REDIS_CONNECT_TIMEOUT', 5, 0),
}

# 匹配置信度
MAX_ROUND_DIFF = 10  # 最大允许轮次差异（放宽以支持更多工具调用）
HASH_CHAIN_TOLERANCE = 8  # 哈希链容错轮数（放宽以支持更多工具调用）

@dataclass
class MatchResult:
    """匹配结果"""
    chatbot_id: str
    match_type: str  # 'exact', 'partial', 'branch', 'fuzzy', 'new'
    round_diff: int
    branch_id: str
    confidence: float  # 匹配置信度 0-1
    matched_slice_index: int  # 匹配到的切片索引

class HashChainManager:
    """Hash-chain session matcher.

    Connections are established lazily on first use rather than at
    construction. Importing this module (or instantiating the manager) is
    therefore side-effect free, which matters when the kernel is embedded in a
    host that may not have MySQL/Redis reachable at boot.
    """

    def __init__(self, autoconnect: bool = False):
        self.mysql_pool = None
        self.redis_client = None
        self._redis_checked = False
        if autoconnect:
            self._init_connections()

    def _init_connections(self):
        """Establish MySQL and Redis connections (best effort, never raises)."""
        if self.mysql_pool is None:
            try:
                self.mysql_pool = pymysql.connect(**MYSQL_CONFIG)
                logger.info("MySQL connection established")
            except Exception as e:
                logger.warning(f"MySQL connection failed: {e}")

        if not self._redis_checked:
            self._redis_checked = True
            try:
                self.redis_client = redis.Redis(**REDIS_CONFIG)
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                # Redis is a cache here; matching still works without it.
                self.redis_client = None
                logger.warning(f"Redis connection failed: {e}")

    def get_mysql_connection(self):
        """Return a live MySQL connection, reconnecting if the socket dropped."""
        if not self.mysql_pool or not self.mysql_pool.open:
            self.mysql_pool = pymysql.connect(**MYSQL_CONFIG)
        return self.mysql_pool
    
    def _compute_message_hash(self, messages: List[Dict], slice_index: int) -> str:
        """
        计算消息切片的哈希值
        slice_index: 切片索引，1表示[:-1]，2表示[:-2]，...，len(messages)表示[:]
        """
        if slice_index > len(messages):
            return ""
        
        # 取前 len(messages) - slice_index + 1 条消息
        # slice_index=1 -> [:] (全部消息)
        # slice_index=2 -> [:-1]
        # slice_index=3 -> [:-2]
        # ...
        if slice_index == 1:
            sliced_messages = messages[:]
        else:
            sliced_messages = messages[:-(slice_index - 1)]
        
        if not sliced_messages:
            return ""
        
        # 序列化消息用于哈希计算
        hash_input = []
        for msg in sliced_messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            # 处理不同类型的content
            if isinstance(content, list):
                content_str = ""
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        content_str += item.get('text', '')
            else:
                content_str = str(content)
            
            # 包含tool_calls信息
            tool_calls = msg.get('tool_calls')
            if tool_calls:
                tool_calls_str = json.dumps(tool_calls, sort_keys=True, ensure_ascii=False)
                hash_input.append(f"{role}:{content_str[:2000]}|tc:{tool_calls_str[:500]}")
            else:
                hash_input.append(f"{role}:{content_str[:2000]}")
        
        hash_str = "||".join(hash_input)
        return hashlib.sha256(hash_str.encode('utf-8')).hexdigest()
    
    def store_hash_chain(self, messages: List[Dict], client_id: str, model_id: str, 
                        client_ip: str, chatbot_id: str, branch_id: str = 'main') -> bool:
        """
        存储完整的哈希链
        为每个切片[:], [:-1], [:-2], ..., [:-n]计算哈希并存储，形成完整链条
        """
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            
            # 计算并存储每个切片的哈希（从完整消息开始，逐步减少）
            prev_hash = None
            total_rounds = len(messages)
            
            # slice_index=1 -> [:] (全部消息)
            # slice_index=2 -> [:-1]
            # slice_index=3 -> [:-2]
            # ...
            # slice_index=n -> [-(n-1):] 即只有第一条消息
            
            for slice_index in range(1, total_rounds + 1):
                hash_value = self._compute_message_hash(messages, slice_index)
                if not hash_value:
                    continue
                
                # 当前切片对应的轮次
                # slice_index=1 -> 完整消息 -> 轮次 = total_rounds
                # slice_index=2 -> 去掉最后一条 -> 轮次 = total_rounds - 1
                # ...
                current_round = total_rounds - slice_index + 1
                
                # 检查是否已存在（同一分支中不重复存储）
                cursor.execute("""
                    SELECT id FROM hash_chain 
                    WHERE hash_value = %s AND client_id = %s AND model_id = %s AND branch_id = %s
                """, (hash_value, client_id, model_id, branch_id))
                
                record_exists = cursor.fetchone()
                
                if not record_exists:
                    # 插入新的哈希链记录
                    cursor.execute("""
                        INSERT INTO hash_chain 
                        (hash_value, client_id, model_id, client_ip, message_slice_index, 
                         prev_hash, chatbot_id, round_count, branch_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (hash_value, client_id, model_id, client_ip, slice_index, 
                          prev_hash, chatbot_id, current_round, branch_id))
                
                prev_hash = hash_value
                
                # 缓存到Redis（只缓存最近的哈希，减少内存占用）
                if slice_index <= 10:  # 只缓存前10个切片
                    cache_key = f"hash:{hash_value}:{client_id}:{model_id}"
                    self.redis_client.setex(cache_key, 3600 * 6, json.dumps({
                        'chatbot_id': chatbot_id,
                        'round_count': current_round,
                        'branch_id': branch_id,
                        'prev_hash': prev_hash
                    }))
            
            # 更新chatbot信息
            cursor.execute("""
                INSERT INTO chatbot_info 
                (chatbot_id, client_id, model_id, client_ip, total_rounds, last_message_hash, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'active')
                ON DUPLICATE KEY UPDATE 
                    updated_at = CURRENT_TIMESTAMP,
                    total_rounds = %s,
                    last_message_hash = %s,
                    status = 'active'
            """, (chatbot_id, client_id, model_id, client_ip, total_rounds, prev_hash, 
                  total_rounds, prev_hash))
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.warning(f"存储哈希链失败: {e}")
            return False
    
    def find_matching_chatbot(self, messages: List[Dict], client_id: str, 
                             model_id: str, client_ip: str) -> Optional[MatchResult]:
        """
        使用完整哈希链查找匹配的chatbot
        返回匹配结果，如果没有匹配则返回None
        """
        try:
            request_rounds = len(messages)
            
            # 1. 首先尝试精确匹配（完整消息哈希）
            exact_match = self._try_exact_match(messages, client_id, model_id, client_ip, request_rounds)
            if exact_match:
                return exact_match
            
            # 2. 尝试部分匹配（遍历所有切片）
            partial_match = self._try_partial_match(messages, client_id, model_id, client_ip, request_rounds)
            if partial_match:
                return partial_match
            
            # 3. 尝试模糊匹配（基于相似度）
            fuzzy_match = self._try_fuzzy_match(messages, client_id, model_id, client_ip, request_rounds)
            if fuzzy_match:
                return fuzzy_match
            
            # 没有匹配，返回None表示需要创建新会话
            return None
            
        except Exception as e:
            logger.warning(f"查找匹配chatbot失败: {e}")
            return None
    
    def _try_exact_match(self, messages: List[Dict], client_id: str, 
                        model_id: str, client_ip: str, request_rounds: int) -> Optional[MatchResult]:
        """尝试精确匹配（完整消息哈希）"""
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            
            # 计算完整消息的哈希（slice_index=1）
            full_hash = self._compute_message_hash(messages, 1)
            
            if not full_hash:
                return None
            
            # 先从Redis缓存查找
            cache_key = f"hash:{full_hash}:{client_id}:{model_id}"
            cached = self.redis_client.get(cache_key)
            if cached:
                try:
                    cached_data = json.loads(cached)
                    chatbot_id = cached_data['chatbot_id']
                    round_count = cached_data['round_count']
                    branch_id = cached_data['branch_id']
                    
                    # 验证缓存数据是否仍然有效
                    cursor.execute("""
                        SELECT status FROM chatbot_info WHERE chatbot_id = %s
                    """, (chatbot_id,))
                    result = cursor.fetchone()
                    if result and result[0] == 'active':
                        round_diff = abs(round_count - request_rounds)
                        self._log_match(client_id, model_id, client_ip, request_rounds, 
                                       chatbot_id, 'exact', round_diff)
                        
                        cursor.close()
                        return MatchResult(
                            chatbot_id=chatbot_id,
                            match_type='exact',
                            round_diff=round_diff,
                            branch_id=branch_id,
                            confidence=1.0,
                            matched_slice_index=1
                        )
                except Exception:
                    pass
            
            # 从数据库查找
            cursor.execute("""
                SELECT hc.chatbot_id, hc.round_count, hc.branch_id, ci.status
                FROM hash_chain hc
                JOIN chatbot_info ci ON hc.chatbot_id = ci.chatbot_id
                WHERE hc.hash_value = %s AND hc.client_id = %s AND hc.model_id = %s
                AND ci.status = 'active'
                ORDER BY hc.round_count DESC
                LIMIT 1
            """, (full_hash, client_id, model_id))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                chatbot_id, round_count, branch_id, status = result
                round_diff = abs(round_count - request_rounds)
                
                self._log_match(client_id, model_id, client_ip, request_rounds, 
                               chatbot_id, 'exact', round_diff)
                
                return MatchResult(
                    chatbot_id=chatbot_id,
                    match_type='exact',
                    round_diff=round_diff,
                    branch_id=branch_id,
                    confidence=1.0,
                    matched_slice_index=1
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"精确匹配失败: {e}")
            return None
    
    def _try_partial_match(self, messages: List[Dict], client_id: str, 
                          model_id: str, client_ip: str, request_rounds: int) -> Optional[MatchResult]:
        """
        尝试部分匹配 - 遍历所有切片查找最佳匹配
        这是核心的哈希链匹配逻辑
        """
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            
            best_match = None
            best_confidence = 0
            
            # 遍历所有切片（从[:-1]开始，即slice_index=2）
            # slice_index=2 -> [:-1]
            # slice_index=3 -> [:-2]
            # ...
            # 直到只剩一条消息
            
            for slice_index in range(2, len(messages) + 1):
                hash_value = self._compute_message_hash(messages, slice_index)
                if not hash_value:
                    continue
                
                # 先从Redis缓存查找
                cache_key = f"hash:{hash_value}:{client_id}:{model_id}"
                cached = self.redis_client.get(cache_key)
                
                if cached:
                    try:
                        cached_data = json.loads(cached)
                        chatbot_id = cached_data['chatbot_id']
                        round_count = cached_data['round_count']
                        branch_id = cached_data['branch_id']
                        
                        # 验证缓存数据
                        cursor.execute("""
                            SELECT total_rounds, status FROM chatbot_info WHERE chatbot_id = %s
                        """, (chatbot_id,))
                        result = cursor.fetchone()
                        if result and result[1] == 'active':
                            total_rounds = result[0]
                            candidate = self._evaluate_partial_match(
                                chatbot_id, branch_id, round_count, total_rounds, 
                                request_rounds, slice_index, conn, cursor
                            )
                            if candidate and candidate.confidence > best_confidence:
                                best_match = candidate
                                best_confidence = candidate.confidence
                    except Exception:
                        pass
                else:
                    # 从数据库查找
                    cursor.execute("""
                        SELECT hc.chatbot_id, hc.round_count, hc.branch_id, 
                               hc.prev_hash, ci.total_rounds, ci.status
                        FROM hash_chain hc
                        JOIN chatbot_info ci ON hc.chatbot_id = ci.chatbot_id
                        WHERE hc.hash_value = %s AND hc.client_id = %s AND hc.model_id = %s
                        AND ci.status = 'active'
                        ORDER BY hc.round_count DESC
                        LIMIT 5
                    """, (hash_value, client_id, model_id))
                    
                    candidates = cursor.fetchall()
                    
                    for chatbot_id, round_count, branch_id, prev_hash, total_rounds, status in candidates:
                        if status != 'active':
                            continue
                        
                        candidate = self._evaluate_partial_match(
                            chatbot_id, branch_id, round_count, total_rounds, 
                            request_rounds, slice_index, conn, cursor
                        )
                        if candidate and candidate.confidence > best_confidence:
                            best_match = candidate
                            best_confidence = candidate.confidence
            
            cursor.close()
            
            if best_match:
                self._log_match(client_id, model_id, client_ip, request_rounds, 
                               best_match.chatbot_id, best_match.match_type, best_match.round_diff)
                return best_match
            
            return None
            
        except Exception as e:
            logger.warning(f"部分匹配失败: {e}")
            return None
    
    def _evaluate_partial_match(self, chatbot_id: str, branch_id: str, 
                               round_count: int, total_rounds: int, 
                               request_rounds: int, slice_index: int,
                               conn, cursor) -> Optional[MatchResult]:
        """
        评估部分匹配的质量
        关键逻辑：检查哈希链是否可以续接，避免错误匹配到其他会话
        """
        # 计算轮次差异
        round_diff = abs(total_rounds - request_rounds)
        
        # 检查轮次差异是否在可接受范围内
        if round_diff > MAX_ROUND_DIFF:
            return None
        
        # 检查哈希链长度是否合理
        # 如果数据库中该chatbot的轮次数远超请求的轮次数，说明不是目标会话
        chain_length_diff = total_rounds - request_rounds
        
        if chain_length_diff > HASH_CHAIN_TOLERANCE + round_diff:
            # 数据库中的会话比请求的会话长太多，可能是另一个会话
            return None
        
        # 计算置信度：匹配的切片越接近完整消息，置信度越高
        # slice_index=2 -> 去掉最后一条 -> 置信度较高
        # slice_index越大 -> 去掉越多 -> 置信度越低
        max_index = max(request_rounds, 5)
        confidence = max(0.5, 1.0 - (slice_index - 1) * 0.1)
        
        # 根据轮次差异调整置信度
        confidence -= round_diff * 0.05
        
        # 确定匹配类型
        match_type = 'partial' if round_diff == 0 else 'branch'
        
        return MatchResult(
            chatbot_id=chatbot_id,
            match_type=match_type,
            round_diff=round_diff,
            branch_id=branch_id,
            confidence=confidence,
            matched_slice_index=slice_index
        )
    
    def _try_fuzzy_match(self, messages: List[Dict], client_id: str, 
                         model_id: str, client_ip: str, request_rounds: int) -> Optional[MatchResult]:
        """尝试模糊匹配（最后手段）"""
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            
            # 获取当前消息最后几条的哈希
            recent_hashes = []
            for i in range(1, min(3, len(messages) + 1)):
                h = self._compute_message_hash(messages, i)
                if h:
                    recent_hashes.append(h)
            
            # 查找相同客户端和模型的所有活跃chatbot
            cursor.execute("""
                SELECT chatbot_id, total_rounds, last_message_hash, client_ip
                FROM chatbot_info 
                WHERE client_id = %s AND model_id = %s AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 10
            """, (client_id, model_id))
            
            candidates = cursor.fetchall()
            
            for chatbot_id, total_rounds, last_message_hash, stored_client_ip in candidates:
                # 检查IP是否匹配
                if stored_client_ip != client_ip:
                    continue
                
                # 检查轮次差异
                round_diff = abs(total_rounds - request_rounds)
                
                if round_diff > MAX_ROUND_DIFF + 2:
                    continue
                
                # 计算相似度
                current_last_hash = self._compute_message_hash(messages, 2)  # [:-1]
                similarity = self._calculate_hash_similarity(current_last_hash, last_message_hash)
                
                if similarity > 0.75:
                    self._log_match(client_id, model_id, client_ip, request_rounds, 
                                   chatbot_id, 'fuzzy', round_diff)
                    
                    cursor.close()
                    return MatchResult(
                        chatbot_id=chatbot_id,
                        match_type='fuzzy',
                        round_diff=round_diff,
                        branch_id='main',
                        confidence=similarity,
                        matched_slice_index=-1
                    )
            
            cursor.close()
            return None
            
        except Exception as e:
            logger.warning(f"模糊匹配失败: {e}")
            return None
    
    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """计算两个哈希值的相似度"""
        if not hash1 or not hash2:
            return 0.0
        
        # 比较相同字符的数量
        same_chars = sum(1 for a, b in zip(hash1, hash2) if a == b)
        return same_chars / max(len(hash1), len(hash2))
    
    def create_branch(self, parent_chatbot_id: str, branch_point_hash: str, 
                     branch_point_round: int, messages: List[Dict], 
                     client_id: str, model_id: str, client_ip: str) -> str:
        """
        创建分支链
        返回新的分支ID
        """
        try:
            import uuid
            branch_id = f"branch_{uuid.uuid4().hex[:8]}"
            
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            
            # 插入分支信息
            cursor.execute("""
                INSERT INTO branch_info 
                (branch_id, chatbot_id, parent_branch_id, branch_point_hash, branch_point_round)
                VALUES (%s, %s, 'main', %s, %s)
            """, (branch_id, parent_chatbot_id, branch_point_hash, branch_point_round))
            
            # 存储分支的哈希链
            self.store_hash_chain(messages, client_id, model_id, client_ip, 
                                 parent_chatbot_id, branch_id)
            
            cursor.close()
            return branch_id
            
        except Exception as e:
            logger.warning(f"创建分支失败: {e}")
            return 'main'
    
    def _log_match(self, client_id: str, model_id: str, client_ip: str, 
                  request_rounds: int, chatbot_id: str, match_type: str, round_diff: int):
        """记录匹配日志"""
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO match_log 
                (client_id, model_id, client_ip, request_rounds, matched_chatbot_id, match_type, round_diff)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (client_id, model_id, client_ip, request_rounds, chatbot_id, match_type, round_diff))
            
            cursor.close()
            
        except Exception as e:
            logger.warning(f"记录匹配日志失败: {e}")
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """清理过期会话"""
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor()
            
            # 更新过期的chatbot状态
            cursor.execute("""
                UPDATE chatbot_info 
                SET status = 'expired' 
                WHERE updated_at < DATE_SUB(NOW(), INTERVAL %s HOUR)
                AND status = 'active'
            """, (max_age_hours,))
            
            affected_rows = cursor.rowcount
            cursor.close()
            
            logger.info(f"清理了 {affected_rows} 个过期会话")
            
        except Exception as e:
            logger.warning(f"清理过期会话失败: {e}")
    
    def get_chatbot_stats(self, chatbot_id: str) -> Dict:
        """获取chatbot统计信息"""
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            cursor.execute("""
                SELECT * FROM chatbot_info WHERE chatbot_id = %s
            """, (chatbot_id,))
            
            chatbot_info = cursor.fetchone()
            
            # 获取哈希链信息
            cursor.execute("""
                SELECT branch_id, COUNT(*) as hash_count, MAX(round_count) as max_round
                FROM hash_chain 
                WHERE chatbot_id = %s
                GROUP BY branch_id
            """, (chatbot_id,))
            
            chain_stats = cursor.fetchall()
            cursor.close()
            
            return {
                'chatbot_info': chatbot_info,
                'chain_stats': chain_stats
            }
            
        except Exception as e:
            logger.warning(f"获取chatbot统计失败: {e}")
            return {}

# 全局实例
hash_chain_manager = HashChainManager()