-- Schema for harness_kernel.hash_chain
--
-- Three tables:
--   hash_chain   : one row per (context slice, session) — the chain itself
--   chatbot_info : session metadata / liveness
--   branch_info  : forks created via create_branch()
--   match_log    : which degradation tier fired, for observability

-- ===================== the chain =====================
CREATE TABLE IF NOT EXISTS hash_chain (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    hash_value VARCHAR(64) NOT NULL COMMENT 'sha256 of the sliced context',
    client_id VARCHAR(128) NOT NULL COMMENT 'caller identity',
    model_id VARCHAR(64) NOT NULL COMMENT 'model identity',
    client_ip VARCHAR(64) NOT NULL,
    message_slice_index INT NOT NULL COMMENT '1 = full context, 2 = [:-1], ...',
    prev_hash VARCHAR(64) DEFAULT NULL COMMENT 'points at the previous slice',
    chatbot_id VARCHAR(128) NOT NULL COMMENT 'upstream session id',
    round_count INT NOT NULL DEFAULT 0 COMMENT 'conversation round for this slice',
    branch_id VARCHAR(64) DEFAULT 'main' COMMENT 'main or a fork id',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- A given context slice identifies at most one session per caller+model+branch.
    UNIQUE KEY uk_hash_client_model (hash_value, client_id, model_id, branch_id),
    KEY idx_client_model (client_id, model_id),
    KEY idx_chatbot (chatbot_id),
    KEY idx_prev_hash (prev_hash),
    KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='context-pruning hash chain';

-- ===================== session metadata =====================
CREATE TABLE IF NOT EXISTS chatbot_info (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    chatbot_id VARCHAR(128) NOT NULL UNIQUE COMMENT 'upstream session id',
    client_id VARCHAR(128) NOT NULL,
    model_id VARCHAR(64) NOT NULL,
    client_ip VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    status ENUM('active', 'inactive', 'expired') DEFAULT 'active',
    total_rounds INT NOT NULL DEFAULT 0,
    last_message_hash VARCHAR(64) DEFAULT NULL,
    KEY idx_client_model (client_id, model_id),
    KEY idx_status (status),
    KEY idx_updated_at (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='session metadata';

-- ===================== forks =====================
CREATE TABLE IF NOT EXISTS branch_info (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    branch_id VARCHAR(64) NOT NULL UNIQUE,
    chatbot_id VARCHAR(128) NOT NULL,
    parent_branch_id VARCHAR(64) DEFAULT NULL,
    branch_point_hash VARCHAR(64) NOT NULL COMMENT 'hash where the fork happened',
    branch_point_round INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_rounds INT NOT NULL DEFAULT 0,
    KEY idx_chatbot (chatbot_id),
    KEY idx_parent_branch (parent_branch_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='conversation forks';

-- ===================== observability =====================
-- 'partial' is produced when round_diff == 0 resolves via chain walking;
-- omitting it from the ENUM is a classic silent-failure source, so it is
-- listed explicitly here.
CREATE TABLE IF NOT EXISTS match_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    client_id VARCHAR(128) NOT NULL,
    model_id VARCHAR(64) NOT NULL,
    client_ip VARCHAR(64) NOT NULL,
    request_rounds INT NOT NULL,
    matched_chatbot_id VARCHAR(128) DEFAULT NULL,
    match_type ENUM('exact', 'partial', 'branch', 'fuzzy', 'new') NOT NULL,
    hash_chain_diff INT DEFAULT NULL,
    round_diff INT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    KEY idx_client_model (client_id, model_id),
    KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='which tier fired, per lookup';

-- ===================== index tuning =====================
-- Redundant if the table was just created (covered by the UNIQUE KEY), but
-- kept for upgrading an existing deployment.
CREATE INDEX idx_hash_composite ON hash_chain(hash_value, client_id, model_id, chatbot_id);
CREATE INDEX idx_round_count ON hash_chain(round_count);
CREATE INDEX idx_branch_chatbot ON hash_chain(branch_id, chatbot_id);
