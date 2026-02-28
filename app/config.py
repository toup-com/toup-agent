from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # App
    app_name: str = "HexBrain Memory System"
    debug: bool = True
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./hexbrain.db"
    
    # For production PostgreSQL with pgvector:
    # database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/hexbrain"
    
    # Property alias for Alembic compatibility
    @property
    def DATABASE_URL(self) -> str:
        return self.database_url
    
    # Embedding
    embedding_provider: str = "openai"  # "openai" or "local"
    embedding_model: str = "text-embedding-3-small"  # OpenAI model or local model name
    embedding_dimension: int = 1536  # 1536 for text-embedding-3-small, 384 for all-MiniLM-L6-v2
    openai_api_key: Optional[str] = None  # Set via OPENAI_API_KEY env var
    
    # LLM Settings (for chat)
    default_model: str = "gpt-4o-mini"  # Used for memory extraction only (via OpenAI)
    fallback_model: str = "gpt-4o-mini"  # Fallback if main model fails
    max_tokens: int = 4096  # Max response tokens
    temperature: float = 0.7  # Response creativity
    
    # Auth
    jwt_secret: str = "hexbrain-dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 1 week
    
    # API
    api_prefix: str = "/api"
    cors_origins: list[str] = [
        "http://localhost:5173", "http://localhost:3000", 
        "http://localhost:80", "http://localhost",
        "https://toup.ai", "https://brain.toup.ai", "https://brain-ui.toup.ai",
        "https://hex-api.toup.ai",
    ]
    
    # Memory extraction
    max_memories_per_message: int = 10
    similarity_threshold: float = 0.7
    
    # Chat & Session Settings
    memory_recall_limit: int = 15  # How many memories to recall per message
    auto_extract_memories: bool = True  # Auto-extract memories from conversations
    max_history_messages: int = 20  # Max conversation history to include
    
    # Scheduler (for memory decay/consolidation)
    enable_scheduler: bool = True  # Set to False in multi-worker deployments
    decay_interval_hours: int = 6  # How often to run decay
    consolidation_cron_hour: int = 3  # Hour to run consolidation (3 AM)
    
    # Telegram Bot
    telegram_bot_token: Optional[str] = None  # Set via TELEGRAM_BOT_TOKEN env var
    telegram_allowed_user_ids: list[int] = []  # Restrict to specific Telegram user IDs
    telegram_polling_mode: bool = True  # True=polling, False=webhook
    telegram_user_map: dict[str, str] = {}  # Map Telegram user ID → HexBrain user ID
    
    # Anthropic Claude (kept for future use)
    anthropic_api_key: Optional[str] = None  # Set via ANTHROPIC_API_KEY env var
    anthropic_model: str = "claude-opus-4-6"  # Default Claude model
    anthropic_max_tokens: int = 16000
    
    # LLM Provider Keys (set by platform setup wizard)
    llm_mode: str = "manual"  # "manual" or "bundle"
    google_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None

    # Agent Runtime
    agent_model: str = "gpt-5.2"  # Primary agent model
    agent_fallback_model: str = "gpt-4o"  # Fallback if primary model fails
    agent_max_tokens: int = 16000  # Max output tokens for agent
    agent_max_tool_iterations: int = 20  # Max tool call loops before forcing stop
    agent_workspace_dir: str = "/app/workspace"  # Working directory for file operations
    brave_api_key: Optional[str] = None  # For web search
    skills_dir: str = "/app/skills"  # External skills directory

    # Workspace Bootstrap
    workspace_per_user: bool = True  # Create per-user workspace subdirectories
    workspace_create_readme: bool = True  # Create README.md in new workspaces

    # Message Queuing & Debounce (Telegram)
    telegram_debounce_ms: int = 1500  # Debounce delay in ms for rapid messages
    telegram_max_queue: int = 5  # Max messages to queue before forcing flush

    # Heartbeat / Proactive Agent
    heartbeat_enabled: bool = False  # Enable proactive agent heartbeat
    heartbeat_interval_hours: int = 6  # How often to run heartbeat
    heartbeat_prompt: str = (
        "You are running as a proactive heartbeat. Check if there are any "
        "pending reminders, scheduled tasks coming up, or anything you should "
        "proactively tell the user about. If nothing is notable, respond with "
        "exactly: __HEARTBEAT_SKIP__"
    )

    # DM Pairing & Access Control
    telegram_require_pairing: bool = False  # Require /pair before using bot
    telegram_pairing_code: str = ""  # Pairing code users must provide

    # Docker Sandbox
    sandbox_enabled: bool = False  # Route exec through Docker container
    sandbox_image: str = "python:3.12-slim"  # Sandbox container image
    
    # Cross-Encoder Re-ranker (Phase 6)
    enable_reranker: bool = True  # Enable cross-encoder re-ranking after RRF
    cohere_api_key: Optional[str] = None  # Set via COHERE_API_KEY env var
    reranker_model: str = "rerank-v3.5"  # Cohere rerank model
    
    # ── Discord ──────────────────────────────────────────────
    discord_bot_token: Optional[str] = None  # Set via DISCORD_BOT_TOKEN env var
    discord_allowed_guilds: list[str] = []  # Restrict to specific guild IDs
    discord_allowed_users: list[str] = []  # Restrict to specific Discord user IDs

    # ── Slack ────────────────────────────────────────────────
    slack_bot_token: Optional[str] = None  # xoxb-... via SLACK_BOT_TOKEN env var
    slack_app_token: Optional[str] = None  # xapp-... via SLACK_APP_TOKEN env var
    slack_allowed_channels: list[str] = []  # Restrict to specific channel IDs

    # ── WhatsApp (Cloud API) ─────────────────────────────────
    whatsapp_phone_number_id: Optional[str] = None  # via WHATSAPP_PHONE_NUMBER_ID
    whatsapp_access_token: Optional[str] = None  # via WHATSAPP_ACCESS_TOKEN
    whatsapp_verify_token: str = ""  # Webhook verification token
    whatsapp_app_secret: Optional[str] = None  # For payload signature verification
    whatsapp_allowed_numbers: list[str] = []  # Restrict to specific phone numbers

    # ── Thinking / Extended Thinking ─────────────────────────
    thinking_budget_default: int = 0  # 0 = disabled, >0 = max thinking tokens
    thinking_model_override: Optional[str] = None  # Model to use when thinking enabled

    # ── Tool Policies ────────────────────────────────────────
    tool_deny_list: list[str] = []  # Tools completely blocked (e.g. ["exec", "write_file"])
    tool_elevated_list: list[str] = ["exec", "apply_patch", "process"]  # Tools requiring user confirmation
    tool_max_output_chars: int = 60000  # Global tool output truncation limit
    tool_timeout_default: int = 30  # Default per-tool timeout in seconds
    tool_timeout_overrides: dict[str, int] = {  # Per-tool timeout overrides
        "exec": 120, "web_fetch": 60, "web_search": 30,
        "browser": 120, "spawn": 300, "process": 300,
    }

    # ── DM / Group Policy ────────────────────────────────────
    # dm_policy: pairing | allowlist | open | disabled
    dm_policy: str = "allowlist"  # Default: only allowed user IDs
    # group_policy: open | allowlist | disabled
    group_policy: str = "open"  # Default: respond in any group where added
    group_require_mention: bool = True  # Require @mention in groups


    # ── TTS Auto-mode ────────────────────────────────────────
    tts_auto_mode: str = "off"  # off | always | inbound | tagged
    tts_default_voice: str = "alloy"  # OpenAI TTS voice
    tts_model: str = "gpt-4o-mini-tts"  # TTS model
    tts_speed: float = 1.0  # TTS speed multiplier

    # ── Config Hot-Reload ────────────────────────────────────
    config_reload_enabled: bool = True  # Allow hot-reload from /config

    # ── Agent Lanes ──────────────────────────────────────────
    # Lanes: main, subagent, cron, hook — separate execution contexts
    lane_max_concurrent: int = 5  # Max concurrent agent runs across all lanes
    lane_cron_model: Optional[str] = None  # Override model for cron lane
    lane_hook_model: Optional[str] = None  # Override model for hook lane

    # ── Enhanced Telegram ────────────────────────────────────
    telegram_forum_support: bool = True  # Support Telegram forum topics
    telegram_topic_routing: bool = True  # Route by topic thread_id
    telegram_reactions_enabled: bool = True  # Send reactions (e.g., 👍)
    telegram_inline_buttons: bool = True  # Allow inline keyboard buttons
    telegram_polls_enabled: bool = True  # Allow sending polls

    # ── Moderation ───────────────────────────────────────────
    moderation_enabled: bool = False  # Enable moderation tools
    moderation_log_channel: Optional[str] = None  # Channel to log mod actions

    # ── TTS Providers ────────────────────────────────────────
    tts_provider: str = "openai"  # openai | elevenlabs | edge
    elevenlabs_api_key: Optional[str] = None  # Set via ELEVENLABS_API_KEY
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    tts_per_user_prefs: bool = True  # Save per-user TTS preferences

    # ── Platform / Agent Architecture ──────────────────────────
    platform_api_url: str = "https://toup.ai/api"   # Agent calls this to reach the platform
    agent_api_url: str = ""                          # Platform calls this to proxy chat to user's Agent VPS
    supabase_url: str = ""                           # Supabase project URL (for edge-case direct access)
    supabase_anon_key: str = ""                      # Supabase anon key
    run_mode: str = "monolith"                       # "monolith" | "platform" | "agent"
    agent_api_key: str = ""                          # API key for authenticating requests to this Agent VPS
    user_id: str = ""                                # Owner user ID (set on Agent VPS via cloud-init)
    toup_token: str = ""                             # Connect token from toup.ai dashboard (for tunnel auth)

    # ── VPS Provisioning (AWS + Stripe) ──────────────────────
    aws_access_key_id: Optional[str] = None        # Set via AWS_ACCESS_KEY_ID
    aws_secret_access_key: Optional[str] = None    # Set via AWS_SECRET_ACCESS_KEY
    aws_region: str = "us-east-1"
    aws_ami_id: str = ""                           # Custom AMI with platform pre-installed
    aws_key_pair_name: str = ""                    # EC2 key pair name
    aws_security_group_id: str = ""                # Security group allowing SSH + HTTP(S)
    stripe_secret_key: Optional[str] = None        # Set via STRIPE_SECRET_KEY
    stripe_webhook_secret: Optional[str] = None    # Set via STRIPE_WEBHOOK_SECRET
    stripe_starter_price_id: str = ""              # Stripe Price ID for Starter plan
    stripe_standard_price_id: str = ""             # Stripe Price ID for Standard plan
    stripe_pro_price_id: str = ""                  # Stripe Price ID for Pro plan
    stripe_llm_bundle_price_id: str = ""           # Stripe Price ID for $40/mo LLM bundle
    vps_provisioning_enabled: bool = False         # Gate: set True once AWS creds are configured

    # ── Agent Spawn Policies ─────────────────────────────────
    allow_agents: list[str] = []  # If non-empty, only these agent IDs can be spawned

    # ── Custom Model Providers ───────────────────────────────
    # Map model prefix to base URL: {"ollama": "http://localhost:11434/v1", "groq": "https://..."}
    custom_model_providers: dict[str, str] = {}
    # Map model name to provider: {"llama3": "ollama", "mixtral": "groq"}
    custom_model_map: dict[str, str] = {}

    # ── Multi-Agent Routing ──────────────────────────────────
    multi_agent_enabled: bool = False  # Enable persona-based routing
    multi_agent_default: str = "default"  # Default persona name

    # Pricing per 1K tokens (USD)
    pricing_per_1k: dict[str, dict[str, float]] = {
        "gpt-5.2": {"input": 0.003, "output": 0.012},
        "gpt-5": {"input": 0.003, "output": 0.012},
        "gpt-4.1": {"input": 0.002, "output": 0.008},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "claude-opus-4-6": {"input": 0.015, "output": 0.075},
        "claude-sonnet-4-5-20250514": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
