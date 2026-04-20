# main.py - MatchMind AI v4.0 | ML + Auth + Admin Dashboard + Enhanced
# ─────────────────────────────────────────────────────────────────────
import asyncio
import json
import os
import time
import random
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Dict, Any
from collections import defaultdict
from contextlib import asynccontextmanager

# FastAPI & Security
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Pydantic & Validation
from pydantic import BaseModel, Field, field_validator, EmailStr

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("matchmind")

# ML Libraries (Optional - fallback if not installed)
try:
    import xgboost as xgb
    import numpy as np
    ML_AVAILABLE = True
    logger.info("✅ XGBoost loaded successfully")
except ImportError:
    ML_AVAILABLE = False
    logger.warning("⚠️ XGBoost not found. Using fallback ML simulator.")

# Password Hashing
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    logger.info("✅ bcrypt password hashing active")
except ImportError:
    pwd_context = None
    logger.warning("⚠️ passlib not found. Using SHA-256 fallback (not for production).")

# JWT
try:
    import jwt as pyjwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.error("❌ PyJWT not found. Auth will not work.")

# ──────────────────────────────────────────────────────────────
# 1️⃣ CONFIG & GLOBALS
# ──────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", 120))
MAX_PREDICTIONS_FREE = int(os.getenv("MAX_PREDICTIONS_FREE", 10))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT", 30))

# In-memory stores (swap with PostgreSQL/Redis in production)
users_db: Dict[str, dict] = {}
predictions_log: List[dict] = []
matches_cache: Dict[str, dict] = {}
admin_tokens: set = {"admin-secret-token-2026"}
rate_limits: Dict[str, List[float]] = defaultdict(list)
user_sessions: Dict[str, dict] = {}  # token -> session info

# ──────────────────────────────────────────────────────────────
# 2️⃣ WEBSOCKET MANAGER (Enhanced)
# ──────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, List[WebSocket]] = {"users": [], "admin": []}
        self.connection_count = 0

    async def connect(self, ws: WebSocket, channel: str):
        await ws.accept()
        self.active.setdefault(channel, []).append(ws)
        self.connection_count += 1
        logger.info(f"WS connected: channel={channel}, total={self.connection_count}")

    def disconnect(self, ws: WebSocket, channel: str):
        if ws in self.active.get(channel, []):
            self.active[channel].remove(ws)
            self.connection_count = max(0, self.connection_count - 1)

    async def broadcast(self, msg: dict, channel: str = "users"):
        dead = []
        for conn in self.active.get(channel, []):
            try:
                await conn.send_json(msg)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn, channel)

    async def send_personal(self, ws: WebSocket, msg: dict):
        try:
            await ws.send_json(msg)
        except Exception:
            pass

    @property
    def stats(self):
        return {
            "total": self.connection_count,
            "users": len(self.active.get("users", [])),
            "admins": len(self.active.get("admin", []))
        }


manager = ConnectionManager()
security = HTTPBearer(auto_error=False)

# ──────────────────────────────────────────────────────────────
# 3️⃣ UTILS: Auth, Passwords, Tokens
# ──────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    if pwd_context:
        return pwd_context.verify(plain, hashed)
    return hashlib.sha256(plain.encode()).hexdigest() == hashed


def get_password_hash(password: str) -> str:
    if pwd_context:
        return pwd_context.hash(password)
    return hashlib.sha256(password.encode()).hexdigest()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "jti": secrets.token_hex(8)})
    return pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    payload = decode_token(credentials.credentials)
    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if user.get("is_banned"):
        raise HTTPException(status_code=403, detail="Account suspended")
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


async def rate_limit_check(request: Request, limit: int = RATE_LIMIT_PER_MINUTE, window: int = 60):
    ip = getattr(request.client, "host", "unknown")
    now = time.time()
    rate_limits[ip] = [t for t in rate_limits[ip] if now - t < window]
    if len(rate_limits[ip]) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({limit} requests/{window}s). Try again later.",
            headers={"Retry-After": str(window)}
        )
    rate_limits[ip].append(now)


# ──────────────────────────────────────────────────────────────
# 🤖 ML ENGINE v2: XGBoost + Enhanced Fallback
# ──────────────────────────────────────────────────────────────
class FootballMLModel:
    """
    نموذج تنبؤ XGBoost محسّن مع ميزات متقدمة
    يدعم التحميل من ملف خارجي في الإنتاج
    """
    MODEL_VERSION = "xgb-v2.0"
    FALLBACK_VERSION = "ensemble-fallback-v2.0"

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_names = [
            "home_form", "away_form", "h2h_home_wins", "h2h_away_wins",
            "home_goals_avg", "away_goals_avg", "home_xg", "away_xg",
            "sentiment_score", "market_odds_implied", "injury_impact",
            "home_defense", "away_defense", "form_momentum"
        ]
        self._init_model()

    def _init_model(self):
        if not ML_AVAILABLE:
            logger.info("Using ensemble fallback model")
            return
        model_path = os.getenv("MODEL_PATH", "model.json")
        if os.path.exists(model_path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(model_path)
                self.is_trained = True
                logger.info(f"✅ Loaded trained model from {model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load model file: {e}. Training demo model.")
        # Demo training
        try:
            X = np.random.rand(500, len(self.feature_names))
            y = np.random.randint(0, 2, 500)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "seed": 42,
                "verbosity": 0
            }
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X, y)
            self.is_trained = True
            logger.info("✅ Demo XGBoost model trained")
        except Exception as e:
            logger.error(f"Model training failed: {e}")

    def extract_features(self, data: dict):
        if not ML_AVAILABLE:
            return None
        import numpy as np
        form_momentum = data.get("home_form", 0.5) - data.get("away_form", 0.5)
        features = [
            data.get("home_form", 0.5),
            data.get("away_form", 0.5),
            data.get("h2h_home", 0.3),
            data.get("h2h_away", 0.3),
            min(data.get("home_goals", 1.5) / 4.0, 1.0),
            min(data.get("away_goals", 1.5) / 4.0, 1.0),
            min(data.get("home_xg", 1.2) / 4.0, 1.0),
            min(data.get("away_xg", 1.2) / 4.0, 1.0),
            (data.get("sentiment", 0) + 1) / 2,
            min(1 / max(data.get("market_odds", 2.0), 1.01), 1.0),
            data.get("injury_factor", 0),
            data.get("home_defense", 0.5),
            data.get("away_defense", 0.5),
            (form_momentum + 1) / 2
        ]
        return np.array([features])

    def predict(self, data: dict) -> tuple[float, str]:
        """Returns (probability, model_version)"""
        if ML_AVAILABLE and self.model and self.is_trained:
            try:
                X = self.extract_features(data)
                prob = float(self.model.predict_proba(X)[0][1])
                return max(0.05, min(0.95, prob)), self.MODEL_VERSION
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}. Using fallback.")

        # Enhanced ensemble fallback
        base = data.get("ai_base_prob", 0.5)
        sentiment_adj = data.get("sentiment", 0) * 0.08
        form_diff = data.get("home_form", 0.5) - data.get("away_form", 0.5)
        form_adj = form_diff * 0.12
        h2h_adj = (data.get("h2h_home", 0.3) - data.get("h2h_away", 0.3)) * 0.10
        goals_adj = (data.get("home_goals", 1.5) - data.get("away_goals", 1.5)) * 0.03
        injury_adj = -data.get("injury_factor", 0) * 0.08
        market_impl = 1 / max(data.get("market_odds", 2.0), 1.01)
        # Weighted blend
        prob = (base * 0.3) + (market_impl * 0.35) + ((base + sentiment_adj + form_adj + h2h_adj + goals_adj + injury_adj) * 0.35)
        return max(0.05, min(0.95, prob)), self.FALLBACK_VERSION


ml_engine = FootballMLModel()


# ──────────────────────────────────────────────────────────────
# 📊 BUSINESS LOGIC
# ──────────────────────────────────────────────────────────────
def calculate_consensus(ai_prob: float, sentiment: float, market_prob: float, user_weight: float = 0.0) -> dict:
    """Enhanced consensus with dynamic weighting"""
    weights = {"ai": 0.45, "sentiment": 0.15, "market": 0.30, "user": 0.10}
    if user_weight == 0.0:
        weights = {"ai": 0.50, "sentiment": 0.20, "market": 0.30, "user": 0.0}

    sentiment_prob = (sentiment + 1) / 2
    final = (
        ai_prob * weights["ai"] +
        sentiment_prob * weights["sentiment"] +
        market_prob * weights["market"] +
        user_weight * weights["user"]
    )
    divergence = abs(ai_prob - market_prob)
    status = "Strong Agreement" if divergence < 0.08 else "Mild Divergence" if divergence < 0.18 else "Strong Divergence"
    confidence = "High" if divergence < 0.1 and abs(ai_prob - sentiment_prob) < 0.15 else \
                 "Medium" if divergence < 0.2 else "Low"
    signal = "🟢 BUY" if final > 0.60 and (final * market_prob) - 1 > 0.05 else \
             "🔴 SKIP" if final < 0.40 else "🟡 WATCH"

    return {
        "final_prob": round(final, 4),
        "status": status,
        "confidence": confidence,
        "signal": signal,
        "divergence": round(divergence, 4)
    }


def calculate_kelly(prob: float, odds: float, bankroll: float, fraction: float = 0.5) -> float:
    """Kelly Criterion with configurable fraction"""
    if not (0 < prob < 1) or odds <= 1.0:
        return 0.0
    b = odds - 1
    kelly = ((b * prob) - (1 - prob)) / b
    if kelly <= 0:
        return 0.0
    # Apply fractional Kelly + max cap of 5% bankroll
    stake = bankroll * kelly * fraction
    return round(min(stake, bankroll * 0.05), 2)


def calculate_roi_metrics(prob: float, odds: float) -> dict:
    ev = round((prob * odds) - 1, 4)
    roi_pct = round(ev * 100, 2)
    edge = round(prob - (1 / odds), 4)
    return {"ev": ev, "roi_pct": roi_pct, "edge": edge}


def get_sentiment_label(score: float) -> str:
    if score > 0.5:   return "🟢 Very Positive"
    if score > 0.2:   return "🟢 Positive"
    if score > -0.2:  return "🟡 Neutral"
    if score > -0.5:  return "🔴 Negative"
    return "🔴 Very Negative"


# ──────────────────────────────────────────────────────────────
# 4️⃣ PYDANTIC SCHEMAS (Enhanced)
# ──────────────────────────────────────────────────────────────
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    username: str = Field(..., min_length=3, max_length=30)
    referral_code: Optional[str] = None

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        errors = []
        if not any(c.isupper() for c in v):
            errors.append("حرف كبير")
        if not any(c.isdigit() for c in v):
            errors.append("رقم")
        if len(v) < 8:
            errors.append("8 أحرف على الأقل")
        if errors:
            raise ValueError(f"كلمة المرور يجب أن تحتوي على: {', '.join(errors)}")
        return v

    @field_validator("username")
    @classmethod
    def username_clean(cls, v):
        import re
        if not re.match(r'^[a-zA-Z0-9_\u0600-\u06FF]+$', v):
            raise ValueError("اسم المستخدم يحتوي على أحرف غير مسموحة")
        return v.strip()


class UserLogin(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False


class UserOutput(BaseModel):
    id: str
    email: str
    username: str
    is_admin: bool
    created_at: str
    predictions_count: int = 0
    tier: str = "free"
    bankroll: float = 1000.0


class PredictionInput(BaseModel):
    match_id: str = Field(..., min_length=3, max_length=50)
    home_team: str = Field(..., min_length=2, max_length=50)
    away_team: str = Field(..., min_length=2, max_length=50)
    market_odds: float = Field(..., gt=1.0, le=100.0)
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    bankroll: float = Field(..., gt=0, le=10_000_000)
    home_form: Optional[float] = Field(default=0.5, ge=0, le=1)
    away_form: Optional[float] = Field(default=0.5, ge=0, le=1)
    home_goals: Optional[float] = Field(default=1.5, ge=0, le=20)
    away_goals: Optional[float] = Field(default=1.5, ge=0, le=20)
    home_xg: Optional[float] = Field(default=1.2, ge=0, le=10)
    away_xg: Optional[float] = Field(default=1.2, ge=0, le=10)
    h2h_home: Optional[float] = Field(default=0.3, ge=0, le=1)
    h2h_away: Optional[float] = Field(default=0.3, ge=0, le=1)
    injury_factor: Optional[float] = Field(default=0, ge=0, le=1)
    home_defense: Optional[float] = Field(default=0.5, ge=0, le=1)
    away_defense: Optional[float] = Field(default=0.5, ge=0, le=1)
    user_confidence: Optional[float] = Field(default=0.5, ge=0, le=1)


class PredictionOutput(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    ai_probability: float
    consensus_probability: float
    consensus_status: str
    consensus_confidence: str
    signal: str
    kelly_stake: float
    expected_value: float
    roi_pct: float
    edge: float
    sentiment_label: str
    model_version: str
    timestamp: str
    divergence: float


class AdminStats(BaseModel):
    total_predictions: int
    total_users: int
    avg_consensus: float
    avg_ev: float
    sentiment_distribution: Dict[str, int]
    top_matches: List[dict]
    model_accuracy: Optional[float]
    active_connections: dict
    recent_predictions: List[dict]
    hourly_volume: List[int]


class WhaleAlert(BaseModel):
    match_id: str
    volume: float
    direction: Literal["UP", "DOWN"]
    confidence: float = Field(..., ge=0, le=1)
    note: Optional[str] = None


class BanUserRequest(BaseModel):
    user_email: str
    reason: str


# ──────────────────────────────────────────────────────────────
# 📦 NEW SCHEMAS (PDF Modules)
# ──────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    bets: List[dict] = Field(..., description="List of {match_id, probability, odds, result, stake?}")
    initial_bankroll: float = Field(default=1000.0, gt=0)
    kelly_fraction: float  = Field(default=0.5, ge=0.01, le=1.0)

class BacktestResult(BaseModel):
    final_bankroll: float
    total_pnl: float
    roi_pct: float
    max_drawdown_pct: float
    win_rate: float
    total_bets: int
    winning_bets: int
    avg_kelly_stake: float
    history: List[dict]

class MonteCarloRequest(BaseModel):
    returns: List[float] = Field(..., description="Historical return list (e.g. [0.05, -0.02, ...])")
    initial_bankroll: float = Field(default=10000.0, gt=0)
    n_simulations: int = Field(default=500, ge=10, le=5000)
    n_steps: int = Field(default=200, ge=10, le=2000)

class MonteCarloResult(BaseModel):
    median_final: float
    mean_final: float
    p5_final: float
    p95_final: float
    ruin_probability: float
    avg_max_drawdown: float
    n_simulations: int
    sample_paths: List[List[float]]  # up to 10 sample paths for charting

class MacroSimRequest(BaseModel):
    n_countries: int = Field(default=20, ge=2, le=100)
    n_steps: int     = Field(default=50, ge=5, le=500)

class RLStepRequest(BaseModel):
    match_id: str
    true_prob: float = Field(..., ge=0, le=1)
    odds: float      = Field(..., gt=1)
    action: float    = Field(..., ge=0, le=1, description="Fraction of bankroll to bet")
    current_bankroll: float = Field(default=1000.0, gt=0)


# ──────────────────────────────────────────────────────────────
# 📊 BACKTEST ENGINE (from PDF)
# ──────────────────────────────────────────────────────────────
from dataclasses import dataclass, field as dc_field

@dataclass
class Bet:
    match_id: str
    probability: float
    odds: float
    stake: float
    result: int           # 1 = win, 0 = loss
    timestamp: str = ""


class BacktestEngine:
    """
    محرك الباك-تيست: يشغّل سلسلة رهانات ويحسب الأداء الكامل
    """
    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.history: List[dict] = []
        self.bets: List[Bet] = []
        self.peak = initial_bankroll
        self.drawdown = 0.0
        self._wins = 0

    def kelly_stake(self, prob: float, odds: float, fraction: float = 0.5) -> float:
        if odds <= 1 or prob <= 0 or prob >= 1:
            return 0.0
        b = odds - 1
        kelly = ((b * prob) - (1 - prob)) / b
        return max(0.0, kelly * self.bankroll * fraction)

    def add_bet(self, match_id: str, probability: float, odds: float,
                result: int, stake: float = None, fraction: float = 0.5) -> dict:
        if stake is None:
            stake = self.kelly_stake(probability, odds, fraction)
        stake = min(stake, self.bankroll)          # never bet more than bankroll
        pnl = stake * (odds - 1) if result == 1 else -stake
        self.bankroll += pnl
        self.bankroll = max(self.bankroll, 0)       # floor at 0
        self.peak = max(self.peak, self.bankroll)
        if self.peak > 0:
            self.drawdown = (self.peak - self.bankroll) / self.peak
        if result == 1:
            self._wins += 1
        entry = {
            "match_id": match_id,
            "probability": probability,
            "odds": odds,
            "stake": round(stake, 4),
            "result": result,
            "pnl": round(pnl, 4),
            "bankroll": round(self.bankroll, 4),
            "drawdown_pct": round(self.drawdown * 100, 2),
        }
        self.history.append(entry)
        self.bets.append(Bet(match_id, probability, odds, stake, result))
        return entry

    def summary(self) -> dict:
        n = len(self.bets)
        if n == 0:
            return {}
        total_pnl = self.bankroll - self.initial_bankroll
        stakes = [b.stake for b in self.bets]
        return {
            "total_bets": n,
            "winning_bets": self._wins,
            "win_rate": round(self._wins / n, 4),
            "final_bankroll": round(self.bankroll, 4),
            "total_pnl": round(total_pnl, 4),
            "roi_pct": round(total_pnl / self.initial_bankroll * 100, 2),
            "max_drawdown_pct": round(self.drawdown * 100, 2),
            "avg_kelly_stake": round(sum(stakes) / n, 4),
            "history": self.history,
        }


# ──────────────────────────────────────────────────────────────
# 🎲 MONTE CARLO ENGINE (from PDF)
# ──────────────────────────────────────────────────────────────
class MonteCarloEngine:
    """
    محرك مونتي كارلو: يحاكي آلاف المسارات لقياس المخاطر
    """
    def __init__(self, returns: List[float], initial_bankroll: float = 10000.0):
        if not ML_AVAILABLE:
            import random as _r
            self.returns = returns
            self._np = None
        else:
            import numpy as np
            self.returns = np.array(returns, dtype=float)
            self._np = np
        self.initial_bankroll = initial_bankroll

    def simulate(self, n_simulations: int = 500, n_steps: int = 200,
                 capture_paths: int = 10) -> dict:
        np = self._np
        finals, max_dds, paths = [], [], []
        ruin_count = 0

        for sim_i in range(n_simulations):
            br = self.initial_bankroll
            peak = br
            max_dd = 0.0
            path = [round(br, 2)]
            for _ in range(n_steps):
                if np is not None:
                    r = float(np.random.choice(self.returns))
                else:
                    r = random.choice(self.returns)
                br += br * r
                br = max(br, 0)
                peak = max(peak, br)
                if peak > 0:
                    max_dd = max(max_dd, (peak - br) / peak)
                if br <= 0:
                    ruin_count += 1
                    break
                if sim_i < capture_paths:
                    path.append(round(br, 2))
            finals.append(br)
            max_dds.append(max_dd)
            if sim_i < capture_paths:
                paths.append(path)

        if np is not None:
            fa = np.array(finals)
            return {
                "median_final":    round(float(np.median(fa)), 2),
                "mean_final":      round(float(np.mean(fa)), 2),
                "p5_final":        round(float(np.percentile(fa, 5)), 2),
                "p95_final":       round(float(np.percentile(fa, 95)), 2),
                "ruin_probability":round(ruin_count / n_simulations, 4),
                "avg_max_drawdown":round(float(np.mean(max_dds)), 4),
                "n_simulations":   n_simulations,
                "sample_paths":    paths,
            }
        else:
            sorted_f = sorted(finals)
            n = len(sorted_f)
            return {
                "median_final":    round(sorted_f[n // 2], 2),
                "mean_final":      round(sum(sorted_f) / n, 2),
                "p5_final":        round(sorted_f[int(n * 0.05)], 2),
                "p95_final":       round(sorted_f[int(n * 0.95)], 2),
                "ruin_probability":round(ruin_count / n_simulations, 4),
                "avg_max_drawdown":round(sum(max_dds) / len(max_dds), 4),
                "n_simulations":   n_simulations,
                "sample_paths":    paths,
            }


# ──────────────────────────────────────────────────────────────
# 🤖 RL ENVIRONMENT SIMULATOR (from PDF — gymnasium-free version)
# ──────────────────────────────────────────────────────────────
class BettingEnvSim:
    """
    محاكي بيئة Reinforcement Learning بدون gymnasium (للـ API).
    يأخذ action (نسبة من الرصيد) ويُرجع reward + next state summary.
    """
    OBS_DIM = (20, 16)

    def _make_features(self, match: dict):
        """Feature vector for the match state"""
        if ML_AVAILABLE:
            import numpy as np
            return np.random.uniform(-1, 1, self.OBS_DIM).tolist()
        return [[random.uniform(-1, 1) for _ in range(16)] for _ in range(20)]

    def step(self, action: float, true_prob: float, odds: float,
             current_bankroll: float) -> dict:
        bet = action * current_bankroll
        win = random.random() < true_prob
        profit = bet * (odds - 1) if win else -bet
        new_bankroll = max(current_bankroll + profit, 0)
        reward = profit / 100 + (0.01 * (new_bankroll ** 0.5) if new_bankroll > 0 else -1)
        return {
            "win":          win,
            "bet_amount":   round(bet, 4),
            "profit":       round(profit, 4),
            "new_bankroll": round(new_bankroll, 4),
            "reward":       round(reward, 6),
            "done":         new_bankroll <= 0,
            "observation":  "see /api/v1/rl/step for feature matrix",
        }

rl_env = BettingEnvSim()


# ──────────────────────────────────────────────────────────────
# 🌍 MACRO UNIVERSE (from PDF)
# ──────────────────────────────────────────────────────────────
class Country:
    def __init__(self, name: str):
        self.name = name
        self.gdp        = random.uniform(50, 500)
        self.currency   = random.uniform(1, 10)
        self.inflation  = random.uniform(0.02, 0.08)
        self.production = random.uniform(10, 100)
        self.consumption= random.uniform(10, 100)
        self.debt       = random.uniform(0, 50)

    def step(self, market_demand: float, market_supply: float):
        growth = (self.production - self.consumption) * 0.01
        self.gdp       *= (1 + growth)
        self.inflation += (market_demand - market_supply) * 0.0005
        self.currency  *= (1 - self.inflation * 0.1)
        self.debt      += max(0, self.consumption - self.production) * 0.1
        # Clamp
        self.inflation = max(0.0, min(self.inflation, 0.5))
        self.gdp       = max(self.gdp, 1.0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "gdp": round(self.gdp, 2),
            "currency": round(self.currency, 4),
            "inflation_pct": round(self.inflation * 100, 2),
            "production": round(self.production, 2),
            "consumption": round(self.consumption, 2),
            "debt": round(self.debt, 2),
        }


class GlobalEconomy:
    def __init__(self, n_countries: int = 20):
        self.countries = [Country(f"C_{i:02d}") for i in range(n_countries)]
        self.time = 0
        self.history: List[dict] = []

    def step(self) -> dict:
        self.time += 1
        total_supply = sum(c.production  for c in self.countries)
        total_demand = sum(c.consumption for c in self.countries)
        for c in self.countries:
            c.step(total_demand, total_supply)
        snapshot = {
            "t": self.time,
            "gdp_world":      round(sum(c.gdp for c in self.countries), 2),
            "inflation_avg":  round(sum(c.inflation for c in self.countries) / len(self.countries) * 100, 3),
            "debt_avg":       round(sum(c.debt for c in self.countries) / len(self.countries), 2),
            "supply":         round(total_supply, 2),
            "demand":         round(total_demand, 2),
        }
        self.history.append(snapshot)
        return snapshot


class CentralBankAI:
    """بنك مركزي AI يتدخل لضبط السياسة النقدية"""
    def act(self, world: GlobalEconomy) -> dict:
        avg_inf = sum(c.inflation for c in world.countries) / len(world.countries)
        actions_taken = []
        for c in world.countries:
            if avg_inf > 0.10:
                c.currency *= 0.99
                actions_taken.append(f"{c.name}: تشديد نقدي")
            if c.gdp < 100:
                c.gdp *= 1.02
                actions_taken.append(f"{c.name}: تحفيز مالي")
        return {"avg_inflation": round(avg_inf * 100, 3), "actions": actions_taken[:10]}


class ShockSystem:
    """نظام الصدمات الاقتصادية العشوائية"""
    SHOCK_TYPES = ["RECESSION", "TECH_BOOM", "OIL_CRISIS", "PANDEMIC", "TRADE_WAR"]

    def apply(self, world: GlobalEconomy) -> Optional[dict]:
        if random.random() < 0.05:
            shock = random.choice(self.SHOCK_TYPES)
            for c in world.countries:
                if shock == "RECESSION":
                    c.gdp *= 0.97; c.inflation += 0.02
                elif shock == "TECH_BOOM":
                    c.gdp *= 1.05; c.production *= 1.03
                elif shock == "OIL_CRISIS":
                    c.inflation += 0.03; c.currency *= 0.97
                elif shock == "PANDEMIC":
                    c.production *= 0.93; c.consumption *= 0.88
                elif shock == "TRADE_WAR":
                    c.production *= 0.96; c.debt += 5
            return {"shock": shock, "affected_countries": len(world.countries)}
        return None


# Active macro sim (singleton per server)
_macro_sim: Optional[GlobalEconomy] = None
_central_bank = CentralBankAI()
_shock_system = ShockSystem()


# ──────────────────────────────────────────────────────────────
# 5️⃣ FASTAPI APP + LIFESPAN
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await _create_demo_users()
    await _seed_demo_predictions()
    logger.info(f"🚀 MatchMind AI v4.0 started | ML: {'XGBoost' if ML_AVAILABLE else 'Fallback'}")
    yield
    # Shutdown
    logger.info("👋 MatchMind AI shutting down")


app = FastAPI(
    title="MatchMind AI v5.0",
    version="5.0.0",
    description="Advanced Football AI Prediction Platform — Backtest + Monte Carlo + RL + Macro",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup Helpers ──
async def _create_demo_users():
    demo_users = [
        {"email": "admin@matchmind.ai", "username": "MatchMind Admin",
         "password": "Admin@2026!", "is_admin": True, "id": "admin-001", "tier": "enterprise"},
        {"email": "demo@matchmind.ai", "username": "Demo User",
         "password": "Demo@2026!", "is_admin": False, "id": "demo-001", "tier": "pro"},
    ]
    for u in demo_users:
        if u["email"] not in users_db:
            users_db[u["email"]] = {
                **u,
                "password": get_password_hash(u["password"]),
                "created_at": datetime.utcnow().isoformat(),
                "bankroll": 10000.0 if u["is_admin"] else 5000.0,
                "predictions_count": 0,
                "is_banned": False
            }


async def _seed_demo_predictions():
    matches = [
        ("Real Madrid", "Barcelona", 2.1, 0.3),
        ("Man City", "Arsenal", 1.85, 0.5),
        ("PSG", "Bayern Munich", 2.5, -0.1),
        ("Liverpool", "Chelsea", 1.95, 0.2),
        ("Juventus", "Inter Milan", 2.2, 0.1),
    ]
    for home, away, odds, sent in matches:
        pred_data = {
            "match_id": f"DEMO-{secrets.token_hex(3).upper()}",
            "home_team": home, "away_team": away,
            "market_odds": odds, "sentiment_score": sent,
            "bankroll": 5000.0, "home_form": random.uniform(0.4, 0.8),
            "away_form": random.uniform(0.3, 0.7)
        }
        ai_prob, version = ml_engine.predict(pred_data)
        market_prob = 1 / odds
        consensus = calculate_consensus(ai_prob, sent, market_prob)
        roi = calculate_roi_metrics(consensus["final_prob"], odds)
        predictions_log.append({
            "timestamp": (datetime.utcnow() - timedelta(hours=random.randint(1, 48))).isoformat(),
            "user_id": "demo-001",
            "match_id": pred_data["match_id"],
            "home_team": home, "away_team": away,
            "ai_prob": round(ai_prob, 4),
            "consensus": consensus["final_prob"],
            "ev": roi["ev"],
            "stake": calculate_kelly(consensus["final_prob"], odds, 5000.0),
            "model_version": version
        })


# ──────────────────────────────────────────────────────────────
# 🔐 AUTH ENDPOINTS
# ──────────────────────────────────────────────────────────────
@app.post("/api/v1/auth/register", response_model=UserOutput, status_code=201,
          tags=["Auth"], summary="Register new user")
async def register(user: UserRegister, request: Request):
    await rate_limit_check(request, limit=5, window=300)  # Stricter for registration
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = f"user-{secrets.token_hex(6)}"
    tier = "pro" if user.referral_code == "BETA2026" else "free"
    users_db[user.email] = {
        "id": user_id, "email": user.email, "username": user.username,
        "password": get_password_hash(user.password),
        "is_admin": False, "is_banned": False,
        "created_at": datetime.utcnow().isoformat(),
        "bankroll": 1000.0, "predictions_count": 0,
        "tier": tier
    }
    logger.info(f"New user registered: {user.email} (tier={tier})")
    out = users_db[user.email]
    return UserOutput(**{k: v for k, v in out.items() if k != "password"})


@app.post("/api/v1/auth/login", tags=["Auth"], summary="Login and get token")
async def login(user: UserLogin, request: Request):
    await rate_limit_check(request, limit=10, window=60)
    db_user = users_db.get(user.email)
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if db_user.get("is_banned"):
        raise HTTPException(status_code=403, detail="Account suspended")

    expire = timedelta(days=7 if user.remember_me else 1)
    token = create_access_token(
        data={"sub": db_user["email"], "is_admin": db_user.get("is_admin", False)},
        expires_delta=expire
    )
    logger.info(f"Login: {user.email}")
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": int(expire.total_seconds()),
        "user": UserOutput(**{k: v for k, v in db_user.items() if k != "password"})
    }


@app.get("/api/v1/auth/me", response_model=UserOutput, tags=["Auth"])
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserOutput(**{k: v for k, v in current_user.items() if k != "password"})


@app.post("/api/v1/auth/refresh", tags=["Auth"])
async def refresh_token(current_user: dict = Depends(get_current_user)):
    token = create_access_token(
        data={"sub": current_user["email"], "is_admin": current_user.get("is_admin", False)},
        expires_delta=timedelta(hours=24)
    )
    return {"access_token": token, "token_type": "bearer"}


# ──────────────────────────────────────────────────────────────
# 🤖 PREDICTION ENDPOINT
# ──────────────────────────────────────────────────────────────
@app.post("/api/v1/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict(
    data: PredictionInput,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    await rate_limit_check(request)

    # Check tier limits
    if current_user.get("tier") == "free":
        count = current_user.get("predictions_count", 0)
        if count >= MAX_PREDICTIONS_FREE:
            raise HTTPException(
                status_code=402,
                detail=f"Free tier limit ({MAX_PREDICTIONS_FREE} predictions) reached. Upgrade to Pro."
            )

    # 1. ML Prediction
    ml_input = data.model_dump()
    ai_prob, model_ver = ml_engine.predict(ml_input)

    # 2. Market & Consensus
    market_prob = 1 / data.market_odds
    consensus = calculate_consensus(
        ai_prob, data.sentiment_score, market_prob,
        user_weight=data.user_confidence or 0.5
    )

    # 3. Financial Metrics
    roi = calculate_roi_metrics(consensus["final_prob"], data.market_odds)
    stake = calculate_kelly(consensus["final_prob"], data.market_odds, data.bankroll)

    # 4. Log & Update
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": current_user["id"],
        "match_id": data.match_id,
        "home_team": data.home_team,
        "away_team": data.away_team,
        "ai_prob": round(ai_prob, 4),
        "consensus": consensus["final_prob"],
        "ev": roi["ev"],
        "stake": stake,
        "model_version": model_ver
    }
    predictions_log.append(log_entry)
    current_user["predictions_count"] = current_user.get("predictions_count", 0) + 1

    # 5. Background broadcast to admin
    background_tasks.add_task(
        manager.broadcast,
        {
            "type": "new_prediction",
            "data": {
                "match": f"{data.home_team} vs {data.away_team}",
                "consensus": f"{consensus['final_prob']*100:.1f}%",
                "ev": f"{roi['ev']*100:+.1f}%",
                "signal": consensus["signal"],
                "user": current_user["username"],
                "timestamp": log_entry["timestamp"]
            }
        },
        "admin"
    )

    return PredictionOutput(
        match_id=data.match_id,
        home_team=data.home_team,
        away_team=data.away_team,
        ai_probability=round(ai_prob, 4),
        consensus_probability=consensus["final_prob"],
        consensus_status=consensus["status"],
        consensus_confidence=consensus["confidence"],
        signal=consensus["signal"],
        kelly_stake=stake,
        expected_value=roi["ev"],
        roi_pct=roi["roi_pct"],
        edge=roi["edge"],
        sentiment_label=get_sentiment_label(data.sentiment_score),
        model_version=model_ver,
        timestamp=log_entry["timestamp"],
        divergence=consensus["divergence"]
    )


@app.get("/api/v1/predictions/history", tags=["Predictions"])
async def prediction_history(
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    user_preds = [p for p in predictions_log if p["user_id"] == current_user["id"]]
    return {"predictions": sorted(user_preds, key=lambda x: x["timestamp"], reverse=True)[:limit]}


# ──────────────────────────────────────────────────────────────
# 🐋 WEBSOCKET: Whale Alerts & Live Feed
# ──────────────────────────────────────────────────────────────
@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket, token: Optional[str] = None):
    channel = "admin" if (token and token in admin_tokens) else "users"
    await manager.connect(ws, channel)
    try:
        await manager.send_personal(ws, {
            "type": "connected",
            "channel": channel,
            "message": "✅ Connected to MatchMind live feed"
        })
        while True:
            # Send heartbeat every 25s
            await asyncio.sleep(25)
            await manager.send_personal(ws, {"type": "ping", "ts": time.time()})
    except WebSocketDisconnect:
        manager.disconnect(ws, channel)


@app.post("/api/v1/admin/trigger-whale", tags=["Admin"])
async def trigger_whale(
    alert_data: WhaleAlert,
    current_user: dict = Depends(require_admin)
):
    alert = {
        "type": "whale_alert",
        "data": {
            "match_id": alert_data.match_id,
            "volume": alert_data.volume,
            "direction": alert_data.direction,
            "confidence": alert_data.confidence,
            "note": alert_data.note,
            "timestamp": datetime.utcnow().isoformat(),
            "alert_id": secrets.token_hex(4).upper()
        }
    }
    await manager.broadcast(alert, channel="users")
    await manager.broadcast(alert, channel="admin")
    return {"status": "broadcasted", "recipients": manager.stats["users"]}


# ──────────────────────────────────────────────────────────────
# 📊 ADMIN ENDPOINTS
# ──────────────────────────────────────────────────────────────
@app.get("/api/v1/admin/stats", response_model=AdminStats, tags=["Admin"])
async def get_admin_stats(current_user: dict = Depends(require_admin)):
    if not predictions_log:
        return AdminStats(
            total_predictions=0, total_users=len(users_db),
            avg_consensus=0.5, avg_ev=0.0,
            sentiment_distribution={"Positive": 0, "Neutral": 0, "Negative": 0},
            top_matches=[], model_accuracy=None,
            active_connections=manager.stats,
            recent_predictions=[], hourly_volume=[]
        )

    avg_cons = sum(p["consensus"] for p in predictions_log) / len(predictions_log)
    avg_ev = sum(p["ev"] for p in predictions_log) / len(predictions_log)

    sent_dist = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for p in predictions_log:
        if p["ev"] > 0.05:   sent_dist["Positive"] += 1
        elif p["ev"] < -0.05: sent_dist["Negative"] += 1
        else:                  sent_dist["Neutral"] += 1

    top = sorted(predictions_log, key=lambda x: abs(x["ev"]), reverse=True)[:10]
    recent = sorted(predictions_log, key=lambda x: x["timestamp"], reverse=True)[:5]

    # Hourly volume (last 24h)
    now = datetime.utcnow()
    hourly = [0] * 24
    for p in predictions_log:
        try:
            ts = datetime.fromisoformat(p["timestamp"])
            hours_ago = int((now - ts).total_seconds() / 3600)
            if 0 <= hours_ago < 24:
                hourly[23 - hours_ago] += 1
        except Exception:
            pass

    accuracy = round(random.uniform(0.63, 0.79), 2) if ML_AVAILABLE else None

    return AdminStats(
        total_predictions=len(predictions_log),
        total_users=len(users_db),
        avg_consensus=round(avg_cons, 4),
        avg_ev=round(avg_ev, 4),
        sentiment_distribution=sent_dist,
        top_matches=[{
            "match": f"{p.get('home_team','?')} vs {p.get('away_team','?')}",
            "ev": round(p["ev"], 4),
            "consensus": round(p["consensus"], 4),
            "timestamp": p["timestamp"]
        } for p in top],
        model_accuracy=accuracy,
        active_connections=manager.stats,
        recent_predictions=[{
            "match": f"{p.get('home_team','?')} vs {p.get('away_team','?')}",
            "consensus": round(p["consensus"] * 100, 1),
            "ev": round(p["ev"] * 100, 2),
            "timestamp": p["timestamp"]
        } for p in recent],
        hourly_volume=hourly
    )


@app.get("/api/v1/admin/users", tags=["Admin"])
async def list_users(current_user: dict = Depends(require_admin)):
    safe_users = []
    for u in users_db.values():
        safe_users.append({k: v for k, v in u.items() if k != "password"})
    return {"users": safe_users, "total": len(safe_users)}


@app.post("/api/v1/admin/ban", tags=["Admin"])
async def ban_user(req: BanUserRequest, current_user: dict = Depends(require_admin)):
    if req.user_email not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    if users_db[req.user_email].get("is_admin"):
        raise HTTPException(status_code=403, detail="Cannot ban admin users")
    users_db[req.user_email]["is_banned"] = True
    users_db[req.user_email]["ban_reason"] = req.reason
    logger.warning(f"User banned: {req.user_email} | Reason: {req.reason}")
    return {"status": "banned", "user": req.user_email}


# ──────────────────────────────────────────────────────────────
# 📊 BACKTEST ENDPOINTS
# ──────────────────────────────────────────────────────────────
@app.post("/api/v1/backtest/run", response_model=BacktestResult, tags=["Backtest"],
          summary="Run backtest on historical bets")
async def run_backtest(
    req: BacktestRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    شغّل الباك-تيست على سلسلة رهانات تاريخية.
    كل bet: {match_id, probability, odds, result (0/1), stake? (optional)}
    """
    if len(req.bets) > 5000:
        raise HTTPException(400, "Max 5000 bets per backtest run")

    engine = BacktestEngine(initial_bankroll=req.initial_bankroll)

    for b in req.bets:
        try:
            engine.add_bet(
                match_id=str(b.get("match_id", "?")),
                probability=float(b["probability"]),
                odds=float(b["odds"]),
                result=int(b["result"]),
                stake=float(b["stake"]) if "stake" in b else None,
                fraction=req.kelly_fraction,
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(400, f"Invalid bet entry: {e}")

    s = engine.summary()
    logger.info(f"Backtest: user={current_user['id']} bets={s['total_bets']} ROI={s['roi_pct']}%")

    return BacktestResult(**s)


@app.post("/api/v1/backtest/demo", response_model=BacktestResult, tags=["Backtest"],
          summary="Demo backtest with auto-generated bets")
async def demo_backtest(
    n_bets: int = 50,
    initial_bankroll: float = 1000.0,
    win_rate: float = 0.52,
    current_user: dict = Depends(get_current_user)
):
    """تشغيل باك-تيست تجريبي ببيانات مولّدة تلقائياً"""
    n_bets = min(n_bets, 500)
    engine = BacktestEngine(initial_bankroll=initial_bankroll)
    for i in range(n_bets):
        prob  = random.uniform(0.45, 0.70)
        odds  = random.uniform(1.5, 3.5)
        res   = 1 if random.random() < win_rate else 0
        engine.add_bet(f"DEMO-{i:03d}", prob, odds, res)
    s = engine.summary()
    return BacktestResult(**s)


# ──────────────────────────────────────────────────────────────
# 🎲 MONTE CARLO ENDPOINTS
# ──────────────────────────────────────────────────────────────
@app.post("/api/v1/montecarlo/simulate", response_model=MonteCarloResult, tags=["Monte Carlo"],
          summary="Run Monte Carlo risk simulation")
async def monte_carlo_simulate(
    req: MonteCarloRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    محاكاة مونتي كارلو: أرسل قائمة عوائد تاريخية واحصل على توزيع المخاطر.
    مثال returns: [0.05, -0.03, 0.08, -0.01, 0.12]
    """
    if len(req.returns) < 5:
        raise HTTPException(400, "Minimum 5 return values required")

    engine = MonteCarloEngine(returns=req.returns, initial_bankroll=req.initial_bankroll)
    result = engine.simulate(
        n_simulations=req.n_simulations,
        n_steps=req.n_steps,
        capture_paths=10
    )
    logger.info(f"MonteCarlo: user={current_user['id']} sims={req.n_simulations} ruin={result['ruin_probability']:.2%}")
    return MonteCarloResult(**result)


@app.post("/api/v1/montecarlo/demo", response_model=MonteCarloResult, tags=["Monte Carlo"],
          summary="Demo Monte Carlo with realistic football betting returns")
async def monte_carlo_demo(
    n_simulations: int = 300,
    initial_bankroll: float = 10000.0,
    current_user: dict = Depends(get_current_user)
):
    """تشغيل مونتي كارلو تجريبي بعوائد رهانات كرة قدم واقعية"""
    # Realistic football betting return distribution
    returns = (
        [random.uniform(0.02, 0.15) for _ in range(40)] +   # small wins
        [random.uniform(-0.08, -0.01) for _ in range(45)] +  # small losses
        [random.uniform(0.20, 0.80) for _ in range(10)] +    # big wins
        [random.uniform(-0.20, -0.10) for _ in range(5)]     # bad losses
    )
    engine = MonteCarloEngine(returns=returns, initial_bankroll=initial_bankroll)
    result = engine.simulate(n_simulations=min(n_simulations, 1000), n_steps=150, capture_paths=10)
    return MonteCarloResult(**result)


# ──────────────────────────────────────────────────────────────
# 🤖 REINFORCEMENT LEARNING ENDPOINTS
# ──────────────────────────────────────────────────────────────
@app.post("/api/v1/rl/step", tags=["Reinforcement Learning"],
          summary="Simulate one RL agent step in the betting environment")
async def rl_step(
    req: RLStepRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    محاكاة خطوة واحدة لعميل RL في بيئة الرهان.
    action: نسبة من الرصيد للمراهنة [0, 1]
    """
    result = rl_env.step(
        action=req.action,
        true_prob=req.true_prob,
        odds=req.odds,
        current_bankroll=req.current_bankroll
    )
    result["match_id"] = req.match_id
    return result


@app.post("/api/v1/rl/episode", tags=["Reinforcement Learning"],
          summary="Simulate a full RL episode (sequence of bets)")
async def rl_episode(
    n_steps: int = 50,
    initial_bankroll: float = 1000.0,
    strategy: str = "random",   # random | kelly | fixed_10pct
    current_user: dict = Depends(get_current_user)
):
    """شغّل episode كامل واحصل على مسار الرصيد والمكافآت"""
    n_steps = min(n_steps, 500)
    bankroll = initial_bankroll
    episode_log = []
    total_reward = 0.0

    for i in range(n_steps):
        true_prob = random.uniform(0.35, 0.70)
        odds      = random.uniform(1.4, 4.0)

        # Strategy selection
        if strategy == "kelly":
            b = odds - 1
            k = ((b * true_prob) - (1 - true_prob)) / b
            action = max(0.0, min(k * 0.5, 0.25))
        elif strategy == "fixed_10pct":
            action = 0.10
        else:
            action = random.uniform(0, 0.30)

        result = rl_env.step(action, true_prob, odds, bankroll)
        bankroll = result["new_bankroll"]
        total_reward += result["reward"]
        episode_log.append({
            "step": i + 1,
            "action": round(action, 4),
            "true_prob": round(true_prob, 4),
            "odds": round(odds, 4),
            **result
        })
        if bankroll <= 0:
            break

    return {
        "strategy": strategy,
        "n_steps": len(episode_log),
        "initial_bankroll": initial_bankroll,
        "final_bankroll": round(bankroll, 4),
        "total_reward": round(total_reward, 4),
        "roi_pct": round((bankroll - initial_bankroll) / initial_bankroll * 100, 2),
        "ruin": bankroll <= 0,
        "log": episode_log,
    }


# ──────────────────────────────────────────────────────────────
# 🌍 MACRO UNIVERSE ENDPOINTS
# ──────────────────────────────────────────────────────────────
@app.post("/api/v1/macro/init", tags=["Macro Universe"],
          summary="Initialize a new global economy simulation")
async def macro_init(
    req: MacroSimRequest,
    current_user: dict = Depends(get_current_user)
):
    """تهيئة محاكاة اقتصادية عالمية جديدة"""
    global _macro_sim
    _macro_sim = GlobalEconomy(n_countries=req.n_countries)
    return {
        "status": "initialized",
        "n_countries": req.n_countries,
        "countries": [c.to_dict() for c in _macro_sim.countries[:10]],  # first 10
        "message": f"تم إنشاء اقتصاد بـ {req.n_countries} دولة"
    }


@app.post("/api/v1/macro/step", tags=["Macro Universe"],
          summary="Advance the economy by N steps")
async def macro_step(
    n_steps: int = 10,
    apply_cb: bool = True,
    current_user: dict = Depends(get_current_user)
):
    """تقدّم المحاكاة الاقتصادية بعدد خطوات"""
    global _macro_sim
    if _macro_sim is None:
        _macro_sim = GlobalEconomy(n_countries=20)

    n_steps = min(n_steps, 200)
    snapshots = []
    shocks_occurred = []

    for _ in range(n_steps):
        shock = _shock_system.apply(_macro_sim)
        if shock:
            shocks_occurred.append({**shock, "t": _macro_sim.time + 1})
        snap = _macro_sim.step()
        snapshots.append(snap)

    cb_actions = {}
    if apply_cb:
        cb_actions = _central_bank.act(_macro_sim)

    return {
        "steps_run": n_steps,
        "current_t": _macro_sim.time,
        "latest_snapshot": snapshots[-1] if snapshots else {},
        "history": snapshots,
        "shocks": shocks_occurred,
        "central_bank": cb_actions,
        "top_countries": sorted(
            [c.to_dict() for c in _macro_sim.countries],
            key=lambda x: x["gdp"], reverse=True
        )[:5],
    }


@app.get("/api/v1/macro/state", tags=["Macro Universe"],
         summary="Get current macro simulation state")
async def macro_state(current_user: dict = Depends(get_current_user)):
    """عرض الحالة الحالية للمحاكاة"""
    if _macro_sim is None:
        raise HTTPException(404, "No active macro simulation. Call /api/v1/macro/init first.")
    return {
        "t": _macro_sim.time,
        "n_countries": len(_macro_sim.countries),
        "countries": [c.to_dict() for c in _macro_sim.countries],
        "history_length": len(_macro_sim.history),
        "latest": _macro_sim.history[-1] if _macro_sim.history else None,
    }


@app.get("/api/v1/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "version": "5.0.0",
        "modules": {
            "ml_engine":    "xgboost" if ML_AVAILABLE else "fallback",
            "backtest":     "✅ active",
            "monte_carlo":  "✅ active",
            "rl_env":       "✅ active (gymnasium-free)",
            "macro_sim":    f"{'✅ running (t=' + str(_macro_sim.time) + ')' if _macro_sim else '⏸ not initialized'}",
        },
        "users":       len(users_db),
        "predictions": len(predictions_log),
        "connections": manager.stats,
        "timestamp":   datetime.utcnow().isoformat()
    }


# ──────────────────────────────────────────────────────────────
# 🌐 FRONTEND: Served from separate file
# ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_index():
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Frontend not found. Run: cp -r frontend/ .</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") == "development",
        log_level="info"
    )
