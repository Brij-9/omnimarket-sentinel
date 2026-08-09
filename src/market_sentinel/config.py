"""Validated, fail-safe environment settings for OmniMarket Sentinel."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from market_sentinel.domain.enums import OperatingMode

SAFE_MAX_TRADE_RISK_FRACTION = Decimal("0.005")
SAFE_MAX_POSITION_FRACTION = Decimal("0.10")
SAFE_MAX_GROSS_EXPOSURE_FRACTION = Decimal("0.50")
SAFE_MAX_DAILY_LOSS_FRACTION = Decimal("0.02")
SAFE_MAX_DRAWDOWN_FRACTION = Decimal("0.10")


class RiskSettings(BaseModel):
    """Risk limits consumed by the future risk engine, never by target reporting."""

    model_config = ConfigDict(frozen=True)

    max_trade_risk_fraction: Decimal = Field(gt=Decimal("0"), le=Decimal("1"))
    max_position_fraction: Decimal = Field(gt=Decimal("0"), le=Decimal("1"))
    max_gross_exposure_fraction: Decimal = Field(gt=Decimal("0"), le=Decimal("1"))
    max_daily_loss_fraction: Decimal = Field(gt=Decimal("0"), le=Decimal("1"))
    max_drawdown_fraction: Decimal = Field(gt=Decimal("0"), le=Decimal("1"))


class TargetProgress(BaseModel):
    """Reporting-only progress toward an aspirational capital target."""

    model_config = ConfigDict(frozen=True)

    starting_capital: Decimal
    current_equity: Decimal
    aspirational_target: Decimal
    required_multiple: Decimal
    achieved_multiple: Decimal
    remaining_gap: Decimal


class Settings(BaseSettings):
    """Environment settings with conservative live-small risk bounds."""

    model_config = SettingsConfigDict(
        env_prefix="MARKET_SENTINEL_",
        extra="ignore",
        frozen=True,
    )

    mode: OperatingMode = OperatingMode.RESEARCH
    database_url: str = "sqlite+pysqlite:///data/market_sentinel.db"
    starting_capital: Decimal = Field(default=Decimal("10"), gt=Decimal("0"))
    aspirational_target: Decimal = Field(default=Decimal("1000000"), gt=Decimal("0"))
    research_provider: str = "tauric"
    llm_provider: str = "ollama"
    llm_model: str = ""
    primary_broker: str = "groww"

    max_trade_risk: Decimal = Field(
        default=SAFE_MAX_TRADE_RISK_FRACTION,
        gt=Decimal("0"),
        le=Decimal("1"),
    )
    max_position: Decimal = Field(
        default=SAFE_MAX_POSITION_FRACTION,
        gt=Decimal("0"),
        le=Decimal("1"),
    )
    max_gross_exposure: Decimal = Field(
        default=SAFE_MAX_GROSS_EXPOSURE_FRACTION,
        gt=Decimal("0"),
        le=Decimal("1"),
    )
    max_daily_loss: Decimal = Field(
        default=SAFE_MAX_DAILY_LOSS_FRACTION,
        gt=Decimal("0"),
        le=Decimal("1"),
    )
    max_drawdown: Decimal = Field(
        default=SAFE_MAX_DRAWDOWN_FRACTION,
        gt=Decimal("0"),
        le=Decimal("1"),
    )

    alpaca_live_trading_enabled: bool = Field(
        default=False,
        validation_alias="ALPACA_LIVE_TRADING_ENABLED",
    )
    alpaca_real_api_enabled: bool = Field(default=False, validation_alias="ALPACA_REAL_API_ENABLED")
    alpaca_trading_endpoint: str = Field(
        default="https://paper-api.alpaca.markets",
        validation_alias="ALPACA_TRADING_ENDPOINT",
    )
    alpaca_account_id: str = Field(default="", validation_alias="ALPACA_ACCOUNT_ID")
    alpaca_key_id: str = Field(default="", validation_alias="ALPACA_KEY_ID")
    alpaca_secret_key: str = Field(default="", validation_alias="ALPACA_SECRET_KEY")

    india_live_trading_enabled: bool = Field(
        default=False,
        validation_alias="INDIA_LIVE_TRADING_ENABLED",
    )
    india_algo_compliance_verified: bool = Field(
        default=False,
        validation_alias="INDIA_ALGO_COMPLIANCE_VERIFIED",
    )
    groww_real_api_enabled: bool = Field(default=False, validation_alias="GROWW_REAL_API_ENABLED")
    groww_api_subscription_active: bool = Field(
        default=False,
        validation_alias="GROWW_API_SUBSCRIPTION_ACTIVE",
    )
    groww_protected_order_client: bool = Field(
        default=False,
        validation_alias="GROWW_PROTECTED_ORDER_CLIENT",
    )
    groww_static_outbound_ip: str = Field(default="", validation_alias="GROWW_STATIC_OUTBOUND_IP")
    groww_static_ip_allowlisted: bool = Field(
        default=False,
        validation_alias="GROWW_STATIC_IP_ALLOWLISTED",
    )
    groww_algo_id: str = Field(default="", validation_alias="GROWW_ALGO_ID")
    groww_access_token: str = Field(default="", validation_alias="GROWW_ACCESS_TOKEN")
    groww_api_key: str = Field(default="", validation_alias="GROWW_API_KEY")
    groww_secret_key: str = Field(default="", validation_alias="GROWW_SECRET_KEY")

    ccxt_live_trading_enabled: bool = Field(
        default=False,
        validation_alias="CCXT_LIVE_TRADING_ENABLED",
    )
    ccxt_real_api_enabled: bool = Field(default=False, validation_alias="CCXT_REAL_API_ENABLED")
    ccxt_exchange_id: str = Field(default="", validation_alias="CCXT_EXCHANGE_ID")
    ccxt_sandbox: bool = Field(default=True, validation_alias="CCXT_SANDBOX")
    ccxt_api_key: str = Field(default="", validation_alias="CCXT_API_KEY")
    ccxt_secret: str = Field(default="", validation_alias="CCXT_SECRET")
    ccxt_withdrawals_disabled_confirmed: bool = Field(
        default=False,
        validation_alias="CCXT_WITHDRAWALS_DISABLED_CONFIRMED",
    )
    ccxt_ip_restricted_confirmed: bool = Field(
        default=False,
        validation_alias="CCXT_IP_RESTRICTED_CONFIRMED",
    )
    ccxt_no_sandbox_acknowledged: bool = Field(
        default=False,
        validation_alias="CCXT_NO_SANDBOX_ACKNOWLEDGED",
    )

    @property
    def required_multiple(self) -> Decimal:
        """Return the reporting multiple needed to reach the aspirational target."""
        return self.aspirational_target / self.starting_capital

    @property
    def risk(self) -> RiskSettings:
        """Return risk limits, ensuring live-small cannot relax approved caps."""
        if self.mode is OperatingMode.LIVE_SMALL:
            return RiskSettings(
                max_trade_risk_fraction=min(self.max_trade_risk, SAFE_MAX_TRADE_RISK_FRACTION),
                max_position_fraction=min(self.max_position, SAFE_MAX_POSITION_FRACTION),
                max_gross_exposure_fraction=min(
                    self.max_gross_exposure,
                    SAFE_MAX_GROSS_EXPOSURE_FRACTION,
                ),
                max_daily_loss_fraction=min(self.max_daily_loss, SAFE_MAX_DAILY_LOSS_FRACTION),
                max_drawdown_fraction=min(self.max_drawdown, SAFE_MAX_DRAWDOWN_FRACTION),
            )
        return RiskSettings(
            max_trade_risk_fraction=self.max_trade_risk,
            max_position_fraction=self.max_position,
            max_gross_exposure_fraction=self.max_gross_exposure,
            max_daily_loss_fraction=self.max_daily_loss,
            max_drawdown_fraction=self.max_drawdown,
        )

    def target_progress(self, current_equity: Decimal) -> TargetProgress:
        """Create reporting-only target progress without changing any risk limit."""
        return TargetProgress(
            starting_capital=self.starting_capital,
            current_equity=current_equity,
            aspirational_target=self.aspirational_target,
            required_multiple=self.required_multiple,
            achieved_multiple=current_equity / self.starting_capital,
            remaining_gap=self.aspirational_target - current_equity,
        )
