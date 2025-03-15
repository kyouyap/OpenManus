import threading
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """プロジェクトのルートディレクトリを取得します"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


class LLMSettings(BaseModel):
    model: str = Field(..., description="モデル名")
    base_url: str = Field(..., description="APIのベースURL")
    api_key: str = Field(..., description="APIキー")
    max_tokens: int = Field(4096, description="リクエストごとの最大トークン数")
    max_input_tokens: int | None = Field(
        None,
        description="全リクエストで使用する最大入力トークン数（無制限の場合はNone）",
    )
    temperature: float = Field(1.0, description="サンプリング温度")
    api_type: str = Field(..., description="AzureOpenaiまたはOpenai")
    api_version: str = Field(..., description="AzureOpenaiの場合のAPIバージョン")


class ProxySettings(BaseModel):
    server: str = Field(None, description="プロキシサーバーのアドレス")
    username: str | None = Field(None, description="プロキシのユーザー名")
    password: str | None = Field(None, description="プロキシのパスワード")


class SearchSettings(BaseModel):
    engine: str = Field(default="Google", description="LLMが使用する検索エンジン")


class BrowserSettings(BaseModel):
    headless: bool = Field(
        False, description="ブラウザをヘッドレスモードで実行するかどうか"
    )
    disable_security: bool = Field(
        True, description="ブラウザのセキュリティ機能を無効化する"
    )
    extra_chromium_args: list[str] = Field(
        default_factory=list, description="ブラウザに渡す追加の引数"
    )
    chrome_instance_path: str | None = Field(
        None, description="使用するChromeインスタンスのパス"
    )
    wss_url: str | None = Field(
        None, description="WebSocketを介してブラウザインスタンスに接続"
    )
    cdp_url: str | None = Field(
        None, description="CDPを介してブラウザインスタンスに接続"
    )
    proxy: ProxySettings | None = Field(None, description="ブラウザのプロキシ設定")


class AppConfig(BaseModel):
    llm: dict[str, LLMSettings]
    browser_config: BrowserSettings | None = Field(None, description="ブラウザの設定")
    search_config: SearchSettings | None = Field(None, description="検索の設定")

    class Config:
        arbitrary_types_allowed = True


class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("設定ディレクトリに設定ファイルが見つかりません")

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }

        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "max_input_tokens": base_llm.get("max_input_tokens"),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", ""),
            "api_version": base_llm.get("api_version", ""),
        }

        # ブラウザ設定の処理
        browser_config = raw_config.get("browser", {})
        browser_settings = None

        if browser_config:
            # プロキシ設定の処理
            proxy_config = browser_config.get("proxy", {})
            proxy_settings = None

            if proxy_config and proxy_config.get("server"):
                proxy_settings = ProxySettings(
                    **{
                        k: v
                        for k, v in proxy_config.items()
                        if k in ["server", "username", "password"] and v
                    }
                )

            # 有効なブラウザ設定パラメータをフィルタリング
            valid_browser_params = {
                k: v
                for k, v in browser_config.items()
                if k in BrowserSettings.__annotations__ and v is not None
            }

            # プロキシ設定がある場合、パラメータに追加
            if proxy_settings:
                valid_browser_params["proxy"] = proxy_settings

            # 有効なパラメータがある場合のみBrowserSettingsを作成
            if valid_browser_params:
                browser_settings = BrowserSettings(**valid_browser_params)

        search_config = raw_config.get("search", {})
        search_settings = None
        if search_config:
            search_settings = SearchSettings(**search_config)

        config_dict = {
            "llm": {
                "default": default_settings,
                **{
                    name: {**default_settings, **override_config}
                    for name, override_config in llm_overrides.items()
                },
            },
            "browser_config": browser_settings,
            "search_config": search_settings,
        }

        self._config = AppConfig(**config_dict)

    @property
    def llm(self) -> dict[str, LLMSettings]:
        return self._config.llm

    @property
    def browser_config(self) -> BrowserSettings | None:
        return self._config.browser_config

    @property
    def search_config(self) -> SearchSettings | None:
        return self._config.search_config


config = Config()
