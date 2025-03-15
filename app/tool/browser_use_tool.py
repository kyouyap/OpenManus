import asyncio
import json

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.tool.base import BaseTool, ToolResult

MAX_LENGTH = 2000

_BROWSER_DESCRIPTION = """
ブラウザを操作して、ナビゲーション、要素の操作、コンテンツの抽出、タブ管理などの様々なアクションを実行します。
サポートされているアクション:
- 'navigate': 指定したURLに移動
- 'click': インデックス指定で要素をクリック
- 'input_text': 要素にテキストを入力
- 'screenshot': スクリーンショットを撮影
- 'get_html': ページのHTML内容を取得
- 'get_text': ページのテキスト内容を取得
- 'read_links': ページ内のすべてのリンクを取得
- 'execute_js': JavaScriptコードを実行
- 'scroll': ページをスクロール
- 'switch_tab': 指定したタブに切り替え
- 'new_tab': 新しいタブを開く
- 'close_tab': 現在のタブを閉じる
- 'refresh': 現在のページを更新
"""


class BrowserUseTool(BaseTool):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate",
                    "click",
                    "input_text",
                    "screenshot",
                    "get_html",
                    "get_text",
                    "execute_js",
                    "scroll",
                    "switch_tab",
                    "new_tab",
                    "close_tab",
                    "refresh",
                ],
                "description": "実行するブラウザアクション",
            },
            "url": {
                "type": "string",
                "description": "'navigate'または'new_tab'アクション用のURL",
            },
            "index": {
                "type": "integer",
                "description": "'click'または'input_text'アクション用の要素インデックス",
            },
            "text": {
                "type": "string",
                "description": "'input_text'アクション用のテキスト",
            },
            "script": {
                "type": "string",
                "description": "'execute_js'アクション用のJavaScriptコード",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "'scroll'アクション用のスクロールピクセル数（正の値で下、負の値で上）",
            },
            "tab_id": {
                "type": "integer",
                "description": "'switch_tab'アクション用のタブID",
            },
        },
        "required": ["action"],
        "dependencies": {
            "navigate": ["url"],
            "click": ["index"],
            "input_text": ["index", "text"],
            "execute_js": ["script"],
            "switch_tab": ["tab_id"],
            "new_tab": ["url"],
            "scroll": ["scroll_amount"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: BrowserUseBrowser | None = Field(default=None, exclude=True)
    context: BrowserContext | None = Field(default=None, exclude=True)
    dom_service: DomService | None = Field(default=None, exclude=True)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """ブラウザとコンテキストが初期化されていることを確認します。"""
        if self.browser is None:
            browser_config_kwargs = {"headless": False, "disable_security": True}

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # handle proxy settings.
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()

            # if there is context config in the config, use it.
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def execute(
        self,
        action: str,
        url: str | None = None,
        index: int | None = None,
        text: str | None = None,
        script: str | None = None,
        scroll_amount: int | None = None,
        tab_id: int | None = None,
        **kwargs,
    ) -> ToolResult:
        """指定されたブラウザアクションを実行します。

        引数:
            action: 実行するブラウザアクション
            url: ナビゲーションまたは新規タブ用のURL
            index: クリックまたは入力アクション用の要素インデックス
            text: 入力アクション用のテキスト
            script: 実行するJavaScriptコード
            scroll_amount: スクロールアクション用のピクセル数
            tab_id: switch_tabアクション用のタブID
            **kwargs: 追加の引数

        戻り値:
            アクションの出力またはエラーを含むToolResult
        """
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()

                if action == "navigate":
                    if not url:
                        return ToolResult(error="URL is required for 'navigate' action")
                    await context.navigate_to(url)
                    return ToolResult(output=f"Navigated to {url}")

                if action == "click":
                    if index is None:
                        return ToolResult(error="Index is required for 'click' action")
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    download_path = await context._click_element_node(element)
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                if action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                if action == "screenshot":
                    screenshot = await context.take_screenshot(full_page=True)
                    return ToolResult(
                        output=f"Screenshot captured (base64 length: {len(screenshot)})",
                        system=screenshot,
                    )

                if action == "get_html":
                    html = await context.get_page_html()
                    truncated = (
                        html[:MAX_LENGTH] + "..." if len(html) > MAX_LENGTH else html
                    )
                    return ToolResult(output=truncated)

                if action == "get_text":
                    text = await context.execute_javascript("document.body.innerText")
                    return ToolResult(output=text)

                if action == "read_links":
                    links = await context.execute_javascript(
                        "document.querySelectorAll('a[href]').forEach((elem) => {if (elem.innerText) {console.log(elem.innerText, elem.href)}})"
                    )
                    return ToolResult(output=links)

                if action == "execute_js":
                    if not script:
                        return ToolResult(
                            error="Script is required for 'execute_js' action"
                        )
                    result = await context.execute_javascript(script)
                    return ToolResult(output=str(result))

                if action == "scroll":
                    if scroll_amount is None:
                        return ToolResult(
                            error="Scroll amount is required for 'scroll' action"
                        )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {scroll_amount});"
                    )
                    direction = "down" if scroll_amount > 0 else "up"
                    return ToolResult(
                        output=f"Scrolled {direction} by {abs(scroll_amount)} pixels"
                    )

                if action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    return ToolResult(output=f"Switched to tab {tab_id}")

                if action == "new_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'new_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with URL {url}")

                if action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                if action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {e!s}")

    async def get_current_state(self) -> ToolResult:
        """現在のブラウザの状態をToolResultとして取得します。"""
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()
                state = await context.get_state()
                state_info = {
                    "url": state.url,
                    "title": state.title,
                    "tabs": [tab.model_dump() for tab in state.tabs],
                    "interactive_elements": state.element_tree.clickable_elements_to_string(),
                }
                return ToolResult(output=json.dumps(state_info))
            except Exception as e:
                return ToolResult(error=f"Failed to get browser state: {e!s}")

    async def cleanup(self):
        """ブラウザリソースをクリーンアップします。"""
        async with self.lock:
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    def __del__(self):
        """オブジェクトが破棄される時にクリーンアップを確実に実行します。"""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()
