import asyncio
import logging
from typing import Type, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from browser_use.agent.service import BrowserContext
from browser_use.browser.browser import Browser
from browser_use.controller.service import Controller
from browser_use.dom.service import DomService
import tiktoken 
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
# ----------------------------------------------------
# 1) ヘルパー: DOM情報を取得して返す
# ----------------------------------------------------

async def get_page_dom_info(
    ctx: BrowserContext, highlight_elements: bool = False, max_length: int = 3000
) -> str:
    """現在のページ状態を更新後、clickable elementsをテキスト化して返す"""
    # ページの state を更新
    await ctx.get_state()

    # DomService で DOM 情報を取得
    dom_service = DomService(page=await ctx.get_current_page())
    dom_state = await dom_service.get_clickable_elements(
        highlight_elements=highlight_elements
    )

    # clickable_elements_to_string()でインデックスつきのテキスト生成
    dom_text = dom_state.element_tree.clickable_elements_to_string()

    # 長すぎる場合は先頭の一部だけ返す
    if len(dom_text) > max_length:
        dom_text = dom_text[:max_length] + "...(truncated)"
    return dom_text

# ----------------------------------------------------
# 2) 各Tool定義 (共通BrowserUseManagerを参照)
# ----------------------------------------------------
from browser_use.controller.registry.views import ActionModel
from browser_use.agent.views import ActionResult
from browser_use.controller.views import (
    SearchGoogleAction,
    GoToUrlAction,
    ClickElementAction,
    InputTextAction,
    SwitchTabAction,
    OpenTabAction,
    ExtractPageContentAction,
    DoneAction,
    ScrollAction,
    SendKeysAction,
)

# ----------------------------------------------------
# 3) BrowserUseManager (Controller/BrowserContextまとめ)
# ----------------------------------------------------
class BrowserUseManager:
    def __init__(self):
        # まとめて初期化
        self.controller = Controller()
        self.browser = Browser()
        self.browser_context = BrowserContext(browser=self.browser)
        self.DynamicActionModel = self.controller.registry.create_action_model()
        # 履歴を保持するリスト
        self.history = []
        # トークナイザーの初期化（使用しているモデルに合わせて設定）
        self.tokenizer = tiktoken.get_encoding("p50k_base")  # 例としてGPT-4を使用
        # トークン数の上限
        self.max_tokens = 100000
         # Executorの初期化
        self.executor = ThreadPoolExecutor(max_workers=4)  # 必要に応じて調整


    async def shutdown(self):
        """後始末用: browser_contextやbrowserを閉じる。"""
        # BrowserContext をclose
        if self.browser_context and self.browser_context.session is not None:
            await self.browser_context.close()
        # Browser自体をclose
        if self.browser:
            await self.browser.close()

    def add_history(self, action: str, result: str):
        """履歴にアクションと結果を追加する"""
        self.history.append({"action": action, "result": result})

    async def get_full_history(self) -> str:
        """これまでの履歴をテキスト形式で取得する"""
        history_text = ""
        for idx, entry in enumerate(self.history, start=1):
            history_text += f"{idx}. アクション: {entry['action']}\n   結果: {entry['result']}\n"
        #print(f"\n\n history_text: {history_text}")
        return history_text
    
    def count_tokens_sync(self, text: str) -> int:
        """同期的にトークン数を計測"""
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    async def count_tokens_async(self, text: str) -> int:
        """非同期的にトークン数を計測"""
        loop = asyncio.get_event_loop()
        tokens = await loop.run_in_executor(self.executor, self.tokenizer.encode, text)
        return len(tokens)
    
    async def execute_action(self, action_name: str, action_model: ActionModel, action_description: str) -> str:
        """共通アクション実行ロジック"""
        try:
            # アクションを実行し、結果を取得
            result: ActionResult = await self.controller.act(action_model, self.browser_context)
            if result.error:
                action_result = f"Error: {result.error}"
            else:
                action_result = result.extracted_content or "No content extracted."

            # 履歴に追加
            self.add_history(action_description, action_result)
            action_history = await self.get_full_history()

            # DOM情報を取得
            dom_text = await get_page_dom_info(self.browser_context)

            # トークン数を計測
            history_tokens = self.count_tokens_sync(action_history)
            dom_tokens = self.count_tokens_sync(dom_text)
            action_result_tokens = self.count_tokens_sync(action_result)

            # トークン数の合計計算
            total_tokens = (
                history_tokens +
                dom_tokens +
                action_result_tokens +
                len(self.tokenizer.encode("\nCurrent page clickable elements:\n"))
            )

            # トークンが不足する場合の処理
            if total_tokens > self.max_tokens:
                # 必要なトークン数を計算
                excess_tokens = total_tokens - self.max_tokens

                if action_result_tokens > excess_tokens:
                    # `action_result`を削除またはトリミング
                    truncated_action_result_tokens = self.tokenizer.encode(action_result)[
                        :-excess_tokens
                    ]
                    action_result = self.tokenizer.decode(truncated_action_result_tokens)
                else:
                    # `action_result`を完全に削除
                    action_result = "[TRUNCATED DUE TO TOKEN LIMIT]"
                    excess_tokens -= action_result_tokens

                # DOM情報の削除（残りのトークン数が不足する場合）
                if dom_tokens > excess_tokens:
                    truncated_dom_tokens = self.tokenizer.encode(dom_text)[:-excess_tokens]
                    dom_text = self.tokenizer.decode(truncated_dom_tokens)
                else:
                    dom_text = "[TRUNCATED DUE TO TOKEN LIMIT]"

            # 最終的な履歴テキスト
            action_history_with_current_page_dom = (
                f"{action_history}\nAction result:\n{action_result}\n"
                f"Current page clickable elements:\n{dom_text}"
            )
            await asyncio.sleep(0.1)

            return action_history_with_current_page_dom
        except Exception as e:
            error_message = f"Exception in {action_name}: {e}"
            self.add_history(action_description, error_message)
            return error_message

# 0引数アクション用クラス
class GoBackAction(BaseModel):
    pass

class ScrollToTextAction(BaseModel):
    text: str = Field(..., description="The text to scroll into view")

class GetDropdownOptionsAction(BaseModel):
    index: int = Field(..., description="The highlight index of the <select> element")

class SelectDropdownOptionAction(BaseModel):
    index: int = Field(..., description="The highlight index of the <select> element")
    text: str = Field(..., description="The exact option text to select")

# ここで、グローバルに1つの manager を作る (実運用ではシングルトン等にする場合もある)
manager = BrowserUseManager()

class SearchGoogleInput(BaseModel):
    query: str = Field(..., description="Query string to search in Google")

class SearchGoogleTool(BaseTool):
    name: str = "search_google"
    description: str = "Search Google with a query string. Input: {'query': '<string>'}"
    args_schema: Type[BaseModel] = SearchGoogleInput

    def _run(self, query: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, query: str) -> str:
        action_description = f"search_google(query={query})"
        action_model = manager.DynamicActionModel(search_google=SearchGoogleAction(query=query))  # type: ignore
        return await manager.execute_action("search_google", action_model, action_description)
    

class SearchAmazonInput(BaseModel):
    query: str = Field(..., description="Query string to search in Amazon")

class SearchAmazonTool(BaseTool):
    name: str = "search_amazon"
    description: str = "Search Amazon with a query string. Input: {'query': '<string>'}"
    args_schema: Type[BaseModel] = SearchAmazonInput

    def _run(self, query: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, query: str) -> str:
        url = "https://www.amazon.co.jp/s?k=" + query
        action_description = f"go_to_url({url})"
        action_model = manager.DynamicActionModel(go_to_url=GoToUrlAction(url=url))   # type: ignore
        return await manager.execute_action("go_to_url", action_model, action_description)

class GoToUrlInput(BaseModel):
    url: str

class GoToUrlTool(BaseTool):
    name: str = "go_to_url"
    description: str = "Navigate to a specified URL in the current tab. Input: {'url': '<string>'}"
    args_schema: Type[BaseModel] = GoToUrlInput

    def _run(self, url: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, url: str) -> str:
        action_description = f"go_to_url(url={url})"
        action_model = manager.DynamicActionModel(go_to_url=GoToUrlAction(url=url))  # type: ignore
        return await manager.execute_action("go_to_url", action_model, action_description)

class GoBackTool(BaseTool):
    name: str = "go_back"
    description: str = "Go back to the previous page."
    args_schema: Type[BaseModel] = GoBackAction

    def _run(self) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self) -> str:
        action_description = "go_back()"
        action_model = manager.DynamicActionModel(go_back=GoBackAction())  # type: ignore
        return await manager.execute_action("go_back", action_model, action_description)

class ClickElementInput(BaseModel):
    index: int = Field(..., description="The highlight index")

class ClickElementTool(BaseTool):
    name: str = "click_element"
    description: str = "Click the element with the given highlight index."
    args_schema: Type[BaseModel] = ClickElementInput

    def _run(self, index: int) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, index: int) -> str:
        action_description = f"click_element(index={index})"
        action_model = manager.DynamicActionModel(click_element=ClickElementAction(index=index))  # type: ignore
        return await manager.execute_action("click_element", action_model, action_description)

class InputTextInput(BaseModel):
    index: int
    text: str

class InputTextTool(BaseTool):
    name: str = "input_text"
    description:str =  "Input text into a element with a given highlight index."
    args_schema: Type[BaseModel] = InputTextInput

    def _run(self, index: int, text: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, index: int, text: str) -> str:
        action_description = f"input_text(index={index}, text={text})"
        action_model = manager.DynamicActionModel(input_text=InputTextAction(index=index, text=text))  # type: ignore
        return await manager.execute_action("input_text", action_model, action_description)

class SwitchTabInput(BaseModel):
    page_id: int

class SwitchTabTool(BaseTool):
    name: str = "switch_tab"
    description: str = "Switch to a tab with a given page_id."
    args_schema: Type[BaseModel] = SwitchTabInput

    def _run(self, page_id: int) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, page_id: int) -> str:
        action_description = f"switch_tab(page_id={page_id})"
        action_model = manager.DynamicActionModel(switch_tab=SwitchTabAction(page_id=page_id))  # type: ignore
        return await manager.execute_action("switch_tab", action_model, action_description)

class OpenTabInput(BaseModel):
    url: Optional[str] = None

class OpenTabTool(BaseTool):
    name: str = "open_tab"
    description:str =  "Open a new tab and optionally go to a URL."
    args_schema: Type[BaseModel] = OpenTabInput

    def _run(self, url: Optional[str] = None) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, url: Optional[str] = None) -> str:
        action_description = f"open_tab(url={url})"
        action_model = manager.DynamicActionModel(open_tab=OpenTabAction(url=url or ""))  # type: ignore
        return await manager.execute_action("open_tab", action_model, action_description)

class ExtractContentInput(BaseModel):
    value: str = Field("text", description="One of 'text','markdown','html'")

class ExtractContentTool(BaseTool):
    name: str = "extract_content"
    description: str = "Extract page content as text/markdown/html."
    args_schema: Type[BaseModel] = ExtractContentInput

    def _run(self, value: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, value: str) -> str:
        action_description = f"extract_content(value={value})"
        action_model = manager.DynamicActionModel(
            extract_content=ExtractPageContentAction(value=value)  # type: ignore
        )
        return await manager.execute_action("extract_content", action_model, action_description)

class DoneInput(BaseModel):
    text: str

class DoneTool(BaseTool):
    name: str = "done"
    description: str = "Signal the task is done."
    args_schema: Type[BaseModel] = DoneInput

    def _run(self, text: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, text: str) -> str:
        action_description = f"done(text={text})"
        action_model = manager.DynamicActionModel(done=DoneAction(text=text))  # type: ignore
        return await manager.execute_action("done", action_model, action_description)

class ScrollDownInput(BaseModel):
    amount: Optional[int] = None

class ScrollDownTool(BaseTool):
    name: str = "scroll_down"
    description: str = "Scroll down the page."
    args_schema: Type[BaseModel] = ScrollDownInput

    def _run(self, amount: Optional[int] = None) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, amount: Optional[int] = None) -> str:
        action_description = f"scroll_down(amount={amount})"
        action_model = manager.DynamicActionModel(scroll_down=ScrollAction(amount=amount))  # type: ignore
        return await manager.execute_action("scroll_down", action_model, action_description)

class ScrollUpInput(BaseModel):
    amount: Optional[int] = None

class ScrollUpTool(BaseTool):
    name: str = "scroll_up"
    description: str = "Scroll up the page."
    args_schema: Type[BaseModel] = ScrollUpInput

    def _run(self, amount: Optional[int] = None) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, amount: Optional[int] = None) -> str:
        action_description = f"scroll_up(amount={amount})"
        action_model = manager.DynamicActionModel(scroll_up=ScrollAction(amount=amount))  # type: ignore
        return await manager.execute_action("scroll_up", action_model, action_description)

class SendKeysInput(BaseModel):
    keys: str

class SendKeysTool(BaseTool):
    name: str = "send_keys"
    description: str = "Send special keys or keyboard shortcuts."
    args_schema: Type[BaseModel] = SendKeysInput

    def _run(self, keys: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, keys: str) -> str:
        action_description = f"send_keys(keys={keys})"
        action_model = manager.DynamicActionModel(send_keys=SendKeysAction(keys=keys))  # type: ignore
        return await manager.execute_action("send_keys", action_model, action_description)

class ScrollToTextInput(BaseModel):
    text: str

class ScrollToTextTool(BaseTool):
    name: str = "scroll_to_text"
    description: str = "Scroll until the specified text is in view."
    args_schema: Type[BaseModel] = ScrollToTextInput

    def _run(self, text: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, text: str) -> str:
        action_description = f"scroll_to_text(text={text})"
        action_model = manager.DynamicActionModel(scroll_to_text=ScrollToTextAction(text=text))  # type: ignore
        return await manager.execute_action("scroll_to_text", action_model, action_description)

class GetDropdownOptionsInput(BaseModel):
    index: int

class GetDropdownOptionsTool(BaseTool):
    name: str = "get_dropdown_options"
    description: str = "Get all options from a native dropdown."
    args_schema: Type[BaseModel] = GetDropdownOptionsInput

    def _run(self, index: int) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, index: int) -> str:
        action_description = f"get_dropdown_options(index={index})"
        action_model = manager.DynamicActionModel(get_dropdown_options=GetDropdownOptionsAction(index=index))  # type: ignore
        return await manager.execute_action("get_dropdown_options", action_model, action_description)

class SelectDropdownOptionInput(BaseModel):
    index: int
    text: str

class SelectDropdownOptionTool(BaseTool):
    name: str = "select_dropdown_option"
    description: str = "Select an option in a dropdown by text."
    args_schema: Type[BaseModel] = SelectDropdownOptionInput

    def _run(self, index: int, text: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, index: int, text: str) -> str:
        action_description = f"select_dropdown_option(index={index}, text={text})"
        action_model = manager.DynamicActionModel(
            select_dropdown_option=SelectDropdownOptionAction(index=index, text=text)  # type: ignore
        )
        return await manager.execute_action("select_dropdown_option", action_model, action_description)