"""
LangChain用のBrowserUse Tools一式を管理するサンプル。

[構成]
1) BrowserUseManager: controllerやbrowser, browser_context, DynamicActionModelをまとめて保持
2) 各Tool: BrowserUseManagerのインスタンスを利用して実装
3) メイン(main)でエージェントを作り、各Toolをまとめて使用する
"""

import logging
from typing import Type, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

# ----------------------------------------------------
# 1) BrowserUseManager (Controller/BrowserContextまとめ)
# ----------------------------------------------------
from browser_use.agent.service import BrowserContext
from browser_use.browser.browser import Browser
from browser_use.controller.service import Controller

logger = logging.getLogger(__name__)

class BrowserUseManager:
    def __init__(self):
        # まとめて初期化
        self.controller = Controller()
        self.browser = Browser()
        self.browser_context = BrowserContext(browser=self.browser)
        self.DynamicActionModel = self.controller.registry.create_action_model()

    async def shutdown(self):
        """後始末用: browser_contextやbrowserを閉じる。"""
        # BrowserContext をclose
        if self.browser_context and self.browser_context.session is not None:
            await self.browser_context.close()
        # Browser自体をclose
        if self.browser:
            await self.browser.close()


# ----------------------------------------------------
# 2) ヘルパー: DOM情報を取得して返す
# ----------------------------------------------------
from browser_use.dom.service import DomService


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

    # clickable_elements_to_string() でインデックスつきのテキスト生成
    dom_text = dom_state.element_tree.clickable_elements_to_string()

    # 長すぎる場合は先頭の一部だけ返す
    if len(dom_text) > max_length:
        dom_text = dom_text[:max_length] + "...(truncated)"
    return dom_text


# ----------------------------------------------------
# 3) 各Tool定義 (共通BrowserUseManagerを参照)
# ----------------------------------------------------
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


from browser_use.controller.registry.views import ActionModel


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
        try:
            # ActionModel作成
            action_model = manager.DynamicActionModel(search_google=SearchGoogleAction(query=query))  # type: ignore
            # 実行
            result: ActionResult = await manager.controller.act(
                action_model, manager.browser_context
            )
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "No content extracted."
            # DOM情報
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"

        except Exception as e:
            return f"Exception in search_google: {e}"


class GoToUrlInput(BaseModel):
    url: str


class GoToUrlTool(BaseTool):
    name: str = "go_to_url"
    description: str = (
        "Navigate to a specified URL in the current tab. Input: {'url': '<string>'}"
    )
    args_schema: Type[BaseModel] = GoToUrlInput

    def _run(self, url: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, url: str) -> str:
        try:
            action_model = manager.DynamicActionModel(go_to_url=GoToUrlAction(url=url))  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "No content extracted."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in go_to_url: {e}"


class GoBackTool(BaseTool):
    name: str = "go_back"
    description: str = "Go back to the previous page."
    args_schema: Type[BaseModel] = GoBackAction

    def _run(self) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self) -> str:
        try:
            action_model = ActionModel(go_back=GoBackAction())  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "Back done."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in go_back: {e}"


class ClickElementInput(BaseModel):
    index: int = Field(..., description="The highlight index")


class ClickElementTool(BaseTool):
    name: str = "click_element"
    description: str = "Click the element with the given highlight index."
    args_schema: Type[BaseModel] = ClickElementInput

    def _run(self, index: int) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, index: int) -> str:
        try:
            action_model = manager.DynamicActionModel(click_element=ClickElementAction(index=index))  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "Clicked element."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in click_element: {e}"


class InputTextInput(BaseModel):
    index: int
    text: str


class InputTextTool(BaseTool):
    name: str = "input_text"
    description: str = "Input text into a element with a given highlight index."
    args_schema: Type[BaseModel] = InputTextInput

    def _run(self, index: int, text: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, index: int, text: str) -> str:
        try:
            action_model = manager.DynamicActionModel(input_text=InputTextAction(index=index, text=text))  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "Text input done."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in input_text: {e}"


class SwitchTabInput(BaseModel):
    page_id: int


class SwitchTabTool(BaseTool):
    name: str = "switch_tab"
    description: str = "Switch to a tab with a given page_id."
    args_schema: Type[BaseModel] = SwitchTabInput

    def _run(self, page_id: int) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, page_id: int) -> str:
        try:
            action_model = manager.DynamicActionModel(switch_tab=SwitchTabAction(page_id=page_id))  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or f"Switched to tab {page_id}."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in switch_tab: {e}"


class OpenTabInput(BaseModel):
    url: Optional[str] = None


class OpenTabTool(BaseTool):
    name: str = "open_tab"
    description: str = "Open a new tab and optionally go to a URL."
    args_schema: Type[BaseModel] = OpenTabInput

    def _run(self, url: Optional[str] = None) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, url: Optional[str] = None) -> str:
        try:
            action_model = manager.DynamicActionModel(open_tab=OpenTabAction(url=url or ""))  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "New tab opened."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in open_tab: {e}"


class ExtractContentInput(BaseModel):
    value: str = Field("text", description="One of 'text','markdown','html'")


class ExtractContentTool(BaseTool):
    name: str = "extract_content"
    description: str = "Extract page content as text/markdown/html."
    args_schema: Type[BaseModel] = ExtractContentInput

    def _run(self, value: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, value: str) -> str:
        try:
            action_model = manager.DynamicActionModel(
                extract_content=ExtractPageContentAction(value=value)  # type: ignore
            )
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "No content extracted."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in extract_content: {e}"


class DoneInput(BaseModel):
    text: str


class DoneTool(BaseTool):
    name: str = "done"
    description: str = "Signal the task is done."
    args_schema: Type[BaseModel] = DoneInput

    def _run(self, text: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, text: str) -> str:
        try:
            action_model = manager.DynamicActionModel(done=DoneAction(text=text))  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "Task done."
            return f"{extracted_text}"
        except Exception as e:
            return f"Exception in done: {e}"


class ScrollDownInput(BaseModel):
    amount: Optional[int] = None


class ScrollDownTool(BaseTool):
    name: str = "scroll_down"
    description: str = "Scroll down the page."
    args_schema: Type[BaseModel] = ScrollDownInput

    def _run(self, amount: Optional[int] = None) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, amount: Optional[int] = None) -> str:
        try:
            action_model = manager.DynamicActionModel(scroll_down=ScrollAction(amount=amount))  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "Scrolled down."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in scroll_down: {e}"


class ScrollUpInput(BaseModel):
    amount: Optional[int] = None


class ScrollUpTool(BaseTool):
    name: str = "scroll_up"
    description: str = "Scroll up the page."
    args_schema: Type[BaseModel] = ScrollUpInput

    def _run(self, amount: Optional[int] = None) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, amount: Optional[int] = None) -> str:
        try:
            action_model = manager.DynamicActionModel(scroll_up=ScrollAction(amount=amount))  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "Scrolled up."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in scroll_up: {e}"


class SendKeysInput(BaseModel):
    keys: str


class SendKeysTool(BaseTool):
    name: str = "send_keys"
    description: str = "Send special keys or keyboard shortcuts."
    args_schema: Type[BaseModel] = SendKeysInput

    def _run(self, keys: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, keys: str) -> str:
        try:
            action_model = manager.DynamicActionModel(send_keys=SendKeysAction(keys=keys))  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or f"Sent keys: {keys}"
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in send_keys: {e}"


class ScrollToTextInput(BaseModel):
    text: str


class ScrollToTextTool(BaseTool):
    name: str = "scroll_to_text"
    description: str = "Scroll until the specified text is in view."
    args_schema: Type[BaseModel] = ScrollToTextInput

    def _run(self, text: str) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, text: str) -> str:
        try:
            action_model = manager.DynamicActionModel(scroll_to_text=ScrollToTextAction(text=text))  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "Scrolled to text."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in scroll_to_text: {e}"


class GetDropdownOptionsInput(BaseModel):
    index: int


class GetDropdownOptionsTool(BaseTool):
    name: str = "get_dropdown_options"
    description: str = "Get all options from a native dropdown."
    args_schema: Type[BaseModel] = GetDropdownOptionsInput

    def _run(self, index: int) -> str:
        raise RuntimeError("Use _arun instead.")

    async def _arun(self, index: int) -> str:
        try:
            action_model = manager.DynamicActionModel(get_dropdown_options=GetDropdownOptionsAction(index=index))  # type: ignore
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "No dropdown options found."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in get_dropdown_options: {e}"


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
        try:
            action_model = manager.DynamicActionModel(
                select_dropdown_option=SelectDropdownOptionAction(index=index, text=text)  # type: ignore
            )
            result = await manager.controller.act(action_model, manager.browser_context)
            if result.error:
                return f"Error: {result.error}"

            extracted_text = result.extracted_content or "Dropdown option selected."
            dom_text = await get_page_dom_info(manager.browser_context)
            return f"{extracted_text}\nCurrent page clickable elements:\n{dom_text}"
        except Exception as e:
            return f"Exception in select_dropdown_option: {e}"
