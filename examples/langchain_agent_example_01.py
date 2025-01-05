import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import List

# LangChain関連
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain import hub

# browser_use関連のToolをまとめてインポート
# ※ ここで、browser_use.controller.langchain_tools 内に下記すべてのToolが登録されている想定
from langchain_tools import (
    manager,
    SearchGoogleTool,
    SearchAmazonTool,
    GoToUrlTool,
    GoBackTool,
    ClickElementTool,
    InputTextTool,
    SwitchTabTool,
    OpenTabTool,
    ExtractContentTool,
    DoneTool,
    ScrollDownTool,
    ScrollUpTool,
    SendKeysTool,
    ScrollToTextTool,
    GetDropdownOptionsTool,
    SelectDropdownOptionTool,
)


async def main():
    # 1) LLMを用意
    model = ChatOpenAI(
        temperature=0,
        model="gpt-4o",  # 例えば gpt-4
    )

    # 2) すべてのToolsをリストに格納
    tools = [
        SearchGoogleTool(),
        SearchAmazonTool(),
        GoToUrlTool(),
        GoBackTool(),
        ClickElementTool(),
        InputTextTool(),
        SwitchTabTool(),
        OpenTabTool(),
        ExtractContentTool(),
        DoneTool(),
        ScrollDownTool(),
        ScrollUpTool(),
        SendKeysTool(),
        ScrollToTextTool(),
        GetDropdownOptionsTool(),
        SelectDropdownOptionTool(),
    ]

    # 3) プロンプトを取得 (OpenAI Functionsエージェント用)
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # 4) create_tool_calling_agent でAgent作成
    agent = create_tool_calling_agent(model, tools, prompt)

    # 5) AgentExecutor作成
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    # 6) タスク(プロンプト)を定義
    task = """
    あなたは価格監視のエージェントです。
    5000円以下で10000mA以上のモバイルバッテリーを探してください。

    下記の形式でデータを教えてほしい
    - 価格
    - 送料(なければ0円)
    - クーポン(なければ0円)
    - ポイント(なければ0円)
    - ショップ名
    """

    # 7) AgentExecutorのainvoke() で非同期呼び出し
    response = await agent_executor.ainvoke({"input": task})
    print("Response:", response)
    if "output" in response:
        print("Final output:\n", response["output"])

    await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
