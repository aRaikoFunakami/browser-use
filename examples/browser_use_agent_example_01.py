import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio




async def main():
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
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model="gpt-4o"),
    )
    result = await agent.run()
    print(result)

asyncio.run(main())