import streamlit as st
import os
import json
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
import logging

# ロガーの設定
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# .envファイルから環境変数をロード
load_dotenv()

# 環境変数からAPIキーを取得
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# mcpの設定を読み込む
with open("mcp_config.json", "r") as f:
    config = json.load(f)

system_prompt = r"""
あなたは高度に訓練された AI チャットアシスタントです。  
## 目的 / Purpose
- ユーザーの質問に **まず LLM の内蔵知識** を用いて回答する。  
- 回答が不完全・不確実・最新情報が必要と判断した場合のみ、**MCP 経由でローカルファイルや Web を検索して補足情報を取得** し、その結果を反映して回答する。  
- 取得した情報は **簡潔に要約** し、必要に応じてソース名・URL 等をカッコ書きで示す（過度な引用は避ける）。

## 応答スタイル / Style
1. **日本語を既定** とし、ユーザーが英語で質問した場合のみ英語で返答。  
2. 回答の最後に必ず **1 行のフォローアップ質問** を置き、対話を促進する。

## 情報取得ポリシー / Retrieval Policy
- 外部情報の利用を示すときは「\[情報源: …\]」の形式で記載。URL を貼る場合は公式サイトや一次ソースを優先。
- 検索結果はデータベースにキャッシュされ、同じ質問に対して再度検索する場合は、まずキャッシュを確認します。

## システム動作のヒント（LangGraph/LangChain 実装向け）
- あなたはPlaywrightというツールを用いてブラウザを操作できます
- Filesystemからはローカルに存在するファイルを読み取ることができます
- 必要に応じてこれらを使い分けてください
- ユーザの質問からツールをどういう意図で何回利用しないといけないのかを判断し、必要なら複数回toolを利用して情報収集をしたのち、すべての情報が取得できたら、その情報を元に返答してください。
- なお、サイトのアクセスでエラーが出た場合は、もう一度再施行してください。ネットワーク関連のエラーの場合があります。
- 検索結果は自動的にデータベースに保存され、同じクエリに対しては一定期間キャッシュが利用されます。

あなたの最終目標は **ユーザーが欲しい情報を素早く、正確かつ理解しやすい形で提供すること** である。
"""

def create_prompt_from_system_and_messages(system_prompt: str, messages: List) -> str:
    # ChatPromptTemplateの作成
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),  # システムプロンプト
        MessagesPlaceholder("messages")  # チャット履歴
    ])
    
    return chat_prompt

async def main():
    st.title("MCPチャット")

    # セッション状態の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 過去のメッセージを表示
    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    # ユーザー入力
    user_input = st.chat_input("メッセージを入力してください")

    # ユーザーがメッセージを送信した場合
    if user_input:
        # ユーザーメッセージをセッションに追加して表示
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # APIキーが環境変数に設定されているか確認
        if OPENAI_API_KEY:
            # AIの応答を生成
            with st.chat_message("assistant"):
                with st.spinner("考え中..."):
                    # LangChainでChatGPTを呼び出す
                    llm = ChatOpenAI(
                        model=OPENAI_MODEL,
                        temperature=0.7,
                        openai_api_key=OPENAI_API_KEY
                    )

                    async with MultiServerMCPClient(config["mcpServers"]) as mcp_client:
                        # toolsの読み込み
                        tools = mcp_client.get_tools()

                        # 空のコンテンツを持つメッセージを除外
                        filtered_messages = [msg for msg in st.session_state.messages if msg.content]
                        
                        # tool callingがなくなるまで呼び出し(最大10回)
                        max_iterations = 10
                        count = 0
                        while count < max_iterations:
                            count += 1
                            llm_with_tools = create_prompt_from_system_and_messages(system_prompt, filtered_messages) | llm.bind_tools(tools)
                            response = await llm_with_tools.ainvoke({"messages": filtered_messages})
                            logging.info(response)
                            filtered_messages.append(response)
                            st.markdown(response.content)

                            if response.tool_calls:
                                for tool_call in response.tool_calls:
                                    selected_tool = {tool.name.lower(): tool for tool in tools}[
                                        tool_call["name"].lower()
                                    ]
                                    tool_msg = await selected_tool.ainvoke(tool_call)
                                    # ツール呼び出し結果をメッセージとして追加
                                    filtered_messages.append(tool_msg)
                            else:
                                logging.info(filtered_messages)
                                # 元のメッセージリストを更新
                                st.session_state.messages = filtered_messages
                                break
        else:
            st.error("API keyが環境変数に設定されていません。.envファイルを確認してください。")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())