import streamlit as st
import os
import json
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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

def display_human_message(message: HumanMessage):
    st.markdown(message.content)

def display_ai_message(message: AIMessage):
    st.markdown(message.content)
    # ツール呼び出しがある場合は表示
    if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
        tool_calls = message.additional_kwargs['tool_calls']
        if tool_calls:
            st.info(f"💡 アシスタントがツールを使用しました（{len(tool_calls)}個）")

def display_tool_message(message: ToolMessage):
    # エクスパンダー（デフォルトで閉じた状態）
    with st.expander(f"🔧 ツール実行結果 ({message.tool_call_id})", expanded=False):
        st.code(message.content, language="json")

# メッセージ表示関数
def display_messages(messages):
    for message in messages:
        # メッセージのタイプを判断
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                display_human_message(message)
        
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                display_ai_message(message)
        
        # ツールメッセージをエクスパンダーで表示
        elif isinstance(message, ToolMessage):
            with st.chat_message("assistant"):
                display_tool_message(message)

def process_tool_response(tool_message: ToolMessage) -> ToolMessage:
    raw_content = tool_message.content
    # HTMLが含まれているかチェック
    if isinstance(raw_content, str) and ("<html" in raw_content.lower() or "<!doctype" in raw_content.lower()):
        # BeautifulSoupなどを使ってHTMLから必要なテキストだけを抽出
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw_content, 'html.parser')
        
        # 不要なタグを削除（スクリプト、スタイルなど）
        for tag in soup(['script', 'style', 'meta', 'link', 'svg', 'path']):
            tag.decompose()
        
        # テキストのみを抽出
        text = soup.get_text(separator='\n', strip=True)
        
        # さらにテキストを要約または切り詰め
        if len(text) > 5000:  # 適切な長さに調整
            text = text[:5000] + "...(省略されました)"
        
        return ToolMessage(content=text, tool_call_id=tool_message.tool_call_id)
    return tool_message

async def main():
    st.title("MCPチャット")

    # セッション状態の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 過去のメッセージを表示
    display_messages(st.session_state.messages)

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
                        openai_api_key=OPENAI_API_KEY
                    )

                    # 応答をまとめて表示するためのリスト
                    responses = []

                    async with MultiServerMCPClient(config["mcpServers"]) as mcp_client:
                        # toolsの読み込み
                        tools = mcp_client.get_tools()

                        # 空のコンテンツを持つメッセージを除外
                        #filtered_messages = [msg for msg in st.session_state.messages if msg.content]
                        
                        # tool callingがなくなるまで呼び出し(最大10回)
                        max_iterations = 10
                        count = 0
                        while count < max_iterations:
                            count += 1
                            logging.info(st.session_state.messages)
                            llm_with_tools = create_prompt_from_system_and_messages(system_prompt, st.session_state.messages) | llm.bind_tools(tools)
                            response = await llm_with_tools.ainvoke({"messages": st.session_state.messages})
                            logging.info(response)
                            st.session_state.messages.append(response)
                            responses.append(response)
                            #st.markdown(response.content)

                            if response.tool_calls:
                                for tool_call in response.tool_calls:
                                    selected_tool = {tool.name.lower(): tool for tool in tools}[
                                        tool_call["name"].lower()
                                    ]
                                    tool_messsage = await selected_tool.ainvoke(tool_call)
                                    logging.info(tool_messsage)
                                    processed_tool_message = process_tool_response(tool_messsage)
                                    # ツール呼び出し結果をメッセージとして追加
                                    st.session_state.messages.append(processed_tool_message)
                                    responses.append(tool_messsage)
                            else:
                                logging.info(st.session_state.messages)
                                break
                    # 最初の応答を表示
                    if isinstance(responses[0], AIMessage):
                        display_ai_message(responses[0])
                    elif isinstance(responses[0], ToolMessage):
                        display_tool_message(responses[0])
            # まとめて残りの応答を表示
            display_messages(responses[1:])
        else:
            st.error("API keyが環境変数に設定されていません。.envファイルを確認してください。")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())