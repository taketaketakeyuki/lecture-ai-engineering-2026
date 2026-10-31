"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何か？"
    uv run python -m app.ask  # 質問を省略すると、講座に関連した質問をする
    uv run python -m app.ask --chat "LoRAとは何ですか？"   # 会話履歴を保って続けて質問する

ツールを持たない「素の」呼び出し。
--chat を付けると、これまでのやり取り（履歴）を毎回まとめて送り、続けて質問できる。
モデル自体は会話を覚えていないため、履歴はこのスクリプト側で保持する。
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response) -> None:
    """入力トークン数と出力トークン数を表示する。"""
    usage = response.usage_metadata
    if usage is None:
        print("[tokens] 取得できませんでした")
        return
    print(f"[tokens] 入力 {usage.prompt_token_count} / 出力 {usage.candidates_token_count}")


def ask(question: str) -> str:
    """1回だけ質問"""
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    print_usage(response)
    return response.text


def chat(first_question: str) -> None:
    """会話履歴を保ったまま、続けて質問（空行で終了）。"""
    client = genai.Client(api_key=get_api_key())
    history: list[types.Content] = []  # これまでのやり取りを自分で保持する

    question = first_question
    turn = 1
    while question:
        # 今回の質問を履歴に足し、履歴まるごとを送る
        history.append(types.Content(role="user", parts=[types.Part(text=question)]))
        response = client.models.generate_content(model=MODEL, contents=history)
        answer = response.text or ""
        # モデルの答えも履歴に足す（次のターンで送るため）
        history.append(types.Content(role="model", parts=[types.Part(text=answer)]))

        print(f"\n[{turn}回目の回答]\n{answer}\n")
        print_usage(response)
        print(f"[履歴] {len(history)} 件（質問と回答の合計）")

        try:
            question = input("\n[続けて質問（空行で終了）] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        turn += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保ったまま続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        if args.chat:
            chat(args.question)
        else:
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()