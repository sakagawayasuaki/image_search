"""
Azure AI Search 画像検索デモアプリ
Streamlitを使用したベクトル検索デモ
"""

import os
import base64
from datetime import datetime, timedelta
import traceback
import time
from urllib.parse import urlparse, unquote

import streamlit as st
import certifi
import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from PIL import Image
import io

# 環境変数読み込み
load_dotenv()

# Azure AI Search設定
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Azure OpenAI設定
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT_MINI = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_MINI", "gpt-5-mini")
AZURE_OPENAI_CHAT_DEPLOYMENT_NANO = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NANO", "gpt-5-nano")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_SMALL = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_SMALL", "text-embedding-3-small"
)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE", "text-embedding-3-large"
)

# インデックス切り替え設定
SEARCH_INDEX_NANO = os.getenv("SEARCH_INDEX_NANO", "geek-location-image-search-nano-index")
SEARCH_INDEX_MINI_SMALL = os.getenv("SEARCH_INDEX_MINI_SMALL", "geek-location-image-search-test-index")
SEARCH_INDEX_MINI_LARGE = os.getenv("SEARCH_INDEX_MINI_LARGE", "geek-location-image-search-large-mini-index")
SEARCH_INDEX_NANO_LARGE = os.getenv("SEARCH_INDEX_NANO_LARGE", "geek-location-image-search-large-nano-index")

# 料金（円）
PRICE_GPT5_NANO_INPUT_PER_1M = 7.68
PRICE_GPT5_NANO_OUTPUT_PER_1M = 61.4
PRICE_GPT5_MINI_INPUT_PER_1M = 38.38
PRICE_GPT5_MINI_OUTPUT_PER_1M = 306.99
PRICE_EMBEDDING_SMALL_PER_1K = 0.003838
PRICE_EMBEDDING_LARGE_PER_1K = 0.024252


def get_patterns() -> list[dict]:
    return [
        {
            "id": "mini_large",
            "label": "gpt-5-mini + emb-large",
            "chat": AZURE_OPENAI_CHAT_DEPLOYMENT_MINI,
            "embed": AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE,
            "index": SEARCH_INDEX_MINI_LARGE
        },
        {
            "id": "mini_small",
            "label": "gpt-5-mini + emb-small",
            "chat": AZURE_OPENAI_CHAT_DEPLOYMENT_MINI,
            "embed": AZURE_OPENAI_EMBEDDING_DEPLOYMENT_SMALL,
            "index": SEARCH_INDEX_MINI_SMALL
        },
        {
            "id": "nano_large",
            "label": "gpt-5-nano + emb-large",
            "chat": AZURE_OPENAI_CHAT_DEPLOYMENT_NANO,
            "embed": AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE,
            "index": SEARCH_INDEX_NANO_LARGE
        },
        {
            "id": "nano_small",
            "label": "gpt-5-nano + emb-small",
            "chat": AZURE_OPENAI_CHAT_DEPLOYMENT_NANO,
            "embed": AZURE_OPENAI_EMBEDDING_DEPLOYMENT_SMALL,
            "index": SEARCH_INDEX_NANO
        }
    ]

# Azure Blob Storage設定
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")


def init_openai_client() -> AzureOpenAI:
    """Azure OpenAIクライアントを初期化"""
    http_client = httpx.Client(verify=certifi.where(), timeout=60.0)
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2025-03-01-preview",
        http_client=http_client,
        timeout=60.0  # タイムアウト60秒
    )


def test_connection(client: AzureOpenAI, chat_deployment: str) -> tuple[bool, str]:
    """Azure OpenAI接続テスト"""
    try:
        client.chat.completions.create(
            model=chat_deployment,
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=5
        )
        return True, "接続成功"
    except Exception as e:
        cause = repr(getattr(e, "__cause__", None))
        detail = traceback.format_exc()
        return False, f"{type(e).__name__}: {e}\nCAUSE={cause}\n{detail}"


def init_search_client(index_name: str) -> SearchClient:
    """Azure AI Searchクライアントを初期化"""
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=index_name,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )


def encode_image_to_base64(image_bytes: bytes) -> str:
    """画像をBase64エンコード"""
    return base64.b64encode(image_bytes).decode("utf-8")


def render_result_image(image_bytes: bytes, content_type: str, height_px: int = 220) -> None:
    data_url = f"data:{content_type};base64,{encode_image_to_base64(image_bytes)}"
    st.markdown(
        f"""
        <div class="result-card">
          <img src="{data_url}" class="result-image" style="height:{height_px}px;" />
        </div>
        """,
        unsafe_allow_html=True
    )


def _usage_get(usage, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key) or 0)
    return int(getattr(usage, key, 0) or 0)


def _extract_chat_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    prompt = _usage_get(usage, "prompt_tokens")
    completion = _usage_get(usage, "completion_tokens")
    total = _usage_get(usage, "total_tokens") or (prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total
    }


def _extract_response_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    prompt = _usage_get(usage, "input_tokens")
    completion = _usage_get(usage, "output_tokens")
    total = _usage_get(usage, "total_tokens") or (prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total
    }


def estimate_cost(
    snippet_usage: dict,
    embedding_tokens: int,
    snippet_input_price_per_1m: float,
    snippet_output_price_per_1m: float,
    embedding_price_per_1k: float
) -> dict:
    snippet_input_cost = (snippet_usage.get("prompt_tokens", 0) / 1_000_000) * snippet_input_price_per_1m
    snippet_output_cost = (snippet_usage.get("completion_tokens", 0) / 1_000_000) * snippet_output_price_per_1m
    embedding_cost = (embedding_tokens / 1_000) * embedding_price_per_1k
    total = snippet_input_cost + snippet_output_cost + embedding_cost
    return {
        "snippet_input_cost": snippet_input_cost,
        "snippet_output_cost": snippet_output_cost,
        "embedding_cost": embedding_cost,
        "total": total
    }


def get_snippet_prices(chat_deployment: str) -> tuple[float, float]:
    if chat_deployment == AZURE_OPENAI_CHAT_DEPLOYMENT_NANO:
        return PRICE_GPT5_NANO_INPUT_PER_1M, PRICE_GPT5_NANO_OUTPUT_PER_1M
    return PRICE_GPT5_MINI_INPUT_PER_1M, PRICE_GPT5_MINI_OUTPUT_PER_1M


def get_embedding_price(embedding_deployment: str) -> float:
    if embedding_deployment == AZURE_OPENAI_EMBEDDING_DEPLOYMENT_LARGE:
        return PRICE_EMBEDDING_LARGE_PER_1K
    return PRICE_EMBEDDING_SMALL_PER_1K


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "output_text"):
                    parts.append(part.get("text", ""))
            else:
                if getattr(part, "type", None) in ("text", "output_text"):
                    parts.append(getattr(part, "text", ""))
        return "".join(parts).strip()
    return ""


def _response_to_text(response) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts = []
        for item in output:
            content = None
            if isinstance(item, dict):
                content = item.get("content")
                if item.get("type") in ("output_text", "text") and item.get("text"):
                    parts.append(item.get("text"))
            else:
                content = getattr(item, "content", None)
                item_type = getattr(item, "type", None)
                if item_type in ("output_text", "text"):
                    item_text = getattr(item, "text", None)
                    if item_text:
                        parts.append(item_text)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") in ("output_text", "text"):
                            parts.append(part.get("text", ""))
                    else:
                        if getattr(part, "type", None) in ("output_text", "text"):
                            parts.append(getattr(part, "text", ""))
        return "".join(parts).strip()
    return ""


def generate_snippet_from_image(
    client: AzureOpenAI,
    image_base64: str,
    image_type: str,
    chat_deployment: str
) -> tuple[str, dict]:
    """
    画像からsnippet（説明文）を生成
    選択したモデルで生成
    """
    response = client.chat.completions.create(
        model=chat_deployment,
        messages=[
            {
                "role": "system",
                "content": "You are tasked with generating concise, accurate descriptions of images, figures, diagrams, or charts in documents."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=300
    )
    usage = _extract_chat_usage(response)
    text = _content_to_text(response.choices[0].message.content)
    if text:
        return text, usage

    finish_reason = getattr(response.choices[0], "finish_reason", None)
    responses_error = None

    if hasattr(client, "responses"):
        try:
            resp = client.responses.create(
                model=chat_deployment,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Please describe this image."},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/{image_type};base64,{image_base64}"
                            }
                        ]
                    }
                ],
                max_output_tokens=1000
            )
            usage = _extract_response_usage(resp)
            text = _response_to_text(resp)
            if text:
                return text, usage
        except Exception as e:
            responses_error = f"{type(e).__name__}: {e}"

    refusal = getattr(response.choices[0].message, "refusal", None)
    raise RuntimeError(
        "画像説明が空です。"
        f"finish_reason={finish_reason} "
        f"refusal={refusal} "
        f"responses_error={responses_error}"
    )


def generate_embedding(client: AzureOpenAI, text: str, embedding_deployment: str) -> tuple[list[float], int]:
    """
    テキストをベクトル化
    選択した埋め込みモデルを使用
    """
    response = client.embeddings.create(
        model=embedding_deployment,
        input=text
    )
    usage = getattr(response, "usage", None)
    tokens = _usage_get(usage, "total_tokens") or _usage_get(usage, "prompt_tokens")
    return response.data[0].embedding, tokens


def generate_sas_url(blob_url: str) -> str:
    """
    Blob URLからSASトークン付きURLを生成
    blob_url形式: https://<account>.blob.core.windows.net/<container>/<blob_path>
    """
    if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY:
        return blob_url  # SAS生成不可の場合は元のURLを返す

    try:
        parsed = urlparse(blob_url)
        decoded_path = unquote(parsed.path)
        path_parts = decoded_path.lstrip("/").split("/", 1)

        if len(path_parts) < 2:
            return blob_url

        container_name = path_parts[0]
        blob_name = path_parts[1]

        sas_token = generate_blob_sas(
            account_name=AZURE_STORAGE_ACCOUNT_NAME,
            container_name=container_name,
            blob_name=blob_name,
            account_key=AZURE_STORAGE_ACCOUNT_KEY,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )

        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{sas_token}"
    except Exception as e:
        st.warning(f"SASトークン生成エラー: {e}")
        return blob_url


def search_similar_images(
    search_client: SearchClient,
    embedding: list[float],
    top_k: int = 20
) -> list[dict]:
    """
    ベクトル検索を実行
    snippet_vectorフィールドを使用
    """
    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=top_k,
        fields="snippet_vector"
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["blob_url", "snippet"],
        top=top_k
    )

    search_results = []
    for result in results:
        search_results.append({
            "blob_url": result.get("blob_url", ""),
            "snippet": result.get("snippet", ""),
            "score": result.get("@search.score", 0)
        })

    return search_results


def main():
    """メインアプリケーション"""
    st.set_page_config(
        page_title="画像検索デモ",
        page_icon="🔍",
        layout="wide"
    )
    st.markdown(
        """
        <style>
          .result-image {
            width: 100%;
            object-fit: cover;
            display: block;
            border-radius: 8px;
          }
          .result-card {
            width: 100%;
            margin-bottom: 8px;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🔍 Azure AI Search 画像検索デモ")
    st.markdown("画像をアップロードすると、類似画像を検索します。")

    patterns = get_patterns()

    # サイドバー: 検索設定
    with st.sidebar:
        st.header("検索設定")
        top_k = 5
        st.caption("検索結果は上位5件を比較表示します")

        # st.markdown("---")
        # st.header("比較パターン")
        # for pattern in patterns:
        #     st.caption(f"{pattern['label']} → {pattern['index']}")

        st.markdown("---")
        st.header("料金計算")
        st.markdown(
            "Azure OpenAI 価格表: "
            "https://azure.microsoft.com/ja-jp/pricing/details/azure-openai/"
        )
        st.caption(
            f"GPT-5-mini 入力: {PRICE_GPT5_MINI_INPUT_PER_1M}円/1M, 出力: {PRICE_GPT5_MINI_OUTPUT_PER_1M}円/1M"
        )
        st.caption(
            f"GPT-5-nano 入力: {PRICE_GPT5_NANO_INPUT_PER_1M}円/1M, 出力: {PRICE_GPT5_NANO_OUTPUT_PER_1M}円/1M"
        )

        st.caption(
            f"text-embedding-3-large: {PRICE_EMBEDDING_LARGE_PER_1K}円/1K"
        )
        st.caption(
            f"text-embedding-3-small: {PRICE_EMBEDDING_SMALL_PER_1K}円/1K, "
        )


        st.markdown("### 処理フロー")
        st.markdown("""
        1. 画像をアップロード
        2. gpt-5-mini / gpt-5-nano で画像説明を生成
        3. text-embedding-3-small / text-embedding-3-large でベクトル化
        4. 4パターンのインデックスで類似検索
        5. 上位5件を横並び比較
        """)

    # クライアント初期化
    try:
        openai_client = init_openai_client()
    except Exception as e:
        st.error(f"クライアント初期化エラー: {e}")
        st.info("`.env`ファイルの設定を確認してください。")
        return

    # サイドバー: 接続診断
    st.sidebar.markdown("---")
    st.sidebar.header("接続診断")
    with st.sidebar:
        with st.spinner("接続テスト中..."):
            tests = [
                ("gpt-5-mini", AZURE_OPENAI_CHAT_DEPLOYMENT_MINI),
                ("gpt-5-nano", AZURE_OPENAI_CHAT_DEPLOYMENT_NANO)
            ]
            for label, deployment in tests:
                success, message = test_connection(openai_client, deployment)
                if success:
                    st.success(f"{label}: {message}")
                else:
                    st.error(f"{label}: {message}")
                    st.info(f"エンドポイント: {AZURE_OPENAI_ENDPOINT}")

    # 画像アップロード
    uploaded_file = st.file_uploader(
        "検索する画像をアップロード",
        type=["jpg", "jpeg", "png"],
        help="JPGまたはPNG画像をアップロードしてください"
    )

    if uploaded_file is not None:
        # 画像タイプ判定
        image_type = uploaded_file.type.split("/")[-1] if uploaded_file.type else "png"
        if image_type.lower() == "jpg":
            image_type = "jpeg"

        # 画像プレビュー
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("アップロード画像")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("処理状況")

            # 画像をBase64エンコード
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
            image_base64 = encode_image_to_base64(image_bytes)

            # snippet生成
            with st.spinner("画像を解析中..."):
                snippet_by_chat = {}
                snippet_usage_by_chat = {}
                snippet_errors = {}
                snippet_time_by_chat = {}
                chat_models = [
                    ("gpt-5-mini", AZURE_OPENAI_CHAT_DEPLOYMENT_MINI),
                    ("gpt-5-nano", AZURE_OPENAI_CHAT_DEPLOYMENT_NANO)
                ]
                for label, chat_deployment in chat_models:
                    start_ts = time.monotonic()
                    try:
                        snippet, usage = generate_snippet_from_image(
                            openai_client, image_base64, image_type, chat_deployment
                        )
                        snippet_by_chat[chat_deployment] = snippet
                        snippet_usage_by_chat[chat_deployment] = usage
                    except Exception as e:
                        snippet_errors[chat_deployment] = (label, e)
                    finally:
                        snippet_time_by_chat[chat_deployment] = time.monotonic() - start_ts

                if snippet_by_chat:
                    st.success("画像解析完了")
                else:
                    for label, err in snippet_errors.values():
                        st.error(f"snippet生成エラー ({label}): {type(err).__name__}: {err}")
                    return

            # ベクトル化＆検索（4パターン）
            pattern_results = {}
            pattern_meta = {}
            with st.spinner("ベクトル化と検索中..."):
                for pattern in patterns:
                    pattern_id = pattern["id"]
                    chat_deployment = pattern["chat"]
                    embedding_deployment = pattern["embed"]
                    index_name = pattern["index"]

                    if chat_deployment not in snippet_by_chat:
                        pattern_meta[pattern_id] = {
                            "error": "snippet生成に失敗したためスキップ",
                            "index": index_name
                        }
                        pattern_results[pattern_id] = []
                        continue

                    snippet = snippet_by_chat[chat_deployment]
                    try:
                        embedding, embedding_tokens = generate_embedding(
                            openai_client, snippet, embedding_deployment
                        )
                        search_client = init_search_client(index_name)
                        results = search_similar_images(search_client, embedding, top_k)

                        snippet_input_price, snippet_output_price = get_snippet_prices(chat_deployment)
                        embedding_price = get_embedding_price(embedding_deployment)
                        cost = estimate_cost(
                            snippet_usage_by_chat.get(chat_deployment, {}),
                            embedding_tokens,
                            snippet_input_price,
                            snippet_output_price,
                            embedding_price
                        )
                        pattern_meta[pattern_id] = {
                            "cost": cost,
                            "tokens": {
                                "prompt": snippet_usage_by_chat.get(chat_deployment, {}).get("prompt_tokens", 0),
                                "completion": snippet_usage_by_chat.get(chat_deployment, {}).get("completion_tokens", 0),
                                "embedding": embedding_tokens
                            },
                            "index": index_name
                        }
                        pattern_results[pattern_id] = results
                    except Exception as e:
                        pattern_meta[pattern_id] = {
                            "error": f"{type(e).__name__}: {e}",
                            "index": index_name
                        }
                        pattern_results[pattern_id] = []

        st.markdown("### 生成されたsnippet")
        snippet_cols = st.columns(2, gap="large")
        for col, (label, deployment) in zip(snippet_cols, chat_models):
            with col:
                elapsed = snippet_time_by_chat.get(deployment)
                if elapsed is not None:
                    st.markdown(f"**{label}**（{elapsed:.2f}s）")
                else:
                    st.markdown(f"**{label}**")
                if deployment in snippet_by_chat:
                    st.write(snippet_by_chat[deployment])
                else:
                    err = snippet_errors.get(deployment)
                    if err:
                        st.error(f"{type(err[1]).__name__}: {err[1]}")
                    else:
                        st.caption("未生成")

        # 検索結果比較表示
        st.markdown("---")
        st.subheader("検索結果比較（上位5件）")

        header_cols = st.columns(4, gap="small")
        for col, pattern in zip(header_cols, patterns):
            pattern_id = pattern["id"]
            meta = pattern_meta.get(pattern_id, {})
            with col:
                st.markdown(f"**{pattern['label']}**")
                st.caption(f"index: {pattern['index']}")
                if "error" in meta:
                    st.error(meta["error"])
                else:
                    cost = meta.get("cost", {})
                    st.caption(
                        f"コスト: ¥{cost.get('total', 0):.6f} "
                        f"(入力: ¥{cost.get('snippet_input_cost', 0):.6f}, "
                        f"出力: ¥{cost.get('snippet_output_cost', 0):.6f}, "
                        f"Embedding: ¥{cost.get('embedding_cost', 0):.6f})"
                    )
                    tokens = meta.get("tokens", {})
                    st.caption(
                        f"Tokens: prompt={tokens.get('prompt', 0)}, "
                        f"completion={tokens.get('completion', 0)}, "
                        f"embedding={tokens.get('embedding', 0)}"
                    )

        for rank in range(top_k):
            row_cols = st.columns(4, gap="small")
            for col, pattern in zip(row_cols, patterns):
                pattern_id = pattern["id"]
                results = pattern_results.get(pattern_id, [])
                with col:
                    if rank < len(results):
                        result = results[rank]
                        st.markdown(f"**#{rank + 1}** (スコア: {result['score']:.4f})")

                        sas_url = generate_sas_url(result["blob_url"])
                        try:
                            resp = httpx.get(sas_url, timeout=10.0, verify=certifi.where())
                            content_type = resp.headers.get("content-type", "")
                            if resp.status_code == 200 and content_type.startswith("image/"):
                                render_result_image(resp.content, content_type)
                            else:
                                st.caption("画像を読み込めませんでした")
                        except Exception:
                            st.caption("画像を読み込めませんでした")

                        with st.expander("snippet"):
                            st.write(result["snippet"])
                    else:
                        st.caption(f"#{rank + 1} 該当なし")


if __name__ == "__main__":
    main()
