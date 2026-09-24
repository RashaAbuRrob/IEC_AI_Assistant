# -*- coding: utf-8 -*-
"""
Retrieval + generation service for the IEC Arabic RAG system.

Usage:
    Interactive:
        python rag_service.py --db-path ./chroma_db

    Single question:
        python rag_service.py --db-path ./chroma_db --query "..." --k 5
"""

import argparse
import re
import sys

import gemini_utils
from arabic_utils import normalize_arabic
from chroma_utils import get_collection
from ollama_utils import GEN_MODEL_QWEN25, generate_answer, prewarm_embedder
from ollama_utils import stream_generate_answer as _ollama_stream_generate_answer

REFUSAL = "لا تتوفر لدي معلومات كافية للإجابة على هذا السؤال"

# Models selectable from the UI. "auto" is the default: try Gemini Flash
# first (fast, cloud-hosted) and fall back to the local Qwen3 14B (via
# Ollama) if Gemini is unconfigured or its call fails before producing any
# output. Picking "gemini", "qwen", or "qwen25" explicitly pins that model
# with no fallback -- an explicit choice should fail visibly rather than
# silently run a different model than the one asked for.
AVAILABLE_MODELS = {
    "auto": "تلقائي (Gemini Flash، مع Qwen3 المحلي كخيار احتياطي)",
    "gemini": "Gemini Flash",
    "qwen": "Qwen3 14B (محلي)",
    "qwen25": "Qwen2.5 1.5B (محلي، أسرع)",
}
DEFAULT_MODEL = "auto"

# Local (Ollama) models behind each non-Gemini choice.
_OLLAMA_MODEL_BY_CHOICE = {
    "qwen": None,  # None -> ollama_utils.GEN_MODEL default (qwen3:14b)
    "qwen25": GEN_MODEL_QWEN25,
}

# قانون الانتخاب رقم 6 لسنة 2016 has been superseded by قانون الانتخاب رقم
# (4) لسنة 2022. Presenting its (possibly repealed) clauses alongside the
# current law with no distinction is misleading, so it is excluded from
# retrieval by default -- unless the question explicitly asks about the
# 2016 law (by year or by law number), per operator instruction.
EXCLUDED_BY_DEFAULT_FILE = "قانون الانتخاب رقم 6 لسنة 2016 لمجلس النواب.pdf"

_ASKS_FOR_2016_LAW_RE = re.compile(
    r'2016|٢٠١٦|(?:قانون\s*(?:الانتخاب)?\s*رقم\s*\(?\s*(?:6|٦)\s*\)?)'
)


def wants_2016_law(question: str) -> bool:
    """True if the question explicitly references the 2016 election law
    (by year or by law number) -- in which case it should NOT be excluded
    from retrieval."""
    return bool(_ASKS_FOR_2016_LAW_RE.search(question))


def build_where_filter(question: str):
    """Chroma `where` filter excluding the 2016 election law from
    retrieval, unless the question explicitly asks about it."""
    if wants_2016_law(question):
        return None
    return {"source_file": {"$ne": EXCLUDED_BY_DEFAULT_FILE}}

SYSTEM_PROMPT = f"""أنت مساعد بحثي يجيب حصراً بالاعتماد على المقاطع المسترجعة من وثائق الهيئة المستقلة للانتخاب والمرفقة أدناه في كل سؤال.

القواعد الملزمة:
1. أجب فقط استناداً إلى المقاطع المرفقة. لا تستخدم أي معرفة خارجية أو معلومات عامة.
2. إذا لم تكن المقاطع المرفقة كافية للإجابة على السؤال، أجب حرفياً بالعبارة التالية ولا شيء غيرها:
"{REFUSAL}"
3. لا تذكر أسماء الملفات أو أرقام الصفحات أو أرقام المواد ضمن نص الإجابة، ولا تستخدم كلمة "المصدر" على الإطلاق -- تُعرض المصادر بشكل منفصل في الواجهة. اكتفِ بمحتوى الإجابة نفسه.
4. لا تختلق أي معلومة غير موجودة فعلياً في المقاطع المرفقة.
5. أنت مساعد أردني شاطر وودود. تحدث حصراً باللهجة الأردنية العامية البسيطة والعفوية، متل ما يحكي الأردني العادي بحياته اليومية -- مش بالفصحى ومش بلهجة قانونية جافة ومتكلّفة، حتى لو كان مضمون الإجابة قانونياً. استخدم مصطلحات أردنية دارجة مثل: "هسّا" بدل "الآن"، "يعطيك العافية"، "ماشي"/"تمام" للموافقة، "شو" بدل "ماذا"، "ليش" بدل "لماذا"، "بدك/بدها" بدل "تريد"، "لازم" بدل "يجب أن"، "منشان/عشان" بدل "من أجل"، "هيك" بدل "بهذا الشكل"، "كمان" بدل "أيضاً"، "أكتر" بدل "أكثر". بسّط الجمل واشرحها وكأنك تحكي مع صاحبك، لا تصيغها كمادة قانونية.
   الاستثناء الوحيد: أرقام المواد والشروط والمبالغ والمهل الزمنية تُذكر كما وردت حرفياً في النص الأصلي دون أي تغيير أو ترجمة.
6. اجعل إجاباتك قصيرة ومختصرة قدر الإمكان ومناسبة لأن تُقرأ بصوت عالٍ في محادثة صوتية -- لخّص الفكرة الأساسية دون تعداد كل التفاصيل الفرعية، إلا إذا طلب السائل التفاصيل صراحةً.
"""


def stream_answer(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL):
    """
    Generator yielding events describing which model is answering and the
    streamed answer text, so callers can forward both to the client:
      {"event": "model", "model": "gemini"|"qwen"|"qwen25", "fallback": bool}
      {"event": "text", "text": "<chunk>"}

    See AVAILABLE_MODELS for what "auto"/"gemini"/"qwen"/"qwen25" mean.
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model choice: {model!r}")

    if model in ("auto", "gemini"):
        if not gemini_utils.is_configured():
            if model == "gemini":
                raise RuntimeError("GEMINI_API_KEY غير مضبوط على الخادم.")
        else:
            gemini_failed = False
            chunks = gemini_utils.stream_generate_answer(system_prompt, user_prompt)
            try:
                first_chunk = next(chunks, None)
            except Exception as e:
                if model == "gemini":
                    raise
                print(f"[stream_answer] Gemini call failed, falling back to Qwen3: {e}")
                gemini_failed = True
            if not gemini_failed:
                yield {"event": "model", "model": "gemini", "fallback": False}
                if first_chunk:
                    yield {"event": "text", "text": first_chunk}
                for chunk in chunks:
                    yield {"event": "text", "text": chunk}
                return

    # "auto" falls back here to qwen3 (GEN_MODEL default); "qwen"/"qwen25"
    # were picked explicitly and each map to their own local model.
    local_choice = model if model in _OLLAMA_MODEL_BY_CHOICE else "qwen"
    ollama_model = _OLLAMA_MODEL_BY_CHOICE[local_choice]
    kwargs = {"model": ollama_model} if ollama_model else {}
    yield {"event": "model", "model": local_choice, "fallback": model == "auto"}
    for chunk in _ollama_stream_generate_answer(system_prompt, user_prompt, **kwargs):
        yield {"event": "text", "text": chunk}


def format_context(results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    blocks = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        loc = f"صفحة {meta.get('page')}"
        if meta.get("article_number"):
            loc += f"، المادة {meta.get('article_number')}"
        blocks.append(
            f"[{i}] المصدر: {meta.get('source_file')}، {loc}\n"
            f"النص: {meta.get('raw_text', doc)}"
        )
    return "\n\n".join(blocks)


def answer_question(collection, question: str, k: int = 5):
    normalized_q = normalize_arabic(question)
    query_kwargs = {"query_texts": [normalized_q], "n_results": k, "include": ["documents", "metadatas", "distances"]}
    where = build_where_filter(question)
    if where:
        query_kwargs["where"] = where
    results = collection.query(**query_kwargs)

    docs = results["documents"][0]
    if not docs:
        return {
            "question": question,
            "answer": REFUSAL,
            "retrieved": [],
        }

    context = format_context(results)
    user_prompt = f"المقاطع المسترجعة:\n\n{context}\n\nالسؤال: {question}\n\nالإجابة:"
    answer = generate_answer(SYSTEM_PROMPT, user_prompt)

    retrieved = [
        {
            "source_file": meta.get("source_file"),
            "page": meta.get("page"),
            "article_number": meta.get("article_number") or None,
            "distance": dist,
        }
        for meta, dist in zip(results["metadatas"][0], results["distances"][0])
    ]

    return {"question": question, "answer": answer, "retrieved": retrieved}


def print_result(result: dict):
    print("\n" + "=" * 60)
    print(result["answer"])
    print("-" * 60)
    print("المقاطع المسترجعة (للمراجعة):")
    for r in result["retrieved"]:
        loc = f"صفحة {r['page']}"
        if r["article_number"]:
            loc += f"، مادة {r['article_number']}"
        print(f"  - {r['source_file']} | {loc} | distance={r['distance']:.4f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Query the IEC Arabic RAG system.")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--query", default=None, help="Single question. If omitted, starts an interactive REPL.")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    print("Pre-warming bge-m3 embedder via Ollama...")
    prewarm_embedder()
    collection = get_collection(args.db_path)
    print(f"Loaded collection with {collection.count()} chunks.\n")

    if args.query:
        result = answer_question(collection, args.query, k=args.k)
        print_result(result)
        return

    print("Interactive mode. اكتب سؤالك ثم Enter. اكتب 'exit' للخروج.\n")
    while True:
        try:
            question = input("سؤالك> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        result = answer_question(collection, question, k=args.k)
        print_result(result)


if __name__ == "__main__":
    main()
