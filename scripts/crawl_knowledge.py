#!/usr/bin/env python3
"""知识库采集器（Phase K1）— 多来源抓取 + 可信度分级。

来源:
    1. PubMed (NCBI E-utilities 官方 API): 每类 2 个查询 × 6 篇 → 摘要 (grade B)
    2. ECGpedia (MediaWiki API): 每类 1 页 wikitext (grade C, CC BY-NC-SA)
    3. LITFL / Utah: 已注册但暂不可达（403/结构不稳定），解封后启用

产出: data/knowledge/raw/{class_slug}/doc_*.json
    字段: {doc_id, source, grade, url, title, text, meta, retrieved_at}

合规: 仅公开内容；限速 ~1 req/s；保留来源与许可信息；提炼后知识条目注明出处。

用法:
    python scripts/crawl_knowledge.py --classes "Atrial Fibrillation" --limit-sites  # 试点
    python scripts/crawl_knowledge.py                                               # 全量 27 类 + 总论
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_pipeline.label_extractor import LabelExtractor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent  # ecg-ai-agent/
RAW_DIR = BASE_DIR / "data" / "knowledge" / "raw"
UA = "ECG-AI-Agent-KnowledgeBot/0.1 (research; contact: local)"


def load_sources():
    with open(BASE_DIR / "src" / "knowledge" / "sources.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# LITFL ECG Library 真实页名映射（官方 27 评分类 → litfl.com/<slug>/）
# 抓取时逐个实测，404 会在日志显式告警
LITFL_SLUGS = {
    "Atrial Fibrillation": "atrial-fibrillation",
    "Atrial Flutter": "atrial-flutter",
    "Sinus Bradycardia": "sinus-bradycardia",
    "Bradycardia": "bradycardia-ddx",
    "Sinus Rhythm": "sinus-rhythm",
    "Sinus Tachycardia": "sinus-tachycardia",
    "Sinus Arrhythmia": "sinus-arrhythmia",
    "Premature Atrial Contraction": "premature-atrial-complex-pac",
    "Supraventricular Premature Beats": "premature-atrial-complex-pac",
    "Premature Ventricular Contractions": "premature-ventricular-complex-pvc",
    "Ventricular Premature Beats": "premature-ventricular-complex-pvc",
    "First Degree AV Block": "first-degree-heart-block",
    "Left Bundle Branch Block": "left-bundle-branch-block-lbbb",
    "Right Bundle Branch Block": "right-bundle-branch-block-rbbb",
    "Incomplete Right Bundle Branch Block": "right-bundle-branch-block-rbbb",
    "Complete Right Bundle Branch Block": "right-bundle-branch-block-rbbb",
    "Left Anterior Fascicular Block": "left-anterior-fascicular-block-lafb",
    "Nonspecific Intraventricular Conduction Disorder": "interventricular-conduction-delay-qrs-widening",
    "Q Wave Abnormal": "q-wave",
    "T Wave Abnormal": "t-wave",
    "T Wave Inversion": "t-wave",
    "Low QRS Voltages": "low-qrs-voltage",
    "Left Axis Deviation": "left-axis-deviation",
    "Right Axis Deviation": "right-axis-deviation",
    "Prolonged QT Interval": "qt-interval",   # 3E-Y8：优先 QT 间期专页（LQTS 页为兜底）
    "Prolonged PR Interval": "pr-interval",
    "Pacing Rhythm": "pacemaker-rhythms",
}

# 补充候选页名（按 LITFL 官方 A-Z 索引核对；TInv/SVPB 无专页，复用覆盖性页面）
EXTRA_LITFL_SLUGS = {
    "Pacing Rhythm": ["pacemaker-malfunction"],
    "Supraventricular Premature Beats": ["premature-atrial-complex-pac"],
    "Nonspecific Intraventricular Conduction Disorder":
        ["interventricular-conduction-delay-qrs-widening"],
    "Prolonged QT Interval": ["long-qt-syndrome"],  # 3E-Y8：qt-interval 专页失败时兜底
}

PROXIES = None  # 由 --proxy / HTTP(S)_PROXY 环境变量设置


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RateLimiter:
    def __init__(self, min_interval=1.0):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self):
        dt = time.time() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.time()


RL = RateLimiter(1.0)


def save_doc(class_slug, doc):
    out = RAW_DIR / class_slug
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{doc['doc_id']}.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)


# ============================================================
# PubMed (E-utilities)
# ============================================================
class PubMedCrawler:
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def _get_retry(self, url, params, timeout, attempts=3):
        """2F-O5 修复：PubMed 瞬时失败（网络抖动/429/5xx）指数退避重试，
        旧版单次失败直接丢弃该查询（查询静默缺失）。"""
        for a in range(attempts):
            try:
                r = requests.get(url, params=params, headers={"User-Agent": UA},
                                 timeout=timeout)
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                if a == attempts - 1:
                    raise
                logger.warning(f"  [pubmed] 第{a+1}次失败 {url}: {e}，退避重试")
                time.sleep(3 * (2 ** a))
        raise RuntimeError("unreachable")

    def esearch(self, term, retmax=6):
        RL.wait()
        params = {"db": "pubmed", "term": term, "retmax": retmax,
                  "retmode": "json", "sort": "relevance"}
        r = self._get_retry(f"{self.BASE}/esearch.fcgi", params, timeout=30)
        return r.json()["esearchresult"].get("idlist", [])

    def efetch(self, ids):
        if not ids:
            return []
        RL.wait()
        params = {"db": "pubmed", "id": ",".join(ids),
                  "rettype": "abstract", "retmode": "xml"}
        r = self._get_retry(f"{self.BASE}/efetch.fcgi", params, timeout=60)
        root = ET.fromstring(r.content)
        docs = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID", default="")
            title = art.findtext(".//ArticleTitle", default="").strip()
            ab = art.findall(".//Abstract/AbstractText")
            # 稳健拼接（含子标签文本）
            abstract = " ".join("".join(t.itertext()) for t in ab)
            if not title and not abstract:
                continue
            journal = art.findtext(".//Journal/ISOAbbreviation", default="")
            year = art.findtext(".//JournalIssue/PubDate/Year", default="")
            docs.append({
                "doc_id": f"pubmed_{pmid}",
                "source": "pubmed",
                "grade": "B",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "title": title[:300],
                "text": abstract[:4000],
                "meta": {"pmid": pmid, "journal": journal, "year": year},
                "retrieved_at": now_iso(),
            })
        return docs


# ============================================================
# MediaWiki（ECGpedia）
# ============================================================
class MediaWikiCrawler:
    def __init__(self, api_url):
        self.api = api_url

    def search(self, query, limit=3):
        """用搜索 API 找规范页名。"""
        RL.wait()
        params = {"action": "query", "list": "search", "srsearch": query,
                  "srlimit": limit, "format": "json"}
        try:
            r = requests.get(self.api, params=params,
                             headers={"User-Agent": UA}, timeout=30)
            if r.status_code != 200:
                return []
            return [h["title"] for h in r.json().get("query", {}).get("search", [])]
        except Exception:
            return []

    def fetch_page(self, title):
        RL.wait()
        params = {"action": "parse", "page": title, "prop": "wikitext",
                  "format": "json", "redirects": 1}
        r = requests.get(self.api, params=params,
                         headers={"User-Agent": UA}, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        if "error" in data:
            return None
        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
        if not wikitext.strip():
            return None
        plain = wikitext_to_plain(wikitext)
        return {
            "doc_id": f"ecgpedia_{slugify(title)}",
            "source": "ecgpedia",
            "grade": "C",
            "url": f"https://en.ecgpedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "title": data["parse"].get("title", title),
            "text": plain[:6000],
            "meta": {"page": title},
            "retrieved_at": now_iso(),
        }


def wikitext_to_plain(wt: str) -> str:
    """粗粒度 wikitext → 纯文本（供 LLM 提炼足够）。"""
    # 引用与模板
    wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*/>", "", wt)
    wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)          # 简单模板
    wt = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", wt, flags=re.I)
    wt = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", wt)  # 链接保留显示文本
    wt = re.sub(r"\[(https?://[^\]\s]+)\s+([^\]]+)\]", r"\2 (\1)", wt)
    wt = re.sub(r"'''?([^']+?)'''?", r"\1", wt)
    wt = re.sub(r"^==+\s*(.*?)\s*==+$", r"\n## \1", wt, flags=re.M)
    wt = re.sub(r"^\s*[*#;:]+\s*", "- ", wt, flags=re.M)
    wt = re.sub(r"\{\|.*?\|\}", "", wt, flags=re.S)  # 表格（粗删）
    wt = re.sub(r"<[^>]+>", "", wt)
    wt = re.sub(r"\n{3,}", "\n\n", wt)
    return wt.strip()


# ============================================================
# 主流程
# ============================================================
def crawl_class(name: str, sources_cfg, only_litfl: bool = False) -> list:
    docs = []
    slug = slugify(name)

    if not only_litfl:
        # PubMed: 2 查询 × retmax
        pubmed = PubMedCrawler()
        pm_cfg = sources_cfg["sources"]["pubmed_review"]
        for q in pm_cfg["queries"]:
            term = q.format(name=name)
            try:
                ids = pubmed.esearch(term, retmax=pm_cfg.get("retmax", 6))
                logger.info(f"  [pubmed] '{term}' → {len(ids)} 篇")
                for d in pubmed.efetch(ids):
                    d["class"] = name
                    d["class_slug"] = slug
                    docs.append(d)
            except Exception as e:
                logger.warning(f"  [pubmed] 查询失败 '{term}': {e}")

        # ECGpedia: 直接取页，失败则搜索规范页名
        wiki_cfg = sources_cfg["sources"]["ecgpedia"]
        wiki = MediaWikiCrawler(wiki_cfg["api"])
        candidates = [name.replace(" ", "_")]
        try:
            d = wiki.fetch_page(candidates[0])
            if not d:
                titles = wiki.search(name, limit=3)
                logger.info(f"  [ecgpedia] 搜索 '{name}' → {titles}")
                for t in titles:
                    d = wiki.fetch_page(t)
                    if d:
                        break
            if d:
                d["class"] = name
                d["class_slug"] = slug
                docs.append(d)
                logger.info(f"  [ecgpedia] {d['title']} ✓ ({len(d['text'])} 字符)")
            else:
                logger.info(f"  [ecgpedia] {name} 无匹配页")
        except Exception as e:
            logger.warning(f"  [ecgpedia] 失败 {name}: {e}")

    # LITFL（若启用且可达）
    # R4 修复：URL 用真实页名映射（LITFL 为连字符 + 每类专用后缀，
    # slugify 的下划线模板对不上）；支持 --proxy（默认读 HTTP(S)_PROXY 环境变量）
    litfl_cfg = sources_cfg["sources"]["litfl"]
    if litfl_cfg.get("enabled"):
        slug = LITFL_SLUGS.get(name, slugify(name))
        base = litfl_cfg["url_template"].rstrip("/")
        # LITFL 页有两种路径形态：多数 ECG Library 页带 "-ecg-library" 后缀，
        # 少数为裸路径；失败类追加补充候选页名——全部依次尝试
        slugs = [slug] + EXTRA_LITFL_SLUGS.get(name, [])
        candidates = []
        for s in slugs:
            if s.endswith("-ddx") or s in ("long-qt-syndrome",):
                candidates += [f"{base}/{s}/", f"{base}/{s}-ecg-library/"]
            else:
                candidates += [f"{base}/{s}-ecg-library/", f"{base}/{s}/"]
        # Cloudflare 反爬：浏览器形态请求头 + 会话预热 + 退避重试
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/126.0 Safari/537.36"),
            "Accept": ("text/html,application/xhtml+xml,"
                       "application/xml;q=0.9,*/*;q=0.8"),
            "Accept-Language": "en-US,en;q=0.9",
        }
        got = False
        for url in candidates:
            for attempt in range(5):
                try:
                    RL.wait()
                    sess = requests.Session()
                    sess.proxies = PROXIES or None
                    if attempt == 0:
                        sess.get("https://litfl.com/", headers=headers, timeout=20)
                    r = sess.get(url, headers=headers, timeout=30)
                    if r.status_code == 200:
                        text = html_to_plain(r.text)
                        if len(text) > 500:
                            # 3E-Y7：title 用页面真实标题（剥离 LITFL 站点后缀），
                            # 旧版取类名导致引用标题错配（如 PAC 类显示 Supraventricular Premature Beats）
                            docs.append({
                                "doc_id": f"litfl_{slug}", "source": "litfl",
                                "grade": litfl_cfg.get("grade", "C"), "url": url,
                                "title": html_page_title(r.text) or name, "text": text[:6000],
                                "meta": {}, "class": name,
                                "class_slug": slugify(name),
                                "retrieved_at": now_iso(),
                            })
                            logger.info(f"  [litfl] {url} ✓ ({len(text)} 字符, 第{attempt+1}次)")
                            got = True
                            break
                        else:
                            logger.warning(f"  [litfl] 内容过短: {url}")
                            break
                    else:
                        logger.warning(f"  [litfl] 第{attempt+1}次 HTTP {r.status_code}: {url}")
                        time.sleep(6 + 6 * attempt)
                except Exception as e:
                    logger.warning(f"  [litfl] 第{attempt+1}次失败 {url}: {e}")
                    time.sleep(6 + 6 * attempt)
            if got:
                break
        if not got:
            logger.warning(f"  [litfl] 放弃: {name}（候选: {candidates}）")
        time.sleep(3.0)  # LITFL 额外限速

    return docs


def html_page_title(html: str) -> str:
    """提取 <title> 标签文本并剥离站点后缀（3E-Y7：引用标题用真实页标题而非类名）。"""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    if not m:
        return ""
    t = re.sub(r"\s+", " ", m.group(1)).strip()
    for sep in ("| LITFL", "— LITFL", "- LITFL", "– LITFL"):
        if sep in t:
            t = t.split(sep)[0].strip()
    return t[:200]


def html_to_plain(html: str) -> str:
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

        def handle_endtag(self, tag):
            if tag in ("p", "h1", "h2", "h3", "li", "br"):
                self.parts.append("\n")

    p = P()
    p.feed(html)
    return re.sub(r"\n{3,}", "\n\n", "".join(p.parts)).strip()


def crawl_general(sources_cfg) -> list:
    """总论主题（ECG 解读方法学等）。"""
    docs = []
    pubmed = PubMedCrawler()
    for topic in sources_cfg["general_topics"]:
        try:
            ids = pubmed.esearch(topic, retmax=4)
            for d in pubmed.efetch(ids):
                d["class"] = "_general_"
                d["class_slug"] = "general"
                docs.append(d)
            logger.info(f"[general] '{topic}' → {len(ids)} 篇")
        except Exception as e:
            logger.warning(f"[general] '{topic}' 失败: {e}")
    # ECGpedia 基础页
    wiki = MediaWikiCrawler(sources_cfg["sources"]["ecgpedia"]["api"])
    for page in ["Basics", "Rhythm", "Rate", "P_wave_morphology",
                 "PR_interval", "QRS_morphology", "ST_morphology",
                 "QT_interval", "Electrical_Axis", "Conduction"]:
        try:
            d = wiki.fetch_page(page)
            if d:
                d["class"] = "_general_"
                d["class_slug"] = "general"
                docs.append(d)
                logger.info(f"[general][ecgpedia] {page} ✓")
        except Exception as e:
            logger.warning(f"[general][ecgpedia] {page} 失败: {e}")
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=None,
                        help="仅采集指定类别（试点用）")
    parser.add_argument("--skip-general", action="store_true")
    parser.add_argument("--proxy", default=None,
                        help="HTTP(S) 代理，如 http://127.0.0.1:7890"
                             "（默认读 HTTP_PROXY/HTTPS_PROXY 环境变量）")
    parser.add_argument("--only-litfl", action="store_true",
                        help="只抓 LITFL（跳过 PubMed/ECGpedia，避免重复抓取）")
    args = parser.parse_args()

    import os
    global PROXIES
    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        PROXIES = {"http": proxy, "https": proxy}
        logger.info(f"使用代理: {proxy}")
    else:
        logger.info("无代理（直连）")

    cfg = load_sources()
    le = LabelExtractor(num_classes=27)
    classes = args.classes or le.class_names
    logger.info(f"目标类别: {len(classes)} 个"
                f"{'（仅 LITFL）' if args.only_litfl else ''}")

    summary = {}
    for name in classes:
        logger.info(f"=== 采集: {name} ===")
        docs = crawl_class(name, cfg, only_litfl=args.only_litfl)
        for d in docs:
            save_doc(d["class_slug"], d)
        summary[name] = {
            "n_docs": len(docs),
            "by_source": {s: sum(1 for x in docs if x["source"] == s)
                          for s in ("pubmed", "ecgpedia", "litfl")},
        }
        logger.info(f"  小计: {len(docs)} 条 ({summary[name]['by_source']})")

    if not args.skip_general:
        logger.info("=== 采集: 总论 ===")
        gdocs = crawl_general(cfg)
        for d in gdocs:
            save_doc("general", d)
        summary["_general_"] = {"n_docs": len(gdocs)}

    total = sum(v["n_docs"] for v in summary.values())
    with open(RAW_DIR / "crawl_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "total": total}, f,
                  ensure_ascii=False, indent=1)
    logger.info(f"采集完成: 总计 {total} 条 → {RAW_DIR}")


if __name__ == "__main__":
    main()
