from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Optional

from bs4 import BeautifulSoup
from bs4.element import Tag
import pandas as pd
from tqdm import tqdm

from models import TripAdvisorReview


RAW_DIR = Path("raw_data") / "tripadvisor"
OUT_CSV = Path("dataframes") / "tripadvisor.csv"


_PT_MONTHS = {
    "janeiro": 1,
    "fevereiro": 2,
    "março": 3,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}


@dataclass(frozen=True)
class _CardParseContext:
    file_path: Path


def _parse_pt_br_date(date_str: str, *, ctx: _CardParseContext) -> datetime:
    # Example: "7 de abril de 2025"
    m = re.search(r"(\d{1,2})\s+de\s+([A-Za-zçÇáéíóúâêîôûãõÁÉÍÓÚÂÊÎÔÛÃÕ]+)\s+de\s+(\d{4})", date_str.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"[{ctx.file_path}] Could not parse date from: {date_str!r}")

    day = int(m.group(1))
    month_name = m.group(2).strip().lower()
    year = int(m.group(3))

    month = _PT_MONTHS.get(month_name)
    if month is None:
        raise ValueError(f"[{ctx.file_path}] Unknown pt-BR month: {month_name!r} (from {date_str!r})")

    return datetime(year=year, month=month, day=day)


def _extract_contribuicoes(soup: BeautifulSoup) -> int | None:
    # Find tag like: <span><span class="b">41</span> contribuições</span>
    for span_b in _iter_contribuicoes_number_spans(soup):
        m = re.search(r"\b(\d+)\b", span_b.get_text(" ", strip=True))
        if m:
            return int(m.group(1))
    return None


def _iter_contribuicoes_number_spans(soup: BeautifulSoup):
    # Shared helper: yields <span class="b">..</span> nodes where the enclosing text contains "contribuições".
    for span_b in soup.select("span.b"):
        parent = span_b.parent
        if parent is None:
            continue
        parent_text = parent.get_text(" ", strip=True).lower()
        if "contribui" in parent_text:
            yield span_b


def _extract_cidade_e_estado(soup: BeautifulSoup, *, ctx: _CardParseContext) -> Optional[str]:
    # Example in card HTML: <span>São Roque, SP</span>
    # Strategy:
    # - Anchor near the "contribuições" block (it appears alongside the location)
    # - From that container, find a <span> matching "<city>, <UF>" (UF = 2 uppercase letters)
    city_state_pattern = re.compile(r"^[A-Za-zÀ-ÿ' .\-]+,\s*[A-Z]{2}$")

    def _normalize(s: str) -> str:
        return " ".join(s.replace("\xa0", " ").split())

    # Try to anchor via contributions
    for span_b in _iter_contribuicoes_number_spans(soup):
        container = span_b.find_parent("div")
        if container is None:
            continue

        for span in container.find_all("span"):
            text = _normalize(span.get_text(" ", strip=True))
            if city_state_pattern.match(text):
                return text

    # Fallback: scan all spans
    for span in soup.find_all("span"):
        text = _normalize(span.get_text(" ", strip=True))
        if city_state_pattern.match(text):
            return text

    return None


def _extract_nome(soup: BeautifulSoup, *, ctx: _CardParseContext) -> str:
    # Find an <a> where href starts with "https://www.tripadvisor.com.br/Profile/",
    # then use the text inside the <a> tag.
    profile_prefix = "https://www.tripadvisor.com.br/Profile/"

    anchors = soup.find_all("a", href=True)
    for a in anchors:
        href = a.get("href")
        if not isinstance(href, str) or (not href.startswith(profile_prefix)):
            continue
        text = a.get_text(" ", strip=True)
        if text:
            return text

    raise AssertionError(f"[{ctx.file_path}] Could not extract reviewer name from profile link")


def _extract_nota(soup: BeautifulSoup, *, ctx: _CardParseContext) -> int:
    # Find svg with aria-labelledby and data-automation="bubbleRatingImage". Assert only one.
    svgs = soup.find_all("svg", attrs={"data-automation": "bubbleRatingImage"})
    svgs = [s for s in svgs if s.has_attr("aria-labelledby")]
    if len(svgs) != 1:
        raise AssertionError(f"[{ctx.file_path}] Expected exactly one bubbleRatingImage svg with aria-labelledby, got {len(svgs)}")

    svg = svgs[0]
    title_tag = svg.find("title")
    if title_tag is None:
        raise AssertionError(f"[{ctx.file_path}] bubbleRatingImage svg missing <title>")

    # Example: "4 de 5 círculos" -> grade = 4
    m = re.search(r"\b(\d+)\b", title_tag.get_text(" ", strip=True))
    if not m:
        raise ValueError(f"[{ctx.file_path}] Could not parse grade from title: {title_tag.get_text(strip=True)!r}")
    return int(m.group(1))


def _extract_titulo(soup: BeautifulSoup, *, ctx: _CardParseContext) -> str:
    # Find <div data-test-target="review-title"> then get div.span.div.a.text
    title_div = soup.find("div", attrs={"data-test-target": "review-title"})
    if title_div is None:
        raise AssertionError(f"[{ctx.file_path}] Missing div[data-test-target=review-title]")

    a = title_div.select_one("span div a") or title_div.find("a")
    text = (a.get_text(" ", strip=True) if a else title_div.get_text(" ", strip=True))
    if not text:
        raise AssertionError(f"[{ctx.file_path}] Empty title")
    return text


def _extract_em_companhia_de(soup: BeautifulSoup, *, ctx: _CardParseContext) -> Optional[str]:
    # Find a text like: "out. de 2025\xa0•\xa0Casal"
    # Return substring after the bullet separator.
    # Rule: month with 3 characters + dot + space + (optional 'de ') + year with 4 digits at the beginning.
    pattern = re.compile(
        r"^(?P<mon>[A-Za-zÀ-ÿ]{3})\.\s+(?:de\s+)?(?P<year>\d{4})\s*•\s*(?P<company>.+?)\s*$",
        re.IGNORECASE,
    )

    for text in soup.stripped_strings:
        normalized = " ".join(text.replace("\xa0", " ").split())
        m = pattern.match(normalized)
        if m:
            company = m.group("company").strip()
            if company:
                return company

    return None
    

def _extract_review_text(soup: BeautifulSoup, *, ctx: _CardParseContext) -> str:
    # Find div[data-test-target=review-body]. Go down until find a span, then take span.div.text
    body = soup.find("div", attrs={"data-test-target": "review-body"})
    if body is None:
        raise AssertionError(f"[{ctx.file_path}] Missing div[data-test-target=review-body]")

    span = body.find("span")
    if span is None:
        text = body.get_text(" ", strip=True)
        if not text:
            raise AssertionError(f"[{ctx.file_path}] Empty review body")
        return text

    span_div = span.find("div")
    text = (span_div.get_text(" ", strip=True) if span_div else span.get_text(" ", strip=True))
    if not text:
        raise AssertionError(f"[{ctx.file_path}] Empty review text")
    return text


def _extract_imagens(soup: BeautifulSoup, *, ctx: _CardParseContext) -> int:
    # Count images uploaded in the review.
    # Strategy: count only review-image buttons like:
    # <button aria-label="Ver imagem completa da avaliação"><picture><img .../></picture></button>
    # This intentionally ignores unrelated images (e.g., user profile avatar <img>).
    buttons = soup.find_all("button", attrs={"aria-label": "Ver imagem completa da avaliação"})

    def check_if_button_is_image(btn: Tag) -> bool:
        if getattr(btn, "name", None) != "button":
            return False
        picture = btn.find("picture")
        if picture is None:
            return False
        img = picture.find("img")
        if img is None:
            return False
        return True

    review_images_list = [button for button in buttons if check_if_button_is_image(button)]
    return len(review_images_list)


def _extract_is_parceria_patrocinada(html: str) -> bool:
    return "Avaliação recebida em parceria com este restaurante" in html


def _extract_data_postagem(soup: BeautifulSoup, *, ctx: _CardParseContext) -> datetime:
    # Look for a text like: "Feita em 29 de dezembro de 2025" anywhere in the card.
    # Then parse the substring date part.
    pattern = re.compile(
        r"\bFeita\s+em\s+(\d{1,2}\s+de\s+[A-Za-zçÇáéíóúâêîôûãõÁÉÍÓÚÂÊÎÔÛÃÕ]+\s+de\s+\d{4})\b",
        re.IGNORECASE,
    )

    for text in soup.stripped_strings:
        normalized = " ".join(text.replace("\xa0", " ").split())
        m = pattern.search(normalized)
        if m:
            return _parse_pt_br_date(m.group(1), ctx=ctx)

    raise AssertionError(f"[{ctx.file_path}] Could not find date text matching 'Feita em <date>'")


def _extract_nota_categoria(
    soup: BeautifulSoup,
    *,
    ctx: _CardParseContext,
    label_text: str,
) -> Optional[int]:
    # Strategy (shared by all 4 category grades):
    # - Find a div whose text is exactly label_text
    # - Find the svg right after that div
    # - svg must have aria-labelledby starting with ":lithium-r"
    # - svg must contain <title id="<aria-labelledby>"> like "5,0 de 5 círculos"
    # - Return the grade as int (e.g., 5)

    label_div = None
    for div in soup.find_all("div"):
        text = div.get_text(" ", strip=True)
        if text == label_text:
            label_div = div
            break

    if label_div is None:
        return None

    # Find the next *tag* sibling (skip NavigableString/whitespace)
    next_tag = label_div.next_sibling
    while next_tag is not None and getattr(next_tag, "name", None) is None:
        next_tag = next_tag.next_sibling

    if next_tag is None or (not isinstance(next_tag, Tag)) or next_tag.name != "svg":
        raise AssertionError(f"[{ctx.file_path}] Found '{label_text}' div but no svg right after it")

    svg: Tag = next_tag
    aria_value = svg.get("aria-labelledby")
    if not isinstance(aria_value, str) or (not aria_value.startswith(":lithium-r")):
        raise AssertionError(
            f"[{ctx.file_path}] '{label_text}' svg aria-labelledby expected prefix ':lithium-r', got {aria_value!r}"
        )

    title_tag = svg.find("title", id=aria_value)
    if title_tag is None:
        raise AssertionError(f"[{ctx.file_path}] '{label_text}' svg missing <title id={aria_value!r}>")

    # Example: "5,0 de 5 círculos" -> 5
    title_text = title_tag.get_text(" ", strip=True)
    m = re.search(r"\b(\d+)(?:[\.,]\d+)?\b", title_text)
    if not m:
        raise ValueError(f"[{ctx.file_path}] Could not parse '{label_text}' grade from: {title_text!r}")
    return int(m.group(1))


def parse_tripadvisor_review_card(html: str, *, ctx: _CardParseContext) -> TripAdvisorReview:
    soup = BeautifulSoup(html, "lxml")

    review = TripAdvisorReview()
    review.nome = _extract_nome(soup, ctx=ctx)
    review.cidade_e_estado = _extract_cidade_e_estado(soup, ctx=ctx)
    review.contribuicoes = _extract_contribuicoes(soup) or 0
    review.nota = _extract_nota(soup, ctx=ctx)
    review.titulo = _extract_titulo(soup, ctx=ctx)
    review.em_companhia_de = _extract_em_companhia_de(soup, ctx=ctx)
    review.review = _extract_review_text(soup, ctx=ctx)
    review.imagens = _extract_imagens(soup, ctx=ctx)
    review.nota_custo = _extract_nota_categoria(soup, ctx=ctx, label_text="Custo")
    review.nota_atendimento = _extract_nota_categoria(soup, ctx=ctx, label_text="Atendimento")
    review.nota_comida = _extract_nota_categoria(soup, ctx=ctx, label_text="Comida")
    review.nota_ambiente = _extract_nota_categoria(soup, ctx=ctx, label_text="Ambiente")
    review.is_parceria_patrocinada = _extract_is_parceria_patrocinada(html)
    review.data_postagem = _extract_data_postagem(soup, ctx=ctx)
    return review


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {RAW_DIR}")

    card_files = sorted(RAW_DIR.glob("card_*.html"))
    if not card_files:
        raise FileNotFoundError(f"No card_*.html files found in {RAW_DIR}")

    reviews: list[TripAdvisorReview] = []
    for file_path in tqdm(card_files, desc="Tripadvisor cards", unit="card"):
        html = file_path.read_text(encoding="utf-8")
        ctx = _CardParseContext(file_path=file_path)
        reviews.append(parse_tripadvisor_review_card(html, ctx=ctx))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_dict() for r in reviews])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()

