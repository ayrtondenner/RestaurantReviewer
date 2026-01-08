from pathlib import Path
import re

from selenium.common.exceptions import StaleElementReferenceException
from extractors.browser import start_chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm


FULL_PAGES_DIR = Path("full_page") / "tripadvisor"
OUT_DIR = Path("raw_data") / "tripadvisor"
EXPECTED_TOTAL_CARDS = 70


_VOID_ELEMENTS = {
	"area",
	"base",
	"br",
	"col",
	"embed",
	"hr",
	"img",
	"input",
	"link",
	"meta",
	"param",
	"source",
	"track",
	"wbr",
}


def _wait_page_ready(driver, timeout_seconds: int = 30) -> None:
	WebDriverWait(driver, timeout_seconds).until(
		lambda d: d.execute_script("return document.readyState") == "complete"
	)


def _wait_review_cards(driver, timeout_seconds: int = 30) -> None:
	WebDriverWait(driver, timeout_seconds).until(
		lambda d: len(d.find_elements(By.CSS_SELECTOR, 'div[data-automation="reviewCard"]'))
		> 0
	)


def _scroll_to_element(driver, element) -> None:
	driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)



def _pretty_indent_html(html: str) -> str:
	# Lightweight formatter (no extra dependencies). Not a full HTML parser, but
	# improves readability by inserting newlines and indentation between tags.

	# Normalize tag boundaries to newlines.
	s = re.sub(r">\s+<", "><", html.strip())
	s = s.replace("><", ">\n<")

	lines = [line.strip() for line in s.splitlines() if line.strip()]
	indented: list[str] = []
	indent = 0

	for line in lines:
		if line.startswith("</"):
			indent = max(0, indent - 1)

		indented.append("\t" * indent + line)

		m = re.match(r"^<\s*([a-zA-Z0-9:_-]+)", line)
		tag = (m.group(1).lower() if m else "")

		is_opening = line.startswith("<") and not line.startswith("</") and not line.startswith("<!")
		is_self_closing = line.endswith("/>")
		if is_opening and (tag not in _VOID_ELEMENTS) and (not is_self_closing) and ("</" not in line):
			indent += 1

	return "\n".join(indented) + "\n"


def _write_review_card_html(page_index: int, card_index_in_page: int, html: str) -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	file_path = OUT_DIR / f"card_{page_index}_{card_index_in_page:04d}.html"
	file_path.write_text(_pretty_indent_html(html), encoding="utf-8")


def _scrape_current_page(driver, page_index: int) -> int:
	_wait_page_ready(driver)
	_wait_review_cards(driver)

	cards = driver.find_elements(By.CSS_SELECTOR, 'div[data-automation="reviewCard"]')
	if len(cards) > 15:
		raise AssertionError(f"Expected 15 or fewer review cards, got {len(cards)}")

	saved_this_page = 0
	for card_index in range(len(cards)):
		try:
			cards = driver.find_elements(
				By.CSS_SELECTOR, 'div[data-automation="reviewCard"]'
			)
			if card_index >= len(cards):
				break
			card = cards[card_index]
			_scroll_to_element(driver, card)

			html = card.get_attribute("outerHTML")
			card_index_in_page = card_index + 1
			_write_review_card_html(page_index, card_index_in_page, html)
			saved_this_page += 1

			# Scroll a bit to keep the next card visible
			driver.execute_script("window.scrollBy(0, 250);")
		except StaleElementReferenceException:
			continue

	return saved_this_page


def main():
	driver = start_chrome(lang="pt-BR")

	try:
		if not FULL_PAGES_DIR.exists():
			raise FileNotFoundError(
				f"Missing folder: {FULL_PAGES_DIR}. Put downloaded pages in full_page/tripadvisor/"
			)

		page_files = sorted(FULL_PAGES_DIR.glob("*.html"))
		if not page_files:
			raise FileNotFoundError(
				f"No .html files found in {FULL_PAGES_DIR}"
			)

		total_saved = 0
		page_count = len(page_files)
		for page_index, page_path in tqdm(
			enumerate(page_files, start=1),
			total=page_count,
			desc="Tripadvisor pages",
			unit="page",
		):
			driver.get(page_path.resolve().as_uri())
			total_saved += _scrape_current_page(driver, page_index)

		if total_saved != EXPECTED_TOTAL_CARDS:
			raise AssertionError(
				f"Expected total saved cards to be {EXPECTED_TOTAL_CARDS}, got {total_saved}"
			)
	finally:
		driver.quit()


if __name__ == "__main__":
	main()

