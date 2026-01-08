import time
from pathlib import Path

from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from extractors.browser import start_chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


COMMENTS_REVIEW_URL = "https://www.tripadvisor.com.br/Restaurant_Review-g303631-d23847312-Reviews-Lvtetia_Erick_Jacquin-Sao_Paulo_State_of_Sao_Paulo.html"

OUT_DIR = Path("raw_data") / "tripadvisor"


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


def _write_review_card_html(page_index: int, card_index: int, html: str) -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	file_path = OUT_DIR / f"page_{page_index:03d}_card_{card_index:02d}.html"
	file_path.write_text(html, encoding="utf-8")


def _scrape_current_page(driver, page_index: int) -> int:
	_wait_page_ready(driver)
	_wait_review_cards(driver)

	cards = driver.find_elements(By.CSS_SELECTOR, 'div[data-automation="reviewCard"]')
	if len(cards) > 15:
		raise AssertionError(f"Expected 15 or fewer review cards, got {len(cards)}")

	for card_index in range(len(cards)):
		try:
			cards = driver.find_elements(
				By.CSS_SELECTOR, 'div[data-automation="reviewCard"]'
			)
			if card_index >= len(cards):
				break
			card = cards[card_index]

			_scroll_to_element(driver, card)

			# Expand inside this card if the "Leia mais" button exists
			read_more_buttons = card.find_elements(
				By.XPATH,
				".//button[.//span[contains(normalize-space(.), 'Leia mais')]]",
			)
			if read_more_buttons:
				button = read_more_buttons[0]
				_scroll_to_element(driver, button)
				try:
					button.click()
				except Exception:
					driver.execute_script("arguments[0].click();", button)
				time.sleep(1)

			# Re-acquire the card after potential DOM changes and save its HTML
			cards = driver.find_elements(
				By.CSS_SELECTOR, 'div[data-automation="reviewCard"]'
			)
			if card_index < len(cards):
				card = cards[card_index]
				html = card.get_attribute("outerHTML")
				_write_review_card_html(page_index, card_index, html)

			# Move focus down a bit before next card
			driver.execute_script("window.scrollBy(0, 250);")
		except StaleElementReferenceException:
			continue

	return len(cards)


def _go_to_next_page(driver) -> bool:
	# Next arrow is an <a data-smoke-attr="pagination-next-arrow">
	anchors = driver.find_elements(
		By.CSS_SELECTOR, 'a[data-smoke-attr="pagination-next-arrow"]'
	)
	if not anchors:
		return False

	next_link = anchors[0]
	_scroll_to_element(driver, next_link)
	try:
		next_link.click()
	except Exception:
		driver.execute_script("arguments[0].click();", next_link)
	return True


def main():
	driver = start_chrome(lang="pt-BR")

	try:
		driver.get(COMMENTS_REVIEW_URL)

		page_index = 1
		while True:
			try:
				_scrape_current_page(driver, page_index)
			except TimeoutException:
				# If the page structure changes or content is blocked, exit cleanly.
				break

			# Try to paginate; if no next button found, we're done.
			if not _go_to_next_page(driver):
				break

			page_index += 1
			_wait_page_ready(driver)
	finally:
		driver.quit()


if __name__ == "__main__":
	main()

