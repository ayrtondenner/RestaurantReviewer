from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def start_chrome(lang: str = "pt-BR"):
	options = Options()
	options.add_argument(f"--lang={lang}")

	service = Service(ChromeDriverManager().install())
	return webdriver.Chrome(service=service, options=options)
