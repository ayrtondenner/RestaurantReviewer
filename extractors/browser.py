from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def start_chrome(lang: str = "pt-BR"):
	options = Options()
	options.add_argument(f"--lang={lang}")

	service = Service(ChromeDriverManager().install())
	driver = webdriver.Chrome(service=service, options=options)

	# Best-effort: on Windows, move to the monitor to the right (if any).
	try:
		import ctypes

		if hasattr(ctypes, "windll"):
			user32 = ctypes.windll.user32
			primary_width = user32.GetSystemMetrics(0)
			driver.set_window_position(int(primary_width) + 10, 0)
	except Exception:
		pass

	try:
		driver.maximize_window()
	except Exception:
		pass

	return driver
