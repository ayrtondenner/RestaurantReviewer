from extractors.browser import start_chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


COMMENTS_REVIEW_URL = (
    "https://www.google.com/search?q=lvtetia+sp#lrd=0x94ce59e62ed71977:0xc95d0c439f3e1f01,1,,,,"
)


def main():
    driver = start_chrome(lang="pt-BR")

    try:
        driver.get(COMMENTS_REVIEW_URL)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located(
                (By.XPATH, "//div[contains(normalize-space(.), 'Ordenar por')]"),
            )
        )
        print('Found div containing text: "Ordenar por"')
    finally:
        driver.quit()


if __name__ == "__main__":
    main()

