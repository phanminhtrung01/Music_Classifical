from seleniumwire import webdriver


def interceptor(request):
    # Block PNG, JPEG and GIF images
    if request.path.endswith(('.png', '.jpg', '.gif')):
        request.abort()


def getAllSong():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("headless")
    chrome_options.add_argument("incognito")

    driver = webdriver.Chrome(chrome_options=chrome_options)
    # driver.scopes = [
    #     '.*https://zingmp3.vn/api/v2*.',
    # ]
    driver.get('https://zingmp3.vn/album/Lo-Duyen-Kiep-Nay-Lieu-Co-Kiep-Sau-Single-Kha-Hiep-ACV/6B7WIWAC.html')

    print(driver.page_source)
    # html = driver.find_element(by='tag name', value='html')

    # for request in driver.requests:
    #     if request.response:
    #         print(
    #             request.url,
    #             request.response.status_code,
    #             request.response.headers['Content-Type']
    #         )


getAllSong()
