import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

def document_initialised(driver):
    return driver.execute_script("return initialised")


driver = webdriver.Chrome()
driver2 = webdriver.Chrome()
driver.get("https://huggingface.co/spaces/stabilityai/stable-diffusion")
driver2.get('https://huggingface.co/spaces/flax-community/image-captioning#a-brown-and-white-horse-standing-next-to-a-fence')
# time.sleep(10)
# driver.manage().timeouts().pageLoadTimeout(30, TimeUnit.SECONDS);
iframe = driver.find_element('xpath','//*[@id="iFrameResizer0"]')
iframe2 = driver2.find_element('xpath','//*[@id="iFrameResizer0"]')
driver.switch_to.frame(iframe)
driver2.switch_to.frame(iframe2)
img1_xpath = '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[5]/div/div/div/img'
xpath = '//*[@id="prompt-text-input"]/label/input'
for text in ["a man lying on the beach","a woman lying on the beach"]:
    element = driver.find_element(By.XPATH,xpath)
    button = driver.find_element(By.XPATH,'//*[@id="component-9"]')
    element.clear()
    element.send_keys(text)
    time.sleep(2)
    button.click()
    time.sleep(10)
    img = WebDriverWait(driver, timeout=10000).until(lambda d: d.find_element(By.XPATH,'//*[@id="gallery"]/div[2]/div/button[1]/img'))
    print(img)
    src = img.get_attribute('src')
    # download the image
    urllib.request.urlretrieve(src, fr"D:\pycharmProject\paper\task2\result2\{text}.png")
    time.sleep(5)
