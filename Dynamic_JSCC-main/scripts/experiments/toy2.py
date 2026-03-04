import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, TimeoutException

def document_initialised(driver):
    return driver.execute_script("return initialised")


driver = webdriver.Chrome()
driver.get("https://huggingface.co/spaces/stabilityai/stable-diffusion")
time.sleep(5)
iframe = driver.find_element('xpath','//*[@id="iFrameResizer0"]')
driver.switch_to.frame(iframe)
img1_xpath = '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[5]/div/div/div/img'
xpath = '//*[@id="prompt-text-input"]/label/input'
time.sleep(5)
icon = driver.find_element('xpath','//*[@id="gallery"]/div[2]/div/div')
# bar = driver.find_element('xpath','//*[@id="gallery"]/div[1]/div[1]')
# print(bar)
driver.find_element('xpath','//*[@id="component-9"]').click()
time.sleep(60)
bar = driver.find_element('xpath','//*[@id="gallery"]/div[1]/div[1]')
print(bar)
driver.find_element('xpath','//*[@id="prompt-text-input"]/label/input').send_keys('young boy')
# time.sleep(10)

try:
    driver.find_element('xpath', '//*[@id="gallery"]/div[1]/div/div[2]')
    time.sleep(15)
    print('problemetic keywords')
    driver.refresh()
    time.sleep(20)
    flag = True
    while flag:
        try:
            iframe = driver.find_element('xpath', '//*[@id="iFrameResizer0"]')
            flag = False
        except NoSuchElementException:
            time.sleep(2)
            driver.refresh()
            time.sleep(20)
            iframe = driver.find_element('xpath', '//*[@id="iFrameResizer0"]')
    flag = True
    driver.switch_to.frame(iframe)
except NoSuchElementException:
    print('nothing')
    pass

time.sleep(20)