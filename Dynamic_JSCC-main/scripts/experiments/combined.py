import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import cv2
import numpy as np
import spacy
nlp = spacy.load("en_core_web_lg")
def get_sim(str1,str2):
    doc1,doc2 = nlp(str1),nlp(str2)
    return doc1.similarity(doc2)
def switch2frame(d):
    ifr = d.find_element('xpath', '//*[@id="iFrameResizer0"]')
    d.switch_to.frame(ifr)

driver = webdriver.Chrome()
driver2 = webdriver.Chrome()
driver.get("https://huggingface.co/spaces/stabilityai/stable-diffusion")
# switch2frame(driver)
driver2.get('https://huggingface.co/spaces/flax-community/image-captioning#a-brown-and-white-horse-standing-next-to-a-fence')
# switch2frame(driver2)
# //*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[4]/div/div/p/a
time.sleep(5)
iframe = driver.find_element('xpath','//*[@id="iFrameResizer0"]')
iframe2 = driver2.find_element('xpath','//*[@id="iFrameResizer0"]')
driver.switch_to.frame(iframe)
driver2.switch_to.frame(iframe2)

img1_xpath = '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[5]/div/div/div/img'
# img1_xpath = '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[4]/div/div/p/a'
img1_save_path = r"D:\pycharmProject\paper\task2\origin"
xpath = '//*[@id="prompt-text-input"]/label/input'
                    # //*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[5]/div/div/div/img
for i in range(3):
    # img1 = driver2.find_element('xpath',img1_xpath)
    print(i)
    img1 = WebDriverWait(driver2, timeout=10000).until(
        lambda d: d.find_element(By.XPATH, img1_xpath))
    button2 = driver2.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[4]/div/button/div/p')
    browser = driver2.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[5]/div[1]/div/div[1]/div/section/button')
    upload = driver2.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[5]/div[1]/div/div[2]/div/div/button/div/p')
    # text_xpath = input()
    text = driver2.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[8]/div').text
    src1 = img1.get_attribute('src')
    urllib.request.urlretrieve(src1, fr"D:\pycharmProject\paper\task2\origin\{text}.png")
    # button2.click()
    prompt = driver.find_element(By.XPATH,xpath)
    button = driver.find_element(By.XPATH,'//*[@id="component-9"]')
    prompt.clear()
    prompt.send_keys(text)
    time.sleep(2)
    button.click()
    time.sleep(10)
    img_path = WebDriverWait(driver, timeout=10000).until(lambda d: d.find_element(By.XPATH,'//*[@id="gallery"]/div[2]/div/button[1]/img'))
    src = img_path.get_attribute('src')
    urllib.request.urlretrieve(src, fr"D:\pycharmProject\paper\task2\result2\{text}.png")
    time.sleep(5)
    # browser.send_keys(fr"D:\pycharmProject\paper\task2\result2\{text}.png")
    driver2.find_element('xpath',
                         '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[5]/div[1]/div/div[1]/div/section/input').send_keys(fr"D:\pycharmProject\paper\task2\result2\{text}.png")
    time.sleep(5)
    upload.click()
    time.sleep(10)
    text2 = driver2.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[7]/div/div/div').text
    sim = get_sim(text,text2)
    print(text,text2,sim)

    # urllib.request.urlretrieve(src, fr"D:\pycharmProject\paper\task2\{tmp}.png")

    driver.refresh()
    driver2.refresh()
    iframe = driver.find_element('xpath', '//*[@id="iFrameResizer0"]')
    iframe2 = driver2.find_element('xpath', '//*[@id="iFrameResizer0"]')
    driver.switch_to.frame(iframe)
    driver2.switch_to.frame(iframe2)
    time.sleep(5)
# "css-10trblm e16nr0p30"  <span class="css-10trblm e16nr0p30">a baseball player wearing a catchers mitt on a baseball field</span>