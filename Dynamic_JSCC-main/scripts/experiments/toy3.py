import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, TimeoutException
change_xpath = '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[4]/div/button'
driver2 = webdriver.Edge()
driver2.get('https://huggingface.co/spaces/flax-community/image-captioning#a-brown-and-white-horse-standing-next-to-a-fence')
time.sleep(5)
iframe2 = driver2.find_element('xpath','//*[@id="iFrameResizer0"]')
driver2.switch_to.frame(iframe2)
text2 = driver2.find_element('xpath',
                             '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div[1]/div/div[7]/div/div/div').text
print('Predicted' in text2)
uploaded_img = '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[5]/div[1]/div/div[1]/div/div/ul/li/div'
time.sleep(5)
driver2.find_element('xpath',change_xpath).click()
time.sleep(5)
try:    driver2.find_element('xpath',uploaded_img)
except NoSuchElementException:
    driver2.find_element('xpath',
                         '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[5]/div[1]/div/div[1]/div/section/input').send_keys(r"C:\Users\DELL\Desktop\PSNR vs SNR4.png")
upload = driver2.find_element('xpath',
                                 '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div[1]/div/div[5]/div[1]/div/div[2]/div/div/button/div/p')
not_uploaded = True
while not_uploaded:
    upload.click()
    try:
        driver2.find_element('xpath',uploaded_img)
        print('not uploaded')
    except NoSuchElementException: not_uploaded=False

