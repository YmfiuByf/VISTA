from selenium import webdriver
from selenium.webdriver.common.by import By
import time


test = webdriver.Chrome()
# test.get('https://docs.google.com/forms/d/e/1FAIpQLSeYUmAYYZNtbU8t8MRxwJo-d1zkmSaEHodJXs78RzoG0yFY2w/viewform')
test.get('https://www.youtube.com/watch?v=0vhEi4m6RUw')
time.sleep(5)
# //*[@id="mG61Hd"]/div[2]/div/div[2]/div[1]/div/div/div[2]/div/div[1]/div/div[1]/input
#<input type="text" class="whsOnd zHQkBf" jsname="YPqjbf" autocomplete="off" tabindex="0" aria-labelledby="i1" aria-describedby="i2 i3" required="" dir="auto" data-initial-dir="auto" data-initial-value="">
Name = 'kuch bhi'
# last = test.find_element(By.XPATH,'//*[@id="mG61Hd"]/div[2]/div/div[2]/div[1]/div/div/div[2]/div/div[1]/div/div[1]/input')
last = test.find_element(By.XPATH,'//*[@id="subscribe-button"]/ytd-subscribe-button-renderer/yt-smartimation/yt-button-shape/button/yt-touch-feedback-shape/div/div[2]')
#         '//*[@id="mG61Hd"]/div[2]/div/div[2]/div[1]/div/div/div[2]/div/div[1]/div/div[1]/input'
#         '//*[@id="mG61Hd"]/div[2]/div/div[2]/div[2]/div/div/div[2]/div/div[1]/div/div[1]/input'
#         '//*[@id="mG61Hd"]/div[2]/div/div[2]/div[2]/div/div/div[2]/div/div[1]/div/div[1]/input
# /html/body/div/div[2]/form/div[2]/div/div[2]/div[1]/div/div/div[2]/div/div[1]/div/div[1]/input
# //*[@id="subscribe-button"]/ytd-subscribe-button-renderer/yt-smartimation/yt-button-shape/button/yt-touch-feedback-shape/div/div[2]
# last.send_keys(Name)
time.sleep(10)