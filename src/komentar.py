import scrapy
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from os.path import isfile
from subprocess import call
import os
import datetime
from dateutil import relativedelta


class KomentarSpider(scrapy.Spider):
    name = 'komentar'

    def __init__(self, category=None, *args, **kwargs):
        super(KomentarSpider, self).__init__(*args, **kwargs)
        self.start_urls = [category]
        self.driver = webdriver.Chrome()

    def parse(self, response):
        self.driver.get(self.start_urls[0])

        halaman = 1
        tanggal_sekarang = datetime.datetime.now()
        batas_tanggal = tanggal_sekarang - relativedelta.relativedelta(
            month=tanggal_sekarang.month-1)

        is_next = True

        while True:

            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((
                    By.XPATH, '//div[contains(@class, fyGvr)][text()="Next"]')
                )
            )
            print('HALAMAN:', halaman)

            sel = scrapy.Selector(text=self.driver.page_source)
            comments = sel.css('div._3Q1Bj._14ETA')

            for komentar in comments.css('div.r44Gh'):
                tanggal = komentar.css(
                    '.TbFQ4 ._3ydMH span:first-child::text').extract_first()
                tanggal = tanggal.split(' ')
                hari = tanggal[0]

                if len(hari) == 1:
                    hari = '0' + hari
                    tanggal[0] = hari

                if tanggal[1] == 'Mei':
                    tanggal[1] = 'May'
                if tanggal[1] == 'Agu':
                    tanggal[1] = 'Aug'
                if tanggal[1] == 'Okt':
                    tanggal[1] = 'Oct'
                if tanggal[1] == 'Des':
                    tanggal[1] = 'Dec'

                tanggal = ' '.join(tanggal)

                tanggal = datetime.datetime.strptime(tanggal, '%d %b %Y')

                if tanggal >= batas_tanggal:
                    yield {
                        'text': komentar.css('._1mI1m::text').extract_first(),
                    }
                else:
                    is_next = False
                    break

            if not is_next:
                break

            next = self.driver.find_element_by_xpath(
                '//div[contains(@class, fyGvr)][text()="Next"]')

            try:
                next.click()
                WebDriverWait(self.driver, 5)
                halaman += 1

            except:
                print('halaman habis atau error')
                break

            if halaman > 10:
                break

        self.driver.close()


def crawl_data(url):
    print('crawling...')

    app_path = os.getcwd()
    save_path = app_path+'/data/dinamics/live_data.json'

    if isfile(save_path):
        os.remove(save_path)

    code_path = app_path+'/src/komentar.py'
    call(['scrapy', 'runspider', code_path, '-o',
          save_path, '-a', 'category='+url])
