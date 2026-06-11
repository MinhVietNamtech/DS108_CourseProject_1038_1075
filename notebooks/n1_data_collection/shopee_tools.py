import os
import re
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
import random
from selenium.webdriver.common.action_chains import ActionChains
from datetime import datetime
import pickle
import time

# Write file txt function
def write_file_txt(filename, list_data):
    try:

        with open(filename, "w") as f:
            for item in list_data:
                f.write(item)
    except:
        print('Lỗi không ghi được file')

# Read file txt function
def read_file_txt(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines]

# Simulate human actions
def human_action_simulation(driver):
    total_height = driver.execute_script("return document.body.scrollHeight")
    current_pos = driver.execute_script("return window.pageYOffset")
    
    scroll_steps = random.randint(3, 6)
    for _ in range(scroll_steps):
        scroll_distance = random.randint(200, 600)
        driver.execute_script(f"window.scrollBy(0, {scroll_distance});")
        # Nghỉ ngắn sau mỗi lần cuộn để giả vờ như đang đọc nội dung
        time.sleep(random.uniform(0.5, 2.0))
    try:
        actions = ActionChains(driver)
        # Lấy kích thước cửa sổ trình duyệt
        width = driver.get_window_size()['width']
        height = driver.get_window_size()['height']
        # Di chuyển đến một vài điểm ngẫu nhiên
        for _ in range(random.randint(2, 4)):
            x_offset = random.randint(0, width - 1)
            y_offset = random.randint(0, height - 1)
            try: 
                actions.move_by_offset(x_offset, y_offset).perform() # Nếu offset là hợp lệ
            except:
                pass # Không di chuyển chuột khi OutOfBound
            # Reset vị trí chuột về (0,0) để tránh lỗi di chuyển ra ngoài màn hình ở vòng lặp sau
            actions.move_to_element_with_offset(driver.find_element("tag name", "body"), 0, 0).perform()
            time.sleep(random.uniform(0.2, 0.8))
    except Exception as e:
        print(f"Lỗi khi giả lập chuột: {e}")

    # 3. Nghỉ ngẫu nhiên trước khi thực hiện hành động tiếp theo (Deep Sleep)
    time.sleep(random.uniform(1.5, 4.0))

def get_data(list_link, restaurantid_start, reviewid_start, chrome_options):
    # Place features
    restaurant_id = restaurantid_start
    name_lst = []
    address_lst = []
    type_lst = []
    has_wifi = []
    has_onl = []
    has_table_book = []
    cuisines_lst = []
    price_lst = []
    style_lst = []
    quality_lst = []
    serve_lst = []
    space_lst = []
    avg_rating = []
    total_rates = []
    restaurantid_lst = []
    mon = []
    tue = []
    wed = []
    thu = []
    fri = []
    sat = []
    sun = []
    latitude = []
    longitude = []

    # Review features
    review_id = reviewid_start
    restaurant_review_lst = []
    user_review_lst = []
    time_review_lst = []
    rate_review_lst = []
    comment_review_lst = []
    reviewid_lst = []

    driver = webdriver.Chrome(options = chrome_options)
    driver.maximize_window()
    
    list_links = list_link
    if restaurant_id == 1:
        # This wipes the file completely
        open('../../data/data_raw/shopee_csv/Dtmp.csv', 'w').close()
        open('../../data/data_raw/shopee_csv/Ctmp.csv', 'w').close()

    for link in list_links:
        print(f"current: {restaurant_id}")

        name=''; address=''; price=''; rate = np.nan; total=''; kind=''; cuisines=''; style=''; wifi=''; deli=''; table=''; t2=''; t3=''; t4=''; t5=''; t6=''; t7=''; cn=''; qual = 0.0; ser = 0.0; spa = 0.0; lat=''; long=''; review_user=''; review_time=''; review_rating=0.0; review_comment=''
        link = link.rstrip('\n')
        if link == 'link error':
            continue
        else:
            driver.get(link)
            time.sleep(np.random.uniform(1, 3))
            human_action_simulation(driver)
            # Trong trường hợp hiện thông báo ngoài giờ mở cửa
            try:
                close_button = driver.find_element(By.CLASS_NAME, 'close')
                driver.execute_script("arguments[0].click();", close_button)
            except:
                pass
            try:
                close_button = driver.find_element(By.CLASS_NAME, 'close')
                driver.execute_script("arguments[0].click();", close_button)
            except:
                pass

            # Place ID
            restaurantid_lst.append(restaurant_id)
        
            # Place Name
            try:
                name = driver.find_element(By.XPATH, '//div[@class="detail-restaurant-info"]/h1[@class="name-restaurant"]').text
            except:
                pass
            # Place Address
            try:
                address = driver.find_element(By.XPATH, '//div[@class="detail-restaurant-info"]/div[@class="address-restaurant"]').text
            except:
                pass
            # Place Price
            try:
                price = driver.find_element(By.XPATH, '//div[@class="detail-restaurant-info"]/div[@class="cost-restaurant"]').text
            except:
                pass
            # Place average Rating
            try:
                star_region = driver.find_element(By.CLASS_NAME, 'stars')
                fullstar_icons = star_region.find_elements(By.CLASS_NAME, 'full')
                halfstar_icons = star_region.find_elements(By.CLASS_NAME, 'half')
                rate = len(fullstar_icons) + len(halfstar_icons)*0.5
            except:
                pass
            # Place total rates
            try:
                total = driver.find_element(By.XPATH, '//*[@id="app"]/div/div[1]/div/div[2]/div[3]/span').text
            except:
                pass         
               
            # Review
            try:
                link_review = driver.find_element(By.XPATH, '//div[@class="detail-restaurant-info"]/div[@class="view-more-rating"]/a').get_attribute('href')
            except:
                link_review = ''
            else:
                driver_review = webdriver.Chrome(options = chrome_options)
                driver_review.maximize_window()
                driver_review.get(link_review)
                time.sleep(np.random.uniform(2, 4))
                # Place type
                try:
                    detail = driver_review.find_elements(By.CLASS_NAME, "new-detail-info-area")
                    # Kiểm tra kind có không
                    try:
                        kind = [i.text.split('\n')[1] for i in detail if i.text.split('\n')[0] == 'Thể loại']
                        kind = (','.join(kind))
                    except:
                        pass
                    # Kiểm tra cuisines có không
                    try:
                        cuisines = [i.text.split('\n')[1] for i in detail if i.text.split('\n')[0] == 'Phục vụ các món']
                        cuisines = (','.join(cuisines))
                    except:
                        pass
                    # Kiểm tra phong cách
                    try:
                        style = [i.text.split('\n')[1] for i in detail if i.text.split('\n')[0] == 'Phong cách ẩm thực']
                        style = (','.join(style))
                    except:
                        pass
                except:
                    pass

                # Place services
                util = driver_review.find_elements(By.CLASS_NAME, "none")
                n = [i.text for i in util]
                
                # Wifi
                try:
                    if 'Có wifi' not in n:
                        wifi = 'yes'
                    else:
                        wifi = 'no'
                except:
                    pass
                
                # Online delivery
                try: 
                    if 'Có giao hàng' not in n:
                        deli = 'yes'
                    else:
                        deli = 'no'
                except:
                    pass

                # Has table booking
                try:
                    if 'Nên đặt trước' not in n:
                        table = 'yes'
                    else:
                        table = 'no'
                except:
                    pass

                # Opening time of weekdays
                try:
                    wait = WebDriverWait(driver_review, 2)
                    time_popup_button = driver_review.find_element(By.CLASS_NAME, "opening-time-btn")
                    driver_review.execute_script("arguments[0].click();", time_popup_button)
                    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".date-header.ng-binding")))
                    weekdays_el = driver_review.find_elements(By.CSS_SELECTOR, ".date-header.ng-binding")
                    weekdays = [day.text for day in weekdays_el]
                    time_of_day_el = driver_review.find_elements(By.CSS_SELECTOR, ".date-item.ng-binding.ng-scope")
                    t = [t.text for t in time_of_day_el]                   
                    t2 = t[0]
                    t3 = t[1]
                    t4 = t[2]
                    t5 = t[3]
                    t6 = t[4]
                    t7 = t[5]
                    cn = t[6]
                except Exception as e:
                    pass    

                # Points list
                try:
                    time.sleep(2)
                    human_action_simulation(driver_review)
                    lst = driver_review.find_elements(By.CLASS_NAME, "microsite-top-points")
                    raw_data = [point.text.split('\n') for point in lst if len(point.text.split('\n')) >= 2]
                    points = np.array(raw_data)
                except:
                    points = np.array([])

                # Place product quality
                try:
                    mask = (points[:, 1] == 'Chất lượng')
                    if np.any(mask):
                        val_str = points[mask, 0][0].replace(',', '.') # Thay dấu , bằng . trước khi ép kiểu
                        qual = float(val_str)
                except:
                    pass
                # Place serving quality
                try:
                    mask = (points[:, 1] == 'Phục vụ')
                    if np.any(mask):
                        val_str = points[mask, 0][0].replace(',', '.') # Thay dấu , bằng . trước khi ép kiểu
                        ser = float(val_str)
                except:
                    pass
                # Place interior design
                try:
                    mask = (points[:, 1] == 'Không gian')
                    if np.any(mask):
                        val_str = points[mask, 0][0].replace(',', '.') # Thay dấu , bằng . trước khi ép kiểu
                        spa = float(val_str)
                except:
                    pass
                 # Place Latitude and Longitude
                try:
                    wait = WebDriverWait(driver_review, 2)
                    location_btn = driver_review.find_element(By.CLASS_NAME, "linkmap")
                    driver_review.execute_script("arguments[0].click();", location_btn)
                    map_link = driver_review.find_element(By.XPATH, '//*[@id="iframes"]').get_attribute('src')
                    lat: float
                    long: float
                    if map_link:
                        match = re.search(r'([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)', map_link)                        
                        if match:
                            lat = match.group(1)
                            long= match.group(2)
                        else:
                            match_alt = re.search(r'!3d([-+]?\d+\.\d+)!4d([-+]?\d+\.\d+)', map_link)
                            if match_alt:
                                lat = match_alt.group(1)
                                long = match_alt.group(2)        
                except:
                    pass
                time.sleep(np.random.uniform(1,2))
                while True:                    
                    try:
                        start = time.time()
                        WebDriverWait(driver_review, 5).until(EC.element_to_be_clickable((By.LINK_TEXT, " Bình luận "))).click()
                        time.sleep(2)
                        end = time.time()
                        spend_time = end - start
                        if spend_time > 5:
                            break             
                    except:
                        break
                try:    
                    li_reviews = driver_review.find_elements(By.CLASS_NAME, value='review-item')
                except:
                    li_review = []
                for li_review in li_reviews:

                    # Review ID
                    reviewid_lst.append(review_id)
                    
                    # Reviewer
                    try:
                        review_user = li_review.find_element(By.CLASS_NAME, 'ru-username').text
                        user_review_lst.append(review_user)
                    except:
                        user_review_lst.append('')
                    
                    # Review Time
                    try:
                        review_time = li_review.find_element(By.CLASS_NAME, 'ru-time').text
                        time_review_lst.append(review_time)
                    except:
                        time_review_lst.append('')
        
                    # Review Rate
                    try:
                        review_rating = li_review.find_element(By.CLASS_NAME, 'review-points').text
                        rate_review_lst.append(review_rating)
                    except:
                        rate_review_lst.append(0)
        
                    # Comment
                    try:
                        try:
                            view_more_button = li_review.find_element(By.CLASS_NAME, "view-more")
                            li_review.execute_script("arguments[0].click();", view_more_button)
                        except:
                            pass

                        review_comment = li_review.find_element(By.CLASS_NAME, "rd-des").text
                        comment_review_lst.append(review_comment)
                    except:
                        comment_review_lst.append('')  
        
                    # Restaurant ID of this Review
                    restaurant_review_lst.append(restaurant_id)
                    
                    review_id += 1           
                    
                else:
                    driver_review.close()
        restaurant_id += 1

        name_lst.append(name)
        latitude.append(lat)
        longitude.append(long)
        address_lst.append(address)
        type_lst.append(kind)
        cuisines_lst.append(cuisines)
        mon.append(t2)
        tue.append(t3)
        wed.append(t4)
        thu.append(t5)
        fri.append(t6)
        sat.append(t7)
        sun.append(cn)
        style_lst.append(style)
        has_onl.append(deli)
        has_table_book.append(table)
        has_wifi.append(wifi)
        price_lst.append(price)
        quality_lst.append(qual)
        serve_lst.append(ser)
        space_lst.append(spa)
        avg_rating.append(rate)
        total_rates.append(total)
        unit_d = pd.DataFrame({'RestaurantID' : restaurantid_lst[-1:],'Restaurant Name':name_lst[-1:], 'Latitude': latitude[-1:], 'Longitude': longitude[-1:], 'Address' : address_lst[-1:], 'Type': type_lst[-1:]
                                   ,'Cuisines': cuisines_lst[-1:], 'Monday': mon[-1:], 'Tuesday': tue[-1:], 'Wednesday': wed[-1:], 'Thursday': thu[-1:], 'Friday': fri[-1:], 'Saturday': sat[-1:], 'Sunday': sun[-1:], 'Style': style_lst[-1:]
                                   ,'Has_Online_delivery': has_onl[-1:], 'Has_Table_booking': has_table_book[-1:], 'Has_Wifi': has_wifi[-1:], 'Price' : price_lst[-1:], 'Product_quality(10)': quality_lst[-1:], 'Serving_quality(10)': serve_lst[-1:]
                                   ,'Interior_design(10)': space_lst[-1:], 'Average_rating(5)':avg_rating[-1:], 'Total votes': total_rates[-1:]
                                    })
        
        unit_c = pd.DataFrame({'UserID': reviewid_lst[-1:],'User':user_review_lst[-1:], 'Review Time' : time_review_lst[-1:], 'Rating (10)' : rate_review_lst[-1:], 'Comment' : comment_review_lst[-1:], 'RestaurantID': restaurant_review_lst[-1:]})
        
        # Ghi vào file tạm để lưu dữ liệu đã cào được
        if os.path.exists("../../data/data_raw/shopee_csv/D/Dtmp.csv") and os.path.getsize("../../data/data_raw/shopee_csv/D/Dtmp.csv") > 0:
            unit_d.to_csv("../../data/data_raw/shopee_csv/D/Dtmp.csv", mode='a', index=False, header=False)
        else:
            unit_d.to_csv("../../data/data_raw/shopee_csv/D/Dtmp.csv", index=False)

        if os.path.exists("../../data/data_raw/shopee_csv/C/Ctmp.csv") and os.path.getsize("../../data/data_raw/shopee_csv/C/Ctmp.csv") > 0:
            unit_c.to_csv("../../data/data_raw/shopee_csv/C/Ctmp.csv", mode='a', encoding='utf-8-sig', index=False, header=False)
        else:
            unit_c.to_csv("../../data/data_raw/shopee_csv/C/Ctmp.csv", encoding='utf-8-sig', index=False)


    # Create dataframe
    restaurant_df = pd.DataFrame({'RestaurantID' : restaurantid_lst,'Restaurant Name':name_lst, 'Latitude': latitude, 'Longitude': longitude, 'Address' : address_lst, 'Type': type_lst
                                   ,'Cuisines': cuisines_lst, 'Monday': mon, 'Tuesday': tue, 'Wednesday': wed, 'Thursday': thu, 'Friday': fri, 'Saturday': sat, 'Sunday': sun, 'Style': style_lst
                                   ,'Has_Online_delivery': has_onl, 'Has_Table_booking': has_table_book, 'Has_Wifi': has_wifi, 'Price' : price_lst, 'Product_quality(10)': quality_lst, 'Serving_quality(10)': serve_lst
                                   ,'Interior_design(10)': space_lst, 'Average_rating(5)':avg_rating, 'Total votes': total_rates
                                    })
    restaurant_df = pd.read_csv("../../data/data_raw/shopee_csv/D/Dtmp.csv")

    review_df = pd.read_csv("../../data/data_raw/shopee_csv/C/Ctmp.csv", encoding='utf-8-sig')
        
    driver.close()
    return restaurant_df, review_df