import os
import re
import time
import random
import logging
import json
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from threading import Lock
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
MAX_WORKERS  = 6      # Số luồng scrape song song
DRIVER_POOL  = 6      # Số driver tái sử dụng (= MAX_WORKERS)
OUTPUT_DIR   = r"../../data/data_raw/foody_csv"
TXT_DIR      = "../../data/data_raw/txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TXT_DIR,    exist_ok=True)

CHECKPOINT_RES = os.path.join(OUTPUT_DIR, "checkpoint_restaurants.csv")
CHECKPOINT_REV = os.path.join(OUTPUT_DIR, "checkpoint_reviews.csv")

COOKIE_FILE    = os.path.join(OUTPUT_DIR, "foody_cookies.json")   # Cache session

FOODY_LOGIN_URL = "https://www.foody.vn"

# ─── Thông tin tài khoản Foody ─────────────────────────────────────────────────
# Điền email và mật khẩu vào đây, hoặc đặt biến môi trường FOODY_EMAIL / FOODY_PASS
FOODY_EMAIL = os.getenv("FOODY_EMAIL", "nguyenquangminh772006@gmail.com")
FOODY_PASS  = os.getenv("FOODY_PASS",  "24521075")


# ═══════════════════════════════════════════════════════════════════════════════
# DRIVER FACTORY
# ═══════════════════════════════════════════════════════════════════════════════
def create_driver(headless: bool = True, block_css: bool = False) -> webdriver.Chrome:
    opt = Options()
    if headless:
        opt.add_argument("--headless=new")
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-gpu")
    opt.add_argument("--disable-dev-shm-usage")
    opt.add_argument("--window-size=1920,1080")
    opt.add_argument("--disable-blink-features=AutomationControlled")
    opt.add_experimental_option("excludeSwitches", ["enable-automation"])
    opt.add_experimental_option("useAutomationExtension", False)
    # Tắt ảnh + font; KHÔNG tắt CSS khi lấy link vì Foody dùng CSS sentinel cho lazy-load
    prefs = {
        "profile.managed_default_content_settings.images": 2,  # Chặn ảnh
        "profile.managed_default_content_settings.fonts":  2,  # Chặn font
    }
    if block_css:
        # Chỉ dùng khi scrape trang chi tiết quán (không cần IntersectionObserver)
        prefs["profile.managed_default_content_settings.stylesheets"] = 2
    opt.add_experimental_option("prefs", prefs)
    opt.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=opt)


# ═══════════════════════════════════════════════════════════════════════════════
# ĐĂNG NHẬP FOODY — lưu cookies để tái sử dụng
# ═══════════════════════════════════════════════════════════════════════════════

def login_foody(email: str = FOODY_EMAIL, password: str = FOODY_PASS) -> list[dict]:
    """
    Mở trình duyệt (KHÔNG headless) để đăng nhập Foody một lần duy nhất.
    Sau khi đăng nhập thành công, lưu cookies ra file JSON để các driver sau
    inject lại → không cần login lại mỗi lần chạy.

    Trả về danh sách cookie dict.
    """
    log.info("Bắt đầu đăng nhập Foody (mở trình duyệt hiển thị)...")
    driver = create_driver(headless=False)   # Phải thấy trang để xử lý CAPTCHA nếu có

    try:
        driver.get(FOODY_LOGIN_URL)
        time.sleep(2)

        # ── Bấm nút "Đăng nhập" trên header ──────────────────────────────────
        try:
            btn_login = WebDriverWait(driver, 8).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn-login, a[href*='login'], .btn-header-login"))
            )
            driver.execute_script("arguments[0].click();", btn_login)
            time.sleep(1.5)
        except Exception:
            log.warning("Không tìm thấy nút Đăng nhập trên header, thử vào thẳng trang login...")
            driver.get("https://id.foody.vn/dang-nhap?returnUrl=https%3A%2F%2Fwww.foody.vn%2F")
            time.sleep(2)

        # ── Điền email ────────────────────────────────────────────────────────
        try:
            field_email = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[name='Email'], #Email"))
            )
            field_email.clear()
            field_email.send_keys(email)
        except Exception as e:
            log.error(f"Không tìm thấy ô email: {e}")
            driver.quit()
            return []

        # ── Điền mật khẩu ─────────────────────────────────────────────────────
        try:
            field_pass = driver.find_element(By.CSS_SELECTOR, "input[type='password'], input[name='Password'], #Password")
            field_pass.clear()
            field_pass.send_keys(password)
        except Exception as e:
            log.error(f"Không tìm thấy ô mật khẩu: {e}")
            driver.quit()
            return []

        # ── Click nút Submit ──────────────────────────────────────────────────
        try:
            submit = driver.find_element(By.CSS_SELECTOR, "button[type='submit'], input[type='submit'], .btn-login-submit")
            driver.execute_script("arguments[0].click();", submit)
        except Exception as e:
            log.error(f"Không tìm thấy nút submit: {e}")
            driver.quit()
            return []

        # ── Chờ đăng nhập thành công (avatar hoặc tên user hiện ra) ──────────
        try:
            WebDriverWait(driver, 15).until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".header-user-avatar, .user-avatar, .logout")),
                    EC.url_contains("foody.vn"),
                )
            )
            log.info("✓ Đăng nhập Foody thành công!")
        except Exception:
            log.warning("Chờ login timeout — kiểm tra lại email/mật khẩu hoặc xử lý CAPTCHA thủ công.")
            input("Nếu cần xử lý CAPTCHA thủ công, hãy hoàn tất rồi nhấn Enter để tiếp tục...")

        # ── Lưu cookies ──────────────────────────────────────────────────────
        cookies = driver.get_cookies()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(COOKIE_FILE, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
        log.info(f"✓ Đã lưu {len(cookies)} cookies → {COOKIE_FILE}")
        return cookies

    finally:
        driver.quit()


def load_cookies_to_driver(driver: webdriver.Chrome) -> bool:
    """
    Inject cookies đã lưu vào driver hiện tại.
    Gọi hàm này SAU KHI driver đã mở trang foody.vn (domain phải khớp).
    Trả về True nếu thành công.
    """
    if not os.path.exists(COOKIE_FILE):
        log.warning(f"Không tìm thấy file cookie: {COOKIE_FILE}. Hãy chạy login_foody() trước.")
        return False

    with open(COOKIE_FILE, encoding="utf-8") as f:
        cookies = json.load(f)

    driver.get(FOODY_LOGIN_URL)   # Phải ở đúng domain trước khi add cookie
    time.sleep(1)
    for ck in cookies:
        # Selenium không chấp nhận một số trường không hợp lệ
        ck.pop("sameSite", None)
        try:
            driver.add_cookie(ck)
        except Exception:
            pass

    driver.refresh()
    time.sleep(1.5)
    log.info("✓ Đã inject cookies Foody vào driver.")
    return True


def ensure_foody_cookies() -> bool:
    """
    Kiểm tra xem cookie đã tồn tại chưa. Nếu chưa → gọi login_foody().
    Gọi 1 lần ở đầu chương trình trước khi khởi động DriverPool.
    """
    if os.path.exists(COOKIE_FILE):
        log.info(f"Cookie Foody đã có sẵn: {COOKIE_FILE}")
        return True
    log.info("Chưa có cookie Foody, tiến hành đăng nhập...")
    cookies = login_foody()
    return len(cookies) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# DRIVER POOL — tái sử dụng driver thay vì tạo mới mỗi lần
# ═══════════════════════════════════════════════════════════════════════════════
class DriverPool:
    """
    Pool driver đơn giản: mỗi worker mượn 1 driver, dùng xong trả lại.
    Tránh overhead khởi động Chrome mỗi lần (~2-3s/lần).
    Mỗi driver được inject cookie Foody ngay khi khởi tạo để có session đăng nhập.
    """
    def __init__(self, size: int):
        self._pool: Queue = Queue()
        self._size = size
        log.info(f"Khởi động {size} Chrome drivers (có session Foody)...")
        for _ in range(size):
            d = create_driver(headless=True, block_css=True)
            load_cookies_to_driver(d)
            self._pool.put(d)          # ← BUG CŨ: thiếu dòng này, driver không vào pool!
        log.info(f"Driver pool sẵn sàng ({size} drivers).")

    def acquire(self, timeout: int = 30) -> webdriver.Chrome:
        try:
            return self._pool.get(timeout=timeout)
        except Empty:
            raise RuntimeError(
                f"DriverPool hết driver sau {timeout}s — pool_size={self._size}, "
                "có thể driver bị leak (không release) hoặc pool_size quá nhỏ."
            )

    def release(self, driver: webdriver.Chrome):
        if driver is None:
            return
        try:
            driver.execute_script("window.localStorage.clear(); window.sessionStorage.clear();")
        except Exception:
            pass  # Driver có thể đã crash — vẫn phải put lại để không leak
        finally:
            self._pool.put(driver)  # LUÔN trả về pool dù execute_script lỗi

    def close_all(self):
        while not self._pool.empty():
            try:
                self._pool.get_nowait().quit()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: LẤY DANH SÁCH LINK — tối ưu scroll
# ═══════════════════════════════════════════════════════════════════════════════

# JS: scroll xuống cuối trang 1 phát, trả về chiều cao mới
_JS_SCROLL_BOTTOM = "window.scrollTo(0, document.body.scrollHeight); return document.body.scrollHeight;"

# JS: lấy toàn bộ href của link quán trong 1 lần gọi (nhanh hơn find_elements ~5x)
_JS_GET_LINKS = """
return Array.from(
    document.querySelectorAll('.resname a[target="_blank"], a.resname[target="_blank"]')
).map(a => a.href).filter(h => h && h.includes('foody.vn'));
"""

# JS: cài MutationObserver, resolve khi DOM có thêm node mới (tức lazy load xong)
_JS_WAIT_NEW_NODES = """
return new Promise((resolve) => {
    const observer = new MutationObserver((mutations) => {
        let added = mutations.some(m => m.addedNodes.length > 0);
        if (added) { observer.disconnect(); resolve(true); }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    // Timeout tự giải phóng sau 3s nếu không có node mới (đã hết trang)
    setTimeout(() => { observer.disconnect(); resolve(false); }, 3000);
});
"""


def get_links_fast(area_url: str, max_links: int = 5000) -> list[str]:
    driver = create_driver(headless=True)
    driver.get(area_url)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "resname"))
        )
    except Exception:
        log.warning("Không tìm thấy .resname")

    links: set[str] = set()
    no_change_cnt   = 0
    MAX_NO_CHANGE   = 6
    SCROLL_PAUSE    = 2.0    # Đủ cho AJAX + DOM append xong

    log.info(f"Bắt đầu lấy link: {area_url}")

    while len(links) < max_links:
        before = len(links)

        # ── Ghi nhớ scrollHeight TRƯỚC khi scroll ─────────────────────────────
        height_before = driver.execute_script("return document.body.scrollHeight")

        # ── Scroll từng bước nhỏ để trigger IntersectionObserver ──────────────
        viewport_h = driver.execute_script("return window.innerHeight")
        driver.execute_script(f"window.scrollBy(0, {int(viewport_h * 0.85)});")
        time.sleep(SCROLL_PAUSE)

        # ── Chờ thêm nếu DOM đang thay đổi (AJAX chưa xong) ──────────────────
        # Nếu scrollHeight tăng → trang vừa append thêm nội dung, đợi ổn định
        height_after = driver.execute_script("return document.body.scrollHeight")
        if height_after > height_before:
            time.sleep(0.8)   # Chờ append DOM ổn định hẳn

        # ── Lấy toàn bộ link bằng JS 1 lần ──────────────────────────────────
        new_links = driver.execute_script("""
            return Array.from(
                document.querySelectorAll('.resname a[target="_blank"]')
            ).map(a => a.href).filter(h => h);
        """)
        links.update(new_links)
        gained = len(links) - before

        # ── Kiểm tra at_bottom CHỈ SAU KHI DOM đã ổn định ───────────────────
        # So sánh vị trí scroll với scrollHeight MỚI NHẤT (sau khi AJAX xong)
        at_bottom = driver.execute_script("""
            return (window.innerHeight + Math.round(window.pageYOffset))
                   >= document.body.scrollHeight - 300;
        """)

        log.info(f"  +{gained} | Tổng: {len(links)} | Cuối trang: {at_bottom} | H: {height_before}→{height_after}")

        if gained == 0:
            if at_bottom:
                # Thật sự ở cuối — thử click load-more / next page
                clicked = _try_load_more(driver, links)
                if clicked:
                    no_change_cnt = 0
                    # Chờ URL thay đổi (xác nhận đã sang trang mới)
                    current_url = driver.current_url
                    try:
                        WebDriverWait(driver, 10).until(EC.url_changes(current_url))
                    except Exception:
                        pass
                    # Chờ danh sách quán trên trang MỚI xuất hiện
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "resname"))
                        )
                    except Exception:
                        time.sleep(3.0)
                    # Scroll lên đầu trang mới để bắt đầu lại từ đầu
                    driver.execute_script("window.scrollTo(0, 0);")
                    time.sleep(0.5)
                    # Thu thập ngay link trên trang mới (không chờ scroll)
                    fresh = driver.execute_script("""
                        return Array.from(
                            document.querySelectorAll('.resname a[target="_blank"]')
                        ).map(a => a.href).filter(h => h);
                    """)
                    links.update(fresh)
                    log.info(f"  Trang mới: +{len(fresh)} links ngay | URL: {driver.current_url}")
                    continue
                no_change_cnt += 1
                if no_change_cnt >= MAX_NO_CHANGE:
                    log.info("Đã cuộn hết trang và không còn nút load-more.")
                    break
            else:
                # Chưa ở cuối nhưng không tăng → AJAX chậm, chờ thêm rồi thử lại
                time.sleep(1.5)
                no_change_cnt += 1
                if no_change_cnt >= MAX_NO_CHANGE:
                    log.warning("Không tăng link quá lâu dù chưa ở cuối — có thể Intersection Observer bị block.")
                    # Thử scroll mạnh hơn để force trigger sentinel
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2.5)
                    # Lấy link một lần cuối trước khi bỏ cuộc
                    emergency = driver.execute_script("""
                        return Array.from(
                            document.querySelectorAll('.resname a[target="_blank"]')
                        ).map(a => a.href).filter(h => h);
                    """)
                    links.update(emergency)
                    if len(links) > before:
                        no_change_cnt = 0
                        continue
                    log.info("Dừng — không lấy thêm được link.")
                    break
        else:
            no_change_cnt = 0

    driver.quit()
    result = list(links)[:max_links]
    log.info(f"Tổng: {len(result)} links")
    return result


def _try_load_more(driver, existing_links: set) -> bool:
    """
    Foody HTML thực tế:
      <div id="scrollLoadingPage" class="btn-load-more full-width"
           data-bind="click: handleClickLoadMoreResult, visible: hasMorePage() && viewType() != 'map'">
        <a href="...?page=N#pageN" rel="next"> Xem tiếp kết quả ... </a>
      </div>

    Cấu trúc: div#scrollLoadingPage > a[rel="next"]
    → Phải click thẻ <a> bên trong (hoặc dùng href của nó để navigate),
      KHÔNG click div vì Knockout binding có thể không fire đúng trong headless.
    → Trước khi navigate, thu thập hết link trang hiện tại vào existing_links.
    """
    try:
        # Lấy div wrapper — kiểm tra visible (Knockout binding: hasMorePage())
        wrapper = driver.find_element(By.ID, "scrollLoadingPage")
        if not wrapper.is_displayed():
            log.info("  scrollLoadingPage ẩn (hasMorePage=false) → hết trang.")
            return False

        # Lấy thẻ <a rel="next"> bên trong
        anchor = wrapper.find_element(By.CSS_SELECTOR, "a[rel='next'], a[href*='page']")
        href = anchor.get_attribute("href")

        if not href:
            log.warning("  scrollLoadingPage hiện nhưng <a> không có href.")
            return False

        # Thu thập hết link trang hiện tại TRƯỚC khi rời đi
        current = driver.execute_script("""
            return Array.from(
                document.querySelectorAll('.resname a[target="_blank"]')
            ).map(a => a.href).filter(h => h);
        """)
        existing_links.update(current)

        log.info(f"  → Sang trang mới: {href}  (đã giữ {len(existing_links)} links)")
        driver.get(href)

        # Chờ danh sách quán xuất hiện trên trang mới
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "resname"))
            )
        except Exception:
            time.sleep(3.0)

        return True

    except Exception as e:
        log.debug(f"  _try_load_more: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: SCRAPE TỪNG QUÁN — dùng page_source + BS4 (nhanh hơn find_element)
# ═══════════════════════════════════════════════════════════════════════════════

# Map tên ngày tiếng Việt → tên cột tiếng Anh
_DAY_MAP = {
    "thứ hai":   "Monday",
    "thứ 2":     "Monday",
    "thứ ba":    "Tuesday",
    "thứ 3":     "Tuesday",
    "thứ tư":    "Wednesday",
    "thứ 4":     "Wednesday",
    "thứ năm":   "Thursday",
    "thứ 5":     "Thursday",
    "thứ sáu":   "Friday",
    "thứ 6":     "Friday",
    "thứ bảy":   "Saturday",
    "thứ 7":     "Saturday",
    "chủ nhật":  "Sunday",
    "cn":        "Sunday",
}
_DAY_COLS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def parse_opening_hours(opening_hours: str) -> dict:
    """
    Chuyển chuỗi Opening Hours dạng:
        'Thứ hai: 09:00 - 21:30 | Thứ ba: 09:00 - 21:30 | ... | Chủ nhật: Đóng cửa'
    thành dict 7 cột:
        {'Monday': '09:00 - 21:30', 'Tuesday': '09:00 - 21:30', ..., 'Sunday': 'Đóng cửa'}
    Ngày không tìm thấy → chuỗi rỗng ''.
    """
    result = {day: "" for day in _DAY_COLS}
    if not opening_hours:
        return result
    for segment in opening_hours.split("|"):
        segment = segment.strip()
        if ":" not in segment:
            continue
        # Tách tại dấu ":" đầu tiên — phần trước là tên ngày, phần sau là giờ
        day_vn, _, hours = segment.partition(":")
        day_vn = day_vn.strip().lower()
        hours  = hours.strip()
        eng = _DAY_MAP.get(day_vn)
        if eng:
            result[eng] = hours
    return result


def scrape_single_restaurant(
    link: str,
    res_id: int,
    review_start_id: int,
    pool: DriverPool,
) -> tuple[list, list] | None:
    """
    Scrape 1 quán. Mượn driver từ pool, dùng xong trả lại.
    Dùng page_source + BeautifulSoup thay vì gọi find_element từng cái.
    """
    driver = pool.acquire()
    link   = link.rstrip("\n").strip()
    res_data, rev_data = [], []

    try:
        driver.get(link)
        # Chờ tên quán xuất hiện trước
        try:
            WebDriverWait(driver, 8).until(
                EC.presence_of_element_located((By.CLASS_NAME, "main-info-title"))
            )
        except Exception:
            pass

        # Chờ thêm cho điểm thành phần render (Knockout.js binding load sau)
        # Nếu sau 5s vẫn không có thì thôi — vẫn parse được các field khác
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".microsite-top-points"))
            )
        except Exception:
            pass  # Quán không có điểm thành phần (ít review) → để 0.0 là đúng

        # ── Kiểm tra trang thương hiệu mẹ ────────────────────────────────────
        if driver.find_elements(By.CLASS_NAME, "brand-cover"):
            child_links = driver.execute_script("""
                return Array.from(
                    document.querySelectorAll('.brand-restaurant a[target="_blank"], .ldc-item-h-name a[target="_blank"]')
                ).map(a => a.href).filter(h => h);
            """)
            # Release driver TRƯỚC KHI đệ quy — mỗi lần đệ quy sẽ acquire driver mới
            pool.release(driver)
            driver = None   # Đánh dấu đã release, tránh finally release lần 2

            log.info(f"Thương hiệu mẹ → {len(child_links)} chi nhánh: {link}")
            if not child_links:
                log.warning(f"  Không tìm thấy chi nhánh nào: {link}")
                return [], []

            all_res, all_rev = [], []
            for idx, sub in enumerate(child_links):
                try:
                    # Truyền 0 làm placeholder — ID thật sẽ được gán trong run_parallel
                    result = scrape_single_restaurant(sub, 0, 0, pool)
                    if result:
                        sub_res, sub_rev = result
                        # Đánh dấu index chi nhánh để run_parallel gán ID đúng
                        for r in sub_res:
                            r['_child_idx'] = idx
                        for r in sub_rev:
                            r['_child_idx'] = idx
                        all_res.extend(sub_res)
                        all_rev.extend(sub_rev)
                except Exception as sub_e:
                    import traceback
                    log.error(
                        f"  ✗ Chi nhánh [{sub}]:\n"
                        f"    {type(sub_e).__name__}: {sub_e}\n"
                        f"{traceback.format_exc()}"
                    )
            return all_res, all_rev

        # ── Parse bằng BeautifulSoup (1 lần parse toàn bộ HTML) ──────────────
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Tên — nằm trong h1 bên trong div.main-info-title
        name_el = soup.select_one(".main-info-title h1")
        name    = name_el.get_text(strip=True) if name_el else ""

        # Địa chỉ
        local  = _t(soup, "[itemprop='streetAddress']")
        dist   = _t(soup, "[itemprop='addressLocality']")
        prov   = _t(soup, "[itemprop='addressRegion']")
        addr   = ", ".join(filter(None, [local, dist, prov]))

        # Giá, rating, votes
        price     = _t(soup, "[itemprop='priceRange']").replace("đ", "").strip()
        avg_rate  = _t(soup, ".microsite-point-avg") or np.nan
        total_vote= _t(soup, ".microsite-review-count")

        kind, cuisines, style, open_hours = "", "", "", ""

        for area in soup.select(".new-detail-info-area"):
            label_tag = area.select_one(".new-detail-info-label")
            if not label_tag:
                continue
            k = label_tag.get_text(strip=True)
            sib = label_tag.find_next_sibling()
            if not sib:
                continue
            v = ", ".join(a.get_text(strip=True) for a in sib.select("a")) \
                or sib.get_text(strip=True)

            if k == "Thể loại":              kind     = v
            elif k == "Phục vụ các món":     cuisines = v
            elif k == "Phong cách ẩm thực":  style    = v
        open_hours = ""
        try:
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".opening-time-btn"))
            )
            driver.execute_script("arguments[0].click();", btn)
            WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.ID, "opening-time-popup"))
            )
            open_hours = driver.execute_script("""
                var boxes = document.querySelectorAll('#opening-time-popup .date-box');
                var parts = [];
                boxes.forEach(function(box) {
                    var day   = box.querySelector('.date-header');
                    var times = box.querySelectorAll('.date-item');
                    if (!day) return;
                    var dayText = day.innerText.trim();
                    var ts = Array.from(times).map(function(t) {
                        return t.innerText.trim().replace(/\u00a0/g, ' ').trim();
                    }).filter(Boolean);
                    parts.push(dayText + ': ' + (ts.length ? ts.join(', ') : 'Đóng cửa'));
                });
                return parts.join(' | ');
            """) or ""
            driver.execute_script("""
                var p = document.getElementById('opening-time-popup');
                if (p) p.style.display = 'none';
            """)
        except Exception:
            open_hours = ""

        # Tiện ích
        nones     = {el.get_text(strip=True) for el in soup.select(".none")}
        has_wifi  = "no" if "Có wifi"       in nones else "yes"
        has_onl   = "no" if "Có giao hàng"  in nones else "yes"
        has_table = "no" if "Nên đặt trước" in nones else "yes"
        qual, ser, spa = 0.0, 0.0, 0.0
        for pt in soup.select(".microsite-top-points"):
            val_el = pt.select_one(".avg-txt-highlight")
            lbl_el = pt.select_one(".label")
            if not val_el or not lbl_el:
                continue
            try:
                val = float(val_el.get_text(strip=True).replace(",", "."))
                lbl = lbl_el.get_text(strip=True)
                if lbl == "Chất lượng":   qual = val
                elif lbl == "Phục vụ":    ser  = val
                elif lbl == "Không gian": spa  = val
                else:
                    log.debug(f"  Điểm lạ: '{lbl}' = {val}")
            except ValueError:
                log.debug(f"  Parse điểm lỗi: val='{val_el.get_text(strip=True)}'")

        # Tọa độ — thử lấy từ iframe src ngay trong HTML tĩnh
        lat, lon = "null", "null"
        iframe = soup.select_one("#iframes, iframe[src*='google.com/maps']")
        if iframe:
            src = iframe.get("src", "")
            m = re.search(r"([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)", src)
            if m:
                lat, lon = m.group(1), m.group(2)
            else:
                m2 = re.search(r"!3d([-+]?\d+\.\d+)!4d([-+]?\d+\.\d+)", src)
                if m2:
                    lat, lon = m2.group(1), m2.group(2)
        # Fallback: tìm trong script JSON-LD
        if lat == "null":
            for sc in soup.find_all("script"):
                txt = sc.string or ""
                m = re.search(r'"latitude"\s*:\s*([-\d.]+).*?"longitude"\s*:\s*([-\d.]+)', txt)
                if m:
                    lat, lon = m.group(1), m.group(2)
                    break

        res_data.append({
            "RestaurantID": res_id, "Restaurant Name": name,
            "Latitude": lat, "Longitude": lon, "Address": addr,
            "Type": kind, "Cuisines": cuisines, "Style": style,
            **parse_opening_hours(open_hours),
            "Has_Online_delivery": has_onl, "Has_Table_booking": has_table, "Has_Wifi": has_wifi,
            "Price": price, "Product_quality(10)": qual, "Serving_quality(10)": ser,
            "Interior_design(10)": spa, "Average_rating(5)": avg_rate, "Total votes": total_vote,
        })

        # ── Tab Bình luận ─────────────────────────────────────────────────────
        try:
            tab = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.LINK_TEXT, " Bình luận "))
            )
            driver.execute_script("arguments[0].click();", tab)
            # Chờ review xuất hiện thay vì sleep cố định
            WebDriverWait(driver, 4).until(
                EC.presence_of_element_located((By.CLASS_NAME, "review-item"))
            )
        except Exception:
            pass

        # Parse review bằng BS4
        rev_soup = BeautifulSoup(driver.page_source, "html.parser")
        cur_id   = review_start_id
        for item in rev_soup.select(".review-item"):
            rev_data.append({
                "UserID":      cur_id,
                "User":        _t(item, ".ru-username"),
                "Review Time": _t(item, ".ru-time"),
                "Rating (10)": _t(item, ".review-points") or 0,
                "Comment":     _t(item, ".rd-des"),
                "RestaurantID": res_id,
            })
            cur_id += 1

        log.info(f"✓ [{res_id}] {name} | {len(rev_data)} reviews")

    except Exception as e:
        import traceback
        log.error(
            f"✗ [{res_id}] {link}\n"
            f"  {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()}"
        )
    finally:
        if driver is not None:   # Tránh release lần 2 nếu đã release trong nhánh thương hiệu
            pool.release(driver)

    return res_data, rev_data


def _t(soup, selector: str) -> str:
    el = soup.select_one(selector)
    return el.get_text(strip=True) if el else ""


# ═══════════════════════════════════════════════════════════════════════════════
# PHẦN 3: CHẠY SONG SONG + CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════
_lock = Lock()

def run_parallel_scraper_with_checkpoint(
    list_links: list[str],
    max_workers: int = MAX_WORKERS,
    out_name: str = "foody_v14",
):
    # ── Đọc checkpoint ────────────────────────────────────────────────────────
    all_restaurants, all_reviews = [], []
    res_id_counter = 1
    rev_id_counter = 1

    # ── Lọc link chưa cào ─────────────────────────────────────────────────────
    todo = [l.rstrip("\n").strip() for l in list_links]
    if not todo:
        log.info("Tất cả link đã được cào từ trước!")
        return pd.DataFrame(all_restaurants), pd.DataFrame(all_reviews)

    log.info(f"Cào {len(todo)} link mới với {max_workers} workers...")

    # ── Khởi động pool driver ─────────────────────────────────────────────────
    pool = DriverPool(size=max_workers)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    scrape_single_restaurant,
                    link, res_id_counter + idx, (res_id_counter + idx) * 100, pool
                ): link
                for idx, link in enumerate(todo)
            }

            for future in as_completed(futures):
                link = futures[future]
                try:
                    result = future.result()
                    if not result:
                        continue
                    res_list, rev_list = result
                    if not res_list:
                        continue

                    with _lock:
                        # Nhóm res theo _child_idx để mỗi chi nhánh nhận ID riêng
                        from itertools import groupby
                        res_sorted = sorted(res_list, key=lambda r: r.get('_child_idx', 0))
                        for _, group in groupby(res_sorted, key=lambda r: r.get('_child_idx', 0)):
                            group = list(group)
                            cur_res_id = res_id_counter
                            for res in group:
                                res.pop('_child_idx', None)
                                res["RestaurantID"] = cur_res_id
                                all_restaurants.append(res)

                            for rev in rev_list:
                                if rev.get('_child_idx') == group[0].get('_child_idx', 0):
                                    rev.pop('_child_idx', None)
                                    rev["RestaurantID"] = cur_res_id
                                    rev["UserID"]       = rev_id_counter
                                    all_reviews.append(rev)
                                    rev_id_counter += 1

                            res_id_counter += 1

                except Exception as e:
                    import traceback
                    log.error(
                        f"Future lỗi [{link}]:\n"
                        f"  Loại lỗi : {type(e).__name__}\n"
                        f"  Chi tiết : {e}\n"
                        f"  Traceback:\n{traceback.format_exc()}"
                    )
    finally:
        pool.close_all()

    # ── Xuất file cuối ────────────────────────────────────────────────────────
    df_res = pd.DataFrame(all_restaurants)
    df_rev = pd.DataFrame(all_reviews)
    df_res.to_csv(os.path.join(OUTPUT_DIR, f"D/D{out_name}.csv"), index=False, encoding="utf-8-sig")
    df_rev.to_csv(os.path.join(OUTPUT_DIR, f"C/C{out_name}.csv"), index=False, encoding="utf-8-sig")
    log.info(f"Hoàn thành: {len(df_res)} quán, {len(df_rev)} reviews")
    return df_res, df_rev

