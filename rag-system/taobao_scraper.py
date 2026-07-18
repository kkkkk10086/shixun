"""
模块5：淘宝商品爬虫
从讯飞淘宝专卖店爬取产品数据
"""

import os
import re
import json
import time
import requests
from pathlib import Path
from config import DOCS_DIR, OUTPUT_DIR


class TaobaoScraper:
    """讯飞淘宝专卖店爬虫"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://www.taobao.com/",
        }
        self.products = []

    def scrape_store(self, store_url: str, max_pages: int = 5) -> list:
        """
        爬取店铺所有商品
        store_url: 店铺首页 URL
        """
        print(f"\n开始爬取店铺: {store_url}")

        # 获取商品列表
        product_urls = self._get_product_urls(store_url, max_pages)
        print(f"发现 {len(product_urls)} 个商品")

        # 逐个爬取商品详情
        for i, url in enumerate(product_urls):
            print(f"\n[{i+1}/{len(product_urls)}] 爬取: {url[:60]}...")
            product = self._scrape_product(url)
            if product:
                self.products.append(product)
                print(f"  ✓ {product['name'][:40]}...")
            time.sleep(2)  # 避免被封

        print(f"\n爬取完成，共 {len(self.products)} 个商品")
        return self.products

    def _get_product_urls(self, store_url: str, max_pages: int) -> list:
        """获取店铺内所有商品的 URL"""
        product_urls = []

        # 尝试通过搜索获取商品
        search_url = f"https://s.taobao.com/search?q=讯飞&seller_type=1"
        try:
            resp = self.session.get(search_url, timeout=10)
            # 提取商品链接
            urls = re.findall(r'https?://item\.taobao\.com/item\.htm\?[^"\']+', resp.text)
            urls += re.findall(r'https?://detail\.tmall\.com/item\.htm\?[^"\']+', resp.text)
            product_urls.extend(list(set(urls))[:max_pages * 20])
        except Exception as e:
            print(f"  搜索请求失败: {e}")

        # 如果搜索失败，尝试直接访问店铺
        if not product_urls:
            try:
                resp = self.session.get(store_url, timeout=10)
                urls = re.findall(r'https?://item\.taobao\.com/item\.htm\?[^"\']+', resp.text)
                urls += re.findall(r'https?://detail\.tmall\.com/item\.htm\?[^"\']+', resp.text)
                product_urls.extend(list(set(urls)))
            except Exception as e:
                print(f"  店铺请求失败: {e}")

        return product_urls[:max_pages * 20]

    def _scrape_product(self, url: str) -> dict:
        """爬取单个商品详情"""
        try:
            resp = self.session.get(url, timeout=10)
            html = resp.text

            # 提取商品名称
            name = ""
            name_match = re.search(r'"title"\s*:\s*"([^"]+)"', html)
            if name_match:
                name = name_match.group(1)
            else:
                name_match = re.search(r'<title>([^<]+)</title>', html)
                if name_match:
                    name = name_match.group(1).split('-')[0].strip()

            # 提取价格
            price = ""
            price_match = re.search(r'"price"\s*:\s*"([^"]+)"', html)
            if price_match:
                price = price_match.group(1)

            # 提取商品描述
            desc = ""
            desc_match = re.search(r'"desc"\s*:\s*"([^"]{10,})"', html)
            if desc_match:
                desc = desc_match.group(1)

            # 提取属性/规格
            specs = {}
            spec_matches = re.findall(r'"name"\s*:\s*"([^"]+)"\s*,\s*"value"\s*:\s*"([^"]+)"', html)
            for name_key, value in spec_matches:
                specs[name_key] = value

            if name:
                return {
                    "name": name,
                    "price": price,
                    "description": desc,
                    "specs": specs,
                    "url": url,
                    "source": "taobao"
                }
        except Exception as e:
            print(f"  爬取失败: {e}")

        return None

    def save_to_files(self, output_dir: str = OUTPUT_DIR):
        """将爬取结果保存为文本文件（供 RAG 系统处理）"""
        os.makedirs(output_dir, exist_ok=True)

        for product in self.products:
            # 生成文本内容
            content = self._product_to_text(product)

            # 保存为 txt 文件
            safe_name = re.sub(r'[\\/:*?"<>|]', '_', product['name'])[:50]
            file_path = os.path.join(output_dir, f"{safe_name}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  已保存: {safe_name}.txt")

        # 保存完整 JSON
        json_path = os.path.join(output_dir, "taobao_products.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.products, f, ensure_ascii=False, indent=2)
        print(f"  已保存完整数据: taobao_products.json")

    def _product_to_text(self, product: dict) -> str:
        """将商品信息转为文本"""
        lines = [
            f"产品名称：{product['name']}",
            f"价格：{product.get('price', '未知')} 元",
            f"来源：讯飞淘宝官方旗舰店",
            ""
        ]

        if product.get('specs'):
            lines.append("产品规格：")
            for key, value in product['specs'].items():
                lines.append(f"  {key}：{value}")
            lines.append("")

        if product.get('description'):
            lines.append(f"产品描述：{product['description']}")

        return "\n".join(lines)

    def save_to_mysql(self):
        """将爬取结果存入 MySQL"""
        try:
            from mysql_store import get_connection

            conn = get_connection()
            cursor = conn.cursor()

            # 创建爬虫数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS taobao_products (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(500),
                    price VARCHAR(50),
                    description TEXT,
                    specs JSON,
                    url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            for product in self.products:
                cursor.execute(
                    "INSERT INTO taobao_products (name, price, description, specs, url) VALUES (%s, %s, %s, %s, %s)",
                    (
                        product['name'],
                        product.get('price', ''),
                        product.get('description', ''),
                        json.dumps(product.get('specs', {}), ensure_ascii=False),
                        product.get('url', '')
                    )
                )

            conn.commit()
            cursor.close()
            conn.close()
            print(f"[MySQL] 已存入 {len(self.products)} 个商品")
        except Exception as e:
            print(f"[MySQL] 存储失败: {e}")


def scrape_and_process(store_url: str = "https://iflytek.tmall.com", max_pages: int = 3):
    """爬取淘宝数据并集成到 RAG 系统"""
    scraper = TaobaoScraper()

    # 爬取数据
    products = scraper.scrape_store(store_url, max_pages)

    if not products:
        print("未爬取到任何商品，使用模拟数据演示...")
        products = _get_demo_products()
        scraper.products = products

    # 保存为文件
    scraper.save_to_files()

    # 存入 MySQL
    scraper.save_to_mysql()

    return products


def _get_demo_products() -> list:
    """演示用的模拟讯飞产品数据"""
    return [
        {
            "name": "讯飞智能办公本X2",
            "price": "4999",
            "description": "讯飞智能办公本X2是一款集手写识别、语音转写、会议记录于一体的智能办公设备。采用10.3英寸墨水屏，支持Wacom电磁笔手写，识别准确率达98%。内置讯飞星火大模型，支持会议纪要自动生成、多语种实时翻译。",
            "specs": {
                "屏幕": "10.3英寸 E-Ink 墨水屏",
                "处理器": "八核处理器",
                "内存": "4GB RAM + 64GB ROM",
                "手写笔": "Wacom 电磁笔",
                "电池": "4000mAh",
                "重量": "约380g",
                "特色功能": "语音转写、手写识别、会议纪要、多语种翻译"
            },
            "url": "https://detail.tmall.com/item.htm?id=讯飞办公本X2",
            "source": "demo"
        },
        {
            "name": "讯飞智能办公本X2 LAMY版",
            "price": "5999",
            "description": "讯飞智能办公本X2 LAMY版是与德国LAMY联名的高端版本，配备LAMY EMR电磁笔，书写体验更佳。具备X2全部功能，包括语音转写、手写识别、会议纪要自动生成等。",
            "specs": {
                "屏幕": "10.3英寸 E-Ink 墨水屏",
                "处理器": "八核处理器",
                "内存": "4GB RAM + 128GB ROM",
                "手写笔": "LAMY EMR 电磁笔",
                "电池": "4000mAh",
                "重量": "约380g",
                "特色功能": "LAMY联名、语音转写、手写识别、会议纪要"
            },
            "url": "https://detail.tmall.com/item.htm?id=讯飞办公本X2LAMY",
            "source": "demo"
        },
        {
            "name": "讯飞AI录音卡",
            "price": "399",
            "description": "讯飞AI录音卡是一款便携式智能录音设备，支持高清降噪录音、实时语音转文字。采用双麦克风阵列，支持远场拾音，录音时长可达10小时。通过蓝牙连接手机APP，实现录音文件管理和转写。",
            "specs": {
                "录音": "48kHz/16bit 高清录音",
                "麦克风": "双麦克风阵列",
                "降噪": "AI智能降噪",
                "存储": "32GB 内置存储",
                "电池": "可连续录音10小时",
                "连接": "蓝牙5.0",
                "重量": "约30g",
                "特色功能": "实时语音转文字、AI降噪、蓝牙连接"
            },
            "url": "https://detail.tmall.com/item.htm?id=讯飞录音卡",
            "source": "demo"
        },
        {
            "name": "讯飞双屏翻译机2.0",
            "price": "3499",
            "description": "讯飞双屏翻译机2.0采用主副双屏设计，支持83种语言实时翻译。主屏4.1英寸高清触摸屏，副屏2.2英寸墨水屏可同步显示翻译内容。支持拍照翻译、离线翻译、会议翻译等多种翻译模式。",
            "specs": {
                "主屏": "4.1英寸高清触摸屏",
                "副屏": "2.2英寸墨水屏",
                "翻译语言": "83种语言",
                "离线翻译": "支持中英日韩等主流语言离线翻译",
                "拍照翻译": "800万像素后置摄像头",
                "电池": "支持全天候使用",
                "重量": "约180g",
                "特色功能": "双屏显示、拍照翻译、离线翻译、会议翻译"
            },
            "url": "https://detail.tmall.com/item.htm?id=讯飞翻译机2.0",
            "source": "demo"
        }
    ]


if __name__ == "__main__":
    scrape_and_process()
