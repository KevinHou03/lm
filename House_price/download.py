import hashlib
import os
import requests

DATA_HUB = dict()
DATA_URL = 'https://d2l-data.s3.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件路径"""
    assert name in DATA_HUB, f"{name} 不存在于 DATA_HUB"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])

    # 如果文件存在 -> 检查 SHA1 哈希值
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 已有文件且哈希一致，直接返回

    # or重新下载
    print(f"正在从 {url} 下载 {fname}")
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
    return fname

