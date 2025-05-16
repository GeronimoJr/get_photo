import streamlit as st
import requests
import tempfile
import os
import re
import json
import time
from datetime import datetime
import pandas as pd
import io
import ftplib
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import uuid
import codecs
from bs4 import BeautifulSoup
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict

LOGS_DIR = os.path.join(os.path.expanduser("~"), ".xml_image_processor", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
logger = logging.getLogger("xml_img_processor")
logger.setLevel(logging.INFO)
fh = RotatingFileHandler(os.path.join(LOGS_DIR, "app.log"),
                         maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(threadName)s|%(message)s"))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())

MAX_RETRIES = 10
RETRY_SLEEP = 5
FTP_PARALLEL = 1
DL_PARALLEL = 10


class FTPManager:
    def __init__(self, settings):
        self.settings = settings
        self.ftp = None
        self.connected = False

    def connect(self):
        if self.connected:
            return True
        try:
            self.ftp = ftplib.FTP()
            self.ftp.connect(self.settings["host"], self.settings["port"])
            self.ftp.login(self.settings["username"], self.settings["password"])
            if self.settings["directory"] and self.settings["directory"] != "/":
                try:
                    self.ftp.cwd(self.settings["directory"])
                except ftplib.error_perm:
                    for d in [d for d in self.settings["directory"].strip("/").split("/") if d]:
                        try:
                            self.ftp.cwd(d)
                        except ftplib.error_perm:
                            self.ftp.mkd(d)
                            self.ftp.cwd(d)
            self.connected = True
            return True
        except Exception as e:
            logger.error(str(e))
            self.connected = False
            return False

    def upload_file(self, file_path, remote_filename=None):
        if not self.connected and not self.connect():
            return {"success": False, "error": "ftp connect"}
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return {"success": False, "error": "file missing"}
        try:
            if not remote_filename:
                remote_filename = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                self.ftp.storbinary(f"STOR {remote_filename}", f)
            if self.settings.get("http_path"):
                http_path = self.settings["http_path"].rstrip("/") + "/"
                url = f"{http_path}{remote_filename}"
            else:
                url = f"ftp://{self.settings['host']}{self.settings['directory'].rstrip('/')}/{remote_filename}"
            return {"success": True, "url": url, "filename": remote_filename}
        except Exception as e:
            self.connected = False
            if not self.connect():
                return {"success": False, "error": str(e)}
            try:
                with open(file_path, "rb") as f:
                    self.ftp.storbinary(f"STOR {remote_filename}", f)
                if self.settings.get("http_path"):
                    http_path = self.settings["http_path"].rstrip("/") + "/"
                    url = f"{http_path}{remote_filename}"
                else:
                    url = f"ftp://{self.settings['host']}{self.settings['directory'].rstrip('/')}/{remote_filename}"
                return {"success": True, "url": url, "filename": remote_filename}
            except Exception as e2:
                return {"success": False, "error": str(e2)}

    def close(self):
        if self.connected and self.ftp:
            try:
                self.ftp.quit()
            except Exception:
                pass
            self.connected = False


def authenticate_user():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("Pobieranie zdjęć z XML/CSV - Logowanie")
        u = st.text_input("Login")
        p = st.text_input("Hasło", type="password")
        if st.button("Zaloguj"):
            if u == st.secrets.get("APP_USER") and p == st.secrets.get("APP_PASSWORD"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Błędny login lub hasło")
        st.stop()


def initialize_state():
    defaults = {
        "file_info": None,
        "output_bytes": None,
        "ftp_settings": {"host": "", "port": 21, "username": "",
                         "password": "", "directory": "/", "http_path": ""},
        "processing_params": {"xpath": "", "column_name": "",
                              "new_node_name": "", "new_column_name": "", "separator": ","}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def read_file_content(uploaded):
    if not uploaded:
        return None, "brak pliku"
    try:
        raw = uploaded.read()
        ext = uploaded.name.split(".")[-1].lower()
        if ext not in ("xml", "csv"):
            return None, "typ"
        if ext == "xml":
            if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
                enc = "utf-16-le" if raw.startswith(codecs.BOM_UTF16_LE) else "utf-16-be"
                return {"content": raw.decode(enc), "type": ext, "encoding": "utf-16", "name": uploaded.name}, None
            enc_match = re.search(br'<\?xml[^>]*encoding=["\']([^"\']+)["\']', raw)
            if enc_match:
                enc = enc_match.group(1).decode("ascii").lower()
                return {"content": raw.decode(enc), "type": ext, "encoding": enc, "name": uploaded.name}, None
        for enc_try in ("utf-8", "iso-8859-2", "windows-1250", "utf-16-le", "utf-16-be"):
            try:
                return {"content": raw.decode(enc_try), "type": ext, "encoding": enc_try, "name": uploaded.name}, None
            except UnicodeDecodeError:
                continue
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
            return {"content": df.to_csv(index=False), "type": ext, "encoding": "auto", "name": uploaded.name}, None
        return None, "kodowanie"
    except Exception as e:
        return None, str(e)


def download_image(url, tmp_dir):
    try:
        p = urlparse(url)
        if not p.scheme or not p.netloc:
            return None, "url"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*",
                   "Referer": f"{p.scheme}://{p.netloc}/"}
        if "image_show.php" in url:
            r_html = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            r_html.raise_for_status()
            img_tag = BeautifulSoup(r_html.text, "html.parser").find("img")
            if not img_tag or not img_tag.get("src"):
                return None, "img"
            img_url = img_tag["src"] if img_tag["src"].startswith("http") else \
                f"{p.scheme}://{p.netloc}/{img_tag['src'].lstrip('/')}"
        else:
            img_url = url
        resp = requests.get(img_url, headers=headers, timeout=(3, 15), allow_redirects=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if not ctype.startswith("image/"):
            return None, "ctype"
        ext_map = {"image/jpeg": ".jpg", "image/png": ".png",
                   "image/gif": ".gif", "image/webp": ".webp"}
        ext = ext_map.get(ctype, ".jpg")
        fname = f"img_{uuid.uuid4().hex}{ext}"
        path = os.path.join(tmp_dir, fname)
        with open(path, "wb") as f:
            f.write(resp.content)
        if os.path.getsize(path) < 100:
            return None, "empty"
        return {"path": path, "filename": fname, "original_url": url}, None
    except Exception as e:
        return None, str(e)


def _download(url, tmp_dir):
    img, err = download_image(url, tmp_dir)
    if err:
        raise RuntimeError(err)
    return img


def _upload(img_info, ftp_settings):
    ftp = FTPManager(ftp_settings)
    if not ftp.connect():
        raise RuntimeError("ftp")
    try:
        res = ftp.upload_file(img_info["path"])
        if not res["success"]:
            raise RuntimeError(res["error"])
        return res["url"], res["filename"]
    finally:
        ftp.close()


def process_with_retry(urls, tmp_dir, ftp_settings,
                       dl_workers=DL_PARALLEL, ftp_workers=FTP_PARALLEL):
    attempts = defaultdict(int)
    remaining = set(urls)
    url_map = {}
    downloaded = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=dl_workers) as dl_pool, \
            concurrent.futures.ThreadPoolExecutor(max_workers=ftp_workers) as up_pool:
        while remaining:
            fut_to_url = {dl_pool.submit(_download, u, tmp_dir): u for u in remaining}
            for fut in concurrent.futures.as_completed(fut_to_url):
                url = fut_to_url[fut]
                try:
                    downloaded[url] = fut.result()
                except Exception as e:
                    attempts[url] += 1
                    if attempts[url] > MAX_RETRIES:
                        remaining.remove(url)
                        logger.error(f"Download max {url}")
                    else:
                        time.sleep(RETRY_SLEEP)
                else:
                    remaining.remove(url)
        remaining_up = set(downloaded.keys())
        while remaining_up:
            fut_to_url = {up_pool.submit(_upload, downloaded[u], ftp_settings): u
                          for u in remaining_up}
            for fut in concurrent.futures.as_completed(fut_to_url):
                url = fut_to_url[fut]
                try:
                    url_map[url], _ = fut.result()
                except Exception as e:
                    attempts[url] += 1
                    if attempts[url] > MAX_RETRIES:
                        remaining_up.remove(url)
                        logger.error(f"Upload max {url}")
                    else:
                        time.sleep(RETRY_SLEEP)
                else:
                    remaining_up.remove(url)
    return url_map


def extract_image_urls_from_xml(xml, xpath, sep=","):
    try:
        if xml.startswith("\ufeff"):
            xml = xml[1:]
        xml = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", xml)
        is_attr = "/@" in xpath
        attr = xpath.split("/@")[-1] if is_attr else None
        base = xpath.split("/@")[0] if is_attr else xpath
        root = ET.fromstring(xml)
        xp = base[2:] if base.startswith("//") else base
        els = root.findall(f".//{xp}")
        urls = []
        for el in els:
            txt = el.attrib.get(attr) if is_attr else el.text
            if txt:
                txt = txt.replace("&amp;", "&")
                if sep in txt:
                    urls.extend([u.strip() for u in txt.split(sep)
                                 if u.strip() and ("http://" in u or "https://" in u)])
                elif "http://" in txt or "https://" in txt:
                    urls.append(txt.strip())
        return urls, None
    except Exception as e:
        return None, str(e)


def update_xml(xml, xpath, url_map, new_node, sep=","):
    try:
        is_attr = "/@" in xpath
        attr = xpath.split("/@")[-1] if is_attr else None
        base = xpath.split("/@")[0] if is_attr else xpath
        root = ET.fromstring(xml)
        xp = base[2:] if base.startswith("//") else base
        els = root.findall(f".//{xp}")
        parents = {c: p for p in root.iter() for c in p}
        for el in els:
            parent = parents.get(el)
            if parent is None:
                continue
            ftp_imgs = next((c for c in parent if c.tag == "ftp_images"), None)
            if ftp_imgs is None:
                ftp_imgs = ET.Element("ftp_images")
                parent.append(ftp_imgs)
            orig = el.attrib.get(attr, "").strip() if is_attr else (el.text.strip() if el.text else "")
            if not orig:
                continue
            if sep in orig:
                urls = [u.strip() for u in orig.split(sep)]
                new_urls = [url_map[u] for u in urls if u in url_map]
                if new_urls:
                    new_el = ET.Element(new_node)
                    new_el.text = sep.join(new_urls)
                    ftp_imgs.append(new_el)
            elif orig in url_map:
                new_el = ET.Element(new_node)
                new_el.text = url_map[orig]
                ftp_imgs.append(new_el)
        return ET.tostring(root, encoding="unicode"), None
    except Exception as e:
        return None, str(e)


def extract_image_urls_from_csv(csv_content, col, sep=","):
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        if col not in df.columns:
            return None, f"Brak kolumny {col}"
        urls = []
        for v in df[col]:
            if pd.notna(v):
                v = str(v)
                if sep in v:
                    urls.extend([u.strip() for u in v.split(sep) if u.strip()])
                else:
                    urls.append(v.strip())
        return urls, None
    except Exception as e:
        return None, str(e)


def update_csv(csv_content, col, url_map, new_col, sep=","):
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        if col not in df.columns:
            return None, f"Brak kolumny {col}"
        if new_col not in df.columns:
            df[new_col] = ""
        for idx, v in enumerate(df[col]):
            if pd.notna(v):
                vstr = str(v)
                if sep in vstr:
                    urls = [u.strip() for u in vstr.split(sep)]
                    new_urls = [url_map[u] for u in urls if u in url_map]
                    if new_urls:
                        df.at[idx, new_col] = sep.join(new_urls)
                elif vstr in url_map:
                    df.at[idx, new_col] = url_map[vstr]
        return df.to_csv(index=False), None
    except Exception as e:
        return None, str(e)


def save_to_drive(output_bytes, file_info, url_map):
    try:
        folder = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
        creds_json = st.secrets.get("GOOGLE_DRIVE_CREDENTIALS_JSON")
        if not folder or not creds_json:
            return False, "Brak konfiguracji GD"
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_name = f"map_{now}.txt"
        out_name = f"processed_{now}.{file_info['type']}"
        with tempfile.TemporaryDirectory() as tmp:
            p_out = os.path.join(tmp, out_name)
            p_log = os.path.join(tmp, log_name)
            with open(p_out, "wb") as f:
                f.write(output_bytes)
            with open(p_log, "w", encoding="utf-8") as f:
                for o, n in url_map.items():
                    f.write(f"{o} -> {n}\n")
            creds_dict = json.loads(creds_json) if isinstance(creds_json, str) else creds_json
            credentials = ServiceAccountCredentials.from_json_keyfile_dict(
                creds_dict, ["https://www.googleapis.com/auth/drive"])
            gauth = GoogleAuth()
            gauth.credentials = credentials
            drive = GoogleDrive(gauth)
            f_log = drive.CreateFile({"title": log_name, "parents": [{"id": folder}],
                                      "mimeType": "text/plain"})
            f_log.SetContentFile(p_log)
            f_log.Upload()
            f_res = drive.CreateFile({"title": out_name, "parents": [{"id": folder}],
                                      "mimeType": f"application/{file_info['type']}"})
            f_res.SetContentFile(p_out)
            f_res.Upload()
            return True, "Zapisano na GD"
    except Exception as e:
        return False, str(e)


def save_processing_state(session_id, original_urls, processed_urls,
                          url_map, file_info, processing_params):
    state = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "original_urls": original_urls,
        "processed_urls": list(processed_urls),
        "url_map": url_map,
        "file_info": {k: file_info[k] for k in ("name", "type", "encoding")},
        "processing_params": processing_params,
        "file_content": file_info.get("content", "")
    }
    state_dir = os.path.join(os.path.expanduser("~"), ".xml_image_processor")
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, f"session_{session_id}.json")
    with open(path, "w") as f:
        json.dump(state, f)
    return path


def load_processing_state(session_id=None):
    state_dir = os.path.join(os.path.expanduser("~"), ".xml_image_processor")
    if not os.path.exists(state_dir):
        return None
    if session_id:
        path = os.path.join(state_dir, f"session_{session_id}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None
    files = [f for f in os.listdir(state_dir)
             if f.startswith("session_") and f.endswith(".json")]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(state_dir, x)), reverse=True)
    with open(os.path.join(state_dir, files[0]), "r") as f:
        return json.load(f)


def resume_processing(state, tmp_dir, ftp_settings):
    original = set(state["original_urls"])
    processed = set(state["processed_urls"])
    todo = list(original - processed)
    if not todo:
        st.success("Wszystko już gotowe")
        return True
    pb = st.progress(len(processed) / len(original))
    st.write("Wznawianie...")
    url_map = process_with_retry(todo, tmp_dir, ftp_settings)
    state["url_map"].update(url_map)
    state["processed_urls"].extend(todo)
    save_processing_state(state["session_id"], state["original_urls"],
                          state["processed_urls"], state["url_map"],
                          state["file_info"], state["processing_params"])
    pb.progress(1.0)
    update_file_after_processing(state)
    st.success("Wznowiono")
    return True


def update_file_after_processing(state):
    file_type = state["file_info"]["type"]
    file_content = state["file_content"]
    pp = state["processing_params"]
    if file_type == "xml":
        updated, err = update_xml(file_content,
                                  pp["xpath"],
                                  state["url_map"],
                                  pp["new_node_name"],
                                  pp["separator"])
    else:
        updated, err = update_csv(file_content,
                                  pp["column_name"],
                                  state["url_map"],
                                  pp["new_column_name"],
                                  pp["separator"])
    if err:
        st.error(err)
        return
    bytes_out = updated.encode(state["file_info"]["encoding"])
    st.download_button("📁 Pobierz plik", data=bytes_out,
                       file_name=f"{os.path.splitext(state['file_info']['name'])[0]}_updated.{file_type}",
                       mime="text/plain")
    ok, msg = save_to_drive(bytes_out, state["file_info"], state["url_map"])
    if ok:
        st.success(msg)
    else:
        st.warning(msg)


def list_saved_sessions():
    state_dir = os.path.join(os.path.expanduser("~"), ".xml_image_processor")
    if not os.path.exists(state_dir):
        return []
    ses = []
    for fn in os.listdir(state_dir):
        if fn.startswith("session_") and fn.endswith(".json"):
            try:
                with open(os.path.join(state_dir, fn), "r") as f:
                    s = json.load(f)
                    pct = round(len(s["processed_urls"]) /
                                len(s["original_urls"]) * 100, 1) if s["original_urls"] else 0
                    ses.append({"id": s["session_id"], "ts": s["timestamp"],
                                "file": s["file_info"]["name"],
                                "progress": f"{pct}%"})
            except Exception:
                pass
    ses.sort(key=lambda x: x["ts"], reverse=True)
    return ses


def main():
    st.set_page_config(page_title="Pobieranie zdjęć z XML/CSV")
    authenticate_user()
    initialize_state()

    st.title("Pobieranie zdjęć z XML/CSV")
    tab1, tab2, tab3 = st.tabs(["Nowa sesja", "Wznów", "Info"])

    with tab1:
        up = st.file_uploader("Plik XML/CSV", type=["xml", "csv"])
        if up:
            fi, err = read_file_content(up)
            if err:
                st.error(err)
            else:
                st.session_state.file_info = fi
                st.success(f"Wczytano {fi['name']}")

        if st.session_state.file_info:
            ft = st.session_state.file_info["type"]
            if ft == "xml":
                xpath = st.text_input("XPath", value=st.session_state.processing_params["xpath"])
                new_node = st.text_input("Nazwa węzła", value=st.session_state.processing_params["new_node_name"])
                st.session_state.processing_params.update({"xpath": xpath, "new_node_name": new_node})
            else:
                col = st.text_input("Kolumna URL", value=st.session_state.processing_params["column_name"])
                new_col = st.text_input("Nowa kolumna", value=st.session_state.processing_params["new_column_name"])
                st.session_state.processing_params.update({"column_name": col, "new_column_name": new_col})
            sep = st.text_input("Separator", value=st.session_state.processing_params["separator"])
            st.session_state.processing_params["separator"] = sep

        with st.expander("FTP"):
            fs = st.session_state.ftp_settings
            fs["host"] = st.text_input("Host", value=fs["host"])
            fs["port"] = st.number_input("Port", value=fs["port"], min_value=1, max_value=65535)
            fs["directory"] = st.text_input("Katalog", value=fs["directory"])
            fs["http_path"] = st.text_input("HTTP ścieżka", value=fs["http_path"])
            fs["username"] = st.text_input("User", value=fs["username"])
            fs["password"] = st.text_input("Pass", type="password", value=fs["password"])

        if st.session_state.file_info and st.button("Start"):
            fi = st.session_state.file_info
            sep = st.session_state.processing_params["separator"]
            if fi["type"] == "xml":
                urls, err = extract_image_urls_from_xml(fi["content"], st.session_state.processing_params["xpath"], sep)
            else:
                urls, err = extract_image_urls_from_csv(fi["content"], st.session_state.processing_params["column_name"], sep)
            if err:
                st.error(err)
            elif not urls:
                st.warning("Brak URL")
            else:
                st.success(f"URL-i: {len(urls)}")
                sid = f"{uuid.uuid4().hex}_{int(time.time())}"
                with tempfile.TemporaryDirectory() as tmp:
                    url_map = process_with_retry(urls, tmp, st.session_state.ftp_settings,
                                                 dl_workers=DL_PARALLEL, ftp_workers=FTP_PARALLEL)
                    save_processing_state(sid, urls, urls, url_map, fi, st.session_state.processing_params)
                    state = load_processing_state(sid)
                    update_file_after_processing(state)

    with tab2:
        sessions = list_saved_sessions()
        if not sessions:
            st.info("Brak zapisów")
        else:
            st.table(pd.DataFrame(sessions))
            sel = st.selectbox("ID", [s["id"] for s in sessions])
            if st.button("Wznów"):
                state = load_processing_state(sel)
                if state:
                    with tempfile.TemporaryDirectory() as tmp:
                        resume_processing(state, tmp, st.session_state.ftp_settings)
                else:
                    st.error("Błąd stanu")

    with tab3:
        st.markdown("""
**Kroki:**

1. Wgraj plik XML/CSV  
2. Podaj XPath/kolumnę i separator  
3. Ustaw FTP (host, katalog, http_path)  
4. Start – aplikacja pobierze obrazy równolegle ({} wątki) i kolejkowo wyśle na FTP  
5. Pobierz zaktualizowany plik lub zapisz go na Google Drive  

W razie przerwania procesu użyj zakładki **Wznów**.  
""".format(DL_PARALLEL))


if __name__ == "__main__":
    main()
