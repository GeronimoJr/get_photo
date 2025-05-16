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

class FTPManager:
    def __init__(self, settings):
        self.settings = settings
        self.ftp = None
        self.connected = False
        
    def connect(self):
        if self.connected: return True
        try:
            self.ftp = ftplib.FTP()
            self.ftp.connect(self.settings["host"], self.settings["port"])
            self.ftp.login(self.settings["username"], self.settings["password"])
            if self.settings["directory"] and self.settings["directory"] != "/":
                try:
                    self.ftp.cwd(self.settings["directory"])
                except ftplib.error_perm:
                    for directory in [d for d in self.settings["directory"].strip("/").split("/") if d]:
                        try:
                            self.ftp.cwd(directory)
                        except ftplib.error_perm:
                            self.ftp.mkd(directory)
                            self.ftp.cwd(directory)
            self.connected = True
            return True
        except Exception as e:
            self.connected = False
            return False
            
    def upload_file(self, file_path, remote_filename=None):
        if not self.connected and not self.connect():
            return {"success": False, "error": "Nie można połączyć się z serwerem FTP"}
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return {"success": False, "error": f"Plik nie istnieje lub jest pusty: {file_path}"}
        try:
            if not remote_filename:
                remote_filename = os.path.basename(file_path)
            with open(file_path, 'rb') as file:
                self.ftp.storbinary(f'STOR {remote_filename}', file)
            if self.settings.get("http_path"):
                http_path = self.settings["http_path"].strip()
                if not http_path.endswith('/'): http_path += '/'
                image_url = f"{http_path}{remote_filename}"
            else:
                image_url = f"ftp://{self.settings['host']}"
                if self.settings["directory"] and self.settings["directory"] != "/":
                    if not self.settings["directory"].startswith("/"): image_url += "/"
                    image_url += self.settings["directory"]
                    if not image_url.endswith("/"): image_url += "/"
                else:
                    image_url += "/"
                image_url += remote_filename
            return {"success": True, "url": image_url, "filename": remote_filename}
        except Exception as e:
            self.connected = False
            if not self.connect():
                return {"success": False, "error": f"Utracono połączenie FTP: {str(e)}"}
            try:
                with open(file_path, 'rb') as file:
                    self.ftp.storbinary(f'STOR {remote_filename}', file)
                if self.settings.get("http_path"):
                    http_path = self.settings["http_path"].strip()
                    if not http_path.endswith('/'): http_path += '/'
                    image_url = f"{http_path}{remote_filename}"
                else:
                    image_url = f"ftp://{self.settings['host']}"
                    if self.settings["directory"] and self.settings["directory"] != "/":
                        if not self.settings["directory"].startswith("/"): image_url += "/"
                        image_url += self.settings["directory"]
                        if not image_url.endswith("/"): image_url += "/"
                    else:
                        image_url += "/"
                    image_url += remote_filename
                return {"success": True, "url": image_url, "filename": remote_filename}
            except:
                return {"success": False, "error": "Nie udało się przesłać pliku nawet po ponownym połączeniu"}
    
    def close(self):
        if self.connected and self.ftp:
            try: self.ftp.quit()
            except: pass
            self.connected = False

def authenticate_user():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("Pobieranie zdjęć z XML/CSV - Logowanie")
        user = st.text_input("Login")
        password = st.text_input("Hasło", type="password")
        if st.button("Zaloguj"):
            if user == st.secrets.get("APP_USER") and password == st.secrets.get("APP_PASSWORD"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Nieprawidłowy login lub hasło")
        st.stop()
    return True

def initialize_session_state():
    defaults = {
        "generated_code": "", "edited_code": "", "output_bytes": None,
        "file_info": None, "show_editor": False, "error_info": None,
        "code_fixed": False, "fix_requested": False, "downloaded_images": [],
        "ftp_settings": {
            "host": "", "port": 21, "username": "", "password": "",
            "directory": "/", "http_path": ""
        },
        "processing_params": {
            "xpath": "", "column_name": "", "new_node_name": "",
            "new_column_name": "", "separator": ","
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def read_file_content(uploaded_file):
    if not uploaded_file:
        return None, "Nie wybrano pliku"
    try:
        raw_bytes = uploaded_file.read()
        file_type = uploaded_file.name.split(".")[-1].lower()
        if file_type not in ["xml", "csv"]:
            return None, "Nieobsługiwany typ pliku. Akceptowane formaty to XML i CSV."
        if file_type == "xml":
            if raw_bytes.startswith(codecs.BOM_UTF16_LE) or raw_bytes.startswith(codecs.BOM_UTF16_BE):
                try:
                    encoding = 'utf-16-le' if raw_bytes.startswith(codecs.BOM_UTF16_LE) else 'utf-16-be'
                    return {"content": raw_bytes.decode(encoding), "raw_bytes": raw_bytes, 
                            "type": file_type, "encoding": 'utf-16', "name": uploaded_file.name}, None
                except UnicodeDecodeError:
                    pass
            encoding_match = re.search(br'<\?xml[^>]*encoding=["\']([^"\']+)["\']', raw_bytes)
            if encoding_match:
                try:
                    encoding = encoding_match.group(1).decode('ascii').lower()
                    return {"content": raw_bytes.decode(encoding), "raw_bytes": raw_bytes, 
                            "type": file_type, "encoding": encoding, "name": uploaded_file.name}, None
                except:
                    pass
        for enc in ["utf-8", "iso-8859-2", "windows-1250", "utf-16-le", "utf-16-be"]:
            try:
                return {"content": raw_bytes.decode(enc), "raw_bytes": raw_bytes, 
                        "type": file_type, "encoding": enc, "name": uploaded_file.name}, None
            except UnicodeDecodeError:
                continue
        if file_type == "csv":
            try:
                buffer = io.BytesIO(raw_bytes)
                df = pd.read_csv(buffer, sep=None, engine='python')
                return {"content": df.to_csv(index=False), "raw_bytes": raw_bytes, 
                        "type": file_type, "encoding": "auto-detected", 
                        "name": uploaded_file.name, "dataframe": df}, None
            except:
                pass
        return None, "Nie udało się odczytać pliku – nieznane kodowanie."
    except Exception as e:
        return None, f"Błąd podczas odczytu pliku: {str(e)}"

def download_image(url, temp_dir):
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return None, f"Nieprawidłowy URL: {url}"
        headers = {
            "User-Agent": "Mozilla/5.0", "Accept": "*/*",
            "Referer": f"{parsed_url.scheme}://{parsed_url.netloc}/"
        }
        if "image_show.php" in url:
            html_resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            html_resp.raise_for_status()
            soup = BeautifulSoup(html_resp.text, "html.parser")
            img_tag = soup.find("img")
            if not img_tag or not img_tag.get("src"):
                return None, "Nie znaleziono znacznika <img> w odpowiedzi HTML"
            img_src = img_tag["src"]
            img_url = f"{parsed_url.scheme}://{parsed_url.netloc}/{img_src.lstrip('/')}" if not img_src.startswith("http") else img_src
        else:
            img_url = url
        for retry in range(3):
            try:
                response = requests.get(img_url, headers=headers, stream=False, timeout=15, allow_redirects=True)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/") and retry < 2:
                    continue
                extension = {
                    "image/jpeg": ".jpg", "image/png": ".png",
                    "image/gif": ".gif", "image/webp": ".webp"
                }.get(content_type, ".jpg")
                filename = f"image_{uuid.uuid4().hex}{extension}"
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(response.content)
                if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
                    return {"path": file_path, "filename": filename, "original_url": url}, None
                else:
                    return None, "Pobrano pusty lub niepełny plik"
            except Exception as e:
                if retry == 2:
                    return None, f"Błąd przy pobieraniu: {str(e)}"
    except Exception as e:
        return None, f"Błąd: {str(e)}"

def process_images_sequentially(urls, temp_dir, ftp_settings, debug_container=None):
    new_urls_map = {}
    downloaded_images = []
    failed_urls = []
    ftp_manager = FTPManager(ftp_settings)
    for url in urls:
        image_info, error = download_image(url, temp_dir)
        if error:
            failed_urls.append({"url": url, "error": error})
            if debug_container:
                debug_container.warning(f"Błąd pobierania {url}: {error}")
            continue
        if not ftp_manager.connect():
            failed_urls.append({"url": url, "error": "FTP error"})
            if debug_container:
                debug_container.warning(f"FTP error: {url}")
            continue
        upload_result = ftp_manager.upload_file(image_info["path"])
        if upload_result["success"]:
            new_urls_map[url] = upload_result["url"]
            downloaded_images.append({
                "original_url": url, "ftp_url": upload_result["url"],
                "filename": upload_result["filename"]
            })
            if debug_container:
                debug_container.success(f"Pobrano i przesłano: {url}")
        else:
            failed_urls.append({"url": url, "error": upload_result["error"]})
            if debug_container:
                debug_container.warning(f"Błąd uploadu {url}: {upload_result['error']}")
    ftp_manager.close()
    return new_urls_map, downloaded_images, failed_urls

def extract_image_urls_from_xml(xml_content, xpath_expression, separator=","):
    try:
        if not xml_content or not xml_content.strip():
            return None, "Plik XML jest pusty"
        if xml_content.startswith("\ufeff"):
            xml_content = xml_content[1:]
        xml_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_content)
        is_attribute = '/@' in xpath_expression
        attribute_name = xpath_expression.split('/@')[-1] if is_attribute else None
        xpath_base = xpath_expression.split('/@')[0] if is_attribute else xpath_expression
        if xpath_base in ["//product/image", "product/image"]:
            try:
                pattern_simple = re.compile(r'<image>(.*?)</image>', re.DOTALL)
                pattern_cdata = re.compile(r'<image><!\[CDATA\[(.*?)\]\]></image>', re.DOTALL)
                matches = pattern_simple.findall(xml_content) + pattern_cdata.findall(xml_content)
                urls = []
                for match in matches:
                    match = match.strip()
                    if match and ('http://' in match or 'https://' in match):
                        urls.append(match.replace('&amp;', '&'))
                return urls, None
            except Exception as e:
                return None, f"Błąd przy parsowaniu XML: {str(e)}"
        try:
            root = ET.fromstring(xml_content)
            xpath = f"./{xpath_base[2:]}" if xpath_base.startswith('//') else f"./{xpath_base}" if not xpath_base.startswith('./') else xpath_base
            elements = root.findall(xpath)
            urls = []
            for element in elements:
                element_text = element.attrib.get(attribute_name) if is_attribute else element.text
                if element_text:
                    element_text = element_text.replace('&amp;', '&')
                    if 'http://' in element_text or 'https://' in element_text:
                        if separator in element_text:
                            urls.extend([url.strip() for url in element_text.split(separator) 
                                       if url.strip() and ('http://' in url or 'https://' in url)])
                        else:
                            urls.append(element_text.strip())
            return urls, None
        except ET.ParseError as e:
            return None, f"Błąd przy parsowaniu XML: {str(e)}"
    except Exception as e:
        return None, f"Nieoczekiwany błąd: {str(e)}"

def update_xml_with_new_urls(xml_content, xpath_expression, new_urls_map, new_node_name, separator=","):
    try:
        if not xml_content.strip() or not new_node_name.strip():
            return None, "Plik XML jest pusty lub nazwa węzła jest pusta"
        is_attribute = '/@' in xpath_expression
        attribute_name = xpath_expression.split('/@')[-1] if is_attribute else None
        xpath_base = xpath_expression.split('/@')[0] if is_attribute else xpath_expression
        root = ET.fromstring(xml_content)
        xpath = xpath_base[2:] if xpath_base.startswith('//') else xpath_base
        elements = root.findall(f'.//{xpath}')
        parent_map = {c: p for p in root.iter() for c in p}
        parent_to_elements = {}
        for element in elements:
            parent = parent_map.get(element)
            if parent is None:
                continue
            if parent not in parent_to_elements:
                parent_to_elements[parent] = []
            parent_to_elements[parent].append(element)
        for parent, elements_list in parent_to_elements.items():
            ftp_images = None
            for child in parent:
                if child.tag == "ftp_images":
                    ftp_images = child
                    break
            if ftp_images is None:
                ftp_images = ET.Element("ftp_images")
                parent.append(ftp_images)
            for element in elements_list:
                original_url = element.attrib.get(attribute_name, "").strip() if is_attribute else (element.text.strip() if element.text else "")
                if not original_url:
                    continue
                if separator in original_url:
                    urls = [url.strip() for url in original_url.split(separator)]
                    new_urls = [new_urls_map[url] for url in urls if url in new_urls_map]
                    if new_urls:
                        ftp_node = ET.Element(new_node_name)
                        ftp_node.text = separator.join(new_urls)
                        ftp_images.append(ftp_node)
                elif original_url in new_urls_map:
                    ftp_node = ET.Element(new_node_name)
                    ftp_node.text = new_urls_map[original_url]
                    ftp_images.append(ftp_node)
        return ET.tostring(root, encoding="unicode"), None
    except Exception as e:
        return None, f"Błąd przy aktualizacji XML: {str(e)}"

def extract_image_urls_from_csv(csv_content, column_name, separator=","):
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        if column_name not in df.columns:
            return None, f"Kolumna '{column_name}' nie istnieje w pliku CSV."
        urls = []
        for value in df[column_name]:
            if pd.notna(value):
                if separator in str(value):
                    urls.extend([url.strip() for url in str(value).split(separator) if url.strip()])
                else:
                    urls.append(str(value).strip())
        return urls, None
    except Exception as e:
        return None, f"Błąd przy parsowaniu CSV: {str(e)}"

def update_csv_with_new_urls(csv_content, column_name, new_urls_map, new_column_name, separator=","):
    try:
        if not new_column_name.strip():
            return None, "Nazwa nowej kolumny nie może być pusta"
        df = pd.read_csv(io.StringIO(csv_content))
        if column_name not in df.columns:
            return None, f"Kolumna '{column_name}' nie istnieje w pliku CSV."
        if new_column_name not in df.columns:
            df[new_column_name] = ""
        for idx, value in enumerate(df[column_name]):
            if pd.notna(value):
                value_str = str(value).strip()
                if separator in value_str:
                    urls = [url.strip() for url in value_str.split(separator)]
                    new_urls = [new_urls_map[url] for url in urls if url in new_urls_map]
                    if new_urls:
                        df.at[idx, new_column_name] = separator.join(new_urls)
                elif value_str in new_urls_map:
                    df.at[idx, new_column_name] = new_urls_map[value_str]
                else:
                    for key in new_urls_map:
                        if value_str.replace(" ", "") == key.replace(" ", ""):
                            df.at[idx, new_column_name] = new_urls_map[key]
                            break
        return df.to_csv(index=False), None
    except Exception as e:
        return None, f"Błąd przy aktualizacji CSV: {str(e)}"

def save_to_google_drive(output_bytes, file_info, new_urls_map=None):
    try:
        drive_folder_id = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
        credentials_json = st.secrets.get("GOOGLE_DRIVE_CREDENTIALS_JSON")
        if not drive_folder_id or not credentials_json:
            return False, "Brak konfiguracji Google Drive."
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"image_urls_map_{now}.txt"
        result_filename = f"processed_{now}.{file_info['type']}"
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_result_path = os.path.join(tmpdirname, f"output.{file_info['type']}")
            temp_log_path = os.path.join(tmpdirname, "log.txt")
            with open(temp_result_path, "wb") as f:
                f.write(output_bytes)
            log_content = f"# Raport z przetwarzania obrazów - {now}\n\n## Informacje o pliku\n"
            log_content += f"- Nazwa pliku: {file_info['name']}\n- Typ pliku: {file_info['type'].upper()}\n"
            log_content += f"- Kodowanie: {file_info['encoding']}\n\n## Mapowanie URL-i obrazów\n\n"
            if new_urls_map:
                for i, (original_url, new_url) in enumerate(new_urls_map.items(), 1):
                    log_content += f"### Obraz #{i}\n- Oryginalny URL: {original_url}\n- Nowy URL: {new_url}\n\n"
            else:
                log_content += "Brak mapowania URL-i\n"
            with open(temp_log_path, "w", encoding='utf-8') as f:
                f.write(log_content)
            with st.spinner("Zapisuję na Google Drive..."):
                if isinstance(credentials_json, str):
                    try:
                        creds_dict = json.loads(credentials_json)
                    except:
                        return False, "Błąd dekodowania JSON z credentials"
                else:
                    creds_dict = credentials_json
                scope = ["https://www.googleapis.com/auth/drive"]
                credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                gauth = GoogleAuth()
                gauth.credentials = credentials
                drive = GoogleDrive(gauth)
                try:
                    log_file = drive.CreateFile({
                        "title": log_filename, 
                        "parents": [{"id": drive_folder_id}],
                        "mimeType": "text/plain"
                    })
                    log_file.SetContentFile(temp_log_path)
                    log_file.Upload()
                    result_file = drive.CreateFile({
                        "title": result_filename, 
                        "parents": [{"id": drive_folder_id}],
                        "mimeType": f"application/{file_info['type']}"
                    })
                    result_file.SetContentFile(temp_result_path)
                    result_file.Upload()
                    return True, "Pliki zostały zapisane na Google Drive."
                except Exception as e:
                    return False, f"Błąd podczas wysyłania: {str(e)}"
    except Exception as e:
        return False, f"Błąd Google Drive: {str(e)}"

def save_processing_state(session_id, urls, processed_urls, new_urls_map, file_info, processing_params, failed_urls):
    state = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "file_info": {k: file_info[k] for k in ["name", "type", "encoding"]},
        "processing_params": processing_params,
        "total_urls": len(urls),
        "processed_urls": processed_urls,
        "new_urls_map": new_urls_map,
        "remaining_urls": [url for url in urls if url not in processed_urls and url not in new_urls_map],
        "failed_urls": failed_urls,
        "file_content": file_info.get("content", "")
    }
    state_dir = os.path.join(os.path.expanduser("~"), ".xml_image_processor")
    os.makedirs(state_dir, exist_ok=True)
    state_file = os.path.join(state_dir, f"session_{session_id}.json")
    with open(state_file, "w") as f:
        json.dump(state, f)
    fail_log_path = os.path.join(state_dir, f"fail_log_{session_id}.csv")
    with open(fail_log_path, "w", encoding="utf-8") as f:
        f.write("url,error\n")
        for fail in failed_urls:
            f.write(f"{fail['url']},{fail.get('error','')}\n")
    return state_file

def load_processing_state(session_id=None):
    state_dir = os.path.join(os.path.expanduser("~"), ".xml_image_processor")
    if not os.path.exists(state_dir):
        return None
    if session_id:
        state_file = os.path.join(state_dir, f"session_{session_id}.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                return json.load(f)
        return None
    state_files = [f for f in os.listdir(state_dir) if f.startswith("session_") and f.endswith(".json")]
    if not state_files:
        return None
    state_files.sort(key=lambda x: os.path.getmtime(os.path.join(state_dir, x)), reverse=True)
    with open(os.path.join(state_dir, state_files[0]), "r") as f:
        return json.load(f)

def list_saved_sessions():
    state_dir = os.path.join(os.path.expanduser("~"), ".xml_image_processor")
    if not os.path.exists(state_dir):
        return []
    sessions = []
    for filename in os.listdir(state_dir):
        if filename.startswith("session_") and filename.endswith(".json"):
            try:
                with open(os.path.join(state_dir, filename), "r") as f:
                    state = json.load(f)
                    progress = round(len(state["processed_urls"]) / state["total_urls"] * 100, 1) if state["total_urls"] > 0 else 0
                    sessions.append({
                        "session_id": state["session_id"],
                        "timestamp": state["timestamp"],
                        "file_info": state["file_info"]["name"],
                        "progress": f"{len(state['processed_urls'])}/{state['total_urls']}",
                        "percentage": progress
                    })
            except:
                pass
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return sessions

def resume_processing(state, temp_dir, ftp_settings):
    remaining_urls = state["failed_urls"] + [url for url in state["remaining_urls"] if url not in state["failed_urls"]]
    new_urls_map = state.get("new_urls_map", {})
    file_info = state["file_info"]
    processing_params = state["processing_params"]
    session_id = state["session_id"]
    if not remaining_urls:
        st.success("Wszystkie URL-e zostały już przetworzone.")
        return True
    progress_bar = st.progress(len(state["processed_urls"]) / state["total_urls"])
    status_text = st.empty()
    debug_area = st.empty()
    status_text.text(f"Wznawianie przetwarzania {len(state['processed_urls'])+1}-{state['total_urls']} z {state['total_urls']}...")
    processed_urls = state.get("processed_urls", [])
    batch_size = min(10, len(remaining_urls))
    all_downloaded = []
    failed_urls = state.get("failed_urls", [])
    for i in range(0, len(remaining_urls), batch_size):
        batch_urls = [u for u in remaining_urls[i:i+batch_size] if u not in processed_urls and u not in new_urls_map]
        status_text.text(f"Przetwarzanie {len(processed_urls)+i+1}-{len(processed_urls)+min(i+batch_size, len(remaining_urls))} z {state['total_urls']}...")
        progress_value = (len(processed_urls) + i) / state["total_urls"]
        progress_bar.progress(progress_value)
        batch_result, batch_downloaded, batch_failed = process_images_sequentially(
            batch_urls, 
            temp_dir, 
            ftp_settings,
            debug_container=debug_area
        )
        new_urls_map.update(batch_result)
        processed_urls.extend(batch_urls)
        all_downloaded.extend(batch_downloaded)
        failed_urls.extend(batch_failed)
        save_processing_state(
            session_id, 
            state["remaining_urls"] + state["processed_urls"], 
            processed_urls, 
            new_urls_map, 
            file_info,
            processing_params,
            failed_urls
        )
    progress_bar.progress(1.0)
    status_text.text(f"Zakończono wznowione przetwarzanie. Pobrano i przesłano {len(all_downloaded)} obrazów.")
    if new_urls_map:
        file_type = file_info["type"]
        file_content = state.get("file_content", "")
        if not file_content:
            st.error("Brak treści pliku w zapisanym stanie - nie można zaktualizować pliku.")
            return False
        if file_type == "xml":
            xpath = processing_params.get("xpath", "")
            new_node_name = processing_params.get("new_node_name", "ftp")
            separator = processing_params.get("separator", ",")
            updated_content, error = update_xml_with_new_urls(
                file_content, xpath, new_urls_map, new_node_name, separator
            )
        else:
            column_name = processing_params.get("column_name", "")
            new_column_name = processing_params.get("new_column_name", "ftp_url")
            separator = processing_params.get("separator", ",")
            updated_content, error = update_csv_with_new_urls(
                file_content, column_name, new_urls_map, new_column_name, separator
            )
        if error:
            st.error(f"Błąd podczas aktualizacji pliku: {error}")
            return False
        output_bytes = updated_content.encode(file_info["encoding"])
        st.success("Plik został pomyślnie zaktualizowany!")
        try:
            success, message = save_to_google_drive(output_bytes, file_info, new_urls_map)
            st.success(f"✅ {message}") if success else st.warning(f"⚠️ {message}")
        except Exception as e:
            st.error(f"Błąd Google Drive: {str(e)}")
        base_name = os.path.splitext(file_info["name"])[0]
        st.download_button(
            label=f"📁 Pobierz zaktualizowany plik",
            data=output_bytes,
            file_name=f"{base_name}_updated.{file_type}",
            mime="text/plain"
        )
        state_dir = os.path.join(os.path.expanduser("~"), ".xml_image_processor")
        fail_log_path = os.path.join(state_dir, f"fail_log_{session_id}.csv")
        if os.path.exists(fail_log_path):
            with open(fail_log_path, "rb") as f:
                st.download_button(
                    label="📄 Pobierz log błędów",
                    data=f.read(),
                    file_name=f"fail_log_{session_id}.csv",
                    mime="text/csv"
                )
        return True
    else:
        st.warning("Nie udało się przetworzyć żadnych nowych obrazów.")
        return False

def reset_app_state():
    for key in list(st.session_state.keys()):
        if key not in ["authenticated", "ftp_settings"]:
            del st.session_state[key]
    initialize_session_state()
    st.rerun()

def main():
    st.set_page_config(page_title="Pobieranie zdjęć z XML/CSV", layout="centered")
    authenticate_user()
    initialize_session_state()
    st.title("Pobieranie zdjęć z XML/CSV")
    tab1, tab2, tab3 = st.tabs(["Pobieranie zdjęć", "Wznów przetwarzanie", "Pomoc"])
    with tab1:
        st.markdown("""
        To narzędzie umożliwia pobieranie zdjęć z plików XML lub CSV i zapisywanie ich na serwerze FTP.
        Prześlij plik, wskaż lokalizację linków do zdjęć, podaj dane FTP i pobierz zdjęcia.
        """)
        st.subheader("1. Wczytaj plik źródłowy")
        uploaded_file = st.file_uploader("Wgraj plik XML lub CSV", type=["xml", "csv"])
        if uploaded_file:
            file_info, error = read_file_content(uploaded_file)
            if error:
                st.error(error)
            else:
                st.success(f"Wczytano plik: {file_info['name']} ({file_info['type'].upper()}, {file_info['encoding']})")
                st.session_state.file_info = file_info
        st.subheader("2. Konfiguracja pobierania zdjęć")
        if st.session_state.file_info:
            file_type = st.session_state.file_info["type"]
            if file_type == "xml":
                xpath = st.text_input("XPath do węzła zawierającego URL-e zdjęć", 
                                    placeholder="Np. //product/image lub //image/@url")
                new_node_name = st.text_input("Nazwa nowego węzła dla linków FTP", 
                                            placeholder="Np. ftp")
                st.session_state.processing_params["xpath"] = xpath
                st.session_state.processing_params["new_node_name"] = new_node_name
            else:
                column_name = st.text_input("Nazwa kolumny zawierającej URL-e zdjęć", 
                                          placeholder="Np. image_url")
                new_column_name = st.text_input("Nazwa nowej kolumny dla linków FTP", 
                                              placeholder="Np. ftp_image_url")
                st.session_state.processing_params["column_name"] = column_name
                st.session_state.processing_params["new_column_name"] = new_column_name
            separator = st.text_input("Separator URL-i (jeśli w jednej komórce/węźle jest wiele linków)", value=",")
            st.session_state.processing_params["separator"] = separator
        st.subheader("3. Konfiguracja serwera FTP")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.ftp_settings["host"] = st.text_input("Adres serwera FTP", 
                                                                 value=st.session_state.ftp_settings["host"])
            st.session_state.ftp_settings["port"] = st.number_input("Port", value=st.session_state.ftp_settings["port"],
                                                                   min_value=1, max_value=65535)
            st.session_state.ftp_settings["directory"] = st.text_input("Katalog docelowy", 
                                                                      value=st.session_state.ftp_settings["directory"])
            st.session_state.ftp_settings["http_path"] = st.text_input("Ścieżka HTTP do zdjęć",
                                                                      value=st.session_state.ftp_settings.get("http_path", ""),
                                                                      placeholder="https://example.com/images/")
        with col2:
            st.session_state.ftp_settings["username"] = st.text_input("Nazwa użytkownika", 
                                                                     value=st.session_state.ftp_settings["username"])
            st.session_state.ftp_settings["password"] = st.text_input("Hasło", type="password", 
                                                                     value=st.session_state.ftp_settings["password"])
        st.subheader("4. Pobierz zdjęcia i prześlij na FTP")
        if st.session_state.file_info:
            max_workers = 1
        if st.session_state.file_info and st.button("Pobierz zdjęcia i prześlij na FTP"):
            file_type = st.session_state.file_info["type"]
            file_content = st.session_state.file_info["content"]
            if file_type == "xml":
                xpath = st.session_state.processing_params.get("xpath", "")
                new_node_name = st.session_state.processing_params.get("new_node_name", "")
                separator = st.session_state.processing_params.get("separator", ",")
                if not xpath or not xpath.strip() or not new_node_name or not new_node_name.strip():
                    st.error("Podaj prawidłowy XPath i nazwę nowego węzła!")
                    st.stop()
                urls, error = extract_image_urls_from_xml(file_content, xpath, separator)
            elif file_type == "csv":
                column_name = st.session_state.processing_params.get("column_name", "")
                new_column_name = st.session_state.processing_params.get("new_column_name", "")
                separator = st.session_state.processing_params.get("separator", ",")
                if not column_name or not column_name.strip() or not new_column_name or not new_column_name.strip():
                    st.error("Podaj prawidłową nazwę kolumny i nazwę nowej kolumny!")
                    st.stop()
                urls, error = extract_image_urls_from_csv(file_content, column_name, separator)
            else:
                urls, error = None, "Nie podano ścieżki XPath lub nazwy kolumny."
            if error:
                st.error(error)
            elif not urls:
                st.warning("Nie znaleziono żadnych URL-i zdjęć.")
            else:
                st.success(f"Znaleziono {len(urls)} URL-i zdjęć")
                with st.expander("Podgląd znalezionych URL-i"):
                    for i, url in enumerate(urls[:5]):
                        st.write(f"{i+1}. {url}")
                    if len(urls) > 5:
                        st.write(f"... oraz {len(urls)-5} więcej.")
                if not st.session_state.ftp_settings["host"] or not st.session_state.ftp_settings["username"]:
                    st.error("Podaj dane serwera FTP.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    debug_area = st.empty()
                    session_id = f"{uuid.uuid4().hex}_{int(time.time())}"
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        processed_urls = []
                        new_urls_map = {}
                        failed_urls = []
                        batch_size = 10
                        for i in range(0, len(urls), batch_size):
                            batch_urls = [u for u in urls[i:i+batch_size] if u not in processed_urls and u not in new_urls_map]
                            status_text.text(f"Przetwarzanie obrazów {i+1}-{min(i+batch_size, len(urls))} z {len(urls)}...")
                            progress_bar.progress(i / len(urls))
                            batch_result, _, batch_failed = process_images_sequentially(
                                batch_urls, 
                                tmpdirname, 
                                st.session_state.ftp_settings,
                                debug_container=debug_area
                            )
                            new_urls_map.update(batch_result)
                            processed_urls.extend(batch_urls)
                            failed_urls.extend(batch_failed)
                            save_processing_state(
                                session_id, urls, processed_urls, new_urls_map, 
                                st.session_state.file_info, st.session_state.processing_params,
                                failed_urls
                            )
                        progress_bar.progress(1.0)
                        status_text.text(f"Zakończono przetwarzanie. Pobrano i przesłano {len(new_urls_map)} z {len(urls)} obrazów.")
                        if new_urls_map:
                            if file_type == "xml":
                                updated_content, error = update_xml_with_new_urls(
                                    file_content, xpath, new_urls_map, new_node_name, separator
                                )
                            else:
                                updated_content, error = update_csv_with_new_urls(
                                    file_content, column_name, new_urls_map, new_column_name, separator
                                )
                            if error:
                                st.error(f"Błąd podczas aktualizacji pliku: {error}")
                            else:
                                st.session_state.output_bytes = updated_content.encode(
                                    st.session_state.file_info["encoding"]
                                )
                                st.success("Plik został zaktualizowany o nowe linki FTP.")
                                try:
                                    success, message = save_to_google_drive(
                                        st.session_state.output_bytes,
                                        st.session_state.file_info,
                                        new_urls_map
                                    )
                                    if success:
                                        st.success(f"✅ {message}")
                                    else:
                                        st.warning(f"⚠️ {message}")
                                except Exception as e:
                                    st.error(f"Błąd Google Drive: {str(e)}")
                                original_name = st.session_state.file_info["name"]
                                base_name = os.path.splitext(original_name)[0]
                                st.download_button(
                                    label="📁 Pobierz zaktualizowany plik",
                                    data=st.session_state.output_bytes,
                                    file_name=f"{base_name}_updated.{file_type}",
                                    mime="text/plain"
                                )
                                state_dir = os.path.join(os.path.expanduser("~"), ".xml_image_processor")
                                fail_log_path = os.path.join(state_dir, f"fail_log_{session_id}.csv")
                                if os.path.exists(fail_log_path):
                                    with open(fail_log_path, "rb") as f:
                                        st.download_button(
                                            label="📄 Pobierz log błędów",
                                            data=f.read(),
                                            file_name=f"fail_log_{session_id}.csv",
                                            mime="text/csv"
                                        )
                        if st.button("Rozpocznij nową operację"):
                            reset_app_state()
    with tab2:
        st.subheader("Wznów wcześniej przerwane przetwarzanie")
        saved_sessions = list_saved_sessions()
        if not saved_sessions:
            st.info("Nie znaleziono zapisanych sesji przetwarzania.")
        else:
            st.write("Wybierz sesję do wznowienia:")
            sessions_df = pd.DataFrame(saved_sessions)
            if not sessions_df.empty:
                st.dataframe(sessions_df[["timestamp", "file_info", "progress", "percentage"]])
                selected_session_id = st.selectbox(
                    "Wybierz ID sesji do wznowienia:", 
                    options=[s["session_id"] for s in saved_sessions],
                    format_func=lambda x: f"{next((s['timestamp'] for s in saved_sessions if s['session_id'] == x), '')} - {next((s['file_info'] for s in saved_sessions if s['session_id'] == x), '')}"
                )
                if st.button("Wznów przetwarzanie"):
                    state = load_processing_state(selected_session_id)
                    if state:
                        with tempfile.TemporaryDirectory() as tmpdirname:
                            resume_processing(state, tmpdirname, st.session_state.ftp_settings)
                    else:
                        st.error("Nie udało się załadować stanu sesji.")
    with tab3:
        st.markdown("""
        ### Jak korzystać z aplikacji

        1. **Wgraj plik XML lub CSV** - aplikacja automatycznie wykryje kodowanie
        2. **Skonfiguruj pobieranie zdjęć**:
           - Dla XML: Podaj XPath do węzła zawierającego URL-e zdjęć
           - Dla CSV: Podaj nazwę kolumny zawierającej URL-e zdjęć
           - Określ separator, jeśli w jednej komórce/węźle znajduje się wiele URL-i
        3. **Skonfiguruj serwer FTP** - podaj dane dostępowe do serwera FTP
           - Podaj ścieżkę HTTP, pod którą będą dostępne zdjęcia (np. https://example.com/images/)
        4. **Pobierz zdjęcia i prześlij na FTP** - aplikacja pobierze zdjęcia i prześle je na serwer FTP
           - Możesz dostosować liczbę równoległych procesów pobierania dla przyspieszenia procesu
        5. **Pobierz zaktualizowany plik** - plik źródłowy zostanie zaktualizowany o nowe linki HTTP/FTP
        6. **Wznawianie przerwanego procesu**:
           - W przypadku błędu lub przerwania procesu, przejdź do zakładki "Wznów przetwarzanie"
           - Wybierz odpowiednią sesję i kontynuuj pobieranie od miejsca przerwania

        ### Przykłady konfiguracji

        #### XML

        - XPath: `//product/image`
        - Nazwa nowego węzła: `ftp`
        - Ścieżka HTTP: `https://example.com/images/`

        #### CSV

        - Nazwa kolumny: `image_url`
        - Nazwa nowej kolumny: `ftp_image_url`
        - Ścieżka HTTP: `https://example.com/images/`

        ### Obsługiwane formaty

        - **XML** - pliki XML z linkami do zdjęć w określonych węzłach
        - **CSV** - pliki CSV z linkami do zdjęć w określonych kolumnach
        """)

if __name__ == "__main__":
    main()
