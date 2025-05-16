import streamlit as st
import requests
import tempfile
import os
import re
import traceback
import json
from datetime import datetime
import pandas as pd
import io
import ftplib
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, parse_qs
import shutil
import uuid
import codecs
from bs4 import BeautifulSoup

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
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = ""
    if "edited_code" not in st.session_state:
        st.session_state.edited_code = ""
    if "output_bytes" not in st.session_state:
        st.session_state.output_bytes = None
    if "file_info" not in st.session_state:
        st.session_state.file_info = None
    if "show_editor" not in st.session_state:
        st.session_state.show_editor = False
    if "error_info" not in st.session_state:
        st.session_state.error_info = None
    if "code_fixed" not in st.session_state:
        st.session_state.code_fixed = False
    if "fix_requested" not in st.session_state:
        st.session_state.fix_requested = False
    if "downloaded_images" not in st.session_state:
        st.session_state.downloaded_images = []
    if "ftp_settings" not in st.session_state:
        st.session_state.ftp_settings = {
            "host": "",
            "port": 21,
            "username": "",
            "password": "",
            "directory": "/",
            "url_path": ""  # Dodane nowe pole dla ścieżki URL
        }

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
                    if raw_bytes.startswith(codecs.BOM_UTF16_LE):
                        file_contents = raw_bytes.decode('utf-16-le')
                    else:
                        file_contents = raw_bytes.decode('utf-16-be')

                    return {"content": file_contents, "raw_bytes": raw_bytes, 
                            "type": file_type, "encoding": 'utf-16', "name": uploaded_file.name}, None
                except UnicodeDecodeError:
                    pass

            encoding_declared = re.search(br'<\?xml[^>]*encoding=["\']([^"\']+)["\']', raw_bytes)
            if encoding_declared:
                declared_encoding = encoding_declared.group(1).decode('ascii').lower()
                try:
                    if declared_encoding.startswith('utf-16'):
                        try:
                            file_contents = raw_bytes.decode('utf-16-le')
                        except UnicodeDecodeError:
                            file_contents = raw_bytes.decode('utf-16-be')
                    else:
                        file_contents = raw_bytes.decode(declared_encoding)

                    return {"content": file_contents, "raw_bytes": raw_bytes, 
                            "type": file_type, "encoding": declared_encoding, "name": uploaded_file.name}, None
                except (UnicodeDecodeError, LookupError):
                    pass

        encodings_to_try = ["utf-8", "iso-8859-2", "windows-1250", "utf-16-le", "utf-16-be"]

        for enc in encodings_to_try:
            try:
                file_contents = raw_bytes.decode(enc)
                return {"content": file_contents, "raw_bytes": raw_bytes, 
                        "type": file_type, "encoding": enc, "name": uploaded_file.name}, None
            except UnicodeDecodeError:
                continue

        if file_type == "csv":
            try:
                buffer = io.BytesIO(raw_bytes)
                df = pd.read_csv(buffer, sep=None, engine='python')
                file_contents = df.to_csv(index=False)
                return {"content": file_contents, "raw_bytes": raw_bytes, 
                        "type": file_type, "encoding": "auto-detected", 
                        "name": uploaded_file.name, "dataframe": df}, None
            except Exception:
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
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Referer": f"{parsed_url.scheme}://{parsed_url.netloc}/",
        }

        # Obsługa image_show.php z RM Gastro
        if "image_show.php" in url:
            html_resp = requests.get(url, headers=headers, timeout=10)
            html_resp.raise_for_status()

            soup = BeautifulSoup(html_resp.text, "html.parser")
            img_tag = soup.find("img")

            if not img_tag or not img_tag.get("src"):
                return None, "Nie znaleziono znacznika <img> w odpowiedzi HTML"

            # Budujemy pełny URL do obrazka
            img_src = img_tag["src"]
            if not img_src.startswith("http"):
                img_url = f"{parsed_url.scheme}://{parsed_url.netloc}/{img_src.lstrip('/')}"
            else:
                img_url = img_src
        else:
            img_url = url  # standardowy przypadek

        # Pobieramy właściwy obraz
        response = requests.get(img_url, headers=headers, stream=True, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return None, f"Nieprawidłowy Content-Type: {content_type}"

        # Nadajemy nazwę pliku
        extension = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp"
        }.get(content_type, ".jpg")

        filename = f"image_{uuid.uuid4().hex}{extension}"
        file_path = os.path.join(temp_dir, filename)

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

        if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
            return {"path": file_path, "filename": filename, "original_url": url}, None
        else:
            return None, "Pobrano pusty lub niepełny plik"

    except Exception as e:
        return None, f"Błąd: {str(e)}"


def upload_to_ftp(file_path, ftp_settings, remote_filename=None):
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"Plik nie istnieje: {file_path}"}

        if os.path.getsize(file_path) == 0:
            return {"success": False, "error": f"Plik jest pusty: {file_path}"}

        ftp = ftplib.FTP()
        ftp.connect(ftp_settings["host"], ftp_settings["port"])
        ftp.login(ftp_settings["username"], ftp_settings["password"])

        if ftp_settings["directory"] and ftp_settings["directory"] != "/":
            try:
                ftp.cwd(ftp_settings["directory"])
            except ftplib.error_perm:
                dirs = ftp_settings["directory"].strip("/").split("/")
                for directory in dirs:
                    if directory:
                        try:
                            ftp.cwd(directory)
                        except ftplib.error_perm:
                            ftp.mkd(directory)
                            ftp.cwd(directory)

        if not remote_filename:
            remote_filename = os.path.basename(file_path)

        with open(file_path, 'rb') as file:
            ftp.storbinary(f'STOR {remote_filename}', file)

        # Tworzenie URL do zdjęcia - użyj zdefiniowanej ścieżki URL jeśli istnieje
        if ftp_settings.get("url_path") and ftp_settings["url_path"].strip():
            # Upewnij się, że ścieżka kończy się slashem
            url_path = ftp_settings["url_path"]
            if not url_path.endswith('/'):
                url_path += '/'
            image_url = f"{url_path}{remote_filename}"
        else:
            # Fallback do FTP URL jeśli nie podano ścieżki URL
            image_url = f"ftp://{ftp_settings['host']}"
            if ftp_settings["directory"] and ftp_settings["directory"] != "/":
                if not ftp_settings["directory"].startswith("/"):
                    image_url += "/"
                image_url += ftp_settings["directory"]
                if not image_url.endswith("/"):
                    image_url += "/"
            else:
                image_url += "/"
            image_url += remote_filename

        ftp.quit()

        return {"success": True, "url": image_url, "filename": remote_filename}

    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_image_urls_from_xml(xml_content, xpath_expression, separator=","):
    try:
        if not xml_content or not xml_content.strip():
            return None, "Plik XML jest pusty"

        if xml_content.startswith("\ufeff"):
            xml_content = xml_content[1:]

        xml_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_content)

        if xpath_expression == "//product/image" or xpath_expression == "product/image":
            try:
                pattern = re.compile(r'<image>(.*?)</image>', re.DOTALL)
                matches = pattern.findall(xml_content)

                urls = []
                for match in matches:
                    match = match.strip()
                    if match and ('http://' in match or 'https://' in match):
                        match = match.replace('&amp;', '&')
                        urls.append(match)

                return urls, None
            except Exception as e:
                return None, f"Błąd podczas parsowania XML z wyrażeniami regularnymi: {str(e)}"

        try:
            root = ET.fromstring(xml_content)

            if xpath_expression.startswith('//'):
                xpath_expression = f"./{xpath_expression[2:]}"
            elif not xpath_expression.startswith('./'):
                xpath_expression = f"./{xpath_expression}"

            elements = root.findall(xpath_expression)

            urls = []
            for element in elements:
                element_text = element.text
                if element_text:
                    element_text = element_text.replace('&amp;', '&')
                    if 'http://' in element_text or 'https://' in element_text:
                        if separator in element_text:
                            for url in element_text.split(separator):
                                url = url.strip()
                                if url and ('http://' in url or 'https://' in url):
                                    urls.append(url)
                        else:
                            urls.append(element_text.strip())

            return urls, None
        except ET.ParseError as e:
            return None, f"Błąd podczas parsowania XML z ElementTree: {str(e)}"

    except Exception as e:
        return None, f"Nieoczekiwany błąd: {str(e)}"

def update_xml_with_new_urls(xml_content, xpath_expression, new_urls_map, new_node_name):
    try:
        if not xml_content or not xml_content.strip():
            return None, "Plik XML jest pusty"

        if xml_content.startswith("\ufeff"):
            xml_content = xml_content[1:]

        xml_content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", xml_content)

        if xpath_expression == "//product/image" or xpath_expression == "product/image":
            try:
                updated_xml = xml_content

                for original_url, new_url in new_urls_map.items():
                    escaped_url = re.escape(original_url)
                    pattern = f"(<image>{escaped_url}</image>)"
                    replacement = f"\\1\n<{new_node_name}>{new_url}</{new_node_name}>"
                    updated_xml = re.sub(pattern, replacement, updated_xml)

                return updated_xml, None
            except Exception as e:
                return (
                    None,
                    f"Błąd podczas aktualizacji XML z wyrażeniami regularnymi: {str(e)}",
                )

        try:
            parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
            root = ET.fromstring(xml_content, parser=parser)

            if xpath_expression.startswith("//"):
                xpath_parts = xpath_expression[2:].split("/")
                search_path = ".//" + xpath_parts[-1]
            elif not xpath_expression.startswith("./"):
                search_path = f"./{xpath_expression}"
            else:
                search_path = xpath_expression

            for element in root.findall(search_path):
                element_text = element.text.strip() if element.text else ""

                if element_text in new_urls_map:
                    parent = None
                    for p in root.iter():
                        if element in list(p):
                            parent = p
                            break

                    if parent is not None:
                        new_element = ET.Element(new_node_name)
                        new_element.text = new_urls_map[element_text]

                        idx = list(parent).index(element)
                        parent.insert(idx + 1, new_element)

            return ET.tostring(root, encoding="unicode", method="xml"), None
        except ET.ParseError as e:
            return None, f"Błąd podczas aktualizacji XML z ElementTree: {str(e)}"

    except Exception as e:
        return None, f"Nieoczekiwany błąd: {str(e)}"

def extract_image_urls_from_csv(csv_content, column_name, separator=","):
    try:
        df = pd.read_csv(io.StringIO(csv_content))

        if column_name not in df.columns:
            return None, f"Kolumna '{column_name}' nie istnieje w pliku CSV."

        urls = []

        for value in df[column_name]:
            if pd.notna(value):
                if separator in str(value):
                    for url in str(value).split(separator):
                        url = url.strip()
                        if url:
                            urls.append(url)
                else:
                    urls.append(str(value).strip())

        return urls, None

    except Exception as e:
        return None, f"Błąd podczas parsowania CSV: {str(e)}"

def update_csv_with_new_urls(csv_content, column_name, new_urls_map, new_column_name):
    try:
        df = pd.read_csv(io.StringIO(csv_content))

        if column_name not in df.columns:
            return None, f"Kolumna '{column_name}' nie istnieje w pliku CSV."

        # Dodaj nową kolumnę
        df[new_column_name] = ""

        # Dla każdego wiersza sprawdź, czy URL w kolumnie źródłowej ma odpowiednik w mapie nowych URL-i
        for idx, row in df.iterrows():
            if pd.notna(row[column_name]) and row[column_name] in new_urls_map:
                df.at[idx, new_column_name] = new_urls_map[row[column_name]]

        return df.to_csv(index=False), None

    except Exception as e:
        return None, f"Błąd podczas aktualizacji CSV: {str(e)}"

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

    tab1, tab2 = st.tabs(["Pobieranie zdjęć", "Pomoc"])

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
                xpath = st.text_input(
                    "XPath do węzła zawierającego URL-e zdjęć", 
                    placeholder="Np. //product/image"
                )
                new_node_name = st.text_input(
                    "Nazwa nowego węzła dla linków FTP", 
                    placeholder="Np. ftp_image"
                )
            else:
                column_name = st.text_input(
                    "Nazwa kolumny zawierającej URL-e zdjęć", 
                    placeholder="Np. image_url"
                )
                new_column_name = st.text_input(
                    "Nazwa nowej kolumny dla linków FTP", 
                    placeholder="Np. ftp_image_url"
                )

            separator = st.text_input(
                "Separator URL-i (jeśli w jednej komórce/węźle jest wiele linków)", 
                value=","
            )

        st.subheader("3. Konfiguracja serwera FTP")

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.ftp_settings["host"] = st.text_input(
                "Adres serwera FTP", 
                value=st.session_state.ftp_settings["host"]
            )
            st.session_state.ftp_settings["port"] = st.number_input(
                "Port", 
                value=st.session_state.ftp_settings["port"],
                min_value=1, 
                max_value=65535
            )
            st.session_state.ftp_settings["directory"] = st.text_input(
                "Katalog docelowy", 
                value=st.session_state.ftp_settings["directory"]
            )
            # Nowe pole dla ścieżki URL
            st.session_state.ftp_settings["url_path"] = st.text_input(
                "Ścieżka URL do zdjęć (np. https://geronimo.hosting24.pl/sellstar.pl/getphoto/)", 
                value=st.session_state.ftp_settings.get("url_path", "")
            )

        with col2:
            st.session_state.ftp_settings["username"] = st.text_input(
                "Nazwa użytkownika", 
                value=st.session_state.ftp_settings["username"]
            )
            st.session_state.ftp_settings["password"] = st.text_input(
                "Hasło", 
                type="password", 
                value=st.session_state.ftp_settings["password"]
            )

        st.subheader("4. Pobierz zdjęcia i prześlij na FTP")

        if st.session_state.file_info and st.button("Pobierz zdjęcia i prześlij na FTP"):
            file_type = st.session_state.file_info["type"]
            file_content = st.session_state.file_info["content"]

            if file_type == "xml" and xpath:
                urls, error = extract_image_urls_from_xml(file_content, xpath, separator)
            elif file_type == "csv" and column_name:
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

                    with tempfile.TemporaryDirectory() as tmpdirname:
                        downloaded_images = []
                        new_urls_map = {}

                        for i, url in enumerate(urls):
                            status_text.text(f"Przetwarzanie {i+1}/{len(urls)}: {url}")
                            progress_bar.progress((i) / len(urls))

                            debug_area.info(f"Pobieranie obrazu: {url}")
                            image_info, error = download_image(url, tmpdirname)

                            if error:
                                debug_area.warning(f"Nie udało się pobrać {url}: {error}")
                                continue
                            else:
                                debug_area.success(f"Pobrano obraz: {image_info['filename']} ({os.path.getsize(image_info['path'])} bajtów)")

                            debug_area.info(f"Przesyłanie na FTP: {image_info['filename']}")
                            upload_result = upload_to_ftp(
                                image_info["path"], 
                                st.session_state.ftp_settings
                            )

                            if upload_result["success"]:
                                downloaded_images.append({
                                    "original_url": url,
                                    "ftp_url": upload_result["url"],
                                    "filename": upload_result["filename"]
                                })

                                debug_area.success(f"Przesłano na FTP: {upload_result['filename']}")

                                # Używamy nowego URL zwróconego przez funkcję upload_to_ftp
                                new_urls_map[url] = upload_result["url"]
                            else:
                                debug_area.warning(f"Nie udało się przesłać {image_info['filename']} na FTP: {upload_result['error']}")

                        progress_bar.progress(1.0)
                        status_text.text(f"Zakończono przetwarzanie. Pobrano i przesłano {len(downloaded_images)} z {len(urls)} zdjęć.")
                        debug_area.empty()

                        st.session_state.downloaded_images = downloaded_images

                        if new_urls_map:
                            if file_type == "xml":
                                updated_content, error = update_xml_with_new_urls(
                                    file_content, 
                                    xpath, 
                                    new_urls_map, 
                                    new_node_name
                                )
                            else:
                                updated_content, error = update_csv_with_new_urls(
                                    file_content, 
                                    column_name, 
                                    new_urls_map, 
                                    new_column_name
                                )

                            if error:
                                st.error(f"Błąd podczas aktualizacji pliku: {error}")
                            else:
                                st.session_state.output_bytes = updated_content.encode(
                                    st.session_state.file_info["encoding"]
                                )
                                st.success("Plik został zaktualizowany o nowe linki FTP.")

                                original_name = st.session_state.file_info["name"]
                                base_name = os.path.splitext(original_name)[0]

                                st.download_button(
                                    label=f"📁 Pobierz zaktualizowany plik",
                                    data=st.session_state.output_bytes,
                                    file_name=f"{base_name}_updated.{file_type}",
                                    mime="text/plain"
                                )

                        if st.button("Rozpocznij nową operację"):
                            reset_app_state()

    with tab2:
        st.markdown("""
        ### Jak korzystać z aplikacji

        1. **Wgraj plik XML lub CSV** - aplikacja automatycznie wykryje kodowanie
        2. **Skonfiguruj pobieranie zdjęć**:
           - Dla XML: Podaj XPath do węzła zawierającego URL-e zdjęć
           - Dla CSV: Podaj nazwę kolumny zawierającej URL-e zdjęć
           - Określ separator, jeśli w jednej komórce/węźle znajduje się wiele URL-i
        3. **Skonfiguruj serwer FTP** - podaj dane dostępowe do serwera FTP
        4. **Pobierz zdjęcia i prześlij na FTP** - aplikacja pobierze zdjęcia i prześle je na serwer FTP
        5. **Pobierz zaktualizowany plik** - plik źródłowy zostanie zaktualizowany o nowe linki FTP

        ### Przykłady konfiguracji

        #### XML

        - XPath: `//product/image`
        - Nazwa nowego węzła: `ftp_image`

        #### CSV

        - Nazwa kolumny: `image_url`
        - Nazwa nowej kolumny: `ftp_image_url`

        ### Obsługiwane formaty

        - **XML** - pliki XML z linkami do zdjęć w określonych węzłach
        - **CSV** - pliki CSV z linkami do zdjęć w określonych kolumnach
        """)

if __name__ == "__main__":
    main()
